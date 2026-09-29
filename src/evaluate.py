"""
Comprehensive evaluation metrics module for Image Captioning.
Calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, and ROUGE-L against 5 reference captions.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.config import Config
from src.utils import setup_logger, save_json

logger = setup_logger("Evaluation")


def compute_metrics(
    predictions: List[str],
    references_list: List[List[str]]
) -> Dict[str, float]:
    """
    Compute official NLP metrics comparing predicted captions against multiple references.
    
    Args:
        predictions: List of generated caption strings (len = N).
        references_list: List of lists containing ground truth captions (len = N, each inner list len = 5).
        
    Returns:
        Dictionary containing BLEU-1..4, METEOR, and ROUGE-L scores (percentage scale: 0 - 100).
    """
    assert len(predictions) == len(references_list), "Mismatch between predictions and references count!"
    
    # Tokenize hypotheses and references for BLEU & METEOR
    tokenized_preds = [pred.lower().strip().split() for pred in predictions]
    tokenized_refs = [
        [ref.lower().strip().split() for ref in refs]
        for refs in references_list
    ]
    
    # BLEU Scores with Smoothing Function
    smoothing = SmoothingFunction().method1
    
    # Weights for n-grams
    w_bleu1 = (1.0, 0, 0, 0)
    w_bleu2 = (0.5, 0.5, 0, 0)
    w_bleu3 = (0.333, 0.333, 0.333, 0)
    w_bleu4 = (0.25, 0.25, 0.25, 0.25)
    
    b1 = corpus_bleu(tokenized_refs, tokenized_preds, weights=w_bleu1, smoothing_function=smoothing) * 100.0
    b2 = corpus_bleu(tokenized_refs, tokenized_preds, weights=w_bleu2, smoothing_function=smoothing) * 100.0
    b3 = corpus_bleu(tokenized_refs, tokenized_preds, weights=w_bleu3, smoothing_function=smoothing) * 100.0
    b4 = corpus_bleu(tokenized_refs, tokenized_preds, weights=w_bleu4, smoothing_function=smoothing) * 100.0
    
    # METEOR Score (Word-level precision/recall with stemming & synonym matching)
    meteor_scores = []
    for pred_tokens, refs_tokens in zip(tokenized_preds, tokenized_refs):
        # Best meteor score across the 5 references
        score = meteor_score(refs_tokens, pred_tokens)
        meteor_scores.append(score)
    mean_meteor = float(np.mean(meteor_scores) * 100.0) if meteor_scores else 0.0
    
    # ROUGE-L (Longest Common Subsequence)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []
    for pred, refs in zip(predictions, references_list):
        # Compute maximum ROUGE-L against 5 references
        best_r_l = max(scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs)
        rouge_l_scores.append(best_r_l)
    mean_rouge_l = float(np.mean(rouge_l_scores) * 100.0) if rouge_l_scores else 0.0
    
    return {
        "BLEU-1": round(b1, 2),
        "BLEU-2": round(b2, 2),
        "BLEU-3": round(b3, 2),
        "BLEU-4": round(b4, 2),
        "METEOR": round(mean_meteor, 2),
        "ROUGE-L": round(mean_rouge_l, 2),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    method: str = "beam",
    num_beams: int = 5,
    max_samples: Optional[int] = None
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate model over an entire evaluation DataLoader.
    
    Returns:
        metrics_dict, detailed_predictions_list
    """
    model.eval()
    all_predictions = []
    all_references = []
    detailed_results = []
    
    total_processed = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({method}, beams={num_beams})"):
            pixel_values = batch["pixel_values"].to(device)
            image_names = batch["image_names"]
            references = batch["references"]
            
            # Generate predictions
            if hasattr(model, "generate_caption"):
                preds = model.generate_caption(
                    pixel_values,
                    method=method,
                    num_beams=num_beams,
                    device=device
                )
            else:
                # Raw model fallback
                out = model.generate(pixel_values=pixel_values, num_beams=num_beams, max_length=32)
                preds = [model.processor.decode(o, skip_special_tokens=True).strip() for o in out]
                
            if isinstance(preds, str):
                preds = [preds]
                
            all_predictions.extend(preds)
            all_references.extend(references)
            
            for img_name, pred, refs in zip(image_names, preds, references):
                detailed_results.append({
                    "image": img_name,
                    "prediction": pred,
                    "references": refs
                })
                
            total_processed += len(preds)
            if max_samples and total_processed >= max_samples:
                all_predictions = all_predictions[:max_samples]
                all_references = all_references[:max_samples]
                detailed_results = detailed_results[:max_samples]
                break
                
    metrics = compute_metrics(all_predictions, all_references)
    return metrics, detailed_results


def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format multiple model evaluation results as a Markdown table."""
    headers = ["Model / Setup", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"]
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    rows = []
    for model_name, metrics in results.items():
        row = [
            model_name,
            f"{metrics.get('BLEU-1', 0.0):.2f}",
            f"{metrics.get('BLEU-2', 0.0):.2f}",
            f"{metrics.get('BLEU-3', 0.0):.2f}",
            f"{metrics.get('BLEU-4', 0.0):.2f}",
            f"{metrics.get('METEOR', 0.0):.2f}",
            f"{metrics.get('ROUGE-L', 0.0):.2f}",
        ]
        rows.append("| " + " | ".join(row) + " |")
        
    return "\n".join([header_line, divider_line] + rows)


if __name__ == "__main__":
    from src.model import BLIPCaptioningModel
    from src.dataset import get_dataloaders
    
    device = Config.get_device()
    logger.info(f"Using device: {device}")
    
    # Evaluate Zero-shot Baseline
    logger.info("Initializing Baseline Salesforce/blip-image-captioning-base...")
    model = BLIPCaptioningModel().to(device)
    
    try:
        _, _, test_loader = get_dataloaders()
        logger.info(f"Loaded Test DataLoader with {len(test_loader.dataset)} samples.")
        metrics, details = evaluate_model(model, test_loader, device=device, num_beams=5)
        logger.info(f"Baseline Results:\n{format_metrics_table({'BLIP Zero-Shot': metrics})}")
        
        save_json(metrics, Config.paths.OUTPUTS_DIR / "baseline_metrics.json")
        save_json(details, Config.paths.OUTPUTS_DIR / "baseline_predictions.json")
    except Exception as e:
        logger.warning(f"Could not run test evaluation (dataset might need setup): {e}")
