"""
Evaluation script for ViT-GPT2 on Flickr8k test set.
Computes BLEU-1..4, METEOR, and ROUGE-L using the identical protocol as BLIP.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.vit_gpt2_captioner import ViTGPT2Captioner
from src.evaluate import compute_metrics
from src.utils import setup_logger

logger = setup_logger("ViTGPT2Evaluation")


def load_flickr8k_references(captions_file: Path) -> Dict[str, List[str]]:
    """Load image-to-captions mapping from captions.txt."""
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found at: {captions_file}")

    df = pd.read_csv(captions_file)
    image_to_captions = {}
    for _, row in df.iterrows():
        img = str(row["image"]).strip()
        cap = str(row["caption"]).strip()
        image_to_captions.setdefault(img, []).append(cap)
    return image_to_captions


def run_evaluation(
    num_samples: Optional[int] = 10,
    num_beams: int = 3,
    max_length: int = 32,
    resume: bool = False,
    predictions_out: Optional[str] = None,
    metrics_out: Optional[str] = None
) -> Dict[str, Any]:
    test_split_file = Config.paths.SPLITS_DIR / "Flickr_8k.testImages.txt"
    captions_file = Config.paths.CAPTIONS_FILE
    images_dir = Config.paths.IMAGES_DIR

    if not test_split_file.exists():
        raise FileNotFoundError(f"Test split file not found: {test_split_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # 1. Read test image list
    with open(test_split_file, "r", encoding="utf-8") as f:
        all_test_images = [line.strip() for line in f if line.strip()]

    if num_samples is not None:
        target_images = all_test_images[:num_samples]
        tag = f"{num_samples}"
    else:
        target_images = all_test_images
        tag = f"{len(target_images)}"

    logger.info(f"Target test images to evaluate: {len(target_images)} images")

    # 2. Read reference captions
    image_to_refs = load_flickr8k_references(captions_file)

    # Output paths
    if predictions_out is None:
        pred_path = Config.paths.OUTPUTS_DIR / "predictions" / f"vit_gpt2_test_{tag}.json"
    else:
        pred_path = Path(predictions_out)

    if metrics_out is None:
        metrics_path = Config.paths.OUTPUTS_DIR / f"vit_gpt2_test_{tag}.json"
    else:
        metrics_path = Path(metrics_out)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Check for existing partial predictions if resuming
    existing_records: Dict[str, Dict[str, Any]] = {}
    if resume and pred_path.exists():
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    for rec in saved:
                        existing_records[rec["image"]] = rec
            logger.info(f"Resumed {len(existing_records)} existing predictions from {pred_path}")
        except Exception as e:
            logger.warning(f"Could not load resume file {pred_path}: {e}")

    # 4. Initialize model
    captioner = ViTGPT2Captioner()
    device_name = str(captioner.device)

    records = []
    latencies = []
    failed_images = 0

    t_eval_start = time.perf_counter()

    for idx, img_name in enumerate(tqdm(target_images, desc=f"Evaluating ViT-GPT2 ({tag} images)")):
        # Check if already generated
        if img_name in existing_records:
            records.append(existing_records[img_name])
            latencies.append(existing_records[img_name].get("latency_ms", 0.0))
            continue

        img_path = images_dir / img_name
        refs = image_to_refs.get(img_name, [])

        if not img_path.exists():
            logger.error(f"Image not found: {img_path}")
            failed_images += 1
            continue

        if len(refs) == 0:
            logger.warning(f"No references found for image: {img_name}")

        try:
            caption, latency_ms = captioner.generate_caption(
                img_path,
                num_beams=num_beams,
                max_length=max_length,
                return_latency=True
            )
            latencies.append(latency_ms)
            records.append({
                "sample_id": idx + 1,
                "image": img_name,
                "prediction": caption,
                "references": refs,
                "latency_ms": round(latency_ms, 2)
            })
            # Periodic save every 25 images to ensure robust resume capability
            if len(records) % 25 == 0:
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to generate caption for {img_name}: {e}")
            failed_images += 1

    t_eval_end = time.perf_counter()
    total_runtime = t_eval_end - t_eval_start

    # 5. Save detailed predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved predictions to: {pred_path}")

    # 6. Compute metrics
    preds_list = [r["prediction"] for r in records]
    refs_list = [r["references"] for r in records]

    logger.info(f"Computing BLEU-1..4, METEOR, ROUGE-L on {len(preds_list)} test samples...")
    metrics = compute_metrics(preds_list, refs_list)

    avg_latency = float(sum(latencies) / max(1, len(latencies)))

    eval_summary = {
        "model": "nlpconnect/vit-gpt2-image-captioning",
        "architecture": "VisionEncoderDecoder (ViT + GPT-2)",
        "dataset": "Flickr8k",
        "test_split_file": str(test_split_file.relative_to(ROOT)),
        "num_test_images": len(records),
        "references_per_image": 5,
        "total_references": sum(len(r["references"]) for r in records),
        "failed_images": failed_images,
        "device": device_name,
        "decoding": {
            "method": "beam_search",
            "num_beams": num_beams,
            "max_length": max_length
        },
        "total_runtime_seconds": round(total_runtime, 2),
        "avg_latency_ms_per_image": round(avg_latency, 2),
        "metrics": metrics
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved metrics summary to: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"ViT-GPT2 EVALUATION RESULTS ({len(records)} test images)")
    print("=" * 60)
    print(f"Device: {device_name}")
    print(f"Total Runtime: {total_runtime:.2f}s (Avg {avg_latency:.1f} ms/image)")
    print(f"BLEU-1 : {metrics['BLEU-1']}%")
    print(f"BLEU-2 : {metrics['BLEU-2']}%")
    print(f"BLEU-3 : {metrics['BLEU-3']}%")
    print(f"BLEU-4 : {metrics['BLEU-4']}%")
    print(f"METEOR : {metrics['METEOR']}%")
    print(f"ROUGE-L: {metrics['ROUGE-L']}%")
    print("=" * 60 + "\n")

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT-GPT2 on Flickr8k.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test images (default: 10, set None or 1000 for full)")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam width (default: 3)")
    parser.add_argument("--max_length", type=int, default=32, help="Max length (default: 32)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions")
    parser.add_argument("--predictions_out", type=str, default=None, help="Custom predictions path")
    parser.add_argument("--metrics_out", type=str, default=None, help="Custom metrics path")

    args = parser.parse_args()

    # If --num_samples is 0 or negative or 1000, handle appropriately
    num_samples = None if (args.num_samples is not None and args.num_samples >= 1000) else args.num_samples

    run_evaluation(
        num_samples=num_samples,
        num_beams=args.num_beams,
        max_length=args.max_length,
        resume=args.resume,
        predictions_out=args.predictions_out,
        metrics_out=args.metrics_out
    )


if __name__ == "__main__":
    main()
