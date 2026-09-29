"""
Fine-tuning training pipeline for Salesforce/blip-image-captioning-base on Flickr8k.
Supports differential learning rates, Cosine LR scheduling with Warmup,
Gradient Accumulation, Mixed Precision (AMP), and Early Best Checkpoint saving.
"""

import sys
import time
import math
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import Config
from src.utils import set_seed, setup_logger, save_json
from src.dataset import get_dataloaders
from src.model import BLIPCaptioningModel
from src.evaluate import evaluate_model

logger = setup_logger("Training")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """Create a learning rate schedule with linear warmup and cosine decay."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_epoch(
    model: BLIPCaptioningModel,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True)
    
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass with AMP if enabled
        if scaler and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
        total_loss += loss.item() * grad_accum_steps
        
        # Optimizer step with gradient accumulation
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler and device.type == "cuda":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad()
            
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}", "lr": f"{current_lr:.2e}"})
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train(
    strategy: str = "frozen_vision",
    epochs: int = Config.training.EPOCHS,
    batch_size: int = Config.training.BATCH_SIZE,
    lr_decoder: float = Config.training.LR_TEXT_DECODER,
    lr_vision: float = Config.training.LR_VISION_ENCODER,
    checkpoint_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Main training routine.
    
    Args:
        strategy: 'frozen_vision' (freeze ViT, train decoder only) or 'full_finetune' (unfreeze all)
    """
    Config.create_dirs()
    set_seed(Config.training.SEED)
    device = Config.get_device()
    logger.info(f"Using device: {device} | Strategy: {strategy}")
    
    save_dir = checkpoint_dir or (Config.paths.CHECKPOINT_DIR / strategy)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Model
    model = BLIPCaptioningModel(model_name=Config.model.MODEL_NAME).to(device)
    
    if strategy == "frozen_vision":
        logger.info("Strategy: Freezing Vision Encoder (ViT Backbone)...")
        model.freeze_vision_encoder()
    elif strategy == "full_finetune":
        logger.info("Strategy: Full Fine-tuning with Differential Learning Rates...")
        model.unfreeze_vision_encoder()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    param_stats = model.get_trainable_params_stats()
    logger.info(
        f"Parameters: Total = {param_stats['total_parameters']:,} | "
        f"Trainable = {param_stats['trainable_parameters']:,} ({param_stats['trainable_percent']:.2f}%)"
    )
    
    # 2. Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        processor=model.processor,
        batch_size=batch_size,
        data_dir=data_dir
    )
    logger.info(f"Dataset: {len(train_loader.dataset)} train captions, {len(val_loader.dataset)} val images.")
    
    # 3. Optimizer and LR Scheduler
    param_groups = model.get_optimizer_param_groups(
        lr_decoder=lr_decoder,
        lr_vision=lr_vision,
        weight_decay=Config.training.WEIGHT_DECAY
    )
    optimizer = AdamW(param_groups)
    
    total_training_steps = (len(train_loader) // Config.training.GRADIENT_ACCUMULATION_STEPS) * epochs
    warmup_steps = int(total_training_steps * Config.training.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    
    scaler = torch.cuda.amp.GradScaler() if (Config.training.USE_AMP and device.type == "cuda") else None
    
    # 4. Training Loop
    history = {
        "train_loss": [],
        "val_bleu4": [],
        "val_meteor": [],
        "val_rouge_l": [],
        "strategy": strategy,
        "epochs": epochs
    }
    
    best_bleu4 = -1.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            grad_accum_steps=Config.training.GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=Config.training.MAX_GRAD_NORM
        )
        history["train_loss"].append(train_loss)
        
        # Validation Evaluation
        logger.info("Running validation evaluation...")
        val_metrics, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            method="beam",
            num_beams=3,
            max_samples=200  # fast validation subset
        )
        
        val_b4 = val_metrics["BLEU-4"]
        history["val_bleu4"].append(val_b4)
        history["val_meteor"].append(val_metrics["METEOR"])
        history["val_rouge_l"].append(val_metrics["ROUGE-L"])
        
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} finished in {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val BLEU-4: {val_b4:.2f} | "
            f"Val METEOR: {val_metrics['METEOR']:.2f} | Val ROUGE-L: {val_metrics['ROUGE-L']:.2f}"
        )
        
        # Save Best Checkpoint
        if val_b4 > best_bleu4:
            best_bleu4 = val_b4
            logger.info(f"--> New best Val BLEU-4 ({best_bleu4:.2f})! Saving model to {save_dir / 'best_model'}...")
            model.save_pretrained(save_dir / "best_model")
            
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val BLEU-4: {best_bleu4:.2f}")
    
    # Save final history
    save_json(history, save_dir / "training_history.json")
    return history


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "frozen_vision"
    train(strategy=strategy, epochs=3)
