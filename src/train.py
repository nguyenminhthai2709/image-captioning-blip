"""
Fine-Tuning Pipeline for BLIP on Flickr8k Dataset.

Author: Computer Vision Project Team
Features:
1. Load Pretrained BLIP: Salesforce/blip-image-captioning-base
2. Custom Train/Validation DataLoader with Data Augmentations & Zero Leakage
3. Preprocess Images: Resize (384x384), ColorJitter, RandomHorizontalFlip, ImageNet Norm
4. Tokenize Captions: Padding, Truncation (max_length), Label Masking (-100 on PAD)
5. Forward Pass & Causal Language Modeling Loss
6. Backpropagation with Gradient Clipping & Mixed Precision AMP
7. AdamW Optimizer with Differential Learning Rate & Cosine Warmup Schedule
8. Validation Loss Evaluation after each Epoch
9. Automatic Best Checkpoint Saving based on Validation Loss
10. Detailed Training History Logging (JSON + CSV) for Convergence Plotting
"""

import sys
import os
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# Ensure project root is in sys.path without hard-coding
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import set_seed, setup_logger

logger = setup_logger("FinetunePipeline")


# =====================================================================
# 1. DATASET & DATALOADER MODULE
# =====================================================================

class Flickr8kTrainValDataset(Dataset):
    """
    PyTorch Dataset for Fine-tuning BLIP:
    - Train: Vision Augmentations (Flip, Jitter), Tokenization, Label Masking (-100).
    - Val: Deterministic Preprocessing, Tokenization, Validation Loss Computation.
    """

    def __init__(
        self,
        images_dir: Path,
        captions_file: Path,
        split_file: Path,
        processor: BlipProcessor,
        is_train: bool = True,
        max_length: int = 32
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.is_train = is_train
        self.max_length = max_length

        # 1. Read split image IDs
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found at: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            self.split_images = {line.strip() for line in f if line.strip()}

        # 2. Parse captions and filter by split
        self.samples: List[Tuple[str, str]] = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("image,caption"):
                    continue
                if line.startswith('"'):
                    parts = line.split('","')
                    if len(parts) == 2:
                        img = parts[0].replace('"', '').strip()
                        cap = parts[1].replace('"', '').strip()
                        if img in self.split_images and (self.images_dir / img).exists():
                            self.samples.append((img, cap))
                else:
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        img, cap = parts[0].strip(), parts[1].strip()
                        if img in self.split_images and (self.images_dir / img).exists():
                            self.samples.append((img, cap))

        logger.info(f"[{'TRAIN' if is_train else 'VAL'}] Loaded {len(self.samples):,} image-caption pairs.")

        # 3. Vision Data Augmentations (Computer Vision Pipeline)
        if self.is_train:
            self.vision_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self.vision_augment = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name, caption = self.samples[idx]
        img_path = self.images_dir / img_name

        # Load RGB image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 384), color=0)

        # Apply spatial & color augmentations
        if self.vision_augment:
            image = self.vision_augment(image)

        # Process image and tokenize caption with BLIP Processor
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze(0)  # Shape: (3, 384, 384)
        input_ids = encoding["input_ids"].squeeze(0)        # Shape: (max_length,)

        # Mask padding tokens in labels with -100 so Cross-Entropy ignores padding
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function to batch image tensors and token labels."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels
    }


# =====================================================================
# 2. LEARNING RATE SCHEDULER & OPTIMIZER
# =====================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5
) -> LambdaLR:
    """Create learning rate schedule with linear warmup and cosine annealing decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_optimizer_param_groups(
    model: BlipForConditionalGeneration,
    lr_decoder: float,
    lr_vision: float,
    weight_decay: float,
    strategy: str = "frozen_vision"
) -> List[Dict[str, Any]]:
    """
    Construct differential learning rate parameter groups:
    - ViT Vision Encoder: lower LR (or frozen)
    - Cross-Attention Text Decoder: higher LR
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    if strategy == "frozen_vision":
        logger.info("Strategy: Freezing Vision Transformer (ViT Backbone)...")
        for param in model.vision_model.parameters():
            param.requires_grad = False
    elif strategy == "full_finetune":
        logger.info("Strategy: Full Fine-tuning with Differential Learning Rates...")
        for param in model.vision_model.parameters():
            param.requires_grad = True

    params_decay_decoder = []
    params_nodecay_decoder = []
    params_decay_vision = []
    params_nodecay_vision = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_nodecay = any(nd in name for nd in no_decay)
        if "vision_model" in name:
            if is_nodecay:
                params_nodecay_vision.append(param)
            else:
                params_decay_vision.append(param)
        else:
            if is_nodecay:
                params_nodecay_decoder.append(param)
            else:
                params_decay_decoder.append(param)

    groups = []
    if params_decay_decoder:
        groups.append({"params": params_decay_decoder, "lr": lr_decoder, "weight_decay": weight_decay})
    if params_nodecay_decoder:
        groups.append({"params": params_nodecay_decoder, "lr": lr_decoder, "weight_decay": 0.0})
    if params_decay_vision:
        groups.append({"params": params_decay_vision, "lr": lr_vision, "weight_decay": weight_decay})
    if params_nodecay_vision:
        groups.append({"params": params_nodecay_vision, "lr": lr_vision, "weight_decay": 0.0})

    return groups


# =====================================================================
# 3. TRAINING & VALIDATION ROUTINES
# =====================================================================

def train_one_epoch(
    model: BlipForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    max_steps: Optional[int] = None
) -> float:
    """Run one epoch of training."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    steps_run = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=True)

    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Mixed Precision Forward Pass
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

        # Gradient Accumulation & Optimizer Step
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
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
        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "lr": f"{current_lr:.2e}"
        })

        steps_run += 1
        if max_steps and steps_run >= max_steps:
            break

    return total_loss / max(1, steps_run)


def evaluate_val_loss(
    model: BlipForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
    max_eval_batches: Optional[int] = None
) -> float:
    """Calculate average Validation Loss on the Validation Set."""
    model.eval()
    total_val_loss = 0.0
    batches_run = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating [Val Loss]", leave=False)
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            else:
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

            total_val_loss += outputs.loss.item()
            batches_run += 1

            pbar.set_postfix({"val_loss": f"{outputs.loss.item():.4f}"})

            if max_eval_batches and batches_run >= max_eval_batches:
                break

    return total_val_loss / max(1, batches_run)


# =====================================================================
# 4. MAIN TRAINING PIPELINE
# =====================================================================

def run_training(
    learning_rate: float = 5e-5,
    lr_vision: float = 5e-6,
    batch_size: int = 16,
    epochs: int = 5,
    max_length: int = 32,
    weight_decay: float = 0.05,
    strategy: str = "frozen_vision",
    num_workers: int = 0,
    grad_accum_steps: int = 2,
    max_train_steps: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    output_checkpoint_dir: Optional[Path] = None,
    output_log_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Main training routine for BLIP Fine-Tuning.
    """
    set_seed(42)
    Config.create_dirs()

    # 1. Device Setup (CUDA with AMP or CPU Fallback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU Device: {torch.cuda.get_device_name(0)} (CUDA Mixed Precision AMP Enabled)")
        use_amp = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS Device.")
        use_amp = False
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Training on CPU.")
        use_amp = False

    checkpoint_dir = output_checkpoint_dir or (ROOT / "checkpoints" / strategy)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = checkpoint_dir / "best_model"

    log_dir = output_log_dir or (ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Pretrained BLIP Model & Processor
    model_name = Config.model.MODEL_NAME
    logger.info(f"Loading Pretrained BLIP: {model_name} ...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # 3. Construct Train and Validation DataLoaders
    data_dir = Config.paths.DATA_DIR
    images_dir = data_dir / "Images"
    captions_file = data_dir / "captions.txt"
    splits_dir = data_dir / "splits"

    train_split = splits_dir / "train_images.txt"
    val_split = splits_dir / "val_images.txt"

    train_dataset = Flickr8kTrainValDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        split_file=train_split,
        processor=processor,
        is_train=True,
        max_length=max_length
    )

    val_dataset = Flickr8kTrainValDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        split_file=val_split,
        processor=processor,
        is_train=False,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # 4. Parameter Groups & Optimizer
    param_groups = get_optimizer_param_groups(
        model=model,
        lr_decoder=learning_rate,
        lr_vision=lr_vision,
        weight_decay=weight_decay,
        strategy=strategy
    )
    optimizer = AdamW(param_groups)

    # 5. Cosine Scheduler with Linear Warmup
    effective_steps_per_epoch = len(train_loader) // grad_accum_steps
    if max_train_steps:
        total_training_steps = max_train_steps * epochs
    else:
        total_training_steps = effective_steps_per_epoch * epochs

    warmup_steps = int(total_training_steps * Config.training.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # Print summary parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Summary: {trainable_params:,} / {total_params:,} trainable parameters ({trainable_params/total_params*100:.2f}%)")

    # 6. Training Loop & Metric Logging
    history: Dict[str, Any] = {
        "strategy": strategy,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "epoch_times": []
    }

    best_val_loss = float("inf")
    start_time = time.time()

    print("\n" + "=" * 65)
    print("      STARTING BLIP FINE-TUNING ON FLICKR8K")
    print("=" * 65)
    print(f"[*] Strategy:           {strategy.upper()}")
    print(f"[*] Device:             {device}")
    print(f"[*] Batch Size:         {batch_size} (Grad Accum: {grad_accum_steps})")
    print(f"[*] Epochs:             {epochs}")
    print(f"[*] Decoder LR:         {learning_rate:.2e} | ViT LR: {lr_vision:.2e}")
    print(f"[*] Weight Decay:       {weight_decay}")
    print(f"[*] Max Token Length:   {max_length}")
    print("=" * 65 + "\n")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        logger.info(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train one epoch
        avg_train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=Config.training.MAX_GRAD_NORM,
            max_steps=max_train_steps
        )

        # Evaluate on validation set
        avg_val_loss = evaluate_val_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            max_eval_batches=max_eval_batches
        )

        epoch_duration = time.time() - epoch_t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(round(epoch_duration, 2))

        logger.info(
            f"Epoch {epoch} Results | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # 7. Checkpoint Saving based on Best Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"--> New best Val Loss ({best_val_loss:.4f})! Saving model checkpoint to {best_model_dir} ...")
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)

        # Save training logs progressively
        log_json_path = log_dir / f"training_log_{strategy}.json"
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        # Save CSV format
        df_log = pd.DataFrame({
            "epoch": list(range(1, epoch + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "learning_rate": history["learning_rates"],
            "epoch_time_s": history["epoch_times"]
        })
        df_log.to_csv(log_dir / f"training_log_{strategy}.csv", index=False)

    total_training_time = time.time() - start_time
    logger.info(f"\nFine-Tuning completed in {total_training_time/60:.2f} minutes. Best Validation Loss: {best_val_loss:.4f}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP Image Captioning on Flickr8k")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for Text Decoder (default: 5e-5)")
    parser.add_argument("--lr_vision", type=float, default=5e-6, help="Learning rate for Vision Transformer (default: 5e-6)")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--max_length", type=int, default=32, help="Max token sequence length (default: 32)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW (default: 0.05)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="frozen_vision",
        choices=["frozen_vision", "full_finetune"],
        help="Training strategy: 'frozen_vision' (freeze ViT) or 'full_finetune' (train all with differential LR)"
    )
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional max steps per epoch for quick testing")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Optional max eval batches for quick validation")
    args = parser.parse_args()

    run_training(
        learning_rate=args.learning_rate,
        lr_vision=args.lr_vision,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        strategy=args.strategy,
        grad_accum_steps=args.grad_accum,
        max_train_steps=args.max_train_steps,
        max_eval_batches=args.max_eval_batches
    )


if __name__ == "__main__":
    main()
