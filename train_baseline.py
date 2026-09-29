"""
Training pipeline for the Traditional Baseline: ResNet50 + LSTM on Flickr8k.
Supports GPU acceleration, validation loss monitoring, checkpointing, and logging.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
from typing import List, Tuple, Dict, Any
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.config import Config
from src.baseline_cnn_lstm import Vocabulary, CNNtoLSTM
from src.dataset_pipeline import Flickr8kPipeline
from src.utils import setup_logger, save_json, set_seed

logger = setup_logger("TrainBaseline")


class Flickr8kCNNDataset(Dataset):
    """
    Dataset wrapper tailored for CNN+LSTM pipeline.
    Applies torchvision transforms and tokenizes text with Vocabulary.
    """
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        vocab: Vocabulary,
        images_dir: Path,
        transform=None
    ):
        self.samples = samples
        self.vocab = vocab
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name, caption = self.samples[index]
        img_path = self.images_dir / img_name
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # Numericalize caption: <sos> + tokens + <eos>
        numericalized = [self.vocab.sos_idx] + self.vocab.numericalize(caption) + [self.vocab.eos_idx]
        caption_tensor = torch.tensor(numericalized, dtype=torch.long)
        
        return image, caption_tensor, img_name


class CollatePad:
    """Collate function that dynamically pads captions to the batch maximum length."""
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0) # [B, 3, 224, 224]
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx) # [B, max_seq_len]
        
        image_names = [item[2] for item in batch]
        return images, targets, image_names


def train_baseline_pipeline(
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    embed_size: int = 512,
    hidden_size: int = 512,
    train_cnn: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute complete training and validation pipeline for ResNet50 + LSTM."""
    set_seed(seed)
    device = Config.get_device()
    logger.info(f"Using Computation Device: {device}")
    
    # 1. Load Data Splits
    pipeline = Flickr8kPipeline()
    split_data = pipeline.load_all()
    
    train_samples = [(s.image_name, s.caption) for s in split_data["train"]]
    val_samples = [(s.image_name, s.caption) for s in split_data["val"]]
    
    logger.info(f"Loaded {len(train_samples)} Train pairs, {len(val_samples)} Validation pairs.")
    
    # 2. Build and Save Vocabulary
    checkpoint_dir = Config.paths.CHECKPOINT_DIR / "resnet_lstm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = checkpoint_dir / "vocab.json"
    
    all_train_captions = [cap for _, cap in train_samples]
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(all_train_captions)
    vocab.save(vocab_path)
    logger.info(f"Saved vocabulary to: {vocab_path}")
    
    # 3. Instantiate Model
    model = CNNtoLSTM(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        train_cnn=train_cnn
    ).to(device)
    
    # 4. Create DataLoaders
    pad_collate = CollatePad(pad_idx=vocab.pad_idx)
    
    train_dataset = Flickr8kCNNDataset(
        samples=train_samples,
        vocab=vocab,
        images_dir=Config.paths.IMAGES_DIR,
        transform=model.transform
    )
    val_dataset = Flickr8kCNNDataset(
        samples=val_samples,
        vocab=vocab,
        images_dir=Config.paths.IMAGES_DIR,
        transform=model.transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0
    )
    
    # 5. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    # Only train parameters that require grad (Decoder & Projection, optionally CNN)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    # 6. Training & Validation Loop
    logger.info(f"Starting Training for {epochs} Epochs (Batch size={batch_size}, LR={learning_rate})...")
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": []
    }
    
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for images, captions, _ in pbar:
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass: [B, seq_len, vocab_size]
            outputs = model(images, captions)
            
            # Reshape for CrossEntropyLoss:
            # outputs: [B * seq_len, vocab_size], captions: [B * seq_len]
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                captions.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        mean_train_loss = float(sum(train_losses) / len(train_losses))
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, captions, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    captions.reshape(-1)
                )
                val_losses.append(loss.item())
                
        mean_val_loss = float(sum(val_losses) / len(val_losses))
        
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["epochs"].append(epoch)
        
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")
        
        # Save best checkpoint
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            model.save_checkpoint(best_checkpoint_path)
            logger.info(f"[*] New Best Validation Loss: {best_val_loss:.4f}. Saved checkpoint to {best_checkpoint_path}")
            
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.2f} minutes. Best Val Loss: {best_val_loss:.4f}")
    
    # Save training logs
    log_file = Config.paths.LOGS_DIR / "training_log_resnet_lstm.json"
    save_json({
        "model": "ResNet50_LSTM",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_loss": round(best_val_loss, 4),
        "history": history
    }, log_file)
    logger.info(f"Saved training log to: {log_file}")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traditional Baseline (ResNet50 + LSTM) on Flickr8k.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--embed_size", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--hidden_size", type=int, default=512, help="LSTM hidden dimension (default: 512)")
    parser.add_argument("--train_cnn", action="store_true", help="Fine-tune ResNet CNN backbone")
    
    args = parser.parse_args()
    train_baseline_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        train_cnn=args.train_cnn
    )
