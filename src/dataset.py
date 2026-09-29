"""
Dataset and DataLoader module for Flickr8k image-caption dataset.
Supports both training mode (single caption per sample with tokenization)
and evaluation mode (multi-caption ground truth per image).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BlipProcessor

from src.config import Config


class Flickr8kDataset(Dataset):
    """
    Custom Dataset for Flickr8k.
    
    Args:
        images_dir: Path to directory containing images.
        captions_file: Path to captions file (supports CSV or tab/space separated formats).
        split: 'train', 'val', or 'test'.
        split_file: Optional text file listing image filenames for the split.
        processor: Pretrained Hugging Face BlipProcessor.
        is_train: Boolean indicating training mode (applies data augmentation & tokenization).
        max_length: Maximum token length for caption truncation/padding.
    """
    
    def __init__(
        self,
        images_dir: Path,
        captions_file: Path,
        split: str = "train",
        split_file: Optional[Path] = None,
        processor: Optional[BlipProcessor] = None,
        is_train: bool = True,
        max_length: int = 32
    ):
        self.images_dir = Path(images_dir)
        self.captions_file = Path(captions_file)
        self.split = split
        self.is_train = is_train
        self.max_length = max_length
        
        # Initialize processor if not provided
        self.processor = processor or BlipProcessor.from_pretrained(Config.model.MODEL_NAME)
        
        # Load and parse captions
        self.samples, self.image_to_captions = self._load_data(split_file)
        
        # Vision Transforms (Augmentations for training)
        if self.is_train:
            self.vision_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self.vision_transform = None

    def _load_data(
        self, split_file: Optional[Path]
    ) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
        """Parse annotations file and filter by split if provided."""
        if not self.captions_file.exists():
            raise FileNotFoundError(f"Captions file not found at: {self.captions_file}")
        
        # Allowed images for this split
        split_images = set()
        if split_file and split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                split_images = {line.strip() for line in f if line.strip()}
        
        raw_data = []
        image_to_captions: Dict[str, List[str]] = {}
        
        # Read captions file: handles either comma-separated or tab/space-separated
        try:
            # Try reading as CSV with header (image,caption)
            df = pd.read_csv(self.captions_file)
            if "image" in df.columns and "caption" in df.columns:
                for _, row in df.iterrows():
                    img_name = str(row["image"]).strip()
                    caption = str(row["caption"]).strip()
                    if split_images and img_name not in split_images:
                        continue
                    raw_data.append({"image": img_name, "caption": caption})
                    image_to_captions.setdefault(img_name, []).append(caption)
            else:
                raise ValueError("Not standard CSV header")
        except Exception:
            # Fallback: line-by-line parsing (e.g. 1000268201_693b08cb0e.jpg#0 A child in a pink dress...)
            with open(self.captions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        parts = line.split("\t", 1)
                    elif "," in line:
                        parts = line.split(",", 1)
                    else:
                        parts = line.split(" ", 1)
                    
                    if len(parts) == 2:
                        img_id_raw, caption = parts[0].strip(), parts[1].strip()
                        img_name = img_id_raw.split("#")[0].strip()
                        
                        if split_images and img_name not in split_images:
                            continue
                        
                        raw_data.append({"image": img_name, "caption": caption})
                        image_to_captions.setdefault(img_name, []).append(caption)

        # If evaluation mode (is_train=False), dataset items are unique images
        if not self.is_train:
            unique_samples = [
                {"image": img_name, "captions": caps}
                for img_name, caps in image_to_captions.items()
            ]
            return unique_samples, image_to_captions
            
        return raw_data, image_to_captions

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_name = item["image"]
        img_path = self.images_dir / img_name
        
        # Load image safely
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback black image if corrupted
            raw_image = Image.new("RGB", (Config.model.IMAGE_SIZE, Config.model.IMAGE_SIZE), color=0)
            
        if self.vision_transform:
            raw_image = self.vision_transform(raw_image)
            
        if self.is_train:
            caption = item["caption"]
            # Process image and caption via Hugging Face BlipProcessor
            encoding = self.processor(
                images=raw_image,
                text=caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Squeeze batch dimension (1, ...) -> (...)
            pixel_values = encoding["pixel_values"].squeeze(0)
            input_ids = encoding["input_ids"].squeeze(0)
            
            # Labels for LM loss: replace pad tokens with -100 so loss ignores padding
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "labels": labels,
                "image_name": img_name,
                "caption": caption
            }
        else:
            # Evaluation mode: returns image tensor + all reference captions for BLEU/METEOR/ROUGE
            encoding = self.processor(images=raw_image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
            
            return {
                "pixel_values": pixel_values,
                "image_name": img_name,
                "references": item["captions"],  # List of 5 ground-truth captions
                "raw_image": raw_image
            }


def collate_train_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for training DataLoader."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels
    }


def collate_eval_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for evaluation DataLoader."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    image_names = [b["image_name"] for b in batch]
    references = [b["references"] for b in batch]
    raw_images = [b["raw_image"] for b in batch]
    
    return {
        "pixel_values": pixel_values,
        "image_names": image_names,
        "references": references,
        "raw_images": raw_images
    }


def get_dataloaders(
    processor: Optional[BlipProcessor] = None,
    batch_size: int = Config.training.BATCH_SIZE,
    num_workers: int = Config.training.NUM_WORKERS,
    data_dir: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct Train, Validation, and Test DataLoaders.
    """
    base_dir = data_dir or Config.paths.DATA_DIR
    images_dir = base_dir / "Images"
    captions_file = base_dir / "captions.txt"
    splits_dir = base_dir / "splits"
    
    train_split = splits_dir / "train_images.txt" if (splits_dir / "train_images.txt").exists() else None
    val_split = splits_dir / "val_images.txt" if (splits_dir / "val_images.txt").exists() else None
    test_split = splits_dir / "test_images.txt" if (splits_dir / "test_images.txt").exists() else None
    
    proc = processor or BlipProcessor.from_pretrained(Config.model.MODEL_NAME)
    
    train_dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        split="train",
        split_file=train_split,
        processor=proc,
        is_train=True
    )
    
    val_dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        split="val",
        split_file=val_split,
        processor=proc,
        is_train=False
    )
    
    test_dataset = Flickr8kDataset(
        images_dir=images_dir,
        captions_file=captions_file,
        split="test",
        split_file=test_split,
        processor=proc,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_train_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.eval.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.eval.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval_fn
    )
    
    return train_loader, val_loader, test_loader
