"""
Comprehensive Flickr8k Dataset Processing & Computer Vision Pipeline.

Features:
1. Robust Loading: Handles multiple Flickr8k caption formats (CSV / Tokenized).
2. Data Integrity Checks: Detects and filters missing, corrupted, or unreadable images.
3. Zero Data Leakage Split: Splits at the unique image level (Train/Val/Test).
4. Vision Preprocessing: Resize, Data Augmentations, Color Jitter, ImageNet Normalization.
5. Statistical Analysis: Image dimensions, aspect ratios, caption length distribution.
6. Sample Visualizer: Displays image-caption pairs with multiple ground truths.
"""

import sys
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BlipProcessor

from src.config import Config
from src.utils import setup_logger, set_seed

logger = setup_logger("DatasetPipeline")


# =====================================================================
# 1. DATA INTEGRITY & PARSING
# =====================================================================

class Flickr8kManager:
    """
    Manages loading, validating, splitting, and analyzing the Flickr8k dataset.
    """

    def __init__(self, images_dir: Path, captions_file: Path):
        self.images_dir = Path(images_dir)
        self.captions_file = Path(captions_file)
        self.image_to_captions: Dict[str, List[str]] = {}
        self.corrupted_images: List[str] = []
        self.missing_images: List[str] = []

    def load_and_validate(self) -> Dict[str, List[str]]:
        """
        Parse annotations file, verify image existence and test image readability.
        """
        if not self.captions_file.exists():
            raise FileNotFoundError(f"Captions file not found at: {self.captions_file}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at: {self.images_dir}")

        raw_pairs: List[Tuple[str, str]] = []

        # Try reading standard CSV (image,caption)
        try:
            df = pd.read_csv(self.captions_file)
            if "image" in df.columns and "caption" in df.columns:
                for _, row in df.iterrows():
                    img = str(row["image"]).strip()
                    cap = str(row["caption"]).strip()
                    if img and cap:
                        raw_pairs.append((img, cap))
            else:
                raise ValueError("Not CSV format")
        except Exception:
            # Fallback: tokenized text format (e.g., 1000268201_693b08cb0e.jpg#0 A child in...)
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
                        img_raw, cap = parts[0].strip(), parts[1].strip()
                        img_name = img_raw.split("#")[0].strip()
                        if img_name and cap:
                            raw_pairs.append((img_name, cap))

        # Group captions by image
        temp_dict: Dict[str, List[str]] = {}
        for img_name, caption in raw_pairs:
            temp_dict.setdefault(img_name, []).append(caption)

        logger.info(f"Loaded {len(raw_pairs)} raw caption pairs for {len(temp_dict)} unique image names.")

        # Data integrity check: verify file existence and Pillow readability
        valid_dict: Dict[str, List[str]] = {}
        for img_name, captions in temp_dict.items():
            img_path = self.images_dir / img_name
            if not img_path.exists():
                self.missing_images.append(img_name)
                continue

            # Verify image file is not corrupted
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Fast structural verification
                valid_dict[img_name] = captions
            except Exception:
                self.corrupted_images.append(img_name)

        if self.missing_images:
            logger.warning(f"Detected {len(self.missing_images)} missing image files.")
        if self.corrupted_images:
            logger.warning(f"Detected {len(self.corrupted_images)} corrupted image files.")

        logger.info(f"Verification complete: {len(valid_dict)} valid images ready for use.")
        self.image_to_captions = valid_dict
        return self.image_to_captions

    # =====================================================================
    # 2. ZERO-DATA-LEAKAGE SPLIT
    # =====================================================================

    def create_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        save_dir: Optional[Path] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split at the UNIQUE IMAGE level to strictly guarantee NO DATA LEAKAGE.
        All 5 captions of any single image belong exclusively to one split.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        all_images = sorted(list(self.image_to_captions.keys()))
        rng = random.Random(seed)
        rng.shuffle(all_images)

        total = len(all_images)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        train_imgs = all_images[:n_train]
        val_imgs = all_images[n_train:n_train + n_val]
        test_imgs = all_images[n_train + n_val:]

        # Verification of mutual exclusivity
        set_train = set(train_imgs)
        set_val = set(val_imgs)
        set_test = set(test_imgs)

        assert len(set_train.intersection(set_val)) == 0, "DATA LEAKAGE DETECTED: Train and Val overlap!"
        assert len(set_train.intersection(set_test)) == 0, "DATA LEAKAGE DETECTED: Train and Test overlap!"
        assert len(set_val.intersection(set_test)) == 0, "DATA LEAKAGE DETECTED: Val and Test overlap!"

        logger.info(
            f"Zero-Leakage Split Created:\n"
            f"  - Train Set: {len(train_imgs)} images ({len(train_imgs)*5} captions)\n"
            f"  - Val Set:   {len(val_imgs)} images ({len(val_imgs)*5} captions)\n"
            f"  - Test Set:  {len(test_imgs)} images ({len(test_imgs)*5} captions)"
        )

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "train_images.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(train_imgs))
            with open(save_path / "val_images.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(val_imgs))
            with open(save_path / "test_images.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(test_imgs))
            logger.info(f"Saved split files to {save_path}")

        return train_imgs, val_imgs, test_imgs

    # =====================================================================
    # 3. STATISTICAL ANALYSIS
    # =====================================================================

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute descriptive Computer Vision and NLP statistics on the dataset.
        """
        total_images = len(self.image_to_captions)
        all_captions = [cap for caps in self.image_to_captions.values() for cap in caps]
        total_captions = len(all_captions)
        avg_caps_per_img = total_captions / total_images if total_images > 0 else 0

        # Caption length distribution
        caption_lengths = [len(cap.split()) for cap in all_captions]
        vocab_counter = Counter([w.lower().strip(".,!?;:\"()") for cap in all_captions for w in cap.split()])

        # Image dimension inspection (sample up to 500 images for speed)
        sample_keys = list(self.image_to_captions.keys())[:500]
        widths, heights, aspect_ratios = [], [], []

        for img_name in sample_keys:
            try:
                with Image.open(self.images_dir / img_name) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
            except Exception:
                continue

        stats = {
            "Total Images": total_images,
            "Total Captions": total_captions,
            "Average Captions / Image": round(avg_caps_per_img, 2),
            "Vocabulary Size (Unique Words)": len(vocab_counter),
            "Caption Length (Words)": {
                "Min": int(np.min(caption_lengths)) if caption_lengths else 0,
                "Max": int(np.max(caption_lengths)) if caption_lengths else 0,
                "Mean": round(float(np.mean(caption_lengths)), 2) if caption_lengths else 0,
                "Median": float(np.median(caption_lengths)) if caption_lengths else 0,
            },
            "Image Dimensions (Sample 500)": {
                "Mean Width": round(float(np.mean(widths)), 1) if widths else 0,
                "Mean Height": round(float(np.mean(heights)), 1) if heights else 0,
                "Mean Aspect Ratio (W/H)": round(float(np.mean(aspect_ratios)), 2) if aspect_ratios else 0,
            }
        }
        return stats

    # =====================================================================
    # 4. SAMPLE VISUALIZATION
    # =====================================================================

    def visualize_samples(self, num_samples: int = 3, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Display sample images paired with their ground-truth captions.
        """
        sample_keys = random.sample(list(self.image_to_captions.keys()), min(num_samples, len(self.image_to_captions)))
        fig, axes = plt.subplots(len(sample_keys), 2, figsize=(13, 4.5 * len(sample_keys)), gridspec_kw={"width_ratios": [1, 1.3]})

        if len(sample_keys) == 1:
            axes = np.array([axes])

        for idx, img_name in enumerate(sample_keys):
            img_path = self.images_dir / img_name
            image = Image.open(img_path).convert("RGB")
            captions = self.image_to_captions[img_name]

            # Image column
            axes[idx, 0].imshow(image)
            axes[idx, 0].axis("off")
            axes[idx, 0].set_title(f"Image: {img_name} ({image.size[0]}x{image.size[1]})", fontsize=11, fontweight="bold")

            # Captions column
            axes[idx, 1].axis("off")
            text_lines = [f"**Sample #{idx+1} Ground Truth Captions:**\n"]
            for c_idx, cap in enumerate(captions, 1):
                text_lines.append(f"{c_idx}. \"{cap}\"")

            axes[idx, 1].text(
                0.05, 0.9,
                "\n".join(text_lines),
                transform=axes[idx, 1].transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#f1f5f9", edgecolor="#cbd5e1", alpha=0.95)
            )

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig


# =====================================================================
# 5. PYTORCH DATASET WITH VISION PREPROCESSING
# =====================================================================

class Flickr8kPyTorchDataset(Dataset):
    """
    PyTorch Dataset implementing CV Preprocessing for BLIP:
    - Training: Random Horizontal Flip, Color Jitter, Resize (384x384), Normalization.
    - Inference/Val: Deterministic Resize (384x384), Center Crop, Normalization.
    """

    def __init__(
        self,
        images_dir: Path,
        image_to_captions: Dict[str, List[str]],
        image_list: List[str],
        processor: BlipProcessor,
        is_train: bool = True,
        max_length: int = 32
    ):
        self.images_dir = Path(images_dir)
        self.image_to_captions = image_to_captions
        self.image_list = image_list
        self.processor = processor
        self.is_train = is_train
        self.max_length = max_length

        # Data augmentation pipeline for Vision
        if self.is_train:
            self.vision_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self.vision_transform = None

        # Build flat list for training (image, single_caption)
        if self.is_train:
            self.flat_samples = [
                (img_name, cap)
                for img_name in self.image_list
                for cap in self.image_to_captions.get(img_name, [])
            ]
        else:
            # Evaluation returns (img_name, list_of_all_captions)
            self.flat_samples = [
                (img_name, self.image_to_captions.get(img_name, []))
                for img_name in self.image_list
            ]

    def __len__(self) -> int:
        return len(self.flat_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.is_train:
            img_name, caption = self.flat_samples[idx]
            img_path = self.images_dir / img_name
            raw_image = Image.open(img_path).convert("RGB")

            # Apply CV augmentations
            if self.vision_transform:
                raw_image = self.vision_transform(raw_image)

            # BLIP preprocessing: Resize (384x384), ImageNet norm, tokenization
            encoding = self.processor(
                images=raw_image,
                text=caption,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

            pixel_values = encoding["pixel_values"].squeeze(0)
            input_ids = encoding["input_ids"].squeeze(0)
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
            img_name, reference_captions = self.flat_samples[idx]
            img_path = self.images_dir / img_name
            raw_image = Image.open(img_path).convert("RGB")

            encoding = self.processor(images=raw_image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)

            return {
                "pixel_values": pixel_values,
                "image_name": img_name,
                "references": reference_captions,
                "raw_image": raw_image
            }


# =====================================================================
# 6. PIPELINE VERIFICATION & TEST RUNNER
# =====================================================================

if __name__ == "__main__":
    set_seed(42)
    Config.create_dirs()

    images_dir = Config.paths.IMAGES_DIR
    captions_file = Config.paths.CAPTIONS_FILE

    # Ensure dataset or sample exists
    if not captions_file.exists() or not images_dir.exists():
        logger.info("Initializing sample dataset for verification...")
        from data.setup_flickr8k import create_sample_dataset
        create_sample_dataset(Config.paths.DATA_DIR, num_samples=20)

    # 1. Load & Validate
    manager = Flickr8kManager(images_dir=images_dir, captions_file=captions_file)
    valid_data = manager.load_and_validate()

    # 2. Statistics
    stats = manager.compute_statistics()
    print("\n" + "="*55)
    print("[*] DATASET STATISTICAL REPORT (FLICKR8K)")
    print("="*55)
    print(f"- Total Images:               {stats['Total Images']:,}")
    print(f"- Total Captions:             {stats['Total Captions']:,}")
    print(f"- Avg Captions per Image:     {stats['Average Captions / Image']}")
    print(f"- Unique Vocabulary Size:     {stats['Vocabulary Size (Unique Words)']:,} words")
    print(f"- Caption Length (Words):     Min={stats['Caption Length (Words)']['Min']}, Max={stats['Caption Length (Words)']['Max']}, Mean={stats['Caption Length (Words)']['Mean']}")
    print(f"- Image Dimensions (Mean):    {stats['Image Dimensions (Sample 500)']['Mean Width']} x {stats['Image Dimensions (Sample 500)']['Mean Height']} px (Aspect Ratio: {stats['Image Dimensions (Sample 500)']['Mean Aspect Ratio (W/H)']})")
    print("="*55 + "\n")

    # 3. Create Zero-Leakage Splits
    train_imgs, val_imgs, test_imgs = manager.create_splits(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        save_dir=Config.paths.SPLITS_DIR
    )

    # 4. PyTorch Dataset & DataLoader Test
    processor = BlipProcessor.from_pretrained(Config.model.MODEL_NAME)
    train_ds = Flickr8kPyTorchDataset(images_dir, valid_data, train_imgs, processor=processor, is_train=True)
    sample_batch = train_ds[0]
    print(f"[+] PyTorch Tensor Batch Output:")
    print(f"  - pixel_values shape: {sample_batch['pixel_values'].shape} (Channels x Height x Width)")
    print(f"  - input_ids shape:    {sample_batch['input_ids'].shape} (Token IDs)")
    print(f"  - labels shape:       {sample_batch['labels'].shape} (Target IDs with -100 masking)")
    print(f"  - Sample Caption:     \"{sample_batch['caption']}\"\n")
