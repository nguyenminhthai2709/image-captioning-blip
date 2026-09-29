"""
Flickr8k Exploratory Data Analysis & Visualization Script.

Author: Computer Vision Team
Requirements:
1. Image & caption count statistics.
2. Image dimension distribution (Width, Height).
3. Aspect ratio distribution (Width / Height).
4. Caption length distribution (Word count, summary stats).
5. Grid visualization of 20 random images with captions.
6. Histogram of caption lengths.
7. Image dimension scatter/distribution plot.
8. Save all generated figures into results/figures/.

Tech: Pure Matplotlib (No Seaborn).
"""

import sys
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import Counter

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Project root setup
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import setup_logger, set_seed

logger = setup_logger("DatasetAnalysis")


def load_flickr8k_data(
    images_dir: Path, captions_file: Path
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Parse captions and collect available image files.
    """
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found at: {captions_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    raw_pairs: List[Tuple[str, str]] = []

    # Read CSV format
    try:
        df = pd.read_csv(captions_file)
        if "image" in df.columns and "caption" in df.columns:
            for _, row in df.iterrows():
                img = str(row["image"]).strip()
                cap = str(row["caption"]).strip()
                if img and cap:
                    raw_pairs.append((img, cap))
        else:
            raise ValueError("Not CSV")
    except Exception:
        # Fallback text parsing
        with open(captions_file, "r", encoding="utf-8") as f:
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

    image_to_captions: Dict[str, List[str]] = {}
    for img, cap in raw_pairs:
        image_to_captions.setdefault(img, []).append(cap)

    # Filter only existing images
    valid_images = [img for img in image_to_captions if (images_dir / img).exists()]
    valid_dict = {img: image_to_captions[img] for img in valid_images}

    logger.info(f"Loaded {len(valid_dict)} valid images with {sum(len(c) for c in valid_dict.values())} total captions.")
    return valid_dict, valid_images


def collect_image_metadata(
    images_dir: Path, image_list: List[str]
) -> Tuple[List[int], List[int], List[float]]:
    """
    Extract width, height, and aspect ratio for each image.
    """
    widths, heights, aspect_ratios = [], [], []

    for img_name in image_list:
        try:
            with Image.open(images_dir / img_name) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except Exception as e:
            logger.warning(f"Error reading {img_name}: {e}")

    return widths, heights, aspect_ratios


# =====================================================================
# PLOTTING FUNCTIONS (STRICTLY MATPLOTLIB - NO SEABORN)
# =====================================================================

def plot_caption_length_distribution(
    all_captions: List[str], save_path: Path
) -> None:
    """
    Plot histogram and statistics of caption lengths in words.
    """
    lengths = [len(cap.strip().split()) for cap in all_captions]
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    min_len = int(np.min(lengths))
    max_len = int(np.max(lengths))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Histogram
    bins = np.arange(min_len, max_len + 2) - 0.5
    n, bins_out, patches = ax.hist(
        lengths,
        bins=bins,
        color="#2563EB",
        edgecolor="#1E293B",
        alpha=0.85,
        rwidth=0.85
    )

    # Vertical reference lines
    ax.axvline(mean_len, color="#DC2626", linestyle="--", linewidth=2, label=f"Mean = {mean_len:.2f} words")
    ax.axvline(median_len, color="#16A34A", linestyle="-.", linewidth=2, label=f"Median = {median_len:.1f} words")

    # Styling
    ax.set_title("Distribution of Caption Lengths in Flickr8k", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Caption Length (Number of Words)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (Count)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlim(left=0, right=max(max_len + 2, 25))

    # Info box
    stats_text = (
        f"Total Captions: {len(lengths):,}\n"
        f"Min Length: {min_len} words\n"
        f"Max Length: {max_len} words\n"
        f"Mean Length: {mean_len:.2f} words\n"
        f"Std Dev: {np.std(lengths):.2f}"
    )
    ax.text(
        0.97, 0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95)
    )

    ax.legend(loc="upper left", frameon=True, fontsize=11)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved caption length distribution to: {save_path}")


def plot_image_dimension_distribution(
    widths: List[int], heights: List[int], save_path: Path
) -> None:
    """
    Plot 2D scatter and marginal distribution of image dimensions.
    """
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. Scatter plot of Width vs Height
    ax_scatter.scatter(widths, heights, color="#7C3AED", alpha=0.6, edgecolors="none", s=40)
    
    # Reference 1:1 square line
    max_val = max(max(widths, default=500), max(heights, default=500))
    ax_scatter.plot([0, max_val], [0, max_val], color="#DC2626", linestyle=":", label="Square (1:1 Aspect Ratio)")
    
    ax_scatter.set_title("Image Width vs. Height Distribution", fontsize=13, fontweight="bold")
    ax_scatter.set_xlabel("Width (Pixels)", fontsize=11, fontweight="bold")
    ax_scatter.set_ylabel("Height (Pixels)", fontsize=11, fontweight="bold")
    ax_scatter.grid(True, linestyle="--", alpha=0.5)
    ax_scatter.legend(loc="upper left")

    # 2. Side-by-side histogram for Width and Height
    ax_hist.hist(widths, bins=25, color="#2563EB", alpha=0.65, label=f"Width (Mean={np.mean(widths):.1f}px)", edgecolor="black")
    ax_hist.hist(heights, bins=25, color="#F59E0B", alpha=0.65, label=f"Height (Mean={np.mean(heights):.1f}px)", edgecolor="black")

    ax_hist.set_title("Histogram of Image Dimensions", fontsize=13, fontweight="bold")
    ax_hist.set_xlabel("Dimension (Pixels)", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax_hist.grid(axis="y", linestyle="--", alpha=0.5)
    ax_hist.legend(loc="upper right")

    plt.suptitle("Flickr8k Image Spatial Resolution Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved image dimension distribution to: {save_path}")


def plot_aspect_ratio_distribution(
    aspect_ratios: List[float], save_path: Path
) -> None:
    """
    Plot histogram of Aspect Ratios (Width / Height).
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    mean_ar = float(np.mean(aspect_ratios))
    median_ar = float(np.median(aspect_ratios))

    ax.hist(aspect_ratios, bins=30, color="#059669", edgecolor="#064E3B", alpha=0.85, rwidth=0.9)
    ax.axvline(1.0, color="#DC2626", linestyle="--", linewidth=2, label="1.0 (Square 1:1)")
    ax.axvline(4/3, color="#D97706", linestyle="-.", linewidth=2, label="1.33 (Landscape 4:3)")
    ax.axvline(3/4, color="#7C3AED", linestyle=":", linewidth=2, label="0.75 (Portrait 3:4)")
    ax.axvline(mean_ar, color="#0284C7", linestyle="-", linewidth=2.5, label=f"Mean AR = {mean_ar:.2f}")

    ax.set_title("Distribution of Image Aspect Ratios (Width / Height)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Aspect Ratio (W / H)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved aspect ratio distribution to: {save_path}")


def plot_20_sample_images_grid(
    images_dir: Path,
    image_to_captions: Dict[str, List[str]],
    save_path: Path,
    seed: int = 42
) -> None:
    """
    Display a 4x5 grid of 20 random images with one of their ground-truth captions.
    """
    random.seed(seed)
    all_keys = list(image_to_captions.keys())
    sample_keys = random.sample(all_keys, min(20, len(all_keys)))

    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16), dpi=250)
    axes = axes.flatten()

    for idx in range(rows * cols):
        ax = axes[idx]
        if idx < len(sample_keys):
            img_name = sample_keys[idx]
            img_path = images_dir / img_name
            caption = image_to_captions[img_name][0]  # Take 1st ground truth caption

            # Truncate caption if too long for clean display
            if len(caption) > 55:
                caption_display = caption[:52] + "..."
            else:
                caption_display = caption

            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(
                    f"#{idx+1}: {caption_display}",
                    fontsize=9.5,
                    fontweight="medium",
                    pad=6,
                    wrap=True
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
        ax.axis("off")

    plt.suptitle("20 Random Sample Images with Ground-Truth Captions (Flickr8k)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 20-image sample grid to: {save_path}")


# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    set_seed(42)
    Config.create_dirs()

    # Define paths
    figures_dir = Config.paths.ROOT_DIR / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Config.paths.IMAGES_DIR
    captions_file = Config.paths.CAPTIONS_FILE

    # Ensure dataset or fallback sample exists
    if not captions_file.exists() or not images_dir.exists():
        logger.info("Initializing sample dataset for analysis...")
        from data.setup_flickr8k import create_sample_dataset
        create_sample_dataset(Config.paths.DATA_DIR, num_samples=25)

    # 1. Load Data
    image_to_captions, image_list = load_flickr8k_data(images_dir, captions_file)
    all_captions = [cap for caps in image_to_captions.values() for cap in caps]

    # 2. Collect Metadata
    widths, heights, aspect_ratios = collect_image_metadata(images_dir, image_list)
    caption_lengths = [len(c.strip().split()) for c in all_captions]

    # 3. Print Statistical Summary
    print("\n" + "=" * 60)
    print("      FLICKR8K DATASET EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"1. Total Valid Images:         {len(image_list):,}")
    print(f"2. Total Captions:             {len(all_captions):,}")
    print(f"3. Avg Captions / Image:       {len(all_captions) / len(image_list):.2f}")
    print(f"4. Image Spatial Resolution:   Mean = {np.mean(widths):.1f} x {np.mean(heights):.1f} px")
    print(f"5. Image Aspect Ratio (W/H):   Mean = {np.mean(aspect_ratios):.2f}, Median = {np.median(aspect_ratios):.2f}")
    print(f"6. Caption Word Count Stats:   Min = {np.min(caption_lengths)}, Max = {np.max(caption_lengths)}, Mean = {np.mean(caption_lengths):.2f}, Median = {np.median(caption_lengths):.1f}")
    print("=" * 60 + "\n")

    # 4. Generate Visualizations (Saved into results/figures/)
    logger.info(f"Generating and saving figures into: {figures_dir} ...")

    # Figure 1: Caption Length Histogram
    plot_caption_length_distribution(
        all_captions=all_captions,
        save_path=figures_dir / "caption_length_distribution.png"
    )

    # Figure 2: Image Dimension Distribution (Width vs Height)
    plot_image_dimension_distribution(
        widths=widths,
        heights=heights,
        save_path=figures_dir / "image_dimension_distribution.png"
    )

    # Figure 3: Aspect Ratio Distribution
    plot_aspect_ratio_distribution(
        aspect_ratios=aspect_ratios,
        save_path=figures_dir / "aspect_ratio_distribution.png"
    )

    # Figure 4: 20 Random Sample Images Grid
    plot_20_sample_images_grid(
        images_dir=images_dir,
        image_to_captions=image_to_captions,
        save_path=figures_dir / "sample_20_images_grid.png",
        seed=42
    )

    print(f"\n[+] All 4 analysis figures successfully saved in: {figures_dir}")
    print("    - caption_length_distribution.png")
    print("    - image_dimension_distribution.png")
    print("    - aspect_ratio_distribution.png")
    print("    - sample_20_images_grid.png\n")


if __name__ == "__main__":
    main()
