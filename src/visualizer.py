"""
Visualization module for Computer Vision analysis in Image Captioning.
Provides:
1. Authentic Cross-Attention map overlay (Visual Grounding of words to image regions).
2. Training loss and validation loss convergence curves.
3. Qualitative side-by-side prediction comparisons.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import setup_logger
from src.attention_extractor import BLIPAttentionExtractor

logger = setup_logger("Visualizer")


def overlay_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet"
) -> Image.Image:
    """
    Overlay a 2D attention heatmap onto the original RGB image.
    Delegates directly to BLIPAttentionExtractor for authentic blending.
    """
    return BLIPAttentionExtractor.overlay_heatmap_on_image(
        image=image,
        heatmap_2d=attention_map,
        alpha=alpha,
        colormap=colormap
    )


def plot_training_curves(
    history: Dict[str, Any],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Training Loss and Validation Loss over epochs from training history log.
    """
    train_loss = history.get("train_loss", [])
    if not train_loss:
        raise ValueError("Missing 'train_loss' in history dictionary.")

    val_loss = history.get("val_loss", [])
    lrs = history.get("learning_rates", [])
    epochs = list(range(1, len(train_loss) + 1))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot Train Loss & Val Loss
    line1 = ax1.plot(epochs, train_loss, color="tab:red", marker="o", linewidth=2.5, label="Train Loss")
    lines = line1
    if val_loss and len(val_loss) == len(train_loss):
        line2 = ax1.plot(epochs, val_loss, color="tab:blue", marker="s", linewidth=2.0, linestyle="--", label="Val Loss")
        lines += line2

    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # If learning rates exist, plot on twin axis
    if lrs and len(lrs) == len(train_loss):
        ax2 = ax1.twinx()
        line_lr = ax2.plot(epochs, lrs, color="tab:green", marker="^", linewidth=1.5, linestyle=":", label="Learning Rate")
        ax2.set_ylabel("Learning Rate", color="tab:green", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="tab:green")
        lines += line_lr

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, shadow=True)

    strategy = str(history.get("strategy", "Training")).replace("_", " ").title()
    plt.title(f"Fine-Tuning Convergence ({strategy})", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def visualize_image_attention(
    image_path: Union[str, Path],
    save_dir: Optional[Path] = None,
    model_name: str = Config.model.MODEL_NAME
) -> Tuple[str, List[Path]]:
    """
    Extract authentic Cross-Attention for generated caption using BLIPAttentionExtractor,
    create grid of word-level heatmaps overlaid on image, and save to save_dir.
    """
    img_file = Path(image_path)
    if not img_file.exists() or not img_file.is_file():
        raise FileNotFoundError(f"Image not found at '{img_file}'")

    image = Image.open(img_file).convert("RGB")
    extractor = BLIPAttentionExtractor(model_name=model_name)
    result = extractor.generate_and_extract_attention(image)
    caption = result["caption"]
    words_info = result["words_info"]

    target_dir = save_dir or (ROOT / "results" / "figures" / "attention_maps")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create a multi-panel figure for all words in the caption
    n_words = len(words_info)
    cols = min(4, n_words) if n_words > 0 else 1
    rows = (n_words + cols - 1) // cols if cols > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    if n_words == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    saved_files = []
    for idx, w_info in enumerate(words_info):
        overlay = BLIPAttentionExtractor.overlay_heatmap_on_image(
            image, w_info["resized_heatmap"], alpha=0.55, colormap="jet"
        )
        ax = axes[idx]
        ax.imshow(overlay)
        ax.axis("off")
        sub_tag = f" ({', '.join(w_info['tokens'])})" if w_info['is_multi_token'] else ""
        ax.set_title(f"Word #{w_info['word_index']}: \"{w_info['word']}\"{sub_tag}", fontsize=11, fontweight="bold")

    # Hide unused subplots
    for j in range(n_words, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"BLIP Cross-Attention Grounding:\n\"{caption}\"", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_fig_path = target_dir / f"{img_file.stem}_attention.png"
    fig.savefig(out_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(out_fig_path)

    return caption, saved_files


def plot_qualitative_comparison(
    image: Image.Image,
    zero_shot_caption: str,
    fine_tuned_caption: str,
    ground_truth_captions: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate a high-resolution figure comparing Zero-shot vs. Fine-tuned predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1.3]})

    # Image display
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")

    # Captions comparison card
    axes[1].axis("off")
    text_content = (
        "### PREDICTION COMPARISON\n\n"
        f"🔴 **Zero-Shot Baseline (BLIP):**\n   \"{zero_shot_caption}\"\n\n"
        f"🟢 **Fine-Tuned Model:**\n   \"{fine_tuned_caption}\"\n\n"
    )
    if ground_truth_captions:
        text_content += "📘 **Ground Truth References:**\n"
        for i, ref in enumerate(ground_truth_captions[:3], 1):
            text_content += f"   {i}. \"{ref}\"\n"

    axes[1].text(
        0.05, 0.95,
        text_content,
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95)
    )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualization & Plotting Module for BLIP Image Captioning."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for authentic Cross-Attention Visual Grounding heatmap generation."
    )
    parser.add_argument(
        "--plot_loss",
        type=str,
        default=None,
        help="Path to training log JSON file (e.g. logs/training_log_frozen_vision.json) to plot loss curve."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for saved figure."
    )
    args = parser.parse_args()

    if not args.image and not args.plot_loss:
        print("Usage: python src/visualizer.py --plot_loss <path_to_json> OR --image <image_path>")
        return

    if args.plot_loss:
        json_path = Path(args.plot_loss)
        if not json_path.exists():
            raise FileNotFoundError(f"Log file does not exist at '{json_path}'")
        with open(json_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if "train_loss" not in history:
            raise KeyError(f"Missing 'train_loss' key in log JSON file '{json_path}'.")
        out_path = Path(args.output) if args.output else (ROOT / "results" / "figures" / "loss_curve.png")
        plot_training_curves(history, save_path=out_path)
        print(f"[+] Saved training loss curve figure to: {out_path.resolve()}")

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image does not exist at '{img_path}'")
        out_dir = Path(args.output).parent if (args.output and Path(args.output).suffix) else (Path(args.output) if args.output else None)
        caption, files = visualize_image_attention(img_path, save_dir=out_dir)
        print(f"[+] Generated caption: \"{caption}\"")
        for f in files:
            print(f"[+] Saved authentic Cross-Attention map to: {f.resolve()}")


if __name__ == "__main__":
    main()
