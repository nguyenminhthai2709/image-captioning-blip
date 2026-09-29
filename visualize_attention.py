"""
Command Line Tool for Cross-Attention Visual Grounding with BLIP.

Usage:
  python visualize_attention.py --image data/flickr8k/Images/1000268201_693b08cb0e.jpg
  python visualize_attention.py --image path/to/any_image.jpg --max_words 6
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
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from PIL import Image

from src.config import Config
from src.attention_extractor import BLIPAttentionExtractor, plot_cross_attention_grid
from src.utils import setup_logger

logger = setup_logger("VisualizeAttentionCLI")


def main():
    parser = argparse.ArgumentParser(description="Visualize BLIP Cross-Attention Heatmaps for Image Captioning.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: first image in data/flickr8k/Images/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Config.paths.ROOT_DIR / "results" / "figures" / "attention_maps"),
        help="Directory to save output visualization figures"
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=7,
        help="Maximum number of word heatmaps to plot (default: 7)"
    )
    
    args = parser.parse_args()
    
    # 1. Resolve image path
    if args.image:
        image_path = Path(args.image)
    else:
        sample_images = list(Config.paths.IMAGES_DIR.glob("*.jpg"))
        if not sample_images:
            logger.error("No images found in data/flickr8k/Images! Please specify --image <path>.")
            sys.exit(1)
        image_path = sample_images[0]
        
    if not image_path.exists():
        logger.error(f"Image not found at: {image_path}")
        sys.exit(1)
        
    logger.info(f"Processing Image: {image_path.resolve()}")
    raw_image = Image.open(image_path).convert("RGB")
    
    # 2. Extract Cross-Attention
    extractor = BLIPAttentionExtractor()
    results = extractor.generate_and_extract_attention(raw_image)
    
    print("\n" + "=" * 70)
    print("BLIP CROSS-ATTENTION EXTRACTION RESULTS")
    print("=" * 70)
    print(f"Image File       : {image_path.name}")
    print(f"Generated Caption: \"{results['caption']}\"")
    print(f"Extracted Tokens : {len(results['resized_heatmaps'])} word heatmaps generated.")
    print("=" * 70 + "\n")
    
    # 3. Plot & Save Visualization Grid
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig_path = output_dir / f"{image_path.stem}_attention_grid.png"
    
    plot_cross_attention_grid(
        image=raw_image,
        extraction_results=results,
        save_path=save_fig_path,
        max_words=args.max_words
    )
    
    print(f"[*] Visual Attention Map Grid successfully saved to:")
    print(f"    {save_fig_path.resolve()}\n")


if __name__ == "__main__":
    main()
