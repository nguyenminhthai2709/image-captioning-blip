"""
ViT-GPT2 Image Captioning Module.
Uses nlpconnect/vit-gpt2-image-captioning:
  - Vision Encoder: Vision Transformer (ViT-B/16, google/vit-base-patch16-224-in21k)
  - Text Decoder: GPT-2 Causal LM with Cross-Attention
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import setup_logger

logger = setup_logger("ViTGPT2Captioner")


class ViTGPT2Captioner:
    """
    Independent Image Captioner based on nlpconnect/vit-gpt2-image-captioning.
    Keeps model weights, image processor, and tokenizer completely separate
    from the BLIP architecture.
    """

    MODEL_NAME: str = "nlpconnect/vit-gpt2-image-captioning"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME

        # 1. Device selection
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Initializing ViT-GPT2 ({self.model_name}) on device: {self.device}")

        # 2. Load model, processor, and tokenizer
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.image_processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()
            logger.info("ViT-GPT2 model loaded successfully in evaluation mode.")
        except Exception as e:
            logger.error(f"Failed to load ViT-GPT2 model: {e}")
            raise RuntimeError(f"Could not initialize ViT-GPT2 captioner: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Return basic metadata about the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "architecture": "VisionEncoderDecoder (ViT + GPT-2)",
            "encoder": "google/vit-base-patch16-224-in21k",
            "decoder": "gpt2",
            "total_parameters": total_params,
            "device": str(self.device),
            "input_resolution": "224x224 px"
        }

    def generate_caption(
        self,
        image: Union[Image.Image, Path, str],
        num_beams: int = 3,
        max_length: int = 32,
        min_length: int = 5,
        repetition_penalty: float = 1.0,
        return_latency: bool = False
    ) -> Union[str, tuple[str, float]]:
        """
        Generate a caption for a single image.

        Args:
            image: PIL Image object or file path.
            num_beams: Beam search width (default: 3 to match BLIP evaluation).
            max_length: Maximum tokens generated (default: 32).
            min_length: Minimum tokens generated (default: 5).
            repetition_penalty: Penalty for repeated n-grams.
            return_latency: If True, also return inference time in milliseconds.

        Returns:
            Decoded caption string (or tuple with latency).
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at: {img_path}")
            image = Image.open(img_path)

        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        t_start = time.perf_counter()

        # Preprocess image (resizes to 224x224 and normalizes per ViT spec)
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        gen_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "use_cache": True
        }

        with torch.inference_mode():
            output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        if return_latency:
            return caption, latency_ms
        return caption

    def generate_captions_batch(
        self,
        images: List[Image.Image],
        num_beams: int = 3,
        max_length: int = 32,
        min_length: int = 5
    ) -> List[str]:
        """
        Generate captions for a batch of PIL RGB images.
        """
        if not images:
            return []

        clean_images = []
        for img in images:
            if img.mode != "RGB":
                clean_images.append(img.convert("RGB"))
            else:
                clean_images.append(img)

        pixel_values = self.image_processor(
            images=clean_images,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        gen_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "use_cache": True
        }

        with torch.inference_mode():
            output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [p.strip() for p in preds]


def main():
    """Quick CLI test for single image captioning."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate caption using ViT-GPT2.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--num_beams", type=int, default=3, help="Beam width (default: 3)")
    parser.add_argument("--max_length", type=int, default=32, help="Max token length")
    args = parser.parse_args()

    # Default to first Flickr8k image if none provided
    if args.image:
        image_path = Path(args.image)
    else:
        sample_images = list((ROOT / "data" / "flickr8k" / "Images").glob("*.jpg"))
        if not sample_images:
            print("No images found in data/flickr8k/Images.")
            return
        image_path = sample_images[0]

    captioner = ViTGPT2Captioner()
    print("\n" + "=" * 60)
    print(f"Image: {image_path.name}")
    print(f"Device: {captioner.device}")
    caption, latency = captioner.generate_caption(
        image_path,
        num_beams=args.num_beams,
        max_length=args.max_length,
        return_latency=True
    )
    print(f"Generated Caption: \"{caption}\"")
    print(f"Latency: {latency:.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
