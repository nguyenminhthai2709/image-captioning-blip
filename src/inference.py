"""
Inference Module for Pretrained BLIP Image Captioning.

Model: Salesforce/blip-image-captioning-base
Features:
- Device Auto-Detection: CUDA GPU -> MPS -> CPU fallback.
- Explicit Computer Vision Preprocessing Breakdown:
    Image (PIL) -> RGB -> BLIP Processor -> Tensor [1, 3, 384, 384] -> Vision Encoder (ViT-B/16).
- CLI Support: python src/inference.py --image <image_path>
- Outputs saved into results/predictions/ (JSON & TXT).
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Ensure project root is in sys.path without hard-coding
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("InferenceEngine")


class BLIPInferencePipeline:
    """
    Modular Inference Pipeline for Salesforce/blip-image-captioning-base.
    """

    def __init__(
        self,
        model_name: str = Config.model.MODEL_NAME,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize BLIP processor, model, and computation device.
        """
        # 1. Device Auto-Detection with Fallback
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Device: Apple Silicon MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("Device: CPU (CUDA not detected, falling back to CPU)")

        self.model_name = model_name
        logger.info(f"Loading Pretrained BLIP Model: {model_name} ...")
        self.processor: BlipProcessor = BlipProcessor.from_pretrained(model_name)
        self.model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully in evaluation mode.")

    def log_preprocessing_pipeline(self, image: Image.Image) -> Dict[str, Any]:
        """
        Inspect and display the Computer Vision preprocessing pipeline:
        Image -> RGB -> BLIP Processor -> Tensor -> Vision Encoder Patches.
        """
        orig_w, orig_h = image.size
        orig_mode = image.mode

        # Transform using BLIP Processor
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values: torch.Tensor = encoding.pixel_values  # Shape: (1, 3, 384, 384)

        tensor_min = float(pixel_values.min())
        tensor_max = float(pixel_values.max())
        tensor_mean = float(pixel_values.mean())
        tensor_std = float(pixel_values.std())

        pipeline_info: Dict[str, Any] = {
            "1. Input Image": {
                "Dimensions (W x H)": f"{orig_w} x {orig_h} px",
                "Color Channels": f"{orig_mode} (3 channels)",
                "Aspect Ratio (W/H)": round(orig_w / max(1, orig_h), 2)
            },
            "2. Preprocessing & Tensorization": {
                "Target Resolution": "384 x 384 px",
                "Interpolation Method": "Bicubic Interpolation",
                "Output Tensor Shape [B, C, H, W]": list(pixel_values.shape),
                "Data Type": str(pixel_values.dtype),
                "BLIP Processor Mean (RGB)": [0.48145466, 0.4578275, 0.40821073],
                "BLIP Processor Std (RGB)": [0.26862954, 0.26130258, 0.27577711],
                "Normalization Protocol": "BLIP Processor Normalization (ImageNet-derived parameters)",
                "Normalized Value Range": f"[{tensor_min:.3f}, {tensor_max:.3f}] (μ={tensor_mean:.3f}, σ={tensor_std:.3f})"
            },
            "3. Vision Transformer (ViT-B/16) Architecture": {
                "Patch Size (P x P)": "16 x 16 px",
                "Total Visual Patches (N)": f"({384//16} x {384//16}) = 576 patches",
                "Patch Embedding Dimension (D)": 768,
                "Position Embedding": "Learned 1D Spatial Positional Embeddings (577 x 768)",
                "Vision Encoder Output Shape": "[1, 577, 768] (576 patch tokens + 1 [CLS] token)"
            }
        }
        return pipeline_info

    def generate_caption(
        self,
        image_path: Union[str, Path],
        num_beams: int = 5,
        max_length: int = 32,
        min_length: int = 5,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        save_output: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate natural language caption for an input image.

        Args:
            image_path: Path to image file (JPEG, PNG).
            num_beams: Beam search beam width (k=5).
            max_length: Maximum sequence length.
            min_length: Minimum sequence length.
            repetition_penalty: Penalty factor for repeating n-grams.
            length_penalty: Exponential penalty on sequence length.
            save_output: Whether to serialize output to JSON and TXT.
            output_dir: Target directory for predictions.

        Returns:
            Dictionary containing metadata, latency, and generated caption.
        """
        img_file = Path(image_path)
        if not img_file.exists() or not img_file.is_file():
            raise FileNotFoundError(
                f"Error: Input image does not exist at '{img_file.resolve()}'. "
                "Please verify the file path."
            )

        # 1. Load image using Pillow and ensure RGB
        try:
            image = Image.open(img_file).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error: Unable to read image file '{img_file}': {e}")

        # 2. Extract CV preprocessing details
        cv_pipeline_info = self.log_preprocessing_pipeline(image)

        # 3. Process image to PyTorch Tensor on target device
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 4. Generate caption using BLIP Auto-regressive Text Decoder
        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty
            )
        latency_ms = (time.time() - t0) * 1000.0

        # 5. Decode output tokens to natural language string
        caption: str = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        result: Dict[str, Any] = {
            "image": str(img_file.resolve()),
            "caption": caption,
            "model": self.model_name,
            "device": str(self.device).upper(),
            "inference_time": round(latency_ms, 2),
            # Preserved detailed metadata for compatibility
            "image_path": str(img_file.resolve()),
            "image_name": img_file.name,
            "generated_caption": caption,
            "latency_ms": round(latency_ms, 2),
            "decoding_parameters": {
                "method": "Beam Search" if num_beams > 1 else "Greedy Search",
                "num_beams": num_beams,
                "max_length": max_length,
                "min_length": min_length,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty
            },
            "preprocessing_pipeline": cv_pipeline_info
        }

        # 6. Save generated predictions into results/predictions/
        if save_output:
            out_path = output_dir or (ROOT / "results" / "predictions")
            out_path.mkdir(parents=True, exist_ok=True)

            json_file = out_path / f"{img_file.stem}_prediction.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            txt_file = out_path / f"{img_file.stem}_prediction.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(
                    f"Image Path:        {img_file.resolve()}\n"
                    f"Model:             {self.model_name}\n"
                    f"Device:            {self.device}\n"
                    f"Inference Latency: {latency_ms:.2f} ms\n"
                    f"Generated Caption: \"{caption}\"\n"
                )

            logger.info(f"Saved prediction files:\n  - {json_file}\n  - {txt_file}")

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Pretrained BLIP Image Captioning Inference (Salesforce/blip-image-captioning-base)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (e.g. data/flickr8k/Images/1000268201_693b08cb0e.jpg)"
    )
    parser.add_argument("--num_beams", type=int, default=5, help="Beam width for beam search (default: 5)")
    parser.add_argument("--max_length", type=int, default=32, help="Max caption length (default: 32)")
    args = parser.parse_args()

    # Determine input image path
    if args.image:
        input_image = Path(args.image)
    else:
        images_dir = ROOT / "data" / "flickr8k" / "Images"
        available_imgs = list(images_dir.glob("*.jpg"))
        if not available_imgs:
            print("[-] Error: No images found in data/flickr8k/Images/. Please specify --image <path>.")
            sys.exit(1)
        input_image = available_imgs[0]
        logger.info(f"No --image argument provided. Using sample: {input_image.name}")

    # Run inference
    pipeline = BLIPInferencePipeline()
    try:
        res = pipeline.generate_caption(
            image_path=input_image,
            num_beams=args.num_beams,
            max_length=args.max_length
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # Print clean formatted outputs
    print("\n" + "=" * 65)
    print("      PRETRAINED BLIP IMAGE CAPTIONING INFERENCE")
    print("=" * 65)
    print(f"[*] Device in Use:      {res['device']}")
    print(f"[*] Image Path:         {res['image_path']}")
    print(f"[*] Inference Latency:  {res['latency_ms']} ms")
    print(f"[*] Decoding Strategy:  {res['decoding_parameters']['method']} (num_beams={res['decoding_parameters']['num_beams']})")
    print("-" * 65)
    print(f"[*] GENERATED CAPTION:  \"{res['generated_caption']}\"")
    print("=" * 65)

    print("\n[*] COMPUTER VISION PREPROCESSING & ARCHITECTURE BREAKDOWN:")
    prep = res["preprocessing_pipeline"]
    for step_title, details in prep.items():
        print(f"\n  [{step_title}]")
        for k, v in details.items():
            print(f"    • {k}: {v}")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
