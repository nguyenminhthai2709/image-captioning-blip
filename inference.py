"""
Inference Pipeline for Pretrained BLIP: Salesforce/blip-image-captioning-base.

Features:
- Device Auto-Detection: CUDA GPU -> MPS -> CPU fallback.
- Step-by-step Visual Preprocessing Inspection (Dimensions, Normalization, Tensor Shapes).
- Caption Generation via Beam Search / Greedy Search.
- Output serialization to JSON and TXT.
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

# Project root setup
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("InferenceEngine")


class BLIPInferencePipeline:
    """
    End-to-end Inference Pipeline for BLIP Image Captioning.
    """

    def __init__(
        self,
        model_name: str = Config.model.MODEL_NAME,
        device: Optional[torch.device] = None
    ):
        # 1. Device Auto-Detection with Fallback
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU Device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon MPS Device.")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available. Falling back to CPU.")

        logger.info(f"Loading Pretrained Processor & Model: {model_name} ...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and initialized in EVAL mode.")

    def inspect_preprocessing_pipeline(self, image: Image.Image) -> Dict[str, Any]:
        """
        Step-by-step display of Computer Vision preprocessing.
        """
        orig_w, orig_h = image.size
        orig_mode = image.mode
        
        # Hugging Face BlipProcessor internal transforms
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding.pixel_values  # (1, 3, 384, 384)
        
        tensor_min = float(pixel_values.min())
        tensor_max = float(pixel_values.max())
        tensor_mean = float(pixel_values.mean())
        tensor_std = float(pixel_values.std())

        info = {
            "Raw Image": {
                "Dimensions (W x H)": f"{orig_w} x {orig_h} px",
                "Color Mode": orig_mode,
                "Aspect Ratio (W/H)": round(orig_w / max(1, orig_h), 2)
            },
            "Preprocessed Vision Tensor": {
                "Target Spatial Resolution": "384 x 384 px",
                "Interpolation Method": "Bicubic Interpolation",
                "Tensor Shape (B x C x H x W)": list(pixel_values.shape),
                "Data Type": str(pixel_values.dtype),
                "Normalization Parameters": {
                    "Mean (RGB)": [0.48145466, 0.4578275, 0.40821073],
                    "Std (RGB)": [0.26862954, 0.26130258, 0.27577711],
                },
                "Normalized Pixel Range": f"[{tensor_min:.3f}, {tensor_max:.3f}] (Mean: {tensor_mean:.3f}, Std: {tensor_std:.3f})"
            },
            "Vision Transformer Patches": {
                "Patch Size (P x P)": "16 x 16 px",
                "Total Visual Patches (N)": f"({384//16} x {384//16}) = 576 patches",
                "Visual Feature Embedding Dim (D)": 768,
                "Vision Encoder Output Shape": "(1, 577, 768) including [CLS] token"
            }
        }
        return info

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
        Execute full inference on an input image and save output.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        # 1. Load Image
        image = Image.open(path).convert("RGB")

        # 2. Inspect CV Pipeline
        cv_pipeline_info = self.inspect_preprocessing_pipeline(image)

        # 3. Preprocess Tensor & Transfer to Device
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 4. Generate Caption Auto-regressively
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

        # 5. Decode Tokens to Text
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        result = {
            "image_path": str(path.resolve()),
            "image_name": path.name,
            "generated_caption": caption,
            "latency_ms": round(latency_ms, 2),
            "device": str(self.device).upper(),
            "decoding_parameters": {
                "method": "Beam Search" if num_beams > 1 else "Greedy Search",
                "num_beams": num_beams,
                "max_length": max_length,
                "min_length": min_length,
                "repetition_penalty": repetition_penalty
            },
            "preprocessing_details": cv_pipeline_info
        }

        # 6. Save results to disk
        if save_output:
            out_dir = output_dir or (ROOT / "results" / "captions")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON metadata
            json_file = out_dir / f"{path.stem}_caption.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            # Save plain text
            txt_file = out_dir / f"{path.stem}_caption.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"Image: {path.name}\nGenerated Caption: \"{caption}\"\nInference Latency: {latency_ms:.2f} ms ({self.device})\n")

            logger.info(f"Saved generated caption to:\n  - {json_file}\n  - {txt_file}")

        return result


def main():
    parser = argparse.ArgumentParser(description="BLIP Image Captioning Inference Engine")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image JPEG/PNG. If omitted, takes the first sample image from Flickr8k."
    )
    parser.add_argument("--num_beams", type=int, default=5, help="Beam width for beam search decoding (default: 5)")
    parser.add_argument("--max_length", type=int, default=32, help="Max length of generated sequence (default: 32)")
    args = parser.parse_args()

    # Determine input image
    if args.image:
        input_image_path = Path(args.image)
    else:
        images_dir = ROOT / "data" / "flickr8k" / "Images"
        available_imgs = list(images_dir.glob("*.jpg"))
        if not available_imgs:
            raise FileNotFoundError(f"No images found in {images_dir}. Please specify --image path/to/image.jpg")
        input_image_path = available_imgs[0]

    pipeline = BLIPInferencePipeline()
    result = pipeline.generate_caption(
        image_path=input_image_path,
        num_beams=args.num_beams,
        max_length=args.max_length
    )

    # Pretty print summary
    print("\n" + "=" * 65)
    print("      BLIP IMAGE CAPTIONING INFERENCE RESULTS")
    print("=" * 65)
    print(f"[*] Input Image:        {result['image_name']}")
    print(f"[*] Device Used:        {result['device']}")
    print(f"[*] Inference Latency:  {result['latency_ms']} ms")
    print(f"[*] Decoding Strategy:  {result['decoding_parameters']['method']} (beams={result['decoding_parameters']['num_beams']})")
    print("-" * 65)
    print(f"[*] GENERATED CAPTION:  \"{result['generated_caption']}\"")
    print("=" * 65)

    print("\n[*] COMPUTER VISION PREPROCESSING DETAILS:")
    prep = result["preprocessing_details"]
    for section, details in prep.items():
        print(f"\n  [{section}]")
        for k, v in details.items():
            print(f"    - {k}: {v}")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
