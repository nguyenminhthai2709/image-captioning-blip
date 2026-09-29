"""
Inference module for single or batch image captioning.
Can load either the base Salesforce/blip model or any fine-tuned checkpoint.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Union, Optional, List
from PIL import Image
import torch

from src.config import Config
from src.model import BLIPCaptioningModel


class CaptionPipeline:
    """
    High-level pipeline for generating image captions.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or Config.get_device()
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[Inference] Loading fine-tuned checkpoint from: {checkpoint_path}")
            self.model = BLIPCaptioningModel.from_pretrained_checkpoint(checkpoint_path, device=self.device)
        else:
            print(f"[Inference] Loading pretrained base model: {Config.model.MODEL_NAME}")
            self.model = BLIPCaptioningModel(model_name=Config.model.MODEL_NAME).to(self.device)
            
        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        method: str = "beam",
        num_beams: int = 5,
        max_length: int = 32,
        min_length: int = 5
    ) -> str:
        """
        Generate caption for a single image.
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        caption = self.model.generate_caption(
            image,
            method=method,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            device=self.device
        )
        return caption


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <image_path> [checkpoint_dir]")
        sys.exit(1)
        
    img_path = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    
    pipeline = CaptionPipeline(checkpoint_path=ckpt)
    result = pipeline.predict(img_path)
    print(f"\nGenerated Caption: \"{result}\"")
