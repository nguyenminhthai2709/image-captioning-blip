"""
Authentic Cross-Attention Visual Grounding Extractor for BLIP.

Distinguishes:
1. Vision Encoder Self-Attention (ViT internal patch-to-patch)
2. Text Decoder Self-Attention (token-to-token causal NLP)
3. Cross-Attention between Text Queries (Q) and Vision Keys/Values (K, V) (Visual Grounding)

Extracts authentic, token-specific cross-attention maps for every generated token ID
with contrast-enhanced spatial grounding and debug analytics.
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

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("AttentionExtractor")


class BLIPAttentionExtractor:
    """
    Extracts authentic, token-by-token Cross-Attention from Salesforce/blip-image-captioning-base.
    """
    def __init__(
        self,
        model_name: str = Config.model.MODEL_NAME,
        device: Optional[torch.device] = None
    ):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        self.device = device or Config.get_device()
        logger.info(f"Loading BLIP Attention Extractor on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_and_extract_attention(
        self,
        image: Image.Image,
        num_beams: int = 1,
        max_length: int = 32
    ) -> Dict[str, Any]:
        """
        Generate caption and extract full, token-indexed cross-attention heatmaps.
        
        Returns:
            Dict containing:
              - 'caption': Full generated string
              - 'output_ids': List of integer token IDs
              - 'tokens_info': List of token dictionaries with unique heatmaps and debug statistics
              - 'layer_shape': Attention tensor shape
        """
        img_w, img_h = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values

        # 1. Generate Token Sequence
        with torch.no_grad():
            output_ids_tensor = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True
            ).sequences[0]

        output_ids = output_ids_tensor.tolist()
        full_caption = self.processor.decode(output_ids_tensor, skip_special_tokens=True).strip()

        # 2. Extract Cross-Attention via Decoder Forward Pass
        decoder_input_ids = output_ids_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # A. Vision Encoder forward pass -> Image embeddings (Keys & Values)
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[0]  # [1, 577, 768]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            # B. Text Decoder forward pass with Cross-Attention output
            decoder_outputs = self.model.text_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=True,
                return_dict=True
            )

        # cross_attentions is a tuple of 12 tensors (one per decoder layer)
        # Each tensor shape: [batch=1, num_heads=12, text_seq_len, vision_tokens=577]
        cross_attns = decoder_outputs.cross_attentions
        raw_attn_shape = list(cross_attns[-1].shape)
        
        # 3. Aggregate Semantic Layers (Layers 9, 10, 11) for High-Level Visual Grounding
        # Shape: [num_layers (3), 1, 12, seq_len, 577]
        selected_layers = torch.stack(cross_attns[-3:], dim=0)
        # Mean across selected layers and attention heads -> [seq_len, 577]
        avg_attn = selected_layers.mean(dim=(0, 1, 2)).squeeze(0).cpu().numpy()

        # Compute global cross-token background baseline to emphasize token-specific contrast
        # shape: (577,)
        baseline_spatial = avg_attn.mean(axis=0)

        # 4. Construct Token-by-Token Heatmaps
        tokens_info = []
        for idx, token_id in enumerate(output_ids):
            token_str = self.processor.decode([token_id]).strip()
            
            # Extract 576 spatial patch weights (drop CLS at index 0)
            raw_patch_attn = avg_attn[idx, 1:]  # shape: (576,)
            
            # Specificity contrast: how much this token focuses on a patch MORE than average
            diff_from_baseline = raw_patch_attn - baseline_spatial[1:]
            
            # Positive activation focus
            contrast_attn = np.maximum(diff_from_baseline, 0.0)
            if contrast_attn.max() > 0:
                norm_contrast = contrast_attn / contrast_attn.max()
            else:
                norm_contrast = (raw_patch_attn - raw_patch_attn.min()) / (raw_patch_attn.max() - raw_patch_attn.min() + 1e-8)
                
            # Combine 60% token-specific contrast + 40% raw spatial distribution
            raw_norm = (raw_patch_attn - raw_patch_attn.min()) / (raw_patch_attn.max() - raw_patch_attn.min() + 1e-8)
            final_spatial = 0.65 * norm_contrast + 0.35 * raw_norm
            final_spatial = (final_spatial - final_spatial.min()) / (final_spatial.max() - final_spatial.min() + 1e-8)
            
            # Reshape into 24x24 grid
            grid_24x24 = final_spatial.reshape(24, 24)
            
            # Interpolate to original image dimensions (img_h, img_w)
            grid_tensor = torch.tensor(grid_24x24, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized_tensor = F.interpolate(
                grid_tensor,
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False
            ).squeeze().clamp(0.0, 1.0)
            
            resized_heatmap = resized_tensor.numpy()
            
            tokens_info.append({
                "index": idx,
                "token_id": token_id,
                "token_str": token_str if token_str else f"[{token_id}]",
                "raw_patch_attn": raw_patch_attn,
                "grid_24x24": grid_24x24,
                "resized_heatmap": resized_heatmap,
                "min_val": float(raw_patch_attn.min()),
                "max_val": float(raw_patch_attn.max()),
                "mean_val": float(raw_patch_attn.mean()),
                "std_val": float(raw_patch_attn.std()),
                "peak_patch_index": int(raw_patch_attn.argmax())
            })

        return {
            "caption": full_caption,
            "output_ids": output_ids,
            "raw_attn_shape": raw_attn_shape,
            "tokens_info": tokens_info
        }

    @staticmethod
    def overlay_heatmap_on_image(
        image: Image.Image,
        heatmap_2d: np.ndarray,
        alpha: float = 0.55,
        colormap: str = "jet"
    ) -> Image.Image:
        """Blend a 2D float heatmap [H, W] in [0, 1] onto a PIL RGB image."""
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        
        # Ensure correct size
        if heatmap_2d.shape != (h, w):
            t = torch.tensor(heatmap_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            heatmap_2d = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False).squeeze().numpy()
            
        heatmap_2d = np.clip(heatmap_2d, 0.0, 1.0)
        cmap = cm.get_cmap(colormap)
        colored_rgba = cmap(heatmap_2d) # [H, W, 4]
        colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(colored_rgb)
        
        return Image.blend(img_rgb, heatmap_pil, alpha=alpha)
