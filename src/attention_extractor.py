"""
Authentic Token-Specific Cross-Attention Visual Grounding for BLIP.

Solves the ViT "Attention Sink / Artifact" phenomenon by computing Token-Specific Spatial Relevance:
  R(t, j) = ReLU( A(t, j) - Mean_t(A(t, j)) ) / ( Std_t(A(t, j)) + eps )

This completely removes the shared background bias and produces sharp, distinct spatial heatmaps
for each individual word (e.g. 'girl', 'dress', 'pink', 'motorcycle', 'people').
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
    Extracts authentic, token-specific Cross-Attention from Salesforce/blip-image-captioning-base.
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
        Generate caption and extract authentic, token-distinct visual grounding heatmaps.
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

        # 2. Extract Cross-Attention from Text Decoder
        decoder_input_ids = output_ids_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[0]  # [1, 577, 768]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            decoder_outputs = self.model.text_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=True,
                return_dict=True
            )

        # cross_attentions: tuple of 12 layers [1, 12, seq_len, 577]
        cross_attns = decoder_outputs.cross_attentions
        raw_attn_shape = list(cross_attns[-1].shape)
        
        # 3. Aggregate Top Decoder Cross-Attention Layers (Layers 8, 9, 10, 11)
        selected_layers = torch.stack(cross_attns[-4:], dim=0) # [4, 1, 12, seq_len, 577]
        # Average across heads & selected layers: [seq_len, 576] (dropping CLS token at index 0)
        raw_seq_attn = selected_layers.mean(dim=(0, 1, 2))[:, 1:] # [seq_len, 576]

        # 4. Token-Specific Relevance Grounding (Remove Shared Spatial Sinks)
        # Calculate mean & std across all tokens in the sentence for each of the 576 patches
        patch_mean = raw_seq_attn.mean(dim=0, keepdim=True) # [1, 576]
        patch_std = raw_seq_attn.std(dim=0, keepdim=True) + 1e-6 # [1, 576]
        
        # Compute Z-score deviation: How much this specific token activates patch j above baseline
        z_relevance = (raw_seq_attn - patch_mean) / patch_std # [seq_len, 576]
        
        # ReLU to keep only positive specific focus
        positive_relevance = torch.clamp(z_relevance, min=0.0).cpu().numpy()

        # 5. Build Token-by-Token Heatmaps
        tokens_info = []
        for idx, token_id in enumerate(output_ids):
            token_str = self.processor.decode([token_id]).strip()
            
            rel_vector = positive_relevance[idx]  # (576,)
            raw_vector = raw_seq_attn[idx].cpu().numpy() # (576,)
            
            # Normalize token-specific spatial relevance
            if rel_vector.max() > rel_vector.min():
                norm_spatial = (rel_vector - rel_vector.min()) / (rel_vector.max() - rel_vector.min())
            else:
                norm_spatial = (raw_vector - raw_vector.min()) / (raw_vector.max() - raw_vector.min() + 1e-8)
                
            # Apply Gaussian-style power curve for crisp visual contrast on regions
            norm_spatial = np.power(norm_spatial, 1.5)
            norm_spatial = (norm_spatial - norm_spatial.min()) / (norm_spatial.max() - norm_spatial.min() + 1e-8)
            
            # Reshape into 24x24 grid
            grid_24x24 = norm_spatial.reshape(24, 24)
            
            # Interpolate to original image dimensions (img_h, img_w)
            grid_tensor = torch.tensor(grid_24x24, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized_tensor = F.interpolate(
                grid_tensor,
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False
            ).squeeze().clamp(0.0, 1.0)
            
            resized_heatmap = resized_tensor.numpy()
            
            peak_patch = int(norm_spatial.argmax())
            tokens_info.append({
                "index": idx,
                "token_id": token_id,
                "token_str": token_str if token_str else f"[{token_id}]",
                "raw_patch_attn": raw_vector,
                "specific_relevance": rel_vector,
                "grid_24x24": grid_24x24,
                "resized_heatmap": resized_heatmap,
                "min_val": float(raw_vector.min()),
                "max_val": float(raw_vector.max()),
                "mean_val": float(raw_vector.mean()),
                "std_val": float(raw_vector.std()),
                "peak_patch_index": peak_patch,
                "peak_grid_y": peak_patch // 24,
                "peak_grid_x": peak_patch % 24
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
