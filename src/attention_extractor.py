"""
Real Cross-Attention Visual Grounding Extractor for BLIP (Salesforce/blip-image-captioning-base).
Extracts spatial attention weights from the Text Decoder's Cross-Attention layers (Layer 10 & 11)
and projects the 576 patch activations (24x24) back onto the original image space.
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("AttentionExtractor")


class BLIPAttentionExtractor:
    """
    Extracts authentic Cross-Attention weights between generated text tokens and ViT image patches.
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
        Generate caption and extract full cross-attention heatmaps for each token.
        
        Returns:
            Dict containing:
              - 'caption': Full generated string
              - 'tokens': List of individual string tokens
              - 'attention_maps': Dict mapping token_str -> 2D numpy array [24, 24]
              - 'resized_heatmaps': Dict mapping token_str -> 2D numpy array [H, W]
        """
        img_w, img_h = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values

        # 1. Generate token IDs
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True
            ).sequences[0]

        # 2. Forward pass with output_attentions=True to retrieve cross-attention tensors
        # Prepare decoder input IDs (shifted right with [BOS])
        decoder_input_ids = output_ids.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass through vision encoder first
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[0] # [1, 577, 768]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            # Pass through text decoder with cross_attentions enabled
            decoder_outputs = self.model.text_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=True,
                return_dict=True
            )

        # 3. Extract Cross-Attentions
        # decoder_outputs.cross_attentions is a tuple of length 12 (one per decoder layer)
        # Each tensor is of shape: [1, num_heads (12), seq_len, vision_tokens (577)]
        cross_attns = decoder_outputs.cross_attentions
        
        # Aggregate attention from the top semantic layers (Layers 10 and 11)
        # Shape: [num_layers_to_avg, 1, 12, seq_len, 577]
        top_layers_attn = torch.stack(cross_attns[-2:], dim=0)
        # Average across selected layers and attention heads -> [seq_len, 577]
        avg_attn = top_layers_attn.mean(dim=(0, 1, 2)).squeeze(0) # [seq_len, 577]

        # 4. Tokenize output to match tokens with attention steps
        tokens = [self.processor.decode([t_id]).strip() for t_id in output_ids]
        full_caption = self.processor.decode(output_ids, skip_special_tokens=True).strip()

        attention_maps = {}
        resized_heatmaps = {}

        # 5. Process each token's spatial attention
        for idx, token_str in enumerate(tokens):
            if token_str in ("<unk>", "[PAD]", "[CLS]", "[SEP]", ""):
                continue
                
            # Token attention across 577 vision tokens: token 0 is [CLS], 1..576 are 16x16 patches
            token_attn_vector = avg_attn[idx].cpu().numpy()
            spatial_patches = token_attn_vector[1:] # shape: (576,)
            
            # Reshape into 24x24 patch grid
            grid_24x24 = spatial_patches.reshape(24, 24)
            
            # Min-Max normalize
            grid_norm = (grid_24x24 - grid_24x24.min()) / (grid_24x24.max() - grid_24x24.min() + 1e-8)
            attention_maps[f"{idx}_{token_str}"] = grid_norm
            
            # Bilinear interpolation to original image dimensions (img_h, img_w)
            grid_tensor = torch.tensor(grid_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized_tensor = F.interpolate(
                grid_tensor,
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False
            ).squeeze().clamp(0.0, 1.0)
            
            resized_heatmaps[f"{idx}_{token_str}"] = resized_tensor.numpy()

        return {
            "caption": full_caption,
            "tokens": tokens,
            "raw_tokens_count": len(tokens),
            "attention_maps_24x24": attention_maps,
            "resized_heatmaps": resized_heatmaps
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


def plot_cross_attention_grid(
    image: Image.Image,
    extraction_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    max_words: int = 8
) -> plt.Figure:
    """
    Generate a visual grounding grid: Input Image + Word-by-Word Heatmaps.
    """
    heatmaps = extraction_results["resized_heatmaps"]
    caption = extraction_results["caption"]
    
    # Filter out punctuation and very short stop words if there are many tokens
    stop_words = {"a", "an", "the", "in", "on", "at", "of", "and", ".", ","}
    content_items = [
        (k, v) for k, v in heatmaps.items()
        if k.split("_", 1)[-1].lower() not in stop_words
    ]
    if not content_items:
        content_items = list(heatmaps.items())
        
    content_items = content_items[:max_words]
    num_plots = 1 + len(content_items)
    
    # Layout columns: up to 4 per row
    ncols = min(4, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)
    
    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold", color="#1a1a1a")
    axes[0].axis("off")
    
    # 2. Word Heatmaps
    for idx, (token_key, heatmap) in enumerate(content_items, start=1):
        word_label = token_key.split("_", 1)[-1]
        blended = BLIPAttentionExtractor.overlay_heatmap_on_image(image, heatmap, alpha=0.55, colormap="jet")
        axes[idx].imshow(blended)
        axes[idx].set_title(f'Focus: "{word_label}"', fontsize=12, fontweight="bold", color="#c92a2a")
        axes[idx].axis("off")
        
    # Hide unused subplots
    for ax in axes[num_plots:]:
        ax.axis("off")
        
    plt.suptitle(
        f'BLIP Cross-Attention Visual Grounding\nGenerated Caption: "{caption}"',
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved attention visualization figure to: {save_path}")
        
    return fig
