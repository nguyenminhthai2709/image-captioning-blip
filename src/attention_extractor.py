"""
Word-Level Visual Grounding & Subword Token Aggregation for BLIP.

Why Subword Tokenization & Attention Aggregation are necessary:
1. BLIP uses WordPiece/BERT tokenization. Rare, compound, or morphological words
   (e.g., 'croche', 'motorcycles') are decomposed into multiple subword tokens (e.g., ['cr', '##oche']).
2. Each subword token queries the ViT-B/16 image features individually in the Cross-Attention layer.
3. To visualize the visual grounding of the complete semantic word, the attention vectors
   of all constituent subwords must be aggregated:
       A_word = Mean( A_subword_1, A_subword_2, ... ) ∈ ℝ^(576)
   followed by spatial normalization and bilinear interpolation to the original image dimensions.
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
    Extracts authentic, word-level Cross-Attention from Salesforce/blip-image-captioning-base
    with subword token aggregation and spatial contrast enhancement.
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

    @staticmethod
    def group_subwords_into_words(
        input_ids: List[int],
        tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Group raw subword tokens (e.g., ['cr', '##oche']) into complete human words ('croche').
        Preserves exact word sequence position, token IDs, and subword indices.
        """
        words_data = []
        current_word_tokens = []
        current_word_ids = []
        current_word_indices = []
        current_word_str = ""

        for idx, (tok_id, tok) in enumerate(zip(input_ids, tokens)):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
                
            if tok.startswith("##"):
                current_word_tokens.append(tok)
                current_word_ids.append(tok_id)
                current_word_indices.append(idx)
                current_word_str += tok[2:]
            else:
                if current_word_tokens:
                    words_data.append({
                        "word_index": len(words_data) + 1,
                        "word": current_word_str,
                        "tokens": current_word_tokens,
                        "token_ids": current_word_ids,
                        "token_indices": current_word_indices,
                        "is_multi_token": len(current_word_tokens) > 1,
                        "num_subwords": len(current_word_tokens)
                    })
                current_word_tokens = [tok]
                current_word_ids = [tok_id]
                current_word_indices = [idx]
                current_word_str = tok

        if current_word_tokens:
            words_data.append({
                "word_index": len(words_data) + 1,
                "word": current_word_str,
                "tokens": current_word_tokens,
                "token_ids": current_word_ids,
                "token_indices": current_word_indices,
                "is_multi_token": len(current_word_tokens) > 1,
                "num_subwords": len(current_word_tokens)
            })

        return words_data

    def extract_attention_for_caption(
        self,
        image: Image.Image,
        caption: str
    ) -> Dict[str, Any]:
        """
        Extract authentic Cross-Attention for the EXACT caption and aggregate subwords to words.
        
        Returns:
            Dict containing:
              - 'caption': Full generated caption string
              - 'words_info': List of Word dictionaries with aggregated attention heatmaps
              - 'raw_attn_shape': Shape of the cross-attention tensor
        """
        img_w, img_h = image.size
        
        # 1. Vision Encoder Forward Pass
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values

        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[0]  # [1, 577, 768]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # 2. Tokenize the EXACT Caption
        caption_encoding = self.processor.tokenizer(caption, return_tensors="pt").to(self.device)
        input_ids_tensor = caption_encoding.input_ids[0]
        input_ids = input_ids_tensor.tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)

        # 3. Text Decoder Cross-Attention Pass
        with torch.no_grad():
            decoder_outputs = self.model.text_decoder(
                input_ids=input_ids_tensor.unsqueeze(0),
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=True,
                return_dict=True
            )

        # Tuple of 12 cross-attention tensors: [1, 12, seq_len, 577]
        cross_attns = decoder_outputs.cross_attentions
        raw_attn_shape = list(cross_attns[-1].shape)
        
        # 4. Aggregate High-Level Semantic Layers (Layers 8, 9, 10, 11)
        # Shape: [4, 1, 12, seq_len, 577]
        selected_layers = torch.stack(cross_attns[-4:], dim=0)
        # Mean across 4 layers and 12 heads -> [seq_len, 576] (dropping CLS token at index 0)
        raw_seq_attn = selected_layers.mean(dim=(0, 1, 2)).squeeze(0)[:, 1:] # [seq_len, 576]

        # 5. Token-Specific Relevance Grounding (Z-score to eliminate attention sinks)
        patch_mean = raw_seq_attn.mean(dim=0, keepdim=True)
        patch_std = raw_seq_attn.std(dim=0, keepdim=True) + 1e-6
        z_relevance = (raw_seq_attn - patch_mean) / patch_std
        positive_relevance = torch.clamp(z_relevance, min=0.0) # [seq_len, 576]

        # 6. Group Subwords into Words & Aggregate Attentions
        words_data = self.group_subwords_into_words(input_ids, tokens)
        
        words_info = []
        for w in words_data:
            indices = w["token_indices"]
            
            # Step 7 & 9: Aggregate subwords with mean() or use single token directly
            if len(indices) == 1:
                word_raw_attn = raw_seq_attn[indices[0]].cpu().numpy()
                word_relevance = positive_relevance[indices[0]].cpu().numpy()
            else:
                word_raw_attn = raw_seq_attn[indices].mean(dim=0).cpu().numpy()
                word_relevance = positive_relevance[indices].mean(dim=0).cpu().numpy()
                
            # Normalize spatial relevance
            if word_relevance.max() > word_relevance.min():
                norm_spatial = (word_relevance - word_relevance.min()) / (word_relevance.max() - word_relevance.min())
            else:
                norm_spatial = (word_raw_attn - word_raw_attn.min()) / (word_raw_attn.max() - word_raw_attn.min() + 1e-8)
                
            # Apply power curve for crisp visual contrast on regions
            norm_spatial = np.power(norm_spatial, 1.4)
            norm_spatial = (norm_spatial - norm_spatial.min()) / (norm_spatial.max() - norm_spatial.min() + 1e-8)
            
            # Reshape into 24x24 grid
            grid_24x24 = norm_spatial.reshape(24, 24)
            
            # Bilinear interpolation to original image size
            grid_tensor = torch.tensor(grid_24x24, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized_tensor = F.interpolate(
                grid_tensor,
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False
            ).squeeze().clamp(0.0, 1.0)
            
            resized_heatmap = resized_tensor.numpy()
            peak_patch = int(norm_spatial.argmax())
            
            words_info.append({
                "word_index": w["word_index"],
                "word": w["word"],
                "tokens": w["tokens"],
                "token_ids": w["token_ids"],
                "token_indices": w["token_indices"],
                "num_subwords": w["num_subwords"],
                "is_multi_token": w["is_multi_token"],
                "raw_patch_attn": word_raw_attn,
                "specific_relevance": word_relevance,
                "grid_24x24": grid_24x24,
                "resized_heatmap": resized_heatmap,
                "min_val": float(word_raw_attn.min()),
                "max_val": float(word_raw_attn.max()),
                "mean_val": float(word_raw_attn.mean()),
                "std_val": float(word_raw_attn.std()),
                "peak_patch_index": peak_patch,
                "peak_grid_y": peak_patch // 24,
                "peak_grid_x": peak_patch % 24
            })

        return {
            "caption": caption,
            "raw_attn_shape": raw_attn_shape,
            "words_info": words_info,
            "tokens_info": words_info  # Backward compatibility alias
        }

    def generate_and_extract_attention(
        self,
        image: Image.Image,
        num_beams: int = 5,
        max_length: int = 32
    ) -> Dict[str, Any]:
        """Generate caption first, then extract word-level cross-attention."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values

        with torch.no_grad():
            output_ids_tensor = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams
            )[0]

        caption = self.processor.decode(output_ids_tensor, skip_special_tokens=True).strip()
        return self.extract_attention_for_caption(image=image, caption=caption)

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
        
        if heatmap_2d.shape != (h, w):
            t = torch.tensor(heatmap_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            heatmap_2d = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False).squeeze().numpy()
            
        heatmap_2d = np.clip(heatmap_2d, 0.0, 1.0)
        cmap = cm.get_cmap(colormap)
        colored_rgba = cmap(heatmap_2d)
        colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(colored_rgb)
        
        return Image.blend(img_rgb, heatmap_pil, alpha=alpha)
