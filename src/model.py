"""
Vision-Language Model wrapper module based on Salesforce/blip-image-captioning-base.
Supports flexible fine-tuning strategies (Freeze/Unfreeze Vision Encoder, Differential LR)
and cross-attention extraction for visual grounding analysis.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    BlipConfig
)

from src.config import Config


class BLIPCaptioningModel(nn.Module):
    """
    Modular Wrapper for BLIP Image Captioning Model.
    
    Architecture components:
      1. Vision Encoder: Vision Transformer (ViT-B/16)
      2. Text Decoder: Transformer Decoder with Cross-Attention
      3. Language Modeling Head: Linear projection to vocabulary
    """
    
    def __init__(
        self,
        model_name: str = Config.model.MODEL_NAME,
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        if pretrained:
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        else:
            config = BlipConfig.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration(config)
            
        self.cross_attentions = []

    def freeze_vision_encoder(self) -> None:
        """
        Freeze all parameters in the Vision Transformer (ViT) Backbone.
        Forces the model to rely solely on pretrained visual features while
        adapting the Cross-Attention & Text Decoder layers.
        """
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

    def unfreeze_vision_encoder(self) -> None:
        """Unfreeze the Vision Transformer Backbone for Full Fine-Tuning."""
        for param in self.model.vision_model.parameters():
            param.requires_grad = True

    def freeze_text_decoder(self) -> None:
        """Freeze the text decoder (for ablation / visual representation probing)."""
        for param in self.model.text_decoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters across the entire multimodal network."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params_stats(self) -> Dict[str, Any]:
        """Inspect and return parameter counts (total vs trainable)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        vision_total = sum(p.numel() for p in self.model.vision_model.parameters())
        vision_trainable = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
        
        decoder_total = sum(p.numel() for p in self.model.text_decoder.parameters())
        decoder_trainable = sum(p.numel() for p in self.model.text_decoder.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percent": (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0,
            "vision_encoder": {"total": vision_total, "trainable": vision_trainable},
            "text_decoder": {"total": decoder_total, "trainable": decoder_trainable}
        }

    def get_optimizer_param_groups(
        self,
        lr_decoder: float = Config.training.LR_TEXT_DECODER,
        lr_vision: float = Config.training.LR_VISION_ENCODER,
        weight_decay: float = Config.training.WEIGHT_DECAY
    ) -> List[Dict[str, Any]]:
        """
        Construct differential learning rate parameter groups:
        - Vision Transformer Backbone: lower learning rate (lr_vision)
        - Cross-Attention & Text Decoder: higher learning rate (lr_decoder)
        - Exclude bias and LayerNorm from weight decay
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        vision_params_decay = []
        vision_params_nodecay = []
        decoder_params_decay = []
        decoder_params_nodecay = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            is_nodecay = any(nd in name for nd in no_decay)
            if "vision_model" in name:
                if is_nodecay:
                    vision_params_nodecay.append(param)
                else:
                    vision_params_decay.append(param)
            else:
                if is_nodecay:
                    decoder_params_nodecay.append(param)
                else:
                    decoder_params_decay.append(param)
                    
        groups = []
        if vision_params_decay:
            groups.append({"params": vision_params_decay, "lr": lr_vision, "weight_decay": weight_decay})
        if vision_params_nodecay:
            groups.append({"params": vision_params_nodecay, "lr": lr_vision, "weight_decay": 0.0})
        if decoder_params_decay:
            groups.append({"params": decoder_params_decay, "lr": lr_decoder, "weight_decay": weight_decay})
        if decoder_params_nodecay:
            groups.append({"params": decoder_params_nodecay, "lr": lr_decoder, "weight_decay": 0.0})
            
        return groups

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Forward pass for training with conditional language modeling loss.
        """
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )

    def generate_caption(
        self,
        image_or_pixel_values: Union[Image.Image, torch.Tensor],
        method: str = "beam",
        num_beams: int = Config.model.NUM_BEAMS,
        max_length: int = Config.model.MAX_TEXT_LENGTH,
        min_length: int = Config.model.MIN_TEXT_LENGTH,
        repetition_penalty: float = Config.model.REPETITION_PENALTY,
        length_penalty: float = Config.model.LENGTH_PENALTY,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ) -> Union[str, List[str]]:
        """
        Autoregressively generate captions for input image(s).
        
        Args:
            image_or_pixel_values: PIL Image or preprocessed PyTorch Tensor.
            method: 'greedy', 'beam', or 'sampling'.
            num_beams: Beam search width (active if method='beam').
            max_length: Maximum token length for caption.
            min_length: Minimum token length.
        """
        self.eval()
        dev = device or next(self.parameters()).device
        
        if isinstance(image_or_pixel_values, Image.Image):
            inputs = self.processor(images=image_or_pixel_values, return_tensors="pt").to(dev)
            pixel_values = inputs.pixel_values
            is_single = True
        else:
            pixel_values = image_or_pixel_values.to(dev)
            is_single = pixel_values.dim() == 3 or (pixel_values.dim() == 4 and pixel_values.size(0) == 1)
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
                
        do_sample = method == "sampling"
        actual_beams = 1 if method == "greedy" else num_beams
        
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                num_beams=actual_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0
            )
            
        captions = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        cleaned = [cap.strip() for cap in captions]
        
        return cleaned[0] if is_single and len(cleaned) == 1 else cleaned

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save both model weights and processor."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        load_directory: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> "BLIPCaptioningModel":
        """Load fine-tuned model and processor from checkpoint directory."""
        load_path = Path(load_directory)
        instance = cls(model_name=str(load_path), pretrained=True)
        if device:
            instance.to(device)
        return instance
