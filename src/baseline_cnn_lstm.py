"""
Traditional Baseline Image Captioning Model: ResNet50 Encoder + LSTM Decoder.
Implements the classic "Show and Tell" paradigm (Vinyals et al.) for comparison
against modern Vision-Language Models (BLIP).
"""

import sys
from pathlib import Path
import json
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("BaselineCNNLSTM")


class Vocabulary:
    """
    Vocabulary manager for Token-to-Index and Index-to-Token mapping.
    Includes special tokens: <pad>, <sos>, <eos>, <unk>.
    """
    def __init__(self, freq_threshold: int = 2):
        self.freq_threshold = freq_threshold
        self.itos: Dict[int, str] = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize English text into lowercase clean words."""
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocabulary(self, sentence_list: List[str]) -> None:
        """Build vocabulary from a list of caption strings based on frequency threshold."""
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)

        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
        logger.info(f"Built vocabulary with {len(self.itos)} unique tokens (freq_threshold={self.freq_threshold}).")

    def numericalize(self, text: str) -> List[int]:
        """Convert a sentence string to a list of token indices."""
        tokens = self.tokenize(text)
        return [
            self.stoi.get(token, self.unk_idx)
            for token in tokens
        ]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Convert a list of token indices back to a sentence string."""
        words = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if remove_special and idx in (self.pad_idx, self.sos_idx, self.eos_idx):
                continue
            word = self.itos.get(idx, "<unk>")
            words.append(word)
        return " ".join(words)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to JSON."""
        data = {
            "freq_threshold": self.freq_threshold,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.stoi = data["stoi"]
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        return vocab


class EncoderCNN(nn.Module):
    """
    ResNet50 Convolutional Vision Backbone.
    Extracts global image features (2048-d) and projects to embed_size (512-d).
    """
    def __init__(self, embed_size: int = 512, train_cnn: bool = False):
        super().__init__()
        self.train_cnn = train_cnn
        
        # Load Pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final Classification FC layer, keep everything up to AdaptiveAvgPool2d
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear Projection: 2048 -> embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.freeze_or_unfreeze_resnet(train_cnn)

    def freeze_or_unfreeze_resnet(self, train_cnn: bool) -> None:
        """Control CNN backbone gradient updates."""
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input: images Tensor of shape [B, 3, 224, 224]
        Output: visual feature vector of shape [B, embed_size]
        """
        # [B, 2048, 1, 1]
        features = self.resnet(images)
        # [B, 2048]
        features = features.reshape(features.size(0), -1)
        # [B, embed_size]
        features = self.dropout(self.relu(self.linear(features)))
        return features


class DecoderLSTM(nn.Module):
    """
    Autoregressive LSTM Text Decoder.
    Generates word sequences conditioned on the visual feature vector.
    """
    def __init__(
        self,
        embed_size: int = 512,
        hidden_size: int = 512,
        vocab_size: int = 5000,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Teacher Forcing during training.
        
        Args:
            features: [B, embed_size] visual features from EncoderCNN
            captions: [B, seq_len] token IDs
            
        Returns:
            outputs: [B, seq_len, vocab_size] prediction logits
        """
        # Embed word tokens: [B, seq_len - 1, embed_size]
        # (remove <eos> token from input sequence to match lengths)
        embeddings = self.dropout(self.embedding(captions[:, :-1]))
        
        # Concatenate visual feature as the first time step token: [B, 1, embed_size] + [B, seq_len - 1, embed_size]
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # LSTM forward: [B, seq_len, hidden_size]
        hiddens, _ = self.lstm(inputs)
        
        # Project to vocabulary: [B, seq_len, vocab_size]
        outputs = self.linear(hiddens)
        return outputs


class CNNtoLSTM(nn.Module):
    """
    Complete Baseline Model Pipeline combining ResNet50 Encoder + LSTM Decoder.
    """
    def __init__(
        self,
        embed_size: int = 512,
        hidden_size: int = 512,
        vocab_size: int = 5000,
        num_layers: int = 1,
        train_cnn: bool = False
    ):
        super().__init__()
        self.encoder = EncoderCNN(embed_size=embed_size, train_cnn=train_cnn)
        self.decoder = DecoderLSTM(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers
        )
        
        # Standard CNN preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """End-to-end training forward pass."""
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(
        self,
        image_tensor: torch.Tensor,
        vocab: Vocabulary,
        max_length: int = 32,
        device: torch.device = torch.device("cpu")
    ) -> str:
        """
        Autoregressive Greedy Generation of caption for an input image.
        
        Args:
            image_tensor: [1, 3, 224, 224] or [3, 224, 224]
        """
        self.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
                
            image_tensor = image_tensor.to(device)
            # 1. Trích xuất đặc trưng thị giác
            features = self.encoder(image_tensor) # [1, embed_size]
            
            # 2. Bắt đầu giải mã với visual feature
            states = None
            inputs = features.unsqueeze(1) # [1, 1, embed_size]
            
            predicted_ids = []
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(inputs, states)
                output = self.decoder.linear(hiddens.squeeze(1)) # [1, vocab_size]
                predicted_id = output.argmax(1).item()
                
                if predicted_id == vocab.eos_idx:
                    break
                    
                predicted_ids.append(predicted_id)
                
                # Cập nhật input cho bước thời gian tiếp theo
                inputs = self.decoder.embedding(torch.tensor([[predicted_id]], device=device))
                
        return vocab.decode(predicted_ids, remove_special=True)

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model weights and hyperparameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "embed_size": self.decoder.embed_size,
            "hidden_size": self.decoder.hidden_size,
            "vocab_size": self.decoder.vocab_size,
            "num_layers": self.decoder.num_layers
        }, path)

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: torch.device = torch.device("cpu")) -> "CNNtoLSTM":
        """Load trained baseline checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            embed_size=checkpoint["embed_size"],
            hidden_size=checkpoint["hidden_size"],
            vocab_size=checkpoint["vocab_size"],
            num_layers=checkpoint["num_layers"],
            train_cnn=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
