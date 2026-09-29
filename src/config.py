"""
Configuration module for Image Captioning with BLIP on Flickr8k.
Defines paths, model settings, hyperparameters, and experimental configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import torch


@dataclass
class PathConfig:
    """Project directory and file paths."""
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = ROOT_DIR / "data" / "flickr8k"
    IMAGES_DIR: Path = DATA_DIR / "Images"
    CAPTIONS_FILE: Path = DATA_DIR / "captions.txt"
    SPLITS_DIR: Path = DATA_DIR / "splits"
    
    CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"


@dataclass
class ModelConfig:
    """Pretrained Vision-Language Model configurations."""
    MODEL_NAME: str = "Salesforce/blip-image-captioning-base"
    IMAGE_SIZE: int = 384
    MAX_TEXT_LENGTH: int = 32
    MIN_TEXT_LENGTH: int = 5
    NUM_BEAMS: int = 5
    REPETITION_PENALTY: float = 1.2
    LENGTH_PENALTY: float = 1.0


@dataclass
class TrainingConfig:
    """Hyperparameters for Fine-Tuning."""
    # Seed for reproducibility
    SEED: int = 42
    
    # Batch & Epochs
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 0  # 0 is safe for Windows
    EPOCHS: int = 5
    GRADIENT_ACCUMULATION_STEPS: int = 2
    
    # Differential Learning Rates
    # ViT backbone needs smaller LR to preserve generic visual features
    LR_VISION_ENCODER: float = 5e-6
    LR_TEXT_DECODER: float = 5e-5
    WEIGHT_DECAY: float = 0.05
    WARMUP_RATIO: float = 0.1
    MAX_GRAD_NORM: float = 1.0
    
    # Label smoothing & mixed precision
    LABEL_SMOOTHING: float = 0.1
    USE_AMP: bool = torch.cuda.is_available()
    
    # Strategies: 'frozen_vision' or 'full_finetune'
    STRATEGY: str = "frozen_vision"


@dataclass
class EvaluationConfig:
    """Evaluation settings & metrics."""
    METRICS: List[str] = field(default_factory=lambda: ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"])
    EVAL_BATCH_SIZE: int = 16
    BEAM_SEARCH_BEAMS: List[int] = field(default_factory=lambda: [1, 3, 5])


class Config:
    """Unified configuration class."""
    paths = PathConfig()
    model = ModelConfig()
    training = TrainingConfig()
    eval = EvaluationConfig()
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Auto-detect computation device (CUDA -> MPS -> CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    @classmethod
    def create_dirs(cls) -> None:
        """Ensure all required project directories exist."""
        cls.paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.paths.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
