"""
Utility functions for reproducibility, checkpointing, and logging.
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str = "ImageCaptioning", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure standard console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already added
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


def save_checkpoint(
    state_dict: Dict[str, Any],
    save_path: Path,
    is_best: bool = False,
    best_path: Optional[Path] = None
) -> None:
    """Save PyTorch training checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, save_path)
    if is_best and best_path:
        torch.save(state_dict, best_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load PyTorch training checkpoint onto specified device."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def save_json(data: Any, file_path: Path) -> None:
    """Save structured dictionary to formatted JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
