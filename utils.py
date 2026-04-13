"""
utils.py — Shared utility functions.
"""
import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {path}")


def save_loss_history(loss_history: list, path: str):
    """Save loss history as JSON list of [iter, loss] pairs."""
    data = [[it, loss] for it, loss in loss_history]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved loss history: {path}")


def save_samples(samples: np.ndarray, path: str):
    """Save numpy array of generated samples."""
    np.save(path, samples)
    print(f"  Saved samples: {path}")
