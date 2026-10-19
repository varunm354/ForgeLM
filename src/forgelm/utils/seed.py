"""Reproducible random seeding."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
