"""Runtime device selection."""

import torch


def get_device() -> torch.device:
    """Return CUDA if available, otherwise Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
