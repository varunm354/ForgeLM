"""Small helpers for saving and restoring training checkpoints."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    step: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Save training state and metadata to a checkpoint file."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": dict(config),
            "metrics": dict(metrics),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load training state onto a requested device and return its metadata."""
    checkpoint = torch.load(
        Path(path),
        map_location=torch.device(device),
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")

    required = {"model_state", "optimizer_state", "step", "config", "metrics"}
    missing = required.difference(checkpoint)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Checkpoint is missing required fields: {names}.")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
