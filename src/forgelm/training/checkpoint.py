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
    metadata: Mapping[str, Any] | None = None,
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
            "metadata": dict(metadata or {}),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
    tokenizer_vocab_size: int | None = None,
    tokenizer_fingerprint: str | None = None,
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

    if tokenizer_vocab_size is not None or tokenizer_fingerprint is not None:
        if tokenizer_vocab_size is None or tokenizer_fingerprint is None:
            raise ValueError(
                "Both tokenizer_vocab_size and tokenizer_fingerprint are required."
            )
        verify_tokenizer_compatibility(
            checkpoint,
            vocab_size=tokenizer_vocab_size,
            fingerprint=tokenizer_fingerprint,
        )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def verify_tokenizer_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    vocab_size: int,
    fingerprint: str,
) -> None:
    """Reject a checkpoint created for a different tokenizer."""
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint does not contain tokenizer metadata.")

    saved_vocab_size = metadata.get("tokenizer_vocab_size")
    if saved_vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary mismatch: checkpoint uses {saved_vocab_size}, "
            f"but the loaded tokenizer uses {vocab_size}."
        )

    saved_fingerprint = metadata.get("tokenizer_fingerprint")
    if saved_fingerprint != fingerprint:
        raise ValueError(
            "Tokenizer fingerprint mismatch: the checkpoint and tokenizer "
            "were built from different merge rules."
        )
