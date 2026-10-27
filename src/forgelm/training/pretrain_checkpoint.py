"""Atomic, exact-resume checkpoints for production GPT pretraining."""

import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from forgelm.models.gpt import GPTConfig
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_runtime import ShardBatchSampler

CHECKPOINT_FORMAT = "forgelm.gpt_pretraining"
CHECKPOINT_VERSION = 1


@dataclass
class TrainingProgress:
    """Counters restored exactly at optimizer-step boundaries."""

    optimizer_step: int = 0
    micro_step: int = 0
    tokens_processed: int = 0
    best_validation_loss: float = float("inf")


def capture_rank_state(
    training_sampler: ShardBatchSampler,
    validation_sampler: ShardBatchSampler,
) -> dict[str, Any]:
    """Capture process-local RNG and sampler states."""
    numpy_state = np.random.get_state()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "training_sampler_state": training_sampler.state_dict(),
        "validation_sampler_state": validation_sampler.state_dict(),
    }


def restore_rank_state(
    state: Mapping[str, Any],
    training_sampler: ShardBatchSampler,
    validation_sampler: ShardBatchSampler,
) -> None:
    """Restore process-local RNG and exact sampler positions."""
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "cuda_rng_states",
        "training_sampler_state",
        "validation_sampler_state",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(
            f"Checkpoint rank state is missing: {', '.join(sorted(missing))}"
        )
    random.setstate(state["python_random_state"])
    numpy_state = state["numpy_random_state"]
    if not isinstance(numpy_state, dict):
        raise TypeError("NumPy RNG checkpoint state must be a mapping.")
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    cuda_states = state["cuda_rng_states"]
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    training_sampler.load_state_dict(state["training_sampler_state"])
    validation_sampler.load_state_dict(state["validation_sampler_state"])


def save_pretrain_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    progress: TrainingProgress,
    training_config: PretrainConfig,
    model_config: GPTConfig,
    manifest_identity: str,
    tokenizer_fingerprint: str,
    rank_states: Sequence[Mapping[str, Any]],
    world_size: int,
    precision: str,
) -> None:
    """Atomically write all state needed for exact boundary resumption."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "gradient_scaler_state": None if scaler is None else scaler.state_dict(),
        "optimizer_step": progress.optimizer_step,
        "micro_step": progress.micro_step,
        "tokens_processed": progress.tokens_processed,
        "best_validation_loss": progress.best_validation_loss,
        "training_config": training_config.to_dict(),
        "model_config": model_config.to_dict(),
        "manifest_identity": manifest_identity,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "rank_states": list(rank_states),
        "world_size": world_size,
        "precision": precision,
    }
    try:
        torch.save(checkpoint, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_exact_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    training_config: PretrainConfig,
    model_config: GPTConfig,
    manifest_identity: str,
    tokenizer_fingerprint: str,
    training_sampler: ShardBatchSampler,
    validation_sampler: ShardBatchSampler,
    rank: int,
    world_size: int,
    precision: str,
    device: torch.device,
) -> TrainingProgress:
    """Strictly validate and restore an exact training checkpoint."""
    checkpoint = _load_checkpoint(path, device)
    _validate_compatibility(
        checkpoint,
        training_config=training_config,
        model_config=model_config,
        manifest_identity=manifest_identity,
        tokenizer_fingerprint=tokenizer_fingerprint,
        world_size=world_size,
        precision=precision,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    saved_scaler = checkpoint["gradient_scaler_state"]
    if scaler is None and saved_scaler is not None:
        raise ValueError("Checkpoint requires an FP16 gradient scaler.")
    if scaler is not None:
        if saved_scaler is None:
            raise ValueError("Checkpoint does not contain an FP16 scaler.")
        scaler.load_state_dict(saved_scaler)
    rank_states = checkpoint["rank_states"]
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("Checkpoint rank-state count is incompatible.")
    restore_rank_state(
        rank_states[rank],
        training_sampler,
        validation_sampler,
    )
    return TrainingProgress(
        optimizer_step=_nonnegative_int(checkpoint, "optimizer_step"),
        micro_step=_nonnegative_int(checkpoint, "micro_step"),
        tokens_processed=_nonnegative_int(checkpoint, "tokens_processed"),
        best_validation_loss=float(checkpoint["best_validation_loss"]),
    )


def load_weights_only(
    path: str | Path,
    *,
    model: nn.Module,
    model_config: GPTConfig,
    device: torch.device,
) -> None:
    """Load model weights without optimizer, sampler, counter, or RNG state."""
    checkpoint = _load_checkpoint(path, device)
    if checkpoint.get("model_config") != model_config.to_dict():
        raise ValueError("Weights-only checkpoint model configuration mismatch.")
    model.load_state_dict(checkpoint["model_state"], strict=True)


def _load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            Path(path),
            map_location=device,
            weights_only=False,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not load pretraining checkpoint: {path}") from error
    if not isinstance(checkpoint, dict):
        raise TypeError("Pretraining checkpoint must contain a dictionary.")
    required = {
        "format",
        "version",
        "model_state",
        "optimizer_state",
        "gradient_scaler_state",
        "optimizer_step",
        "micro_step",
        "tokens_processed",
        "best_validation_loss",
        "training_config",
        "model_config",
        "manifest_identity",
        "tokenizer_fingerprint",
        "rank_states",
        "world_size",
        "precision",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Checkpoint is missing fields: {', '.join(sorted(missing))}"
        )
    if (
        checkpoint["format"] != CHECKPOINT_FORMAT
        or checkpoint["version"] != CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported pretraining checkpoint format.")
    return checkpoint


def _validate_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    training_config: PretrainConfig,
    model_config: GPTConfig,
    manifest_identity: str,
    tokenizer_fingerprint: str,
    world_size: int,
    precision: str,
) -> None:
    expected = {
        "training_config": training_config.to_dict(),
        "model_config": model_config.to_dict(),
        "manifest_identity": manifest_identity,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "world_size": world_size,
        "precision": precision,
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"Checkpoint compatibility mismatch: {name}.")


def _nonnegative_int(checkpoint: Mapping[str, Any], name: str) -> int:
    value = checkpoint[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Checkpoint field {name} must be non-negative.")
    return value
