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
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint rank state must be a mapping.")
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

    python_state = state["python_random_state"]
    try:
        python_probe = random.Random()
        python_probe.setstate(python_state)
    except (TypeError, ValueError) as error:
        raise ValueError("Python RNG checkpoint state is malformed.") from error

    numpy_state = state["numpy_random_state"]
    if not isinstance(numpy_state, Mapping):
        raise TypeError("NumPy RNG checkpoint state must be a mapping.")
    numpy_required = {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    numpy_missing = numpy_required - numpy_state.keys()
    if numpy_missing:
        raise ValueError(
            "NumPy RNG checkpoint state is missing: "
            + ", ".join(sorted(numpy_missing))
        )
    try:
        numpy_tuple = (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(numpy_tuple)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("NumPy RNG checkpoint state is malformed.") from error

    cpu_rng_state = normalize_rng_state(
        state["torch_cpu_rng_state"],
        field_name="torch_cpu_rng_state",
    )
    cuda_rng_states_value = state["cuda_rng_states"]
    if not isinstance(cuda_rng_states_value, (list, tuple)):
        raise TypeError("cuda_rng_states must be a list or tuple of tensors.")
    cuda_rng_states = [
        normalize_rng_state(
            cuda_state,
            field_name=f"cuda_rng_states[{index}]",
        )
        for index, cuda_state in enumerate(cuda_rng_states_value)
    ]
    training_sampler_state = _normalize_sampler_state(
        state["training_sampler_state"],
        field_name="training_sampler_state",
    )
    validation_sampler_state = _normalize_sampler_state(
        state["validation_sampler_state"],
        field_name="validation_sampler_state",
    )

    cuda_available = torch.cuda.is_available()
    if cuda_available and len(cuda_rng_states) != torch.cuda.device_count():
        raise ValueError(
            "CUDA RNG checkpoint state count does not match visible CUDA devices."
        )
    if not cuda_available and cuda_rng_states:
        raise ValueError(
            "Checkpoint contains CUDA RNG states, but CUDA is unavailable."
        )

    random.setstate(python_state)
    np.random.set_state(numpy_tuple)
    torch.set_rng_state(cpu_rng_state)
    if cuda_available:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    training_sampler.load_state_dict(training_sampler_state)
    validation_sampler.load_state_dict(validation_sampler_state)


def normalize_rng_state(
    value: Any,
    *,
    field_name: str,
) -> torch.Tensor:
    """Return an RNG state in the CPU ByteTensor form required by PyTorch."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise TypeError(f"{field_name} must be a strided tensor.")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{field_name} must be a non-empty one-dimensional tensor.")

    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if value.dtype not in integer_dtypes:
        raise TypeError(f"{field_name} must contain integer byte values.")
    try:
        cpu_value = value.detach().to(device="cpu")
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"{field_name} could not be copied to CPU.") from error
    if value.dtype != torch.uint8 and (
        torch.any(cpu_value < 0).item() or torch.any(cpu_value > 255).item()
    ):
        raise ValueError(f"{field_name} values must be between 0 and 255.")
    return cpu_value.to(dtype=torch.uint8).contiguous()


def _normalize_sampler_state(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    if "generator_state" not in value:
        raise ValueError(f"{field_name} is missing generator_state.")
    normalized = dict(value)
    normalized["generator_state"] = normalize_rng_state(
        value["generator_state"],
        field_name=f"{field_name}.generator_state",
    )
    return normalized


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
    checkpoint = _load_checkpoint(path)
    _require_fields(
        checkpoint,
        {
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
        },
    )
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
    checkpoint = _load_checkpoint(path)
    _require_fields(checkpoint, {"model_state", "model_config"})
    if checkpoint.get("model_config") != model_config.to_dict():
        raise ValueError("Weights-only checkpoint model configuration mismatch.")
    model.load_state_dict(checkpoint["model_state"], strict=True)


def _load_checkpoint(
    path: str | Path,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            Path(path),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not load pretraining checkpoint: {path}") from error
    if not isinstance(checkpoint, dict):
        raise TypeError("Pretraining checkpoint must contain a dictionary.")
    _require_fields(checkpoint, {"format", "version"})
    if (
        checkpoint["format"] != CHECKPOINT_FORMAT
        or checkpoint["version"] != CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported pretraining checkpoint format.")
    return checkpoint


def _require_fields(
    checkpoint: Mapping[str, Any],
    required: set[str],
) -> None:
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Checkpoint is missing fields: {', '.join(sorted(missing))}"
        )


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
