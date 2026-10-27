"""Runtime primitives for reproducible single-process and DDP pretraining."""

import inspect
import math
import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from forgelm.data.token_shards import (
    MemoryMappedShardDataset,
    load_manifest,
    sha256_file,
)
from forgelm.models.gpt import GPTConfig
from forgelm.training.pretrain_config import PrecisionName, PretrainConfig


@dataclass(frozen=True)
class DistributedEnvironment:
    """Standard torchrun process coordinates."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


@dataclass
class DistributedContext:
    """Initialized process and device state."""

    environment: DistributedEnvironment
    device: torch.device
    backend: str | None
    initialized_here: bool

    @property
    def rank(self) -> int:
        return self.environment.rank

    @property
    def local_rank(self) -> int:
        return self.environment.local_rank

    @property
    def world_size(self) -> int:
        return self.environment.world_size

    @property
    def distributed(self) -> bool:
        return self.environment.distributed

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def shutdown(self) -> None:
        """Destroy only a process group initialized by this context."""
        if self.initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self.initialized_here = False


@dataclass(frozen=True)
class PrecisionPolicy:
    """Selected arithmetic and gradient-scaling policy."""

    name: str
    autocast_dtype: torch.dtype | None
    uses_grad_scaler: bool

    def autocast(
        self,
        device: torch.device,
    ) -> AbstractContextManager[Any]:
        """Return the correct autocast or no-op context."""
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=device.type,
            dtype=self.autocast_dtype,
        )

    def create_grad_scaler(self) -> torch.amp.GradScaler | None:
        """Create a CUDA scaler only for selected float16 training."""
        if not self.uses_grad_scaler:
            return None
        return torch.amp.GradScaler("cuda")


@dataclass
class OptimizerSetup:
    """AdamW plus inspectable grouping metadata."""

    optimizer: torch.optim.AdamW
    decay_parameter_names: tuple[str, ...]
    no_decay_parameter_names: tuple[str, ...]
    fused: bool


class ShardBatchSampler:
    """Stateful deterministic sampling over a memory-mapped split."""

    def __init__(
        self,
        dataset: MemoryMappedShardDataset,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if not isinstance(dataset, MemoryMappedShardDataset):
            raise TypeError("dataset must be a MemoryMappedShardDataset.")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.batches_sampled = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one CPU batch and advance the stored generator."""
        batch = self.dataset.get_batch(
            self.batch_size,
            generator=self.generator,
        )
        self.batches_sampled += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Return exact sampler state for checkpointing."""
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batches_sampled": self.batches_sampled,
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact sampling position after validating fixed settings."""
        if state.get("seed") != self.seed:
            raise ValueError("Sampler seed is incompatible with checkpoint.")
        if state.get("batch_size") != self.batch_size:
            raise ValueError("Sampler batch size is incompatible with checkpoint.")
        batches_sampled = state.get("batches_sampled")
        generator_state = state.get("generator_state")
        if (
            not isinstance(batches_sampled, int)
            or isinstance(batches_sampled, bool)
            or batches_sampled < 0
            or not isinstance(generator_state, torch.Tensor)
        ):
            raise ValueError("Checkpoint sampler state is malformed.")
        self.batches_sampled = batches_sampled
        self.generator.set_state(generator_state.cpu())


def parse_distributed_environment(
    environment: Mapping[str, str] | None = None,
) -> DistributedEnvironment:
    """Parse torchrun coordinates or return the single-process defaults."""
    values = os.environ if environment is None else environment
    names = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    present = [name in values for name in names]
    if not any(present):
        return DistributedEnvironment(rank=0, local_rank=0, world_size=1)
    if not all(present):
        raise ValueError("RANK, LOCAL_RANK, and WORLD_SIZE must be set together.")
    try:
        rank, local_rank, world_size = (int(values[name]) for name in names)
    except ValueError as error:
        raise ValueError("torchrun environment values must be integers.") from error
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank or world-size values.")
    return DistributedEnvironment(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def initialize_distributed(
    device_override: str = "auto",
    *,
    environment: Mapping[str, str] | None = None,
) -> DistributedContext:
    """Select one process device and initialize NCCL or Gloo when required."""
    coordinates = parse_distributed_environment(environment)
    device = _select_device(device_override, coordinates.local_rank)
    backend = None
    initialized_here = False
    if coordinates.distributed:
        if device.type == "mps":
            raise ValueError("Distributed MPS training is not supported.")
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                rank=coordinates.rank,
                world_size=coordinates.world_size,
            )
            initialized_here = True
    return DistributedContext(
        environment=coordinates,
        device=device,
        backend=backend,
        initialized_here=initialized_here,
    )


def select_precision(
    requested: PrecisionName,
    device: torch.device,
    *,
    cuda_bf16_supported: bool | None = None,
) -> PrecisionPolicy:
    """Resolve auto or explicit precision without claiming unsupported devices."""
    if requested not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported precision: {requested}")
    if requested == "float32":
        return PrecisionPolicy("float32", None, False)
    if device.type != "cuda":
        if requested != "auto":
            raise ValueError(
                f"{requested} mixed precision requires a CUDA device."
            )
        return PrecisionPolicy("float32", None, False)

    bf16_supported = (
        torch.cuda.is_bf16_supported()
        if cuda_bf16_supported is None
        else cuda_bf16_supported
    )
    if requested == "auto":
        requested = "bfloat16" if bf16_supported else "float16"
    if requested == "bfloat16":
        if not bf16_supported:
            raise ValueError("CUDA device does not report bfloat16 support.")
        return PrecisionPolicy("bfloat16", torch.bfloat16, False)
    if requested == "float16":
        return PrecisionPolicy("float16", torch.float16, True)
    raise RuntimeError("Unreachable precision selection.")


def create_adamw(
    model: nn.Module,
    config: PretrainConfig,
    *,
    device: torch.device,
) -> OptimizerSetup:
    """Group each unique trainable parameter once with intentional decay."""
    seen: set[tuple[int, int, int]] = set()
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        key = (
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
            parameter.numel(),
        )
        if key in seen:
            continue
        seen.add(key)
        if parameter.ndim >= 2:
            decay.append(parameter)
            decay_names.append(name)
        else:
            no_decay.append(parameter)
            no_decay_names.append(name)
    if not decay or not no_decay:
        raise ValueError("AdamW grouping requires decay and no-decay parameters.")

    fused_supported = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = device.type == "cuda" and fused_supported
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": config.weight_decay,
                "group_name": "decay",
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "group_name": "no_decay",
            },
        ],
        lr=learning_rate_at_step(
            0,
            total_steps=config.optimizer_steps(1),
            warmup_steps=config.warmup_steps,
            maximum=config.learning_rate,
            minimum=config.minimum_learning_rate,
        ),
        betas=config.betas,
        eps=config.epsilon,
        fused=use_fused,
    )
    return OptimizerSetup(
        optimizer=optimizer,
        decay_parameter_names=tuple(decay_names),
        no_decay_parameter_names=tuple(no_decay_names),
        fused=use_fused,
    )


def learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    maximum: float,
    minimum: float,
) -> float:
    """Linear warmup through its boundary, then cosine decay to the minimum."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer.")
    if not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer.")
    if not isinstance(warmup_steps, int) or not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps.")
    if minimum < 0 or maximum <= 0 or minimum > maximum:
        raise ValueError("Invalid learning-rate bounds.")
    bounded_step = min(step, total_steps)
    if warmup_steps > 0 and bounded_step <= warmup_steps:
        return maximum * bounded_step / warmup_steps
    if total_steps == warmup_steps:
        return maximum
    progress = (bounded_step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(minimum, minimum + (maximum - minimum) * cosine)


def validate_training_compatibility(
    manifest_path: os.PathLike[str] | str,
    model_config: GPTConfig,
    training_config: PretrainConfig,
) -> tuple[dict[str, Any], str]:
    """Cross-check model, tokenizer, storage, sequence, and manifest identity."""
    manifest = load_manifest(manifest_path)
    tokenizer = manifest["tokenizer"]
    if tokenizer["vocabulary_size"] != model_config.vocab_size:
        raise ValueError("Model vocabulary does not match shard manifest.")
    if training_config.sequence_length > model_config.block_size:
        raise ValueError("Training sequence length exceeds model block size.")
    if (
        training_config.sequence_length
        > model_config.rope_max_sequence_length
    ):
        raise ValueError("Training sequence length exceeds RoPE capacity.")
    expected = training_config.expected_tokenizer_fingerprint
    if expected is not None and tokenizer["fingerprint"] != expected:
        raise ValueError("Configured tokenizer fingerprint does not match manifest.")
    if manifest["storage"] != {"dtype": "uint16", "byte_order": "little"}:
        raise ValueError("Training requires little-endian uint16 shards.")
    return manifest, sha256_file(manifest_path)


def _select_device(name: str, local_rank: int) -> torch.device:
    if name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {name}")
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        return torch.device("cuda", local_rank)
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(name)
