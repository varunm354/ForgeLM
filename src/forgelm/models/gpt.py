"""Production-oriented decoder-only GPT components for cloud pretraining."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import torch
import yaml
from torch import nn
from torch.nn import functional as F

UINT16_VOCABULARY_LIMIT = 1 << 16


@dataclass(frozen=True)
class GPTConfig:
    """Validated architecture settings for the production GPT model."""

    vocab_size: int = 16_384
    block_size: int = 1_024
    embedding_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    feed_forward_hidden_dim: int = 1_536
    dropout: float = 0.0
    linear_bias: bool = False
    rope_base: float = 10_000.0
    rope_max_sequence_length: int = 1_024
    rms_norm_epsilon: float = 1e-5
    tie_embeddings: bool = True
    initialization_std: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "embedding_dim",
            "num_layers",
            "num_heads",
            "head_dim",
            "feed_forward_hidden_dim",
            "rope_max_sequence_length",
        ):
            _validate_positive_integer(getattr(self, name), name)
        if self.vocab_size > UINT16_VOCABULARY_LIMIT:
            raise ValueError(
                "vocab_size cannot exceed the uint16 pipeline limit of 65536."
            )
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings.")
        if self.embedding_dim != self.num_heads * self.head_dim:
            raise ValueError("embedding_dim must equal num_heads * head_dim.")
        if self.block_size > self.rope_max_sequence_length:
            raise ValueError(
                "block_size cannot exceed rope_max_sequence_length."
            )
        _validate_probability(self.dropout, "dropout")
        _validate_positive_number(self.rope_base, "rope_base")
        _validate_positive_number(self.rms_norm_epsilon, "rms_norm_epsilon")
        _validate_positive_number(self.initialization_std, "initialization_std")
        if not isinstance(self.linear_bias, bool):
            raise TypeError("linear_bias must be a boolean.")
        if self.tie_embeddings is not True:
            raise ValueError("tie_embeddings must be true for this architecture.")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a strict YAML configuration."""
        source = Path(path)
        try:
            values = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"Could not load GPT configuration: {source}") from error
        if not isinstance(values, dict):
            raise TypeError("GPT configuration must be a YAML mapping.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(values.keys() - allowed)
        if unknown:
            raise ValueError(f"Unknown GPT configuration fields: {', '.join(unknown)}")
        try:
            return cls(**values)
        except TypeError as error:
            raise TypeError(f"Invalid GPT configuration in {source}: {error}") from error

    def to_dict(self) -> dict[str, Any]:
        """Return all architecture values as a plain mapping."""
        return asdict(self)


class RMSNorm(nn.Module):
    """Scale inputs by their root-mean-square magnitude without centering."""

    def __init__(self, dimension: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        _validate_positive_integer(dimension, "dimension")
        _validate_positive_number(epsilon, "epsilon")
        self.dimension = dimension
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension while preserving shape and input dtype."""
        _validate_hidden_states(inputs, self.dimension)
        float_inputs = inputs.float()
        inverse_rms = torch.rsqrt(
            float_inputs.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        normalized = (float_inputs * inverse_rms).to(dtype=inputs.dtype)
        return normalized * self.weight.to(dtype=inputs.dtype)


class RotaryEmbedding(nn.Module):
    """Apply cached pairwise rotations to ``(B, H, T, D)`` queries and keys."""

    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        _validate_positive_integer(head_dim, "head_dim")
        _validate_positive_integer(max_sequence_length, "max_sequence_length")
        _validate_positive_number(base, "base")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings.")
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.base = float(base)

        inverse_frequencies = 1.0 / (
            self.base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer(
            "cosine_cache",
            angles.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sine_cache",
            angles.sin(),
            persistent=False,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate query/key feature pairs at each sequence position."""
        self._validate_inputs(queries, keys, position_offset)
        sequence_length = queries.shape[-2]
        end = position_offset + sequence_length
        cosine = self.cosine_cache[position_offset:end].to(
            device=queries.device,
            dtype=queries.dtype,
        )
        sine = self.sine_cache[position_offset:end].to(
            device=queries.device,
            dtype=queries.dtype,
        )
        cosine = cosine.unsqueeze(0).unsqueeze(0)
        sine = sine.unsqueeze(0).unsqueeze(0)
        return (
            _apply_rotary_pairs(queries, cosine, sine),
            _apply_rotary_pairs(keys, cosine, sine),
        )

    def _validate_inputs(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        position_offset: int,
    ) -> None:
        if not isinstance(queries, torch.Tensor) or not isinstance(keys, torch.Tensor):
            raise TypeError("queries and keys must be PyTorch tensors.")
        if queries.shape != keys.shape or queries.ndim != 4:
            raise ValueError("queries and keys must share shape (B, H, T, D).")
        if queries.shape[-1] != self.head_dim:
            raise ValueError(f"Expected head dimension {self.head_dim}.")
        if not queries.is_floating_point() or not keys.is_floating_point():
            raise TypeError("queries and keys must contain floating-point values.")
        if (
            not isinstance(position_offset, int)
            or isinstance(position_offset, bool)
            or position_offset < 0
        ):
            raise ValueError("position_offset must be a non-negative integer.")
        if position_offset + queries.shape[-2] > self.max_sequence_length:
            raise ValueError("Rotary position range exceeds configured capacity.")


class CausalSelfAttention(nn.Module):
    """Fused-QKV multi-head causal attention using PyTorch SDPA."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(
            config.embedding_dim,
            3 * config.num_heads * config.head_dim,
            bias=config.linear_bias,
        )
        self.rotary_embedding = RotaryEmbedding(
            config.head_dim,
            config.rope_max_sequence_length,
            config.rope_base,
        )
        self.output_projection = nn.Linear(
            config.num_heads * config.head_dim,
            config.embedding_dim,
            bias=config.linear_bias,
        )
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, C)`` through causal attention and return ``(B, T, C)``."""
        _validate_hidden_states(inputs, self.embedding_dim)
        batch_size, sequence_length, _ = inputs.shape
        qkv = self.qkv_projection(inputs)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)
        queries, keys = self.rotary_embedding(queries, keys)
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        )
        return self.output_dropout(self.output_projection(attended))


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward network applied independently at each position."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.gate_projection = nn.Linear(
            config.embedding_dim,
            config.feed_forward_hidden_dim,
            bias=config.linear_bias,
        )
        self.value_projection = nn.Linear(
            config.embedding_dim,
            config.feed_forward_hidden_dim,
            bias=config.linear_bias,
        )
        self.output_projection = nn.Linear(
            config.feed_forward_hidden_dim,
            config.embedding_dim,
            bias=config.linear_bias,
        )
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply ``SiLU(gate) * value`` and project back to embedding width."""
        _validate_hidden_states(inputs, self.embedding_dim)
        gated = F.silu(self.gate_projection(inputs))
        values = self.value_projection(inputs)
        return self.output_dropout(self.output_projection(gated * values))


class GPTBlock(nn.Module):
    """Pre-normalized attention and SwiGLU residual block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.attention_norm = RMSNorm(
            config.embedding_dim,
            config.rms_norm_epsilon,
        )
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(
            config.embedding_dim,
            config.rms_norm_epsilon,
        )
        self.feed_forward = SwiGLU(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply both pre-norm residual sublayers."""
        _validate_hidden_states(inputs, self.embedding_dim)
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class GPTLanguageModel(nn.Module):
    """Decoder-only GPT with RoPE, SDPA, SwiGLU, and tied token weights."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if not isinstance(config, GPTConfig):
            raise TypeError("config must be a GPTConfig.")
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )
        self.blocks = nn.ModuleList(
            GPTBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(
            config.embedding_dim,
            config.rms_norm_epsilon,
        )
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )
        self.apply(self._initialize_weights)
        self._scale_residual_projections()
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(B, T, V)`` logits and optional flattened next-token loss."""
        self._validate_token_ids(token_ids, "token_ids", enforce_block_size=True)
        if targets is not None:
            self._validate_token_ids(targets, "targets", enforce_block_size=True)
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids.")
            if targets.device != token_ids.device:
                raise ValueError("targets and token_ids must be on the same device.")

        hidden = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample without a KV cache, cropping to block size."""
        self._validate_token_ids(token_ids, "token_ids", enforce_block_size=False)
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be greater than zero.")
        if top_k is not None and (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError(
                f"top_k must be between 1 and vocab_size "
                f"({self.config.vocab_size})."
            )

        generated = token_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.block_size :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_logits, top_indices = torch.topk(next_token_logits, top_k)
                probabilities = F.softmax(top_logits, dim=-1)
                sampled = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                next_token = top_indices.gather(-1, sampled)
            else:
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initialization_std,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        # Two residual branches are accumulated per layer. The 1/sqrt(2L)
        # scaling keeps their initialization variance controlled with depth.
        standard_deviation = self.config.initialization_std / math.sqrt(
            2 * self.config.num_layers
        )
        for block in self.blocks:
            nn.init.normal_(
                block.attention.output_projection.weight,
                mean=0.0,
                std=standard_deviation,
            )
            nn.init.normal_(
                block.feed_forward.output_projection.weight,
                mean=0.0,
                std=standard_deviation,
            )

    def _validate_token_ids(
        self,
        token_ids: torch.Tensor,
        name: str,
        *,
        enforce_block_size: bool,
    ) -> None:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor.")
        if token_ids.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, sequence).")
        if token_ids.numel() == 0:
            raise ValueError(f"{name} must not be empty.")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must contain integer token IDs.")
        if (
            enforce_block_size
            and token_ids.shape[1] > self.config.block_size
        ):
            raise ValueError(
                f"Sequence length {token_ids.shape[1]} exceeds block_size "
                f"{self.config.block_size}."
            )
        if torch.any(token_ids < 0) or torch.any(
            token_ids >= self.config.vocab_size
        ):
            raise ValueError(
                f"{name} contains IDs outside [0, {self.config.vocab_size})."
            )


@dataclass(frozen=True)
class ParameterMemory:
    """Exact storage estimates for model parameters only."""

    float32_bytes: int
    float16_bytes: int
    bfloat16_bytes: int


@dataclass(frozen=True)
class ParameterReport:
    """Unique parameter counts and dtype-specific storage estimates."""

    total_parameters: int
    trainable_parameters: int
    shared_parameter_references: int
    memory: ParameterMemory


def parameter_report(model: nn.Module) -> ParameterReport:
    """Count shared storage once and estimate parameter-only memory."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch module.")
    references = list(model.named_parameters(remove_duplicate=False))
    unique: dict[tuple[int, int, int], nn.Parameter] = {}
    for _, parameter in references:
        key = (
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
            parameter.numel(),
        )
        unique.setdefault(key, parameter)
    total = sum(parameter.numel() for parameter in unique.values())
    trainable = sum(
        parameter.numel()
        for parameter in unique.values()
        if parameter.requires_grad
    )
    return ParameterReport(
        total_parameters=total,
        trainable_parameters=trainable,
        shared_parameter_references=len(references) - len(unique),
        memory=ParameterMemory(
            float32_bytes=total * 4,
            float16_bytes=total * 2,
            bfloat16_bytes=total * 2,
        ),
    )


def _apply_rotary_pairs(
    inputs: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    even = inputs[..., 0::2]
    odd = inputs[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _validate_hidden_states(inputs: torch.Tensor, dimension: int) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a PyTorch tensor.")
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, sequence, embedding_dim).")
    if inputs.numel() == 0:
        raise ValueError("inputs must not be empty.")
    if inputs.shape[-1] != dimension:
        raise ValueError(f"Expected final dimension {dimension}.")
    if not inputs.is_floating_point():
        raise TypeError("inputs must contain floating-point values.")


def _validate_positive_integer(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_positive_number(value: Any, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


def _validate_probability(value: Any, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value < 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1).")
