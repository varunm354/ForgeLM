"""A decoder-only Transformer language model."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from forgelm.models.transformer import TransformerBlock


class TransformerLanguageModel(nn.Module):
    """Predict next tokens with stacked causal Transformer blocks."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for value, name in (
            (vocab_size, "vocab_size"),
            (embedding_dim, "embedding_dim"),
            (block_size, "block_size"),
            (num_heads, "num_heads"),
            (num_layers, "num_layers"),
        ):
            _validate_positive_integer(value, name)
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        _validate_dropout(dropout)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token logits and optional flattened cross-entropy loss."""
        self._validate_token_ids(token_ids, "token_ids", enforce_block_size=True)
        if targets is not None:
            self._validate_token_ids(targets, "targets", enforce_block_size=True)
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids.")
            if targets.device != token_ids.device:
                raise ValueError("targets and token_ids must be on the same device.")

        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding_table(token_ids)
        hidden = hidden + self.position_embedding_table(positions)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_layer_norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
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
    ) -> torch.Tensor:
        """Sample new tokens autoregressively from a starting context."""
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
            or not 1 <= top_k <= self.vocab_size
        ):
            raise ValueError(
                f"top_k must be between 1 and vocab_size ({self.vocab_size})."
            )

        generated = token_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.block_size :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is None:
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                top_logits, top_indices = torch.topk(next_token_logits, top_k)
                probabilities = F.softmax(top_logits, dim=-1)
                sampled_index = torch.multinomial(probabilities, num_samples=1)
                next_token = top_indices.gather(-1, sampled_index)

            generated = torch.cat((generated, next_token), dim=1)
        return generated

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
            raise ValueError(f"{name} must have shape [batch, sequence].")
        if token_ids.numel() == 0:
            raise ValueError(f"{name} must not be empty.")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must contain integer token IDs.")
        if enforce_block_size and token_ids.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {token_ids.shape[1]} exceeds block_size "
                f"{self.block_size}."
            )
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError(
                f"{name} contains token IDs outside [0, {self.vocab_size})."
            )


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_dropout(dropout: float) -> None:
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0.0 <= dropout <= 1.0
    ):
        raise ValueError("dropout must be a number between 0 and 1.")
