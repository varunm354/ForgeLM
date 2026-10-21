"""Reusable decoder-only Transformer components."""

import torch
from torch import nn

from forgelm.models.attention import MultiHeadCausalSelfAttention


class PositionwiseFeedForward(nn.Module):
    """Transform each sequence position independently."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        _validate_positive_integer(embedding_dim, "embedding_dim")
        _validate_dropout(dropout)

        hidden_dim = 4 * embedding_dim
        self.embedding_dim = embedding_dim
        self.linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the same two-layer network to every token position."""
        _validate_embeddings(inputs, self.embedding_dim)
        hidden = self.activation(self.linear_1(inputs))
        return self.dropout(self.linear_2(hidden))


class TransformerBlock(nn.Module):
    """A pre-normalized decoder-only Transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_integer(embedding_dim, "embedding_dim")
        _validate_dropout(dropout)

        self.embedding_dim = embedding_dim
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = PositionwiseFeedForward(embedding_dim, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention and feed-forward residual sublayers."""
        _validate_embeddings(inputs, self.embedding_dim)
        inputs = inputs + self.attention(self.layer_norm_1(inputs))
        return inputs + self.feed_forward(self.layer_norm_2(inputs))


def _validate_embeddings(inputs: torch.Tensor, embedding_dim: int) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a PyTorch tensor.")
    if inputs.ndim != 3:
        raise ValueError(
            "inputs must have shape [batch, sequence, embedding_dim]."
        )
    if inputs.numel() == 0:
        raise ValueError("inputs must not be empty.")
    if inputs.shape[-1] != embedding_dim:
        raise ValueError(
            f"Expected embedding dimension {embedding_dim}, "
            f"received {inputs.shape[-1]}."
        )
    if not inputs.is_floating_point():
        raise TypeError("inputs must contain floating-point embeddings.")


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
