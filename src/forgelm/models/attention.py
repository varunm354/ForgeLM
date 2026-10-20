"""Explicit single-head causal self-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SingleHeadCausalSelfAttention(nn.Module):
    """Attend to the current and preceding positions with one attention head."""

    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._validate_positive_integer(embedding_size, "embedding_size")
        self._validate_positive_integer(head_size, "head_size")
        self._validate_positive_integer(block_size, "block_size")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not 0.0 <= dropout <= 1.0
        ):
            raise ValueError("dropout must be a number between 0 and 1.")

        self.embedding_size = embedding_size
        self.head_size = head_size
        self.block_size = block_size
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.attention_dropout = nn.Dropout(float(dropout))

        # Each row permits attention only to its position and earlier positions.
        causal_mask = torch.tril(
            torch.ones(block_size, block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply scaled dot-product attention to embedded token sequences."""
        self._validate_inputs(inputs)
        sequence_length = inputs.shape[1]

        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        output = attention_weights @ values

        if return_attention_weights:
            return output, attention_weights
        return output

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a PyTorch tensor.")
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch, sequence, embedding_size]."
            )
        if inputs.numel() == 0:
            raise ValueError("inputs must not be empty.")
        if inputs.shape[-1] != self.embedding_size:
            raise ValueError(
                f"Expected embedding size {self.embedding_size}, "
                f"received {inputs.shape[-1]}."
            )
        if inputs.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {inputs.shape[1]} exceeds block_size "
                f"{self.block_size}."
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must contain floating-point embeddings.")

    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
