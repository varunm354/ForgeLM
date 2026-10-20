"""Utilities for language-model token data."""

import torch


def train_validation_split(
    tokens: torch.Tensor,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1D token tensor into contiguous training and validation sets."""
    _validate_token_tensor(tokens)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_index = int(len(tokens) * train_fraction)
    if split_index == 0 or split_index == len(tokens):
        raise ValueError("The split must leave at least one token in each set.")
    return tokens[:split_index], tokens[split_index:]


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input sequences and their one-token-shifted targets."""
    _validate_token_tensor(tokens)
    _validate_positive_integer(batch_size, "batch_size")
    _validate_positive_integer(block_size, "block_size")
    if len(tokens) <= block_size:
        raise ValueError(
            "The dataset must contain more tokens than block_size "
            "to create shifted targets."
        )

    starts = torch.randint(
        0,
        len(tokens) - block_size,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([tokens[start : start + block_size] for start in starts])
    targets = torch.stack(
        [tokens[start + 1 : start + block_size + 1] for start in starts]
    )
    return inputs, targets


def _validate_token_tensor(tokens: torch.Tensor) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a PyTorch tensor.")
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor.")
    if len(tokens) == 0:
        raise ValueError("tokens must not be empty.")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
