"""Tests for language-model data utilities."""

import pytest
import torch

from forgelm.data.dataset import (
    get_batch,
    train_validation_split,
    train_validation_text_split,
)


def test_raw_text_split_is_deterministic_and_contiguous() -> None:
    text = "abcdefghij"

    first = train_validation_text_split(text, train_fraction=0.7)
    second = train_validation_text_split(text, train_fraction=0.7)

    assert first == ("abcdefg", "hij")
    assert second == first
    assert "".join(first) == text


def test_train_validation_split_is_contiguous() -> None:
    tokens = torch.arange(10)

    train, validation = train_validation_split(tokens, train_fraction=0.7)

    assert torch.equal(train, torch.arange(7))
    assert torch.equal(validation, torch.arange(7, 10))


def test_get_batch_targets_are_shifted_by_one_token() -> None:
    tokens = torch.arange(20)

    inputs, targets = get_batch(
        tokens,
        batch_size=4,
        block_size=5,
        generator=torch.Generator().manual_seed(3),
    )

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert torch.equal(targets, inputs + 1)


def test_get_batch_is_reproducible_with_generator() -> None:
    tokens = torch.arange(30)

    first = get_batch(
        tokens,
        batch_size=3,
        block_size=4,
        generator=torch.Generator().manual_seed(42),
    )
    second = get_batch(
        tokens,
        batch_size=3,
        block_size=4,
        generator=torch.Generator().manual_seed(42),
    )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.parametrize(("batch_size", "block_size"), [(0, 2), (-1, 2), (2, 0)])
def test_get_batch_rejects_invalid_sizes(
    batch_size: int,
    block_size: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        get_batch(torch.arange(10), batch_size, block_size)


def test_get_batch_rejects_short_dataset() -> None:
    with pytest.raises(ValueError, match="more tokens than block_size"):
        get_batch(torch.arange(4), batch_size=2, block_size=4)


def test_data_functions_reject_invalid_token_tensors() -> None:
    with pytest.raises(ValueError, match="1D tensor"):
        train_validation_split(torch.ones((2, 2)))

    with pytest.raises(TypeError, match="PyTorch tensor"):
        get_batch([1, 2, 3], batch_size=1, block_size=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("train_fraction", [0.0, 1.0, -0.1, 1.1])
def test_train_validation_split_rejects_invalid_fraction(
    train_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        train_validation_split(torch.arange(10), train_fraction)


def test_train_validation_split_rejects_empty_partition() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        train_validation_split(torch.arange(2), train_fraction=0.1)
