"""Tests for language-model loss estimation."""

import math

import pytest
import torch
from torch.nn import functional as F

from forgelm.models.bigram import BigramLanguageModel
from forgelm.training.evaluate import (
    estimate_loss,
    evaluate_full_corpus,
    evaluate_full_corpus_bytes,
)


@pytest.mark.parametrize("starts_in_training_mode", [True, False])
def test_estimate_loss_reports_metrics_and_restores_model_mode(
    starts_in_training_mode: bool,
) -> None:
    model = BigramLanguageModel(vocab_size=5)
    model.train(starts_in_training_mode)
    train_tokens = torch.arange(40) % 5
    validation_tokens = torch.arange(20) % 5

    metrics = estimate_loss(
        model,
        train_tokens,
        validation_tokens,
        batch_size=3,
        block_size=4,
        evaluation_batches=3,
        generator=torch.Generator().manual_seed(9),
    )

    assert model.training is starts_in_training_mode
    assert set(metrics) == {"train", "validation"}
    for split_metrics in metrics.values():
        assert math.isfinite(split_metrics["loss"])
        assert split_metrics["perplexity"] == pytest.approx(
            math.exp(split_metrics["loss"])
        )


def test_estimate_loss_rejects_invalid_batch_count() -> None:
    model = BigramLanguageModel(vocab_size=3)
    tokens = torch.arange(12) % 3

    with pytest.raises(ValueError, match="positive integer"):
        estimate_loss(
            model,
            tokens,
            tokens,
            batch_size=2,
            block_size=2,
            evaluation_batches=0,
        )


def test_full_corpus_evaluation_returns_finite_loss_and_perplexity() -> None:
    model = BigramLanguageModel(vocab_size=5)
    metrics = evaluate_full_corpus(
        model,
        torch.arange(17) % 5,
        block_size=4,
        batch_size=3,
    )

    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["perplexity"])
    assert metrics["perplexity"] == pytest.approx(math.exp(metrics["loss"]))


def test_full_corpus_evaluation_uses_token_weighted_mean() -> None:
    model = BigramLanguageModel(vocab_size=3)
    with torch.no_grad():
        model.token_embedding_table.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [5.0, -5.0, -5.0],
                    [-2.0, 1.0, 3.0],
                ]
            )
        )
    tokens = torch.tensor([0, 1, 1, 1, 2, 2, 0, 1, 2, 0])

    metrics = evaluate_full_corpus(
        model,
        tokens,
        block_size=4,
        batch_size=2,
    )
    expected = F.cross_entropy(
        model.token_embedding_table(tokens[:-1]),
        tokens[1:],
    ).item()

    assert metrics["loss"] == pytest.approx(expected)


def test_full_corpus_evaluation_includes_final_partial_window() -> None:
    model = BigramLanguageModel(vocab_size=5)
    batch_shapes: list[tuple[int, ...]] = []

    def record_shape(
        _model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        batch_shapes.append(tuple(inputs[0].shape))

    handle = model.register_forward_pre_hook(record_shape)
    try:
        evaluate_full_corpus(
            model,
            torch.arange(12) % 5,
            block_size=4,
            batch_size=2,
        )
    finally:
        handle.remove()

    assert batch_shapes == [(2, 4), (1, 3)]
    assert sum(batch * sequence for batch, sequence in batch_shapes) == 11


@pytest.mark.parametrize("starts_in_training_mode", [True, False])
def test_full_corpus_evaluation_restores_model_mode(
    starts_in_training_mode: bool,
) -> None:
    model = BigramLanguageModel(vocab_size=5)
    model.train(starts_in_training_mode)

    evaluate_full_corpus(
        model,
        torch.arange(12) % 5,
        block_size=4,
        batch_size=2,
    )

    assert model.training is starts_in_training_mode


def test_full_corpus_evaluation_rejects_invalid_token_rank() -> None:
    model = BigramLanguageModel(vocab_size=5)

    with pytest.raises(ValueError, match="1D tensor"):
        evaluate_full_corpus(
            model,
            torch.ones((2, 3), dtype=torch.long),
            block_size=2,
            batch_size=2,
        )


@pytest.mark.parametrize(("block_size", "batch_size"), [(0, 2), (2, 0)])
def test_full_corpus_evaluation_rejects_invalid_sizes(
    block_size: int,
    batch_size: int,
) -> None:
    model = BigramLanguageModel(vocab_size=5)

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_full_corpus(
            model,
            torch.arange(12) % 5,
            block_size=block_size,
            batch_size=batch_size,
        )


def test_byte_evaluation_accumulates_exact_nll_and_bpb() -> None:
    model = BigramLanguageModel(vocab_size=3)
    with torch.no_grad():
        model.token_embedding_table.weight.copy_(
            torch.tensor(
                [
                    [2.0, 0.0, -1.0],
                    [-1.0, 3.0, 0.0],
                    [0.5, -0.5, 1.0],
                ]
            )
        )
    tokens = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    byte_lengths = {0: 1, 1: 2, 2: 3}

    result = evaluate_full_corpus_bytes(
        model,
        tokens,
        byte_lengths,
        block_size=4,
        batch_size=2,
    )
    expected_losses = F.cross_entropy(
        model.token_embedding_table(tokens[:-1]),
        tokens[1:],
        reduction="none",
    )
    expected_total = expected_losses.double().sum().item()
    expected_bytes = sum(byte_lengths[token_id] for token_id in tokens[1:].tolist())

    assert result.total_negative_log_likelihood == pytest.approx(expected_total)
    assert result.evaluated_target_tokens == len(tokens) - 1
    assert result.evaluated_target_bytes == expected_bytes
    assert result.mean_token_loss == pytest.approx(expected_total / (len(tokens) - 1))
    assert result.token_perplexity == pytest.approx(
        math.exp(result.mean_token_loss)
    )
    assert result.bits_per_byte == pytest.approx(
        expected_total / (expected_bytes * math.log(2))
    )


def test_byte_evaluation_includes_partial_batch_and_window() -> None:
    model = BigramLanguageModel(vocab_size=5)
    batch_shapes: list[tuple[int, ...]] = []

    def record_shape(
        _model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        batch_shapes.append(tuple(inputs[0].shape))

    handle = model.register_forward_pre_hook(record_shape)
    try:
        result = evaluate_full_corpus_bytes(
            model,
            torch.arange(14) % 5,
            {token_id: 1 for token_id in range(5)},
            block_size=4,
            batch_size=2,
        )
    finally:
        handle.remove()

    assert batch_shapes == [(2, 4), (1, 4), (1, 1)]
    assert result.evaluated_target_tokens == 13
    assert result.evaluated_target_bytes == 13


@pytest.mark.parametrize("invalid_length", [0, -1, True])
def test_byte_evaluation_rejects_invalid_byte_lengths(
    invalid_length: int,
) -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_full_corpus_bytes(
            model,
            torch.tensor([0, 1, 2]),
            {0: 1, 1: invalid_length, 2: 1},
            block_size=2,
            batch_size=1,
        )


def test_byte_evaluation_rejects_invalid_token_ids() -> None:
    model = BigramLanguageModel(vocab_size=4)

    with pytest.raises(ValueError, match="No byte length"):
        evaluate_full_corpus_bytes(
            model,
            torch.tensor([0, 1, 3]),
            {0: 1, 1: 1},
            block_size=2,
            batch_size=1,
        )


def test_byte_evaluation_rejects_insufficient_input() -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises(ValueError, match="at least two"):
        evaluate_full_corpus_bytes(
            model,
            torch.tensor([0]),
            {0: 1},
            block_size=2,
            batch_size=1,
        )


@pytest.mark.parametrize(("block_size", "batch_size"), [(0, 1), (1, 0)])
def test_byte_evaluation_rejects_invalid_sizes(
    block_size: int,
    batch_size: int,
) -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_full_corpus_bytes(
            model,
            torch.tensor([0, 1]),
            {0: 1, 1: 1},
            block_size=block_size,
            batch_size=batch_size,
        )


@pytest.mark.parametrize("starts_in_training_mode", [True, False])
def test_byte_evaluation_restores_model_mode(
    starts_in_training_mode: bool,
) -> None:
    model = BigramLanguageModel(vocab_size=3)
    model.train(starts_in_training_mode)

    evaluate_full_corpus_bytes(
        model,
        torch.tensor([0, 1, 2, 0]),
        {0: 1, 1: 1, 2: 1},
        block_size=2,
        batch_size=2,
    )

    assert model.training is starts_in_training_mode


def test_different_token_perplexities_have_comparable_bpb() -> None:
    character_model = BigramLanguageModel(vocab_size=3)
    bpe_model = BigramLanguageModel(vocab_size=3)
    with torch.no_grad():
        character_model.token_embedding_table.weight.zero_()
        bpe_model.token_embedding_table.weight.copy_(
            torch.tensor(
                [
                    [-5.0, 5.0, -5.0],
                    [-5.0, -5.0, 5.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )

    character_result = evaluate_full_corpus_bytes(
        character_model,
        torch.tensor([0, 1, 2, 0]),
        {0: 1, 1: 1, 2: 1},
        block_size=3,
        batch_size=1,
    )
    bpe_result = evaluate_full_corpus_bytes(
        bpe_model,
        torch.tensor([0, 1, 2]),
        {0: 1, 1: 1, 2: 2},
        block_size=2,
        batch_size=1,
    )

    assert character_result.evaluated_target_bytes == 3
    assert bpe_result.evaluated_target_bytes == 3
    assert character_result.token_perplexity != pytest.approx(
        bpe_result.token_perplexity
    )
    assert math.isfinite(character_result.bits_per_byte)
    assert math.isfinite(bpe_result.bits_per_byte)
