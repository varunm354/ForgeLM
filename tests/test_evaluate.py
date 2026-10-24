"""Tests for language-model loss estimation."""

import math

import pytest
import torch
from torch.nn import functional as F

from forgelm.models.bigram import BigramLanguageModel
from forgelm.training.evaluate import estimate_loss, evaluate_full_corpus


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
