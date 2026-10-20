"""Tests for language-model loss estimation."""

import math

import pytest
import torch

from forgelm.models.bigram import BigramLanguageModel
from forgelm.training.evaluate import estimate_loss


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
