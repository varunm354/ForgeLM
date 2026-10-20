"""Evaluation helpers for language models."""

import math

import torch
from torch import nn

from forgelm.data.dataset import get_batch


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    evaluation_batches: int,
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate average loss and perplexity on training and validation data."""
    if (
        not isinstance(evaluation_batches, int)
        or isinstance(evaluation_batches, bool)
        or evaluation_batches <= 0
    ):
        raise ValueError("evaluation_batches must be a positive integer.")

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter.") from error

    was_training = model.training
    model.eval()
    results: dict[str, dict[str, float]] = {}
    try:
        for split, tokens in (
            ("train", train_tokens),
            ("validation", validation_tokens),
        ):
            losses = torch.empty(evaluation_batches)
            for index in range(evaluation_batches):
                inputs, targets = get_batch(
                    tokens,
                    batch_size=batch_size,
                    block_size=block_size,
                    generator=generator,
                )
                _, loss = model(inputs.to(device), targets.to(device))
                if loss is None:
                    raise ValueError("model did not return a loss.")
                losses[index] = loss.detach().cpu()

            average_loss = losses.mean().item()
            results[split] = {
                "loss": average_loss,
                "perplexity": math.exp(average_loss),
            }
    finally:
        model.train(was_training)

    return results
