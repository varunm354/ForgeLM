"""Evaluation helpers for language models."""

import math

import torch
from torch import nn
from torch.nn import functional as F

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


@torch.no_grad()
def evaluate_full_corpus(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate every next-token target in a contiguous token sequence once."""
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a PyTorch tensor.")
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor.")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("tokens must contain integer token IDs.")
    if len(tokens) < 2:
        raise ValueError("tokens must contain at least two token IDs.")
    _validate_positive_integer(block_size, "block_size")
    _validate_positive_integer(batch_size, "batch_size")

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter.") from error

    target_count = len(tokens) - 1
    full_window_count, remaining_targets = divmod(target_count, block_size)
    total_loss = 0.0
    evaluated_targets = 0
    was_training = model.training
    model.eval()

    try:
        for first_window in range(0, full_window_count, batch_size):
            windows_in_batch = min(
                batch_size,
                full_window_count - first_window,
            )
            starts = [
                (first_window + index) * block_size
                for index in range(windows_in_batch)
            ]
            inputs = torch.stack(
                [tokens[start : start + block_size] for start in starts]
            )
            targets = torch.stack(
                [tokens[start + 1 : start + block_size + 1] for start in starts]
            )
            batch_loss = _summed_cross_entropy(model, inputs, targets, device)
            total_loss += batch_loss
            evaluated_targets += targets.numel()

        if remaining_targets:
            start = full_window_count * block_size
            inputs = tokens[start : start + remaining_targets].unsqueeze(0)
            targets = tokens[start + 1 : start + remaining_targets + 1].unsqueeze(0)
            total_loss += _summed_cross_entropy(model, inputs, targets, device)
            evaluated_targets += targets.numel()
    finally:
        model.train(was_training)

    if evaluated_targets != target_count:
        raise RuntimeError("Full-corpus evaluation did not cover every target.")

    mean_loss = total_loss / evaluated_targets
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}


def _summed_cross_entropy(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> float:
    device_inputs = inputs.to(device)
    device_targets = targets.to(device)
    logits, _ = model(device_inputs)
    if logits.ndim != 3 or logits.shape[:2] != device_targets.shape:
        raise ValueError("model logits must have shape [batch, sequence, vocabulary].")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        device_targets.reshape(-1),
        reduction="sum",
    ).item()


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
