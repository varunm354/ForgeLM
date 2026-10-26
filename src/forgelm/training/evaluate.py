"""Evaluation helpers for language models."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from forgelm.data.dataset import get_batch


@dataclass(frozen=True)
class ByteNormalizedEvaluation:
    """Exact token- and byte-normalized full-corpus evaluation metrics."""

    mean_token_loss: float
    token_perplexity: float
    total_negative_log_likelihood: float
    evaluated_target_tokens: int
    evaluated_target_bytes: int
    bits_per_byte: float


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


@torch.no_grad()
def evaluate_full_corpus_bytes(
    model: nn.Module,
    tokens: torch.Tensor,
    token_byte_length: Mapping[int, int] | Callable[[int], int],
    *,
    block_size: int,
    batch_size: int,
) -> ByteNormalizedEvaluation:
    """Evaluate every target once and normalize its exact NLL by source bytes."""
    _validate_evaluation_tokens(tokens)
    _validate_positive_integer(block_size, "block_size")
    _validate_positive_integer(batch_size, "batch_size")
    byte_lengths = _validated_byte_lengths(tokens, token_byte_length)

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter.") from error

    target_count = len(tokens) - 1
    full_window_count, remaining_targets = divmod(target_count, block_size)
    total_nll = 0.0
    evaluated_targets = 0
    evaluated_bytes = 0
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
            target_nll = _unreduced_cross_entropy(
                model,
                inputs,
                targets,
                device,
            )
            total_nll += target_nll.double().sum().item()
            evaluated_targets += targets.numel()
            evaluated_bytes += _target_byte_count(targets, byte_lengths)

        if remaining_targets:
            start = full_window_count * block_size
            inputs = tokens[start : start + remaining_targets].unsqueeze(0)
            targets = tokens[start + 1 : start + remaining_targets + 1].unsqueeze(0)
            target_nll = _unreduced_cross_entropy(
                model,
                inputs,
                targets,
                device,
            )
            total_nll += target_nll.double().sum().item()
            evaluated_targets += targets.numel()
            evaluated_bytes += _target_byte_count(targets, byte_lengths)
    finally:
        model.train(was_training)

    if evaluated_targets != target_count:
        raise RuntimeError("Byte-normalized evaluation did not cover every target.")
    if evaluated_bytes <= 0:
        raise RuntimeError("Byte-normalized evaluation did not score any bytes.")
    if not math.isfinite(total_nll):
        raise ValueError("Evaluation produced a non-finite total NLL.")

    mean_loss = total_nll / evaluated_targets
    token_perplexity = math.exp(mean_loss)
    bits_per_byte = total_nll / (evaluated_bytes * math.log(2))
    if not math.isfinite(token_perplexity) or not math.isfinite(bits_per_byte):
        raise ValueError("Evaluation produced non-finite normalized metrics.")

    return ByteNormalizedEvaluation(
        mean_token_loss=mean_loss,
        token_perplexity=token_perplexity,
        total_negative_log_likelihood=total_nll,
        evaluated_target_tokens=evaluated_targets,
        evaluated_target_bytes=evaluated_bytes,
        bits_per_byte=bits_per_byte,
    )


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


def _unreduced_cross_entropy(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    device_targets = targets.to(device)
    logits, _ = model(inputs.to(device))
    if logits.ndim != 3 or logits.shape[:2] != device_targets.shape:
        raise ValueError("model logits must have shape [batch, sequence, vocabulary].")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        device_targets.reshape(-1),
        reduction="none",
    )


def _validate_evaluation_tokens(tokens: torch.Tensor) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a PyTorch tensor.")
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor.")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("tokens must contain integer token IDs.")
    if len(tokens) < 2:
        raise ValueError("tokens must contain at least two token IDs.")
    if torch.any(tokens < 0):
        raise ValueError("tokens must contain non-negative token IDs.")


def _validated_byte_lengths(
    tokens: torch.Tensor,
    token_byte_length: Mapping[int, int] | Callable[[int], int],
) -> dict[int, int]:
    if not isinstance(token_byte_length, Mapping) and not callable(
        token_byte_length
    ):
        raise TypeError("token_byte_length must be a mapping or callable.")

    byte_lengths: dict[int, int] = {}
    for token_id in set(tokens.tolist()):
        try:
            byte_length = (
                token_byte_length(token_id)
                if callable(token_byte_length)
                else token_byte_length[token_id]
            )
        except (KeyError, IndexError) as error:
            raise ValueError(
                f"No byte length is available for token ID {token_id}."
            ) from error
        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length <= 0
        ):
            raise ValueError(
                f"Byte length for token ID {token_id} must be a positive integer."
            )
        byte_lengths[token_id] = byte_length
    return byte_lengths


def _target_byte_count(
    targets: torch.Tensor,
    byte_lengths: Mapping[int, int],
) -> int:
    return sum(byte_lengths[token_id] for token_id in targets.reshape(-1).tolist())


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
