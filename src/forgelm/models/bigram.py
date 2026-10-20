"""A character-level bigram language model."""

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next token directly from the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if (
            not isinstance(vocab_size, int)
            or isinstance(vocab_size, bool)
            or vocab_size <= 0
        ):
            raise ValueError("vocab_size must be a positive integer.")

        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token logits and, when targets are given, loss."""
        self._validate_token_tensor(input_ids, "input_ids")
        if targets is not None:
            self._validate_token_tensor(targets, "targets")
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids.")
            if targets.device != input_ids.device:
                raise ValueError("targets and input_ids must be on the same device.")

        logits = self.token_embedding_table(input_ids)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Sample tokens autoregressively from a starting sequence."""
        self._validate_token_tensor(input_ids, "input_ids")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer.")

        generated = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self(generated)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    def _validate_token_tensor(self, tokens: torch.Tensor, name: str) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor.")
        if tokens.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, sequence].")
        if tokens.numel() == 0:
            raise ValueError(f"{name} must not be empty.")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must contain integer token IDs.")
        if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
            raise ValueError(
                f"{name} contains token IDs outside [0, {self.vocab_size})."
            )
