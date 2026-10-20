"""Tests for the bigram language model."""

import pytest
import torch

from forgelm.models.bigram import BigramLanguageModel


def test_forward_returns_expected_logit_shape() -> None:
    model = BigramLanguageModel(vocab_size=5)
    input_ids = torch.tensor([[0, 1, 2], [2, 3, 4]])

    logits, loss = model(input_ids)

    assert logits.shape == (2, 3, 5)
    assert loss is None


def test_forward_returns_finite_cross_entropy_loss() -> None:
    model = BigramLanguageModel(vocab_size=4)
    input_ids = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])

    _, loss = model(input_ids, targets)

    assert loss is not None
    assert torch.isfinite(loss)


def test_generate_appends_requested_number_of_tokens() -> None:
    torch.manual_seed(4)
    model = BigramLanguageModel(vocab_size=3)
    input_ids = torch.tensor([[0, 1], [1, 2]])

    generated = model.generate(input_ids, max_new_tokens=5)

    assert generated.shape == (2, 7)
    assert torch.equal(generated[:, :2], input_ids)


@pytest.mark.parametrize(
    ("input_ids", "message"),
    [
        (torch.tensor([0, 1]), r"\[batch, sequence\]"),
        (torch.tensor([[0.0, 1.0]]), "integer token IDs"),
        (torch.tensor([[0, 3]]), "outside"),
    ],
)
def test_forward_rejects_invalid_inputs(
    input_ids: torch.Tensor,
    message: str,
) -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises((TypeError, ValueError), match=message):
        model(input_ids)


def test_forward_rejects_mismatched_targets() -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises(ValueError, match="same shape"):
        model(torch.tensor([[0, 1]]), torch.tensor([[1]]))


def test_generate_rejects_invalid_length() -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises(ValueError, match="non-negative integer"):
        model.generate(torch.tensor([[0]]), max_new_tokens=-1)


def test_repeated_updates_reduce_loss() -> None:
    torch.manual_seed(2)
    model = BigramLanguageModel(vocab_size=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    input_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    targets = torch.tensor([[1, 2, 0], [1, 2, 0]])

    _, initial_loss = model(input_ids, targets)
    assert initial_loss is not None

    for _ in range(40):
        _, loss = model(input_ids, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    _, final_loss = model(input_ids, targets)
    assert final_loss is not None
    assert final_loss.item() < initial_loss.item()
