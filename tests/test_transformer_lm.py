"""Tests for the decoder-only Transformer language model."""

import pytest
import torch
from torch.nn import functional as F

from forgelm.models.transformer_lm import TransformerLanguageModel


def make_model(*, dropout: float = 0.0) -> TransformerLanguageModel:
    """Create a small model suitable for CPU-only tests."""
    return TransformerLanguageModel(
        vocab_size=11,
        embedding_dim=12,
        block_size=6,
        num_heads=3,
        num_layers=2,
        dropout=dropout,
    )


def test_forward_logit_shape_and_optional_loss() -> None:
    model = make_model()

    logits, loss = model(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))

    assert logits.shape == (2, 4, 11)
    assert loss is None


def test_forward_returns_finite_scalar_loss() -> None:
    model = make_model()
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    _, loss = model(token_ids, targets)

    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_matches_flattened_cross_entropy() -> None:
    model = make_model()
    model.eval()
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])

    logits, loss = model(token_ids, targets)
    expected = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))

    assert loss is not None
    assert torch.equal(loss, expected)


def test_sequence_above_block_size_is_rejected() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.zeros((1, 7), dtype=torch.long))


@pytest.mark.parametrize(
    "arguments",
    [
        {"vocab_size": 0},
        {"embedding_dim": 10, "num_heads": 3},
        {"num_layers": 0},
        {"dropout": 1.1},
    ],
)
def test_invalid_model_configuration_is_rejected(
    arguments: dict[str, int | float],
) -> None:
    config: dict[str, int | float] = {
        "vocab_size": 11,
        "embedding_dim": 12,
        "block_size": 6,
        "num_heads": 3,
        "num_layers": 2,
        "dropout": 0.0,
    }
    config.update(arguments)

    with pytest.raises(ValueError):
        TransformerLanguageModel(**config)  # type: ignore[arg-type]


def test_positional_embeddings_distinguish_repeated_tokens() -> None:
    torch.manual_seed(1)
    model = make_model()
    model.eval()

    logits, _ = model(torch.tensor([[4, 4, 4]]))

    assert not torch.equal(logits[:, 0], logits[:, 1])
    assert not torch.equal(logits[:, 1], logits[:, 2])


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(2)
    model = make_model(dropout=0.3)
    model.eval()
    original = torch.tensor([[0, 1, 2, 3, 4, 5]])
    changed = torch.tensor([[0, 1, 2, 8, 9, 10]])

    original_logits, _ = model(original)
    changed_logits, _ = model(changed)

    assert torch.allclose(
        original_logits[:, :3],
        changed_logits[:, :3],
        atol=1e-6,
        rtol=1e-6,
    )


def test_generation_preserves_prefix_and_adds_requested_tokens() -> None:
    model = make_model()
    model.eval()
    prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])

    generated = model.generate(prefix, max_new_tokens=4, top_k=5)

    assert generated.shape == (2, 7)
    assert torch.equal(generated[:, :3], prefix)


def test_generation_crops_context_to_block_size() -> None:
    model = make_model()
    model.eval()
    seen_sequence_lengths: list[int] = []

    def record_sequence_length(
        _model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        seen_sequence_lengths.append(inputs[0].shape[1])

    handle = model.register_forward_pre_hook(record_sequence_length)
    try:
        prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        generated = model.generate(prefix, max_new_tokens=3)
    finally:
        handle.remove()

    assert torch.equal(generated[:, :8], prefix)
    assert seen_sequence_lengths == [6, 6, 6]


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_generation_rejects_invalid_temperature(temperature: float) -> None:
    model = make_model()

    with pytest.raises(ValueError, match="greater than zero"):
        model.generate(torch.tensor([[0]]), 1, temperature=temperature)


@pytest.mark.parametrize("top_k", [0, 12, True])
def test_generation_rejects_invalid_top_k(top_k: int) -> None:
    model = make_model()

    with pytest.raises(ValueError, match="between 1 and vocab_size"):
        model.generate(torch.tensor([[0]]), 1, top_k=top_k)


def test_generation_is_deterministic_with_identical_seeds() -> None:
    model = make_model(dropout=0.5)
    model.eval()
    prefix = torch.tensor([[1, 2, 3]])

    torch.manual_seed(42)
    first = model.generate(prefix, max_new_tokens=8, temperature=0.8, top_k=5)
    torch.manual_seed(42)
    second = model.generate(prefix, max_new_tokens=8, temperature=0.8, top_k=5)

    assert torch.equal(first, second)


def test_gradients_reach_all_model_components() -> None:
    model = make_model()
    token_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    _, loss = model(token_ids, targets)
    assert loss is not None
    loss.backward()

    assert_nonzero_gradient(model.token_embedding_table.weight)
    assert_nonzero_gradient(model.position_embedding_table.weight)
    for block in model.blocks:
        for head in block.attention.heads:
            assert_nonzero_gradient(head.query.weight)
            assert_nonzero_gradient(head.key.weight)
            assert_nonzero_gradient(head.value.weight)
        assert_nonzero_gradient(block.attention.output_projection.weight)
        assert_nonzero_gradient(block.feed_forward.linear_1.weight)
        assert_nonzero_gradient(block.feed_forward.linear_2.weight)
    assert_nonzero_gradient(model.final_layer_norm.weight)
    assert_nonzero_gradient(model.lm_head.weight)


def assert_nonzero_gradient(parameter: torch.Tensor) -> None:
    """Assert that a parameter received a nonzero gradient."""
    assert parameter.grad is not None
    assert torch.count_nonzero(parameter.grad) > 0
