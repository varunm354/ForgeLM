"""Tests for explicit causal self-attention."""

import pytest
import torch

from forgelm.models.attention import SingleHeadCausalSelfAttention


def test_output_and_attention_weight_shapes() -> None:
    attention = SingleHeadCausalSelfAttention(
        embedding_size=6,
        head_size=4,
        block_size=8,
        dropout=0.0,
    )
    inputs = torch.randn(2, 5, 6)

    output, weights = attention(inputs, return_attention_weights=True)

    assert output.shape == (2, 5, 4)
    assert weights.shape == (2, 5, 5)


def test_attention_weights_are_normalized_over_allowed_positions() -> None:
    attention = SingleHeadCausalSelfAttention(6, 4, 8, dropout=0.5)
    attention.eval()

    _, weights = attention(torch.randn(2, 5, 6), return_attention_weights=True)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5))


def test_future_positions_receive_exactly_zero_attention() -> None:
    attention = SingleHeadCausalSelfAttention(6, 4, 8, dropout=0.0)

    _, weights = attention(torch.randn(2, 5, 6), return_attention_weights=True)
    future_positions = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    assert torch.count_nonzero(weights[:, future_positions]) == 0


def test_future_inputs_cannot_change_earlier_outputs() -> None:
    torch.manual_seed(3)
    attention = SingleHeadCausalSelfAttention(6, 4, 8, dropout=0.0)
    inputs = torch.randn(1, 5, 6)
    changed_inputs = inputs.clone()
    changed_inputs[:, 3:, :] = torch.randn(1, 2, 6) * 100

    original_output = attention(inputs)
    changed_output = attention(changed_inputs)

    assert torch.equal(original_output[:, :3], changed_output[:, :3])


def test_gradients_flow_through_all_projections() -> None:
    attention = SingleHeadCausalSelfAttention(6, 4, 8, dropout=0.0)
    output = attention(torch.randn(2, 5, 6))

    output.square().sum().backward()

    for projection in (attention.query, attention.key, attention.value):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_sequence_above_block_size_is_rejected() -> None:
    attention = SingleHeadCausalSelfAttention(6, 4, 4, dropout=0.0)

    with pytest.raises(ValueError, match="exceeds block_size"):
        attention(torch.randn(2, 5, 6))


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (torch.randn(2, 6), r"\[batch, sequence, embedding_size\]"),
        (torch.randn(2, 4, 5), "Expected embedding size 6"),
    ],
)
def test_invalid_rank_or_embedding_size_is_rejected(
    inputs: torch.Tensor,
    message: str,
) -> None:
    attention = SingleHeadCausalSelfAttention(6, 4, 8, dropout=0.0)

    with pytest.raises(ValueError, match=message):
        attention(inputs)


def test_dropout_is_disabled_during_evaluation() -> None:
    attention = SingleHeadCausalSelfAttention(6, 4, 8, dropout=0.9)
    attention.eval()
    inputs = torch.randn(2, 5, 6)

    first_output, first_weights = attention(
        inputs,
        return_attention_weights=True,
    )
    second_output, second_weights = attention(
        inputs,
        return_attention_weights=True,
    )

    assert torch.equal(first_output, second_output)
    assert torch.equal(first_weights, second_weights)
    assert torch.allclose(first_weights.sum(dim=-1), torch.ones(2, 5))
