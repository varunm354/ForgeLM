"""Tests for reusable decoder-only Transformer components."""

import pytest
import torch

from forgelm.models.attention import MultiHeadCausalSelfAttention
from forgelm.models.transformer import PositionwiseFeedForward, TransformerBlock


def test_multi_head_attention_preserves_shape() -> None:
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=12,
        num_heads=3,
        block_size=8,
        dropout=0.0,
    )

    output = attention(torch.randn(2, 5, 12))

    assert output.shape == (2, 5, 12)


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(
        embedding_dim=12,
        num_heads=3,
        block_size=8,
        dropout=0.0,
    )

    output = block(torch.randn(2, 5, 12))

    assert output.shape == (2, 5, 12)


def test_invalid_head_divisibility_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        MultiHeadCausalSelfAttention(
            embedding_dim=10,
            num_heads=3,
            block_size=8,
        )


def test_sequences_above_block_size_are_rejected() -> None:
    inputs = torch.randn(2, 5, 12)
    attention = MultiHeadCausalSelfAttention(12, 3, block_size=4)
    block = TransformerBlock(12, 3, block_size=4)

    with pytest.raises(ValueError, match="exceeds block_size"):
        attention(inputs)
    with pytest.raises(ValueError, match="exceeds block_size"):
        block(inputs)


def test_multi_head_attention_is_causal() -> None:
    torch.manual_seed(11)
    attention = MultiHeadCausalSelfAttention(12, 3, block_size=8, dropout=0.2)
    attention.eval()
    inputs = torch.randn(2, 6, 12)
    changed_inputs = inputs.clone()
    changed_inputs[:, 3:, :] = torch.randn(2, 3, 12) * 20

    original = attention(inputs)
    changed = attention(changed_inputs)

    assert torch.allclose(original[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6)


def test_transformer_block_is_causal() -> None:
    torch.manual_seed(12)
    block = TransformerBlock(12, 3, block_size=8, dropout=0.2)
    block.eval()
    inputs = torch.randn(2, 6, 12)
    changed_inputs = inputs.clone()
    changed_inputs[:, 4:, :] = torch.randn(2, 2, 12) * 20

    original = block(inputs)
    changed = block(changed_inputs)

    assert torch.allclose(original[:, :4], changed[:, :4], atol=1e-6, rtol=1e-6)


def test_gradients_flow_through_every_attention_head() -> None:
    attention = MultiHeadCausalSelfAttention(12, 3, block_size=8, dropout=0.0)

    attention(torch.randn(2, 5, 12)).square().sum().backward()

    for head in attention.heads:
        for projection in (head.query, head.key, head.value):
            assert projection.weight.grad is not None
            assert torch.count_nonzero(projection.weight.grad) > 0


def test_gradients_flow_through_feed_forward_network() -> None:
    feed_forward = PositionwiseFeedForward(embedding_dim=12, dropout=0.0)

    feed_forward(torch.randn(2, 5, 12)).square().sum().backward()

    assert feed_forward.linear_1.weight.grad is not None
    assert torch.count_nonzero(feed_forward.linear_1.weight.grad) > 0
    assert feed_forward.linear_2.weight.grad is not None
    assert torch.count_nonzero(feed_forward.linear_2.weight.grad) > 0


def test_dropout_is_deterministic_during_evaluation() -> None:
    block = TransformerBlock(12, 3, block_size=8, dropout=0.9)
    block.eval()
    inputs = torch.randn(2, 5, 12)

    first = block(inputs)
    second = block(inputs)

    assert torch.equal(first, second)


def test_feed_forward_does_not_mix_token_positions() -> None:
    feed_forward = PositionwiseFeedForward(embedding_dim=12, dropout=0.0)
    inputs = torch.randn(1, 5, 12)
    changed_inputs = inputs.clone()
    changed_inputs[:, 2, :] = torch.randn(1, 12) * 20

    original = feed_forward(inputs)
    changed = feed_forward(changed_inputs)
    unchanged_positions = [0, 1, 3, 4]

    assert torch.equal(
        original[:, unchanged_positions],
        changed[:, unchanged_positions],
    )
