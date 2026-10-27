"""CPU-only tests for the production-oriented GPT architecture."""

import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from forgelm.models.gpt import (
    CausalSelfAttention,
    GPTBlock,
    GPTConfig,
    GPTLanguageModel,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    parameter_report,
)


def small_config(**overrides: object) -> GPTConfig:
    """Return a fast production-GPT configuration for tests."""
    values: dict[str, object] = {
        "vocab_size": 101,
        "block_size": 16,
        "embedding_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 8,
        "feed_forward_hidden_dim": 64,
        "dropout": 0.0,
        "linear_bias": False,
        "rope_base": 10_000.0,
        "rope_max_sequence_length": 16,
        "rms_norm_epsilon": 1e-5,
        "tie_embeddings": True,
        "initialization_std": 0.02,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_dim": 30},
        {"embedding_dim": 28, "num_heads": 4, "head_dim": 7},
        {"num_layers": 0},
        {"dropout": 1.0},
        {"vocab_size": 65_537},
        {"block_size": 17},
        {"linear_bias": 1},
        {"tie_embeddings": False},
    ],
)
def test_configuration_rejects_invalid_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        small_config(**overrides)


def test_yaml_configuration_has_expected_50m_dimensions() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/gpt_50m.yaml"

    config = GPTConfig.load(path)

    assert config.vocab_size == 16_384
    assert config.block_size == 1_024
    assert config.embedding_dim == 512
    assert config.num_layers == 12
    assert config.num_heads == 8
    assert config.head_dim == 64
    assert config.feed_forward_hidden_dim == 1_536


def test_rms_norm_shape_dtype_and_expected_magnitude() -> None:
    norm = RMSNorm(4, epsilon=1e-8)
    inputs = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [-2.0, -1.0, 1.0, 2.0]]],
        dtype=torch.float64,
    )

    output = norm(inputs)

    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype
    assert torch.allclose(
        output.square().mean(dim=-1),
        torch.ones((1, 2), dtype=torch.float64),
        atol=1e-7,
    )
    assert torch.count_nonzero(output) == torch.count_nonzero(inputs)


def test_rms_norm_does_not_subtract_the_mean() -> None:
    output = RMSNorm(4, epsilon=1e-8)(torch.ones(2, 3, 4))

    assert torch.allclose(output, torch.ones_like(output))


def test_rope_is_deterministic_position_dependent_and_norm_preserving() -> None:
    rope = RotaryEmbedding(head_dim=8, max_sequence_length=8)
    queries = torch.randn(2, 3, 4, 8)
    keys = torch.randn(2, 3, 4, 8)

    first_queries, first_keys = rope(queries, keys)
    second_queries, second_keys = rope(queries, keys)

    assert first_queries.shape == queries.shape
    assert first_keys.shape == keys.shape
    assert torch.equal(first_queries, second_queries)
    assert torch.equal(first_keys, second_keys)
    assert torch.equal(first_queries[:, :, 0], queries[:, :, 0])
    assert not torch.equal(first_queries[:, :, 1], queries[:, :, 1])
    assert torch.allclose(
        first_queries.norm(dim=-1),
        queries.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        first_keys.norm(dim=-1),
        keys.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_attention_shape_and_fused_qkv_projection() -> None:
    config = small_config()
    attention = CausalSelfAttention(config)

    output = attention(torch.randn(2, 7, config.embedding_dim))

    assert output.shape == (2, 7, config.embedding_dim)
    assert attention.qkv_projection.out_features == 3 * config.embedding_dim


def test_optimized_attention_matches_explicit_reference() -> None:
    torch.manual_seed(7)
    config = small_config()
    attention = CausalSelfAttention(config)
    attention.eval()
    inputs = torch.randn(2, 6, config.embedding_dim)

    actual = attention(inputs)
    qkv = attention.qkv_projection(inputs).view(
        2,
        6,
        3,
        config.num_heads,
        config.head_dim,
    )
    queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
    queries, keys = attention.rotary_embedding(queries, keys)
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(config.head_dim)
    causal_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    weights = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
    reference = (weights @ values).transpose(1, 2).contiguous().view(2, 6, 32)
    reference = attention.output_projection(reference)

    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)


def test_attention_is_causally_isolated() -> None:
    torch.manual_seed(11)
    attention = CausalSelfAttention(small_config())
    attention.eval()
    inputs = torch.randn(2, 7, 32)
    changed = inputs.clone()
    changed[:, 4:] = torch.randn(2, 3, 32) * 20

    original_output = attention(inputs)
    changed_output = attention(changed)

    assert torch.allclose(
        original_output[:, :4],
        changed_output[:, :4],
        atol=1e-6,
        rtol=1e-5,
    )


def test_attention_dropout_differs_in_training_and_is_off_in_evaluation() -> None:
    attention = CausalSelfAttention(small_config(dropout=0.7))
    inputs = torch.randn(2, 8, 32)
    attention.train()

    torch.manual_seed(1)
    first_training = attention(inputs)
    torch.manual_seed(2)
    second_training = attention(inputs)
    attention.eval()
    first_evaluation = attention(inputs)
    second_evaluation = attention(inputs)

    assert not torch.equal(first_training, second_training)
    assert torch.equal(first_evaluation, second_evaluation)


def test_swiglu_and_transformer_block_preserve_expected_shapes() -> None:
    config = small_config()
    inputs = torch.randn(2, 7, config.embedding_dim)

    feed_forward_output = SwiGLU(config)(inputs)
    block_output = GPTBlock(config)(inputs)

    assert feed_forward_output.shape == (2, 7, config.embedding_dim)
    assert block_output.shape == inputs.shape


def test_model_logits_loss_and_backward_are_finite() -> None:
    model = GPTLanguageModel(small_config())
    token_ids = torch.randint(0, 101, (2, 8))
    targets = torch.randint(0, 101, (2, 8))

    logits, loss = model(token_ids, targets)

    assert logits.shape == (2, 8, 101)
    assert loss is not None
    assert loss.shape == ()
    assert torch.isfinite(loss)
    expected = F.cross_entropy(logits.reshape(-1, 101), targets.reshape(-1))
    assert torch.equal(loss, expected)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_embedding_and_output_weights_share_storage() -> None:
    model = GPTLanguageModel(small_config())

    assert model.lm_head.weight is model.token_embedding.weight
    assert (
        model.lm_head.weight.untyped_storage().data_ptr()
        == model.token_embedding.weight.untyped_storage().data_ptr()
    )


def test_parameter_report_does_not_double_count_tied_weight() -> None:
    model = GPTLanguageModel(small_config())
    report = parameter_report(model)
    duplicated_count = sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=False)
    )

    assert report.total_parameters == report.trainable_parameters == 23_872
    assert report.shared_parameter_references == 1
    assert (
        duplicated_count - report.total_parameters
        == model.token_embedding.weight.numel()
    )
    assert report.memory.float32_bytes == report.total_parameters * 4
    assert report.memory.float16_bytes == report.total_parameters * 2
    assert report.memory.bfloat16_bytes == report.total_parameters * 2


def test_default_model_has_calculated_49m_parameter_count() -> None:
    report = parameter_report(GPTLanguageModel(GPTConfig()))

    assert report.total_parameters == 49_295_872
    assert report.trainable_parameters == 49_295_872


def test_overlong_sequence_is_rejected() -> None:
    model = GPTLanguageModel(small_config())

    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.zeros((1, 17), dtype=torch.long))


def test_short_top_one_generation_is_deterministic() -> None:
    model = GPTLanguageModel(small_config())
    model.eval()
    prefix = torch.tensor([[1, 2, 3]])

    first = model.generate(prefix, max_new_tokens=4, top_k=1)
    second = model.generate(prefix, max_new_tokens=4, top_k=1)

    assert first.shape == (1, 7)
    assert torch.equal(first, second)
    assert torch.equal(first[:, :3], prefix)


def test_state_dict_round_trip_preserves_outputs(tmp_path: Path) -> None:
    config = small_config()
    model = GPTLanguageModel(config)
    model.eval()
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    path = tmp_path / "gpt.pt"
    expected, _ = model(token_ids)
    torch.save(model.state_dict(), path)

    restored = GPTLanguageModel(config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    actual, _ = restored(token_ids)

    assert torch.equal(actual, expected)
    assert restored.lm_head.weight is restored.token_embedding.weight
