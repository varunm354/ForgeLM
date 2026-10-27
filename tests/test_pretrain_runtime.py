"""Tests for pretraining configuration, scheduling, precision, and runtime."""

import math
from pathlib import Path

import pytest
import torch

from forgelm.models.gpt import GPTConfig, GPTLanguageModel
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_runtime import (
    create_adamw,
    initialize_distributed,
    learning_rate_at_step,
    parse_distributed_environment,
    select_precision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _training_config(**overrides: object) -> PretrainConfig:
    values: dict[str, object] = {
        "model_config_path": Path("model.yaml"),
        "manifest_path": Path("manifest.json"),
        "expected_tokenizer_fingerprint": None,
        "seed": 1337,
        "micro_batch_size": 2,
        "sequence_length": 8,
        "gradient_accumulation_steps": 2,
        "target_training_tokens": 96,
        "learning_rate": 1e-3,
        "minimum_learning_rate": 1e-4,
        "warmup_steps": 1,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "epsilon": 1e-8,
        "gradient_clipping": 1.0,
        "logging_interval": 1,
        "validation_interval": 1,
        "validation_batches": 2,
        "checkpoint_interval": 1,
        "precision": "auto",
        "torch_compile": False,
        "compile_allow_fallback": False,
        "output_directory": Path("outputs"),
        "latest_checkpoint_name": "latest.pt",
        "best_checkpoint_name": "best.pt",
        "metrics_filename": "metrics.jsonl",
        "metadata_filename": "run_metadata.json",
        "verify_checksums": True,
        "fixed_validation_batches": True,
    }
    values.update(overrides)
    return PretrainConfig(**values)  # type: ignore[arg-type]


def _model_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        block_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        feed_forward_hidden_dim=32,
        rope_max_sequence_length=8,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"micro_batch_size": 0},
        {"sequence_length": 0},
        {"gradient_accumulation_steps": 0},
        {"minimum_learning_rate": 0.01},
        {"betas": (1.0, 0.9)},
        {"precision": "half"},
        {"warmup_steps": -1},
    ],
)
def test_training_configuration_rejects_invalid_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _training_config(**overrides)


def test_production_configuration_effective_tokens_and_steps() -> None:
    config = PretrainConfig.load(
        PROJECT_ROOT / "configs/pretrain_gpt_50m.yaml",
        project_root=PROJECT_ROOT,
    )

    assert config.effective_batch_tokens(world_size=1) == 131_072
    assert config.optimizer_steps(world_size=1) == 3_815
    assert config.effective_batch_tokens(world_size=4) == 524_288


def test_optimizer_groups_every_unique_parameter_exactly_once() -> None:
    model = GPTLanguageModel(_model_config())
    setup = create_adamw(model, _training_config(), device=torch.device("cpu"))
    grouped = [
        parameter
        for group in setup.optimizer.param_groups
        for parameter in group["params"]
    ]

    assert len(grouped) == len({id(parameter) for parameter in model.parameters()})
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert "token_embedding.weight" in setup.decay_parameter_names
    assert "lm_head.weight" not in setup.decay_parameter_names
    assert all(parameter.ndim >= 2 for parameter in setup.optimizer.param_groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in setup.optimizer.param_groups[1]["params"])
    assert setup.optimizer.param_groups[0]["weight_decay"] == 0.1
    assert setup.optimizer.param_groups[1]["weight_decay"] == 0.0
    assert setup.fused is False


def test_learning_rate_warmup_and_cosine_boundaries() -> None:
    arguments = {
        "total_steps": 10,
        "warmup_steps": 2,
        "maximum": 1e-3,
        "minimum": 1e-4,
    }
    rates = [
        learning_rate_at_step(step, **arguments)
        for step in range(11)
    ]

    assert rates[0] == 0.0
    assert rates[2] == 1e-3
    assert rates[10] == pytest.approx(1e-4)
    assert rates[11 - 1] >= 1e-4
    assert rates[:3] == sorted(rates[:3])
    assert rates[2:] == sorted(rates[2:], reverse=True)
    assert learning_rate_at_step(20, **arguments) == pytest.approx(1e-4)


def test_precision_selection_is_explicit_and_device_safe() -> None:
    assert select_precision("auto", torch.device("cpu")).name == "float32"
    assert (
        select_precision(
            "auto",
            torch.device("cuda"),
            cuda_bf16_supported=True,
        ).name
        == "bfloat16"
    )
    fp16 = select_precision(
        "auto",
        torch.device("cuda"),
        cuda_bf16_supported=False,
    )
    assert fp16.name == "float16"
    assert fp16.uses_grad_scaler
    with pytest.raises(ValueError, match="requires a CUDA"):
        select_precision("float16", torch.device("cpu"))


def test_accumulated_gradients_match_larger_batch() -> None:
    torch.manual_seed(5)
    config = _model_config()
    accumulated = GPTLanguageModel(config)
    combined = GPTLanguageModel(config)
    combined.load_state_dict(accumulated.state_dict())
    first_inputs = torch.randint(0, config.vocab_size, (2, 8))
    first_targets = torch.randint(0, config.vocab_size, (2, 8))
    second_inputs = torch.randint(0, config.vocab_size, (2, 8))
    second_targets = torch.randint(0, config.vocab_size, (2, 8))

    for inputs, targets in (
        (first_inputs, first_targets),
        (second_inputs, second_targets),
    ):
        _, loss = accumulated(inputs, targets)
        assert loss is not None
        (loss / 2).backward()
    _, combined_loss = combined(
        torch.cat((first_inputs, second_inputs)),
        torch.cat((first_targets, second_targets)),
    )
    assert combined_loss is not None
    combined_loss.backward()

    for accumulated_parameter, combined_parameter in zip(
        accumulated.parameters(),
        combined.parameters(),
        strict=True,
    ):
        assert accumulated_parameter.grad is not None
        assert combined_parameter.grad is not None
        assert torch.allclose(
            accumulated_parameter.grad,
            combined_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        )


def test_distributed_environment_parsing_and_single_process_context() -> None:
    default = parse_distributed_environment({})
    parsed = parse_distributed_environment(
        {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
    )

    assert (default.rank, default.local_rank, default.world_size) == (0, 0, 1)
    assert not default.distributed
    assert (parsed.rank, parsed.local_rank, parsed.world_size) == (2, 1, 4)
    assert parsed.distributed
    with pytest.raises(ValueError, match="must be set together"):
        parse_distributed_environment({"RANK": "0"})

    context = initialize_distributed("cpu", environment={})
    assert context.device == torch.device("cpu")
    assert not context.distributed
    assert not torch.distributed.is_initialized()
    context.shutdown()


def test_global_token_accounting_formula() -> None:
    config = _training_config(
        micro_batch_size=3,
        sequence_length=10,
        gradient_accumulation_steps=4,
        target_training_tokens=1_000,
    )

    assert config.effective_batch_tokens(world_size=2) == 240
    assert config.optimizer_steps(world_size=2) == math.ceil(1_000 / 240)
