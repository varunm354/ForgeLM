"""Offline CPU integration tests for the production pretraining engine."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from forgelm.data.token_shards import BinaryTokenShardWriter
from forgelm.models.gpt import GPTConfig
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_engine import PretrainingEngine
from forgelm.training.pretrain_runtime import initialize_distributed

FINGERPRINT = "a" * 64


def _make_manifest(root: Path, *, vocab_size: int = 32) -> Path:
    root.mkdir(parents=True)
    train_writer = BinaryTokenShardWriter(
        root,
        split="train",
        tokens_per_shard=160,
        vocabulary_size=vocab_size,
    )
    validation_writer = BinaryTokenShardWriter(
        root,
        split="validation",
        tokens_per_shard=96,
        vocabulary_size=vocab_size,
    )
    train_writer.append([index % vocab_size for index in range(160)])
    validation_writer.append(
        [(index * 3 + 1) % vocab_size for index in range(96)]
    )
    train_shards = train_writer.finish()
    validation_shards = validation_writer.finish()
    manifest = {
        "format": "forgelm.packed_token_shards",
        "format_version": 1,
        "source": {
            "dataset_name": "local/test",
            "dataset_config": "synthetic",
            "dataset_split": "train",
            "dataset_revision": "fixture-v1",
            "text_field": "text",
        },
        "creation_configuration": {"seed": 1337},
        "tokenizer": {
            "path": "tokenizer.json",
            "fingerprint": FINGERPRINT,
            "vocabulary_size": vocab_size,
            "end_of_document_token": "<|endoftext|>",
            "end_of_document_id": 0,
        },
        "storage": {"dtype": "uint16", "byte_order": "little"},
        "counts": {
            "documents_seen": 2,
            "documents_accepted": 2,
            "documents_rejected": 0,
            "documents_duplicated": 0,
            "train_documents": 1,
            "validation_documents": 1,
            "train_documents_written": 1,
            "validation_documents_written": 1,
            "total_raw_utf8_bytes": 0,
            "rejection_reasons": {},
            "train_tokens": 160,
            "validation_tokens": 96,
        },
        "shards": {
            "train": train_shards,
            "validation": validation_shards,
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _make_assets(tmp_path: Path) -> tuple[Path, Path]:
    model_config = GPTConfig(
        vocab_size=32,
        block_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        feed_forward_hidden_dim=32,
        dropout=0.0,
        rope_max_sequence_length=8,
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(model_config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return model_path, _make_manifest(tmp_path / "shards")


def _config(
    model_path: Path,
    manifest_path: Path,
    output: Path,
    **overrides: Any,
) -> PretrainConfig:
    values: dict[str, Any] = {
        "model_config_path": model_path,
        "manifest_path": manifest_path,
        "expected_tokenizer_fingerprint": FINGERPRINT,
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
        "output_directory": output,
        "latest_checkpoint_name": "latest.pt",
        "best_checkpoint_name": "best.pt",
        "metrics_filename": "metrics.jsonl",
        "metadata_filename": "run_metadata.json",
        "verify_checksums": True,
        "fixed_validation_batches": True,
    }
    values.update(overrides)
    return PretrainConfig(**values)


def _engine(config: PretrainConfig, **kwargs: Any) -> PretrainingEngine:
    context = initialize_distributed("cpu", environment={})
    return PretrainingEngine(config, context, **kwargs)


def test_validation_is_deterministic_and_does_not_change_sampler_state(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    engine = _engine(_config(model_path, manifest_path, tmp_path / "output"))
    training_state = engine.training_sampler.state_dict()
    validation_state = engine.validation_sampler.state_dict()

    first = engine.evaluate()
    second = engine.evaluate()
    inputs, targets = engine.training_sampler.next_batch()

    assert first == second
    assert torch.equal(
        engine.validation_sampler.state_dict()["generator_state"],
        validation_state["generator_state"],
    )
    assert torch.equal(
        training_state["generator_state"],
        _engine(
            _config(model_path, manifest_path, tmp_path / "comparison")
        ).training_sampler.state_dict()["generator_state"],
    )
    assert inputs.shape == targets.shape == (2, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_manifest_compatibility_and_checksum_failures_are_clear(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    wrong_fingerprint = _config(
        model_path,
        manifest_path,
        tmp_path / "wrong-fingerprint",
        expected_tokenizer_fingerprint="b" * 64,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        _engine(wrong_fingerprint)

    incompatible_model = GPTConfig(
        vocab_size=31,
        block_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        feed_forward_hidden_dim=32,
        rope_max_sequence_length=8,
    )
    incompatible_path = tmp_path / "incompatible-model.yaml"
    incompatible_path.write_text(
        yaml.safe_dump(incompatible_model.to_dict()),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vocabulary"):
        _engine(
            _config(
                incompatible_path,
                manifest_path,
                tmp_path / "wrong-vocabulary",
            )
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = manifest_path.parent / manifest["shards"]["train"][0]["name"]
    contents = bytearray(shard_path.read_bytes())
    contents[0] ^= 1
    shard_path.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _engine(_config(model_path, manifest_path, tmp_path / "corrupt"))


def test_checkpoint_metrics_metadata_and_weights_only_loading(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    output = tmp_path / "run"
    config = _config(model_path, manifest_path, output)
    engine = _engine(config)

    summary = engine.run(maximum_optimizer_steps=1)

    expected_files = {
        "latest.pt",
        "best.pt",
        "step_00000001.pt",
        "metrics.jsonl",
        "run_metadata.json",
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    assert not list(output.glob(".*.tmp"))
    checkpoint = torch.load(
        output / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    for field in (
        "model_state",
        "optimizer_state",
        "gradient_scaler_state",
        "optimizer_step",
        "micro_step",
        "tokens_processed",
        "best_validation_loss",
        "training_config",
        "model_config",
        "manifest_identity",
        "tokenizer_fingerprint",
        "rank_states",
        "world_size",
        "precision",
    ):
        assert field in checkpoint
    assert summary["optimizer_step"] == 1
    assert summary["micro_step"] == 2
    assert summary["tokens_processed"] == 32

    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert len(metrics) == 1
    assert metrics[0]["optimizer_step"] == 1
    assert metrics[0]["validation_loss"] is not None
    assert metadata["model_parameters"] > 0
    assert metadata["precision"] == "float32"

    weights_engine = _engine(
        _config(model_path, manifest_path, tmp_path / "weights"),
        weights_only_checkpoint=output / "latest.pt",
    )
    assert weights_engine.progress.optimizer_step == 0
    for expected, actual in zip(
        engine.model.parameters(),
        weights_engine.model.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_exact_resume_restores_all_cpu_rng_and_sampler_states(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    config = _config(model_path, manifest_path, tmp_path / "rng-resume")
    engine = _engine(config)
    engine.run(maximum_optimizer_steps=1)
    checkpoint = torch.load(
        config.output_directory / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    rank_state = checkpoint["rank_states"][0]

    python_probe = random.Random()
    python_probe.setstate(rank_state["python_random_state"])
    expected_python = python_probe.random()
    saved_numpy = rank_state["numpy_random_state"]
    numpy_probe = np.random.RandomState()
    numpy_probe.set_state(
        (
            saved_numpy["bit_generator"],
            np.asarray(saved_numpy["state"], dtype=np.uint32),
            saved_numpy["position"],
            saved_numpy["has_gauss"],
            saved_numpy["cached_gaussian"],
        )
    )
    expected_numpy = numpy_probe.random_sample()
    torch_probe = torch.Generator()
    torch_probe.set_state(rank_state["torch_cpu_rng_state"])
    expected_torch = torch.rand(4, generator=torch_probe)

    random.random()
    np.random.random()
    torch.rand(4)
    resumed = _engine(
        config,
        resume_checkpoint=config.output_directory / "latest.pt",
    )

    assert resumed.progress.optimizer_step == 1
    assert resumed.progress.micro_step == 2
    assert resumed.progress.tokens_processed == 32
    assert (
        resumed.progress.best_validation_loss
        == checkpoint["best_validation_loss"]
    )
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)
    assert rank_state["cuda_rng_states"] == []
    assert resumed.training_sampler.batches_sampled == 2
    assert resumed.validation_sampler.batches_sampled == 0
    assert torch.equal(
        resumed.training_sampler.state_dict()["generator_state"],
        rank_state["training_sampler_state"]["generator_state"],
    )
    assert torch.equal(
        resumed.validation_sampler.state_dict()["generator_state"],
        rank_state["validation_sampler_state"]["generator_state"],
    )


def test_interrupt_request_finishes_boundary_and_saves_interrupt_checkpoint(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    config = _config(model_path, manifest_path, tmp_path / "interrupt")
    engine = _engine(config)
    engine.stop_requested = True

    summary = engine.run(maximum_optimizer_steps=3)

    assert summary["optimizer_step"] == 1
    assert summary["micro_step"] == 2
    assert summary["tokens_processed"] == 32
    assert (config.output_directory / "interrupt_step_00000001.pt").is_file()
    assert (config.output_directory / "latest.pt").is_file()
    assert (config.output_directory / "best.pt").is_file()
    assert (config.output_directory / "step_00000001.pt").is_file()


def test_non_finite_loss_refuses_optimizer_update(tmp_path: Path) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    engine = _engine(_config(model_path, manifest_path, tmp_path / "nonfinite"))

    class NonFiniteModel(torch.nn.Module):
        def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.empty(
                (*inputs.shape, 1),
                device=inputs.device,
            )
            loss = torch.tensor(
                float("nan"),
                device=inputs.device,
                requires_grad=True,
            )
            return logits, loss

    engine.execution_model = NonFiniteModel()
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        engine.train_one_optimizer_step()
    assert engine.progress.optimizer_step == 0


def test_exact_interrupted_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _make_assets(tmp_path)
    reference = _engine(
        _config(model_path, manifest_path, tmp_path / "reference")
    )
    reference.run(maximum_optimizer_steps=3)

    resumed_config = _config(
        model_path,
        manifest_path,
        tmp_path / "resumed",
    )
    interrupted = _engine(resumed_config)
    interrupted.run(maximum_optimizer_steps=1)
    resumed = _engine(
        resumed_config,
        resume_checkpoint=resumed_config.output_directory / "latest.pt",
    )
    resumed_summary = resumed.run(maximum_optimizer_steps=3)

    assert resumed_summary["optimizer_step"] == 3
    assert resumed_summary["micro_step"] == 6
    assert resumed_summary["tokens_processed"] == 96
    for expected, actual in zip(
        reference.model.parameters(),
        resumed.model.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
    _assert_nested_equal(
        reference.optimizer.state_dict(),
        resumed.optimizer.state_dict(),
    )


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_item, actual_item)
    else:
        assert expected == actual
