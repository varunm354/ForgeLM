"""Offline CPU tests for cloud preflight and bounded qualification policy."""

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from forgelm.data.qualification_fixture import create_qualification_fixture
from forgelm.models.gpt import GPTConfig
from forgelm.training import pretrain_checkpoint
from forgelm.training.cloud_preflight import (
    redact_sensitive_fields,
    run_cloud_preflight,
    write_json_report,
)
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.qualification import (
    MAX_QUALIFICATION_STEPS,
    evaluate_acceptance_criteria,
    run_qualification,
    validate_qualification_configuration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_CONFIG = (
    PROJECT_ROOT / "configs/pretrain_gpt_cuda_qualification.yaml"
)


def _tiny_config(tmp_path: Path) -> PretrainConfig:
    model = GPTConfig(
        vocab_size=32,
        block_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        feed_forward_hidden_dim=32,
        rope_max_sequence_length=8,
    )
    model_path = tmp_path / "tiny_model.yaml"
    model_path.write_text(
        yaml.safe_dump(model.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    manifest = create_qualification_fixture(
        tmp_path / "fixture",
        seed=3,
        vocabulary_size=32,
        train_tokens=128,
        validation_tokens=64,
        tokens_per_shard=64,
    )
    return PretrainConfig(
        model_config_path=model_path,
        manifest_path=tmp_path / "fixture/manifest.json",
        expected_tokenizer_fingerprint=manifest["tokenizer"]["fingerprint"],
        seed=1,
        micro_batch_size=1,
        sequence_length=8,
        gradient_accumulation_steps=2,
        target_training_tokens=32,
        learning_rate=1e-3,
        minimum_learning_rate=1e-4,
        warmup_steps=1,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        gradient_clipping=1.0,
        logging_interval=1,
        validation_interval=1,
        validation_batches=1,
        checkpoint_interval=1,
        precision="auto",
        torch_compile=False,
        compile_allow_fallback=False,
        output_directory=tmp_path / "output",
        latest_checkpoint_name="latest.pt",
        best_checkpoint_name="best.pt",
        metrics_filename="metrics.jsonl",
        metadata_filename="metadata.json",
        verify_checksums=True,
        fixed_validation_batches=True,
    )


def _production_qualification_config(tmp_path: Path) -> PretrainConfig:
    create_qualification_fixture(tmp_path / "fixture")
    return PretrainConfig.load(
        QUALIFICATION_CONFIG,
        project_root=PROJECT_ROOT,
    ).with_overrides(
        manifest_path=tmp_path / "fixture/manifest.json",
        output_directory=tmp_path / "output",
    )


def test_cloud_preflight_cpu_diagnostic_needs_no_cuda(tmp_path: Path) -> None:
    report = run_cloud_preflight(
        _tiny_config(tmp_path),
        cpu_diagnostic=True,
        expected_model_parameters=None,
    )

    assert report["passed"]
    assert report["mode"] == "cpu_diagnostic"
    assert report["model"]["parameter_count"] > 0
    assert report["model"]["selected_precision"] == "float32"
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["cuda_detected"]["status"] == "deferred"


def test_secret_fields_are_redacted_and_report_is_strict_json(
    tmp_path: Path,
) -> None:
    value = {
        "api_key": "do-not-write",
        "nested": {
            "password": "do-not-write",
            "tokenizer_fingerprint": "safe-identity",
        },
    }

    redacted = redact_sensitive_fields(value)
    path = tmp_path / "report.json"
    write_json_report(path, value)
    serialized = path.read_text(encoding="utf-8")

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["tokenizer_fingerprint"] == "safe-identity"
    assert "do-not-write" not in serialized
    assert json.loads(serialized) == redacted


def test_acceptance_criteria_distinguish_required_and_optional() -> None:
    evidence = {
        key: True
        for key in (
            "cuda_detected",
            "full_model_loaded",
            "production_sequence_length",
            "mixed_precision_worked",
            "finite_forward_loss",
            "finite_gradients",
            "optimizer_updates_completed",
            "validation_completed",
            "checkpoint_save_load",
            "interrupted_resume",
            "post_resume_step",
            "peak_vram_recorded",
            "artifacts_uncorrupted",
            "metrics_metadata_written",
        )
    }
    evidence["compilation_requested"] = False

    criteria = evaluate_acceptance_criteria(
        evidence,
        cpu_diagnostic=False,
    )

    assert all(
        item["status"] == "pass" for item in criteria if item["required"]
    )
    assert next(item for item in criteria if item["key"] == "compilation")[
        "status"
    ] == "not_requested"
    assert next(item for item in criteria if item["key"] == "multi_gpu")[
        "status"
    ] == "not_tested"


def test_aborted_qualification_distinguishes_failed_and_unfinished_checks() -> None:
    criteria = evaluate_acceptance_criteria(
        {
            "cuda_detected": True,
            "full_model_loaded": True,
            "mixed_precision_worked": False,
            "checkpoint_save_load": False,
        },
        cpu_diagnostic=False,
        failure={
            "type": "TypeError",
            "message": "RNG state must be a torch.ByteTensor",
        },
    )
    by_key = {item["key"]: item for item in criteria}

    assert by_key["cuda_detected"]["outcome"] == "passed"
    assert by_key["mixed_precision_worked"]["outcome"] == "failed"
    assert by_key["checkpoint_save_load"]["outcome"] == "failed"
    assert by_key["interrupted_resume"]["status"] == "fail"
    assert by_key["interrupted_resume"]["outcome"] == "not_completed"
    assert "TypeError" in by_key["interrupted_resume"]["reason"]


def test_qualification_configuration_is_full_model_and_strictly_bounded(
    tmp_path: Path,
) -> None:
    config = _production_qualification_config(tmp_path)

    validated = validate_qualification_configuration(
        config,
        maximum_optimizer_steps=MAX_QUALIFICATION_STEPS,
    )

    assert validated["configured_optimizer_steps"] == 20
    assert validated["effective_batch_tokens"] == 2_048
    with pytest.raises(ValueError, match="between 2 and 20"):
        validate_qualification_configuration(
            config,
            maximum_optimizer_steps=21,
        )
    with pytest.raises(ValueError, match="production token target"):
        validate_qualification_configuration(
            config.with_overrides(target_training_tokens=500_000_000),
            maximum_optimizer_steps=20,
        )


def test_qualification_rejects_production_source_manifest(tmp_path: Path) -> None:
    config = _production_qualification_config(tmp_path)
    manifest_path = config.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["dataset_name"] = "HuggingFaceFW/fineweb-edu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must not read the production dataset"):
        validate_qualification_configuration(
            config,
            maximum_optimizer_steps=20,
        )


def test_cuda_requirement_refuses_training_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _production_qualification_config(tmp_path)
    fake_preflight: dict[str, Any] = {
        "passed": True,
        "model": {"parameter_count": 49_295_872},
        "sdpa": {},
    }
    monkeypatch.setattr(
        "forgelm.training.qualification.run_cloud_preflight",
        lambda *args, **kwargs: fake_preflight,
    )
    monkeypatch.setattr(
        "forgelm.training.qualification.torch.cuda.is_available",
        lambda: False,
    )

    report = run_qualification(
        config,
        maximum_optimizer_steps=2,
        cpu_diagnostic=False,
        resume_checkpoint=None,
        force=False,
    )

    assert report["passed"] is False
    assert report["failure"]["type"] == "CUDAUnavailable"
    criteria = {item["key"]: item for item in report["acceptance_criteria"]}
    assert criteria["cuda_detected"]["outcome"] == "failed"
    assert criteria["full_model_loaded"]["outcome"] == "passed"
    assert criteria["checkpoint_save_load"]["outcome"] == "not_completed"
    assert "CUDAUnavailable" in criteria["checkpoint_save_load"]["reason"]
    assert not (config.output_directory / config.metrics_filename).exists()


def test_qualification_crosses_simulated_cuda_checkpoint_resume_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _production_qualification_config(tmp_path)
    fake_preflight: dict[str, Any] = {
        "passed": True,
        "model": {"parameter_count": 49_295_872},
        "sdpa": {},
    }
    restored_cuda_states: list[torch.Tensor] = []

    class FakeSampler:
        def load_state_dict(self, state: dict[str, Any]) -> None:
            generator_state = state["generator_state"]
            assert generator_state.device.type == "cpu"
            assert generator_state.dtype == torch.uint8

    class FakeContext:
        world_size = 1
        device = torch.device("cuda")

        def shutdown(self) -> None:
            return None

    class FakeEngine:
        instances = 0

        def __init__(
            self,
            engine_config: PretrainConfig,
            _context: FakeContext,
            *,
            resume_checkpoint: Path | None = None,
        ) -> None:
            type(self).instances += 1
            self.config = engine_config
            self.model = object()
            self.precision = SimpleNamespace(name="bfloat16")
            self.progress = SimpleNamespace(optimizer_step=0)
            if resume_checkpoint is not None:
                checkpoint = pretrain_checkpoint._load_checkpoint(
                    resume_checkpoint
                )
                sampler = FakeSampler()
                pretrain_checkpoint.restore_rank_state(
                    checkpoint["rank_states"][0],
                    sampler,  # type: ignore[arg-type]
                    sampler,  # type: ignore[arg-type]
                )
                self.progress.optimizer_step = checkpoint["optimizer_step"]

        def run(self, *, maximum_optimizer_steps: int) -> dict[str, Any]:
            self.progress.optimizer_step = maximum_optimizer_steps
            output = self.config.output_directory
            output.mkdir(parents=True, exist_ok=True)
            numpy_state = np.random.get_state()
            generator_state = torch.Generator().manual_seed(7).get_state()
            torch.save(
                {
                    "format": pretrain_checkpoint.CHECKPOINT_FORMAT,
                    "version": pretrain_checkpoint.CHECKPOINT_VERSION,
                    "optimizer_step": maximum_optimizer_steps,
                    "rank_states": [
                        {
                            "python_random_state": random.getstate(),
                            "numpy_random_state": {
                                "bit_generator": numpy_state[0],
                                "state": numpy_state[1].tolist(),
                                "position": numpy_state[2],
                                "has_gauss": numpy_state[3],
                                "cached_gaussian": numpy_state[4],
                            },
                            "torch_cpu_rng_state": (
                                torch.get_rng_state().to(torch.int16)
                            ),
                            "cuda_rng_states": [
                                torch.get_rng_state().to(torch.int16)
                            ],
                            "training_sampler_state": {
                                "generator_state": generator_state.to(torch.int16)
                            },
                            "validation_sampler_state": {
                                "generator_state": generator_state.to(torch.int16)
                            },
                        }
                    ],
                },
                output / self.config.latest_checkpoint_name,
            )
            (output / self.config.metadata_filename).write_text(
                "{}",
                encoding="utf-8",
            )
            with (output / self.config.metrics_filename).open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "step_time_seconds": 1.0,
                            "tokens_per_second": 2_048.0,
                            "validation_loss": 1.0,
                        }
                    )
                    + "\n"
                )
            return {
                "optimizer_step": maximum_optimizer_steps,
                "micro_step": maximum_optimizer_steps * 2,
                "tokens_processed": maximum_optimizer_steps * 2_048,
                "training_loss": 1.0,
                "validation_loss": 1.0,
                "validation_perplexity": 1.0,
                "gradient_norm": 1.0,
            }

    monkeypatch.setattr(
        "forgelm.training.qualification.run_cloud_preflight",
        lambda *args, **kwargs: fake_preflight,
    )
    monkeypatch.setattr(
        "forgelm.training.qualification.initialize_distributed",
        lambda _device: FakeContext(),
    )
    monkeypatch.setattr(
        "forgelm.training.qualification.PretrainingEngine",
        FakeEngine,
    )
    monkeypatch.setattr(
        "forgelm.training.qualification.parameter_report",
        lambda _model: SimpleNamespace(total_parameters=49_295_872),
    )
    monkeypatch.setattr(
        "forgelm.training.qualification._memory_probe",
        lambda _engine: {
            "completed": True,
            "finite_loss": True,
            "finite_gradients": True,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: restored_cuda_states.extend(states),
    )

    report = run_qualification(
        config,
        maximum_optimizer_steps=6,
        cpu_diagnostic=False,
        resume_checkpoint=None,
        force=False,
    )

    assert report["passed"] is True
    assert report["failure"] is None
    assert report["training"]["resume_start_step"] == 5
    assert FakeEngine.instances == 2
    assert len(restored_cuda_states) == 1
    assert restored_cuda_states[0].device.type == "cpu"
    assert restored_cuda_states[0].dtype == torch.uint8
