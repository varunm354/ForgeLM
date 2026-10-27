"""Offline CPU tests for cloud preflight and bounded qualification policy."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from forgelm.data.qualification_fixture import create_qualification_fixture
from forgelm.models.gpt import GPTConfig
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
    assert not (config.output_directory / config.metrics_filename).exists()
