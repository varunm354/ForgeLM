"""Offline tests for the bounded production-data CUDA canary."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.production_canary import (
    CANARY_TARGET_TOKENS,
    MAX_CANARY_OPTIMIZER_STEPS,
    run_production_canary,
    verify_canary_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "2" * 64
LOCK_IDENTITY = "a" * 64


def _data_config(tmp_path: Path) -> PretrainingDataConfig:
    return PretrainingDataConfig.load(
        PROJECT_ROOT / "configs/pretraining_data_smoke.yaml",
        project_root=PROJECT_ROOT,
    ).with_overrides(
        output_directory=tmp_path / "unused-data",
        tokenizer_path=tmp_path / "unused-tokenizer.json",
    )


def _config(tmp_path: Path) -> tuple[PretrainConfig, Path]:
    production_root = tmp_path / "production"
    manifest = production_root / "shards/manifest.json"
    lock = production_root / "manifests/production-data-lock.json"
    tokenizer = production_root / "tokenizer/tokenizer.json"
    model = tmp_path / "model.yaml"
    for path, contents in (
        (manifest, "offline manifest"),
        (lock, "offline lock"),
        (tokenizer, "offline tokenizer"),
        (model, "offline model"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    config = PretrainConfig(
        model_config_path=model,
        manifest_path=manifest,
        expected_tokenizer_fingerprint=FINGERPRINT,
        production_data_lock_path=lock,
        seed=1337,
        micro_batch_size=1,
        sequence_length=1024,
        gradient_accumulation_steps=2,
        target_training_tokens=CANARY_TARGET_TOKENS,
        learning_rate=6e-4,
        minimum_learning_rate=6e-5,
        warmup_steps=2,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        gradient_clipping=1.0,
        logging_interval=1,
        validation_interval=1,
        validation_batches=1,
        checkpoint_interval=2,
        precision="bfloat16",
        torch_compile=False,
        compile_allow_fallback=False,
        output_directory=tmp_path / "canary-output",
        latest_checkpoint_name="latest.pt",
        best_checkpoint_name="best.pt",
        metrics_filename="metrics.jsonl",
        metadata_filename="run_metadata.json",
        verify_checksums=True,
        fixed_validation_batches=True,
    )
    return config, production_root


def _validated(production_root: Path) -> dict[str, Any]:
    return {
        "maximum_optimizer_steps": 4,
        "hard_maximum_optimizer_steps": 4,
        "configured_optimizer_steps": 4,
        "effective_batch_tokens": 2_048,
        "maximum_training_tokens": 8_192,
        "production_data_root": str(production_root),
        "production_lock_identity": LOCK_IDENTITY,
        "production_lock_sha256": "b" * 64,
        "production_configuration_identity": "c" * 64,
        "manifest_sha256": "d" * 64,
        "tokenizer_fingerprint": FINGERPRINT,
        "tokenizer_manifest_identity": "e" * 64,
        "source": {
            "type": "huggingface_streaming",
            "dataset_name": "HuggingFaceFW/fineweb-edu",
        },
        "counts": {
            "train_tokens": 500_000_000,
            "validation_tokens": 5_000_000,
            "train_shards": 50,
            "validation_shards": 1,
        },
        "train_shard_names": ["train-00000.bin"],
        "validation_shard_names": ["validation-00000.bin"],
    }


def _patch_successful_cuda(
    monkeypatch: pytest.MonkeyPatch,
    production_root: Path,
) -> None:
    class FakeContext:
        world_size = 1
        device = torch.device("cuda")

        def shutdown(self) -> None:
            return None

    class FakeEngine:
        def __init__(
            self,
            config: PretrainConfig,
            _context: FakeContext,
            *,
            resume_checkpoint: Path | None = None,
        ) -> None:
            self.config = config
            self.model = object()
            self.precision = SimpleNamespace(name="bfloat16")
            self.training_dataset = SimpleNamespace(split="train")
            self.validation_dataset = SimpleNamespace(split="validation")
            self.stop_requested = False
            self.stop_signal = None
            step = 0
            if resume_checkpoint is not None:
                step = json.loads(resume_checkpoint.read_text())["optimizer_step"]
            self.progress = SimpleNamespace(
                optimizer_step=step,
                micro_step=step * 2,
                tokens_processed=step * 2_048,
            )

        def run(self, *, maximum_optimizer_steps: int) -> dict[str, Any]:
            start = self.progress.optimizer_step
            end = start + 1 if self.stop_requested else maximum_optimizer_steps
            output = self.config.output_directory
            output.mkdir(parents=True, exist_ok=True)
            for step in range(start + 1, end + 1):
                with (output / self.config.metrics_filename).open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        json.dumps(
                            {
                                "optimizer_step": step,
                                "micro_step": step * 2,
                                "tokens_processed": step * 2_048,
                                "learning_rate": 6e-4,
                                "training_loss": 9.5,
                                "validation_loss": 9.6,
                                "validation_perplexity": 14_765.0,
                                "gradient_norm": 1.0,
                                "elapsed_seconds": float(step),
                                "step_time_seconds": 0.05,
                                "tokens_per_second": 40_960.0,
                                "device_type": "cuda",
                                "precision": "bfloat16",
                                "world_size": 1,
                            }
                        )
                        + "\n"
                    )
            checkpoint = json.dumps({"optimizer_step": end})
            (output / self.config.latest_checkpoint_name).write_text(checkpoint)
            if self.stop_requested:
                (output / f"interrupt_step_{end:08d}.pt").write_text(checkpoint)
            (output / self.config.metadata_filename).write_text("{}")
            self.progress = SimpleNamespace(
                optimizer_step=end,
                micro_step=end * 2,
                tokens_processed=end * 2_048,
            )
            return {
                "optimizer_step": end,
                "micro_step": end * 2,
                "tokens_processed": end * 2_048,
                "training_loss": 9.5,
                "validation_loss": 9.6,
                "validation_perplexity": 14_765.0,
                "gradient_norm": 1.0,
            }

    monkeypatch.setattr(
        "forgelm.training.production_canary.validate_canary_configuration",
        lambda *args, **kwargs: _validated(production_root),
    )
    monkeypatch.setattr(
        "forgelm.training.production_canary.run_cloud_preflight",
        lambda *args, **kwargs: {
            "passed": True,
            "environment": {"cuda_available": True},
            "model": {
                "parameter_count": 49_295_872,
                "sequence_length": 1_024,
                "selected_precision": "bfloat16",
            },
        },
    )
    monkeypatch.setattr(
        "forgelm.training.production_canary.initialize_distributed",
        lambda _device: FakeContext(),
    )
    monkeypatch.setattr(
        "forgelm.training.production_canary.PretrainingEngine",
        FakeEngine,
    )
    monkeypatch.setattr(
        "forgelm.training.production_canary.parameter_report",
        lambda _model: SimpleNamespace(total_parameters=49_295_872),
    )
    monkeypatch.setattr(
        "forgelm.training.production_canary._memory_probe",
        lambda _engine: {
            "completed": True,
            "finite_loss": True,
            "finite_gradients": True,
            "loss": 9.7,
        },
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 1_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2_000)


def test_canary_forces_interruption_exact_resume_and_verifies_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, production_root = _config(tmp_path)
    _patch_successful_cuda(monkeypatch, production_root)

    report = run_production_canary(
        config,
        _data_config(tmp_path),
        expected_production_lock_identity=LOCK_IDENTITY,
        maximum_optimizer_steps=4,
        resume_checkpoint=None,
    )

    assert report["passed"] is True
    assert report["main_training_started"] is False
    assert report["training"]["simulated_interruption"] is True
    assert report["training"]["resume_start_step"] == 1
    assert report["training"]["post_resume_optimizer_steps"] == 3
    assert report["training"]["validation_used_for_training"] is False
    assert report["training"]["final"]["tokens_processed"] == 8_192
    assert report["metrics"]["record_count"] == 4
    assert all(
        item["status"] == "pass" for item in report["acceptance_criteria"]
    )

    verification = verify_canary_report(
        config.output_directory / "canary_report.json",
        expected_production_lock_identity=LOCK_IDENTITY,
        expected_tokenizer_fingerprint=FINGERPRINT,
    )
    assert verification["passed"] is True
    assert verification["optimizer_steps"] == 4
    assert verification["main_training_started"] is False

    with pytest.raises(FileExistsError, match="already passed"):
        run_production_canary(
            config,
            _data_config(tmp_path),
            expected_production_lock_identity=LOCK_IDENTITY,
            maximum_optimizer_steps=4,
            resume_checkpoint=(
                config.output_directory / "interrupt_step_00000001.pt"
            ),
        )

    with (config.output_directory / config.metrics_filename).open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="metrics"):
        verify_canary_report(
            config.output_directory / "canary_report.json",
            expected_production_lock_identity=LOCK_IDENTITY,
            expected_tokenizer_fingerprint=FINGERPRINT,
        )


def test_canary_refuses_cpu_fallback_and_writes_failure_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, production_root = _config(tmp_path)
    monkeypatch.setattr(
        "forgelm.training.production_canary.validate_canary_configuration",
        lambda *args, **kwargs: _validated(production_root),
    )
    monkeypatch.setattr(
        "forgelm.training.production_canary.run_cloud_preflight",
        lambda *args, **kwargs: {
            "passed": False,
            "environment": {"cuda_available": False},
            "model": {
                "parameter_count": 49_295_872,
                "sequence_length": 1_024,
                "selected_precision": None,
            },
        },
    )
    initialized = False

    def initialize(_device: str) -> None:
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(
        "forgelm.training.production_canary.initialize_distributed",
        initialize,
    )

    report = run_production_canary(
        config,
        _data_config(tmp_path),
        expected_production_lock_identity=LOCK_IDENTITY,
        maximum_optimizer_steps=4,
        resume_checkpoint=None,
    )

    assert report["passed"] is False
    assert report["training_started"] is False
    assert report["failure"]["type"] == "RuntimeError"
    assert initialized is False
    assert (config.output_directory / "canary_report.json").is_file()


def test_fresh_canary_never_overwrites_existing_output(tmp_path: Path) -> None:
    config, _ = _config(tmp_path)
    config.output_directory.mkdir(parents=True)
    marker = config.output_directory / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destructive overwrite"):
        run_production_canary(
            config,
            _data_config(tmp_path),
            expected_production_lock_identity=LOCK_IDENTITY,
            maximum_optimizer_steps=4,
            resume_checkpoint=None,
        )

    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_checked_canary_config_has_immutable_production_bounds() -> None:
    config = PretrainConfig.load(
        PROJECT_ROOT / "configs/pretrain_gpt_production_canary.yaml",
        project_root=PROJECT_ROOT,
    )

    assert config.expected_tokenizer_fingerprint == (
        "2f420b82fe239f8a80283ac8adb4abe956e3a7192321da1fbc4b36854144a740"
    )
    assert config.production_data_lock_path == Path(
        "/workspace/forgelm-production/fineweb-edu-500m-v1/"
        "manifests/production-data-lock.json"
    )
    assert config.precision == "bfloat16"
    assert config.sequence_length == 1_024
    assert config.micro_batch_size == 1
    assert config.gradient_accumulation_steps == 2
    assert config.target_training_tokens == CANARY_TARGET_TOKENS == 8_192
    assert config.optimizer_steps(1) == MAX_CANARY_OPTIMIZER_STEPS == 4
    assert config.verify_checksums is True
    assert config.fixed_validation_batches is True
