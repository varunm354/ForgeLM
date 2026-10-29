"""Fail-closed, bounded CUDA canary over verified production token shards."""

from __future__ import annotations

import gc
import json
import math
import signal
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.production_data import (
    ProductionPaths,
    production_configuration_identity,
    verify_production_stage,
)
from forgelm.data.token_shards import load_manifest, sha256_file
from forgelm.models.gpt import GPTConfig, parameter_report
from forgelm.training.cloud_preflight import run_cloud_preflight, write_json_report
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_engine import PretrainingEngine
from forgelm.training.pretrain_runtime import initialize_distributed
from forgelm.training.qualification import (
    PRODUCTION_PARAMETER_COUNT,
    PRODUCTION_SEQUENCE_LENGTH,
    _memory_probe,
)

CANARY_FORMAT = "forgelm.production_data_cuda_canary"
CANARY_VERSION = 1
CANARY_RUN_MARKER_FORMAT = "forgelm.production_data_cuda_canary_run"
MAX_CANARY_OPTIMIZER_STEPS = 4
CANARY_MICRO_BATCH_SIZE = 1
CANARY_GRADIENT_ACCUMULATION_STEPS = 2
CANARY_EFFECTIVE_BATCH_TOKENS = 2_048
CANARY_TARGET_TOKENS = (
    MAX_CANARY_OPTIMIZER_STEPS * CANARY_EFFECTIVE_BATCH_TOKENS
)

_REQUIRED_CRITERIA = (
    ("production_artifacts_verified", "Production lock and all artifact checks passed"),
    ("production_lock_pinned", "Expected production-lock identity matched"),
    ("production_tokenizer_pinned", "Expected production tokenizer matched"),
    ("production_source", "Pinned FineWeb-Edu production source used"),
    ("bounded_configuration", "Canary remained within its hard step/token bounds"),
    ("train_validation_separated", "Training and validation shards stayed separate"),
    ("cuda_detected", "CUDA was detected without CPU fallback"),
    ("full_model_loaded", "Full 49.3M-parameter model loaded"),
    ("production_sequence_length", "Sequence length was 1,024"),
    ("bf16_amp", "BF16 automatic mixed precision was active"),
    ("finite_forward_loss", "Forward probe produced finite loss"),
    ("finite_gradients", "Backward probe and training gradients were finite"),
    ("optimizer_updates_completed", "Bounded optimizer updates completed"),
    ("validation_completed", "Validation completed on the validation split"),
    ("interruption_checkpoint", "An interruption-boundary checkpoint was saved"),
    ("checkpoint_save_load", "Checkpoint save and strict load succeeded"),
    ("exact_resume", "Optimizer, RNG, counters, and samplers resumed exactly"),
    ("post_resume_step", "At least one optimizer update completed after resume"),
    ("metrics_complete", "Loss, LR, gradient, throughput, token, and timing metrics exist"),
    ("peak_vram_recorded", "Peak allocated and reserved VRAM were recorded"),
    ("artifacts_hashed", "Canary evidence and checkpoints were hashed"),
)


def validate_canary_configuration(
    config: PretrainConfig,
    data_config: PretrainingDataConfig,
    *,
    expected_production_lock_identity: str,
    maximum_optimizer_steps: int,
) -> dict[str, Any]:
    """Validate every production identity and reject non-canary configurations."""
    _validate_sha256(
        expected_production_lock_identity,
        "expected_production_lock_identity",
    )
    if (
        not isinstance(maximum_optimizer_steps, int)
        or isinstance(maximum_optimizer_steps, bool)
        or not 2 <= maximum_optimizer_steps <= MAX_CANARY_OPTIMIZER_STEPS
    ):
        raise ValueError(
            "maximum_optimizer_steps must be between 2 and "
            f"{MAX_CANARY_OPTIMIZER_STEPS}."
        )
    if config.production_data_lock_path is None:
        raise ValueError("Production canary requires production_data_lock_path.")
    if config.expected_tokenizer_fingerprint is None:
        raise ValueError("Production canary requires a pinned tokenizer fingerprint.")
    if config.precision != "bfloat16":
        raise ValueError("Production canary requires explicit bfloat16 precision.")
    if config.torch_compile:
        raise ValueError("Production canary must use the bounded eager path.")
    if not config.verify_checksums:
        raise ValueError("Production canary requires shard checksum verification.")
    if not config.fixed_validation_batches:
        raise ValueError("Production canary requires fixed validation batches.")
    if (
        config.micro_batch_size != CANARY_MICRO_BATCH_SIZE
        or config.gradient_accumulation_steps
        != CANARY_GRADIENT_ACCUMULATION_STEPS
        or config.sequence_length != PRODUCTION_SEQUENCE_LENGTH
        or config.target_training_tokens != CANARY_TARGET_TOKENS
        or config.optimizer_steps(1) != MAX_CANARY_OPTIMIZER_STEPS
        or config.validation_interval != 1
        or config.validation_batches != 1
    ):
        raise ValueError("Production canary training bounds or validation policy changed.")
    if maximum_optimizer_steps > config.optimizer_steps(1):
        raise ValueError("Requested canary limit exceeds configured optimizer steps.")
    if config.target_training_tokens >= data_config.target_tokens:
        raise ValueError("Canary must remain smaller than the production training target.")

    model_config = GPTConfig.load(config.model_config_path)
    if model_config != GPTConfig():
        raise ValueError("Production canary requires the checked production GPT model.")

    lock_path = config.production_data_lock_path.resolve()
    production_root = _production_root_from_lock(lock_path)
    paths = ProductionPaths.from_root(production_root)
    if config.manifest_path.resolve() != paths.shard_manifest_path:
        raise ValueError("Canary manifest path is outside the production-lock root.")
    if not lock_path.is_file() or not paths.verification_report_path.is_file():
        raise ValueError(
            "Canary requires the existing Stage E lock and verification report."
        )
    lock = verify_production_stage(data_config, production_root)
    if lock["identity"] != expected_production_lock_identity:
        raise ValueError("Production-lock identity does not match explicit approval.")
    if (
        lock["configuration_identity"]
        != production_configuration_identity(data_config)
    ):
        raise ValueError("Production-lock data configuration identity mismatch.")
    if (
        lock["tokenizer"]["fingerprint"]
        != config.expected_tokenizer_fingerprint
    ):
        raise ValueError("Canary tokenizer fingerprint does not match production lock.")

    manifest = load_manifest(config.manifest_path)
    source = manifest["source"]
    if (
        source.get("type") != "huggingface_streaming"
        or source.get("dataset_name") != "HuggingFaceFW/fineweb-edu"
        or source.get("streaming") is not True
        or source.get("dataset_revision") != data_config.dataset_revision
        or "qualification_fixture" in manifest
    ):
        raise ValueError("Canary requires the pinned production FineWeb-Edu stream.")
    train_names = {entry["name"] for entry in manifest["shards"]["train"]}
    validation_names = {
        entry["name"] for entry in manifest["shards"]["validation"]
    }
    if not train_names or not validation_names or train_names & validation_names:
        raise ValueError("Training and validation shard identities must be disjoint.")

    return {
        "maximum_optimizer_steps": maximum_optimizer_steps,
        "hard_maximum_optimizer_steps": MAX_CANARY_OPTIMIZER_STEPS,
        "configured_optimizer_steps": config.optimizer_steps(1),
        "effective_batch_tokens": config.effective_batch_tokens(1),
        "maximum_training_tokens": (
            maximum_optimizer_steps * config.effective_batch_tokens(1)
        ),
        "production_data_root": str(production_root),
        "production_lock_identity": lock["identity"],
        "production_lock_sha256": sha256_file(lock_path),
        "production_configuration_identity": lock["configuration_identity"],
        "manifest_sha256": sha256_file(config.manifest_path),
        "tokenizer_fingerprint": lock["tokenizer"]["fingerprint"],
        "tokenizer_manifest_identity": lock["tokenizer"]["manifest_identity"],
        "source": source,
        "counts": lock["counts"],
        "train_shard_names": sorted(train_names),
        "validation_shard_names": sorted(validation_names),
    }


def run_production_canary(
    config: PretrainConfig,
    data_config: PretrainingDataConfig,
    *,
    expected_production_lock_identity: str,
    maximum_optimizer_steps: int,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    """Run a fresh simulated interruption/resume or safely resume one checkpoint."""
    output = config.output_directory.resolve()
    report_path = output / "canary_report.json"
    run_marker_path = output / ".canary-run.json"
    if resume_checkpoint is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Fresh canary output is not empty: {output}. "
            "Use a new directory; destructive overwrite is not supported."
        )
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        if resume_checkpoint.parent != output:
            raise ValueError("Resume checkpoint must be inside the canary output directory.")
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        if report_path.is_file():
            try:
                prior_report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("Existing canary report is malformed.") from error
            if isinstance(prior_report, dict) and prior_report.get("passed") is True:
                raise FileExistsError(
                    "Canary already passed; verify it or use a new output directory."
                )
    output.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is None:
        marker = {
            "format": CANARY_RUN_MARKER_FORMAT,
            "version": CANARY_VERSION,
            "expected_production_lock_identity": (
                expected_production_lock_identity
            ),
            "output_directory": str(output),
        }
        with run_marker_path.open("x", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        try:
            marker = json.loads(run_marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Canary resume marker is missing or malformed.") from error
        if (
            not isinstance(marker, dict)
            or marker.get("format") != CANARY_RUN_MARKER_FORMAT
            or marker.get("version") != CANARY_VERSION
            or marker.get("expected_production_lock_identity")
            != expected_production_lock_identity
            or marker.get("output_directory") != str(output)
        ):
            raise ValueError("Canary resume marker is incompatible.")

    started = time.perf_counter()
    evidence: dict[str, Any] = {}
    report: dict[str, Any] = {
        "format": CANARY_FORMAT,
        "version": CANARY_VERSION,
        "passed": False,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "configuration": config.to_dict(),
        "bounds": None,
        "preflight": None,
        "probe": None,
        "training": None,
        "metrics": None,
        "artifacts": None,
        "failure": None,
        "main_training_started": False,
    }
    context = None
    engine: PretrainingEngine | None = None
    training_started = False
    try:
        validation = validate_canary_configuration(
            config,
            data_config,
            expected_production_lock_identity=expected_production_lock_identity,
            maximum_optimizer_steps=maximum_optimizer_steps,
        )
        report["bounds"] = validation
        evidence.update(
            production_artifacts_verified=True,
            production_lock_pinned=True,
            production_tokenizer_pinned=True,
            production_source=True,
            bounded_configuration=True,
            train_validation_separated=True,
        )

        preflight = run_cloud_preflight(
            config,
            cpu_diagnostic=False,
            output_directory=output,
        )
        report["preflight"] = preflight
        if not preflight["passed"]:
            raise RuntimeError("Mandatory production CUDA preflight checks failed.")
        evidence["cuda_detected"] = (
            preflight["environment"]["cuda_available"] is True
        )
        evidence["full_model_loaded"] = (
            preflight["model"]["parameter_count"] == PRODUCTION_PARAMETER_COUNT
        )
        evidence["production_sequence_length"] = (
            preflight["model"]["sequence_length"] == PRODUCTION_SEQUENCE_LENGTH
        )
        evidence["bf16_amp"] = (
            preflight["model"]["selected_precision"] == "bfloat16"
        )

        context = initialize_distributed("cuda")
        if context.world_size != 1:
            raise ValueError("Production canary requires exactly one visible rank.")
        torch.cuda.reset_peak_memory_stats(context.device)
        engine = PretrainingEngine(
            config,
            context,
            resume_checkpoint=resume_checkpoint,
        )
        training_started = True
        evidence["full_model_loaded"] = (
            parameter_report(engine.model).total_parameters
            == PRODUCTION_PARAMETER_COUNT
        )
        evidence["train_validation_separated"] = (
            engine.training_dataset.split == "train"
            and engine.validation_dataset.split == "validation"
            and engine.training_dataset is not engine.validation_dataset
        )
        probe = _memory_probe(engine)
        report["probe"] = probe
        evidence["finite_forward_loss"] = probe["finite_loss"]
        evidence["finite_gradients"] = probe["finite_gradients"]
        evidence["bf16_amp"] = engine.precision.name == "bfloat16"

        first_summary: dict[str, Any] | None = None
        if resume_checkpoint is None:
            engine.stop_requested = True
            engine.stop_signal = signal.SIGTERM
            first_summary = engine.run(
                maximum_optimizer_steps=maximum_optimizer_steps
            )
            interrupt_step = first_summary["optimizer_step"]
            interrupt_checkpoint = (
                output / f"interrupt_step_{interrupt_step:08d}.pt"
            )
            if interrupt_step != 1 or not interrupt_checkpoint.is_file():
                raise RuntimeError(
                    "Simulated interruption did not publish its boundary checkpoint."
                )
            evidence["interruption_checkpoint"] = True
            interruption_hash = sha256_file(interrupt_checkpoint)
            del engine
            engine = None
            gc.collect()
            torch.cuda.empty_cache()
            loaded_checkpoint = interrupt_checkpoint
        else:
            interrupt_candidates = sorted(output.glob("interrupt_step_*.pt"))
            evidence["interruption_checkpoint"] = bool(interrupt_candidates)
            interrupt_checkpoint = (
                interrupt_candidates[-1]
                if interrupt_candidates
                else resume_checkpoint
            )
            interruption_hash = sha256_file(interrupt_checkpoint)
            loaded_checkpoint = resume_checkpoint

        loaded_checkpoint_hash = sha256_file(loaded_checkpoint)
        engine = PretrainingEngine(
            config,
            context,
            resume_checkpoint=loaded_checkpoint,
        )
        resume_start_step = engine.progress.optimizer_step
        metrics_path = output / config.metrics_filename
        if metrics_path.is_file():
            existing_metrics = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if any(
                record.get("optimizer_step", -1) > resume_start_step
                for record in existing_metrics
            ):
                raise ValueError(
                    "Metrics are ahead of the resume checkpoint; refusing "
                    "to append duplicate or conflicting evidence."
                )
        evidence["checkpoint_save_load"] = True
        evidence["exact_resume"] = (
            resume_start_step >= 1
            and engine.progress.micro_step
            == resume_start_step * config.gradient_accumulation_steps
            and engine.progress.tokens_processed
            == resume_start_step * config.effective_batch_tokens(1)
        )
        final_summary = engine.run(
            maximum_optimizer_steps=maximum_optimizer_steps
        )
        evidence["post_resume_step"] = (
            final_summary["optimizer_step"] > resume_start_step
        )
        evidence["optimizer_updates_completed"] = (
            final_summary["optimizer_step"] == maximum_optimizer_steps
        )

        torch.cuda.synchronize(context.device)
        peak_allocated = torch.cuda.max_memory_allocated(context.device)
        peak_reserved = torch.cuda.max_memory_reserved(context.device)
        evidence["peak_vram_recorded"] = (
            peak_allocated > 0 and peak_reserved > 0
        )
        metrics = _load_metrics(output / config.metrics_filename)
        report["metrics"] = metrics
        evidence["metrics_complete"] = metrics["complete"]
        evidence["validation_completed"] = metrics["validation_records"] >= 1
        final_checkpoint = output / config.latest_checkpoint_name
        artifacts = _artifact_records(
            {
                "production_lock": config.production_data_lock_path,
                "shard_manifest": config.manifest_path,
                "production_tokenizer": Path(
                    validation["production_data_root"]
                )
                / "tokenizer/tokenizer.json",
                "model_configuration": config.model_config_path,
                "run_marker": run_marker_path,
                "metrics": output / config.metrics_filename,
                "run_metadata": output / config.metadata_filename,
                "interruption_checkpoint": interrupt_checkpoint,
                "final_checkpoint": final_checkpoint,
            }
        )
        report["artifacts"] = artifacts
        evidence["artifacts_hashed"] = all(
            item["sha256"] for item in artifacts.values()
        )
        report["training"] = {
            "simulated_interruption": first_summary is not None,
            "first_phase": first_summary,
            "interruption_signal": signal.Signals(signal.SIGTERM).name,
            "interruption_checkpoint": str(interrupt_checkpoint),
            "interruption_checkpoint_sha256": interruption_hash,
            "loaded_checkpoint": str(loaded_checkpoint),
            "loaded_checkpoint_sha256_before_resume": loaded_checkpoint_hash,
            "resume_start_step": resume_start_step,
            "final": final_summary,
            "post_resume_optimizer_steps": (
                final_summary["optimizer_step"] - resume_start_step
            ),
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "precision": engine.precision.name,
            "training_split": engine.training_dataset.split,
            "validation_split": engine.validation_dataset.split,
            "validation_used_for_training": False,
        }
    except torch.OutOfMemoryError as error:
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "suggestion": (
                "Do not retry automatically or reduce the approved model. "
                "Use a GPU with enough VRAM."
            ),
        }
    except Exception as error:  # noqa: BLE001 - preserve a machine-readable failure
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        if engine is not None:
            del engine
        if context is not None:
            context.shutdown()

    report["acceptance_criteria"] = _acceptance_criteria(
        evidence,
        report["failure"],
    )
    report["passed"] = (
        report["failure"] is None
        and all(
            item["status"] == "pass"
            for item in report["acceptance_criteria"]
        )
    )
    report["status"] = "passed" if report["passed"] else "failed"
    report["training_started"] = training_started
    report["elapsed_seconds"] = time.perf_counter() - started
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json_report(report_path, report)
    return report


def verify_canary_report(
    report_path: str | Path,
    *,
    expected_production_lock_identity: str,
    expected_tokenizer_fingerprint: str,
) -> dict[str, Any]:
    """Re-hash a passing canary report and reject stale or altered evidence."""
    _validate_sha256(
        expected_production_lock_identity,
        "expected_production_lock_identity",
    )
    _validate_sha256(
        expected_tokenizer_fingerprint,
        "expected_tokenizer_fingerprint",
    )
    source = Path(report_path).expanduser().resolve()
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load canary report: {source}") from error
    if not isinstance(report, dict):
        raise TypeError("Canary report must be a JSON object.")
    if (
        report.get("format") != CANARY_FORMAT
        or report.get("version") != CANARY_VERSION
        or report.get("passed") is not True
        or report.get("status") != "passed"
        or report.get("main_training_started") is not False
    ):
        raise ValueError("Canary report is not a passing bounded-canary report.")
    criteria = report.get("acceptance_criteria")
    expected_criteria = {key for key, _label in _REQUIRED_CRITERIA}
    if (
        not isinstance(criteria, list)
        or len(criteria) != len(_REQUIRED_CRITERIA)
        or any(item.get("status") != "pass" for item in criteria)
        or {item.get("key") for item in criteria} != expected_criteria
    ):
        raise ValueError("Canary acceptance criteria are incomplete.")
    bounds = report.get("bounds")
    if not isinstance(bounds, dict):
        raise TypeError("Canary report bounds must be an object.")
    if bounds.get("production_lock_identity") != expected_production_lock_identity:
        raise ValueError("Canary report production-lock identity mismatch.")
    if bounds.get("tokenizer_fingerprint") != expected_tokenizer_fingerprint:
        raise ValueError("Canary report tokenizer fingerprint mismatch.")
    optimizer_steps = bounds.get("maximum_optimizer_steps")
    maximum_tokens = bounds.get("maximum_training_tokens")
    if (
        bounds.get("hard_maximum_optimizer_steps")
        != MAX_CANARY_OPTIMIZER_STEPS
        or not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or not 2 <= optimizer_steps <= MAX_CANARY_OPTIMIZER_STEPS
        or maximum_tokens
        != optimizer_steps * CANARY_EFFECTIVE_BATCH_TOKENS
    ):
        raise ValueError("Canary report exceeds hard bounds.")
    configuration = report.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("target_training_tokens") != CANARY_TARGET_TOKENS
        or configuration.get("sequence_length") != PRODUCTION_SEQUENCE_LENGTH
        or configuration.get("micro_batch_size") != CANARY_MICRO_BATCH_SIZE
        or configuration.get("gradient_accumulation_steps")
        != CANARY_GRADIENT_ACCUMULATION_STEPS
        or configuration.get("precision") != "bfloat16"
        or configuration.get("verify_checksums") is not True
        or configuration.get("fixed_validation_batches") is not True
    ):
        raise ValueError("Canary report configuration is incompatible.")
    training = report.get("training")
    if (
        not isinstance(training, dict)
        or training.get("validation_used_for_training") is not False
        or training.get("precision") != "bfloat16"
        or training.get("final", {}).get("optimizer_step") != optimizer_steps
        or training.get("final", {}).get("tokens_processed") != maximum_tokens
        or training.get("post_resume_optimizer_steps", 0) < 1
    ):
        raise ValueError("Canary report training evidence is incompatible.")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Canary report has no artifact hashes.")
    for name, item in artifacts.items():
        if not isinstance(item, dict):
            raise TypeError(f"Canary artifact {name!r} must be an object.")
        path = Path(item.get("path", ""))
        expected_hash = item.get("sha256")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Canary artifact checksum mismatch: {name}.")
    return {
        "passed": True,
        "report_path": str(source),
        "production_lock_identity": expected_production_lock_identity,
        "tokenizer_fingerprint": expected_tokenizer_fingerprint,
        "verified_artifacts": len(artifacts),
        "optimizer_steps": training["final"]["optimizer_step"],
        "tokens_processed": training["final"]["tokens_processed"],
        "main_training_started": False,
    }


def _production_root_from_lock(lock_path: Path) -> Path:
    if (
        lock_path.name != "production-data-lock.json"
        or lock_path.parent.name != "manifests"
    ):
        raise ValueError("Production lock must use the verified artifact layout.")
    return lock_path.parent.parent


def _acceptance_criteria(
    evidence: dict[str, Any],
    failure: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    criteria = []
    for key, label in _REQUIRED_CRITERIA:
        passed = evidence.get(key) is True
        criteria.append(
            {
                "key": key,
                "label": label,
                "required": True,
                "status": "pass" if passed else "fail",
                "reason": (
                    None
                    if passed
                    else (
                        f"Canary aborted with {failure['type']}: "
                        f"{failure['message']}"
                        if failure is not None
                        else "No passing evidence was recorded."
                    )
                ),
            }
        )
    return criteria


def _load_metrics(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "optimizer_step",
        "micro_step",
        "tokens_processed",
        "learning_rate",
        "training_loss",
        "gradient_norm",
        "elapsed_seconds",
        "step_time_seconds",
        "tokens_per_second",
        "device_type",
        "precision",
    }
    complete = bool(records)
    for record in records:
        complete = complete and required <= record.keys()
        for name in (
            "learning_rate",
            "training_loss",
            "gradient_norm",
            "elapsed_seconds",
            "step_time_seconds",
            "tokens_per_second",
        ):
            value = record.get(name)
            complete = complete and isinstance(value, (int, float))
            complete = complete and math.isfinite(float(value))
        complete = complete and record.get("device_type") == "cuda"
        complete = complete and record.get("precision") == "bfloat16"
    latencies = [float(record["step_time_seconds"]) for record in records]
    throughputs = [float(record["tokens_per_second"]) for record in records]
    return {
        "complete": complete,
        "records": records,
        "record_count": len(records),
        "validation_records": sum(
            record.get("validation_loss") is not None for record in records
        ),
        "median_optimizer_step_seconds": statistics.median(latencies),
        "median_tokens_per_second": statistics.median(throughputs),
    }


def _artifact_records(paths: dict[str, Path | None]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, candidate in paths.items():
        if candidate is None:
            raise ValueError(f"Required canary artifact path is missing: {name}.")
        path = candidate.resolve()
        if not path.is_file():
            raise ValueError(f"Required canary artifact is missing: {path}.")
        records[name] = {
            "path": str(path),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hexadecimal.")
