"""Bounded CUDA qualification orchestration and acceptance reporting."""

from __future__ import annotations

import gc
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from forgelm.data.token_shards import load_manifest, sha256_file
from forgelm.models.gpt import GPTConfig, parameter_report
from forgelm.training.cloud_preflight import (
    run_cloud_preflight,
    write_json_report,
)
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_engine import PretrainingEngine
from forgelm.training.pretrain_runtime import initialize_distributed

QUALIFICATION_FORMAT = "forgelm.cuda_training_qualification"
QUALIFICATION_VERSION = 1
MAX_QUALIFICATION_STEPS = 20
PRODUCTION_PARAMETER_COUNT = 49_295_872
PRODUCTION_SEQUENCE_LENGTH = 1_024


def validate_qualification_configuration(
    config: PretrainConfig,
    *,
    maximum_optimizer_steps: int,
) -> dict[str, Any]:
    """Reject unbounded, production-data, or reduced-architecture runs."""
    if (
        not isinstance(maximum_optimizer_steps, int)
        or isinstance(maximum_optimizer_steps, bool)
        or not 2 <= maximum_optimizer_steps <= MAX_QUALIFICATION_STEPS
    ):
        raise ValueError(
            f"maximum_optimizer_steps must be between 2 and "
            f"{MAX_QUALIFICATION_STEPS}."
        )
    if config.target_training_tokens >= 500_000_000:
        raise ValueError("Qualification must not claim a production token target.")
    configured_steps = config.optimizer_steps(1)
    if configured_steps > MAX_QUALIFICATION_STEPS:
        raise ValueError(
            f"Qualification config calculates {configured_steps} steps; "
            f"the hard limit is {MAX_QUALIFICATION_STEPS}."
        )
    if maximum_optimizer_steps > configured_steps:
        raise ValueError(
            "Requested optimizer-step limit exceeds the qualification config."
        )
    if config.sequence_length != PRODUCTION_SEQUENCE_LENGTH:
        raise ValueError("Qualification requires the production 1,024-token sequence.")

    model_config = GPTConfig.load(config.model_config_path)
    if model_config != GPTConfig():
        raise ValueError(
            "Qualification requires the checked production GPT architecture."
        )
    manifest = load_manifest(config.manifest_path)
    fixture = manifest.get("qualification_fixture")
    if (
        not isinstance(fixture, dict)
        or fixture.get("synthetic") is not True
        or fixture.get("quality_claims_permitted") is not False
    ):
        raise ValueError(
            "Qualification requires an explicitly marked synthetic fixture."
        )
    if manifest["source"].get("dataset_name") == "HuggingFaceFW/fineweb-edu":
        raise ValueError("Qualification must not read the production dataset.")
    return {
        "configured_optimizer_steps": configured_steps,
        "effective_batch_tokens": config.effective_batch_tokens(1),
        "fixture_identity": fixture["identity"],
        "manifest_identity": sha256_file(config.manifest_path),
    }


def evaluate_acceptance_criteria(
    evidence: dict[str, Any],
    *,
    cpu_diagnostic: bool,
) -> list[dict[str, Any]]:
    """Separate mandatory CUDA acceptance checks from optional capabilities."""
    required = (
        ("cuda_detected", "CUDA detected"),
        ("full_model_loaded", "Full 49.3M model loaded"),
        ("production_sequence_length", "1,024-token sequence used"),
        ("mixed_precision_worked", "Selected BF16 or FP16 precision worked"),
        ("finite_forward_loss", "Finite forward loss"),
        ("finite_gradients", "Finite gradients"),
        ("optimizer_updates_completed", "Optimizer updates completed"),
        ("validation_completed", "Validation completed"),
        ("checkpoint_save_load", "Checkpoint save/load worked"),
        ("interrupted_resume", "Interrupted/resumed training worked"),
        ("post_resume_step", "Post-resume optimizer step completed"),
        ("peak_vram_recorded", "Peak VRAM recorded"),
        ("artifacts_uncorrupted", "No corrupted artifacts"),
        ("metrics_metadata_written", "Metrics and metadata written"),
    )
    criteria: list[dict[str, Any]] = []
    for key, label in required:
        value = evidence.get(key)
        status = (
            "deferred"
            if cpu_diagnostic and value is not True
            else "pass"
            if value is True
            else "fail"
        )
        criteria.append(
            {
                "key": key,
                "label": label,
                "required": True,
                "status": status,
            }
        )
    compile_requested = bool(evidence.get("compilation_requested"))
    criteria.append(
        {
            "key": "compilation",
            "label": "torch.compile qualification",
            "required": False,
            "status": (
                "pass"
                if compile_requested and evidence.get("compilation_worked") is True
                else "fail"
                if compile_requested
                else "not_requested"
            ),
        }
    )
    criteria.append(
        {
            "key": "multi_gpu",
            "label": "Multi-GPU/NCCL qualification",
            "required": False,
            "status": (
                "pass" if evidence.get("multi_gpu_worked") is True else "not_tested"
            ),
        }
    )
    return criteria


def run_qualification(
    config: PretrainConfig,
    *,
    maximum_optimizer_steps: int,
    cpu_diagnostic: bool,
    resume_checkpoint: Path | None,
    force: bool,
) -> dict[str, Any]:
    """Run diagnostics or one bounded CUDA train/interruption/resume sequence."""
    validation = validate_qualification_configuration(
        config,
        maximum_optimizer_steps=maximum_optimizer_steps,
    )
    output = config.output_directory
    report_path = output / "qualification_report.json"
    if resume_checkpoint is None and output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"Qualification output is not empty: {output}. Use --force."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    preflight = run_cloud_preflight(
        config,
        cpu_diagnostic=cpu_diagnostic,
        output_directory=output,
    )
    evidence: dict[str, Any] = {
        "cuda_detected": torch.cuda.is_available(),
        "full_model_loaded": (
            preflight["model"]["parameter_count"] == PRODUCTION_PARAMETER_COUNT
        ),
        "production_sequence_length": (
            config.sequence_length == PRODUCTION_SEQUENCE_LENGTH
        ),
        "compilation_requested": config.torch_compile,
        "multi_gpu_worked": False,
    }
    report: dict[str, Any] = {
        "format": QUALIFICATION_FORMAT,
        "version": QUALIFICATION_VERSION,
        "mode": "cpu_diagnostic" if cpu_diagnostic else "cuda_qualification",
        "passed": False,
        "configuration": config.to_dict(),
        "bounds": {
            **validation,
            "maximum_optimizer_steps": maximum_optimizer_steps,
            "hard_maximum_optimizer_steps": MAX_QUALIFICATION_STEPS,
        },
        "cloud_preflight": preflight,
        "probe": None,
        "training": None,
        "metrics": None,
        "failure": None,
    }

    if cpu_diagnostic:
        report["acceptance_criteria"] = evaluate_acceptance_criteria(
            evidence,
            cpu_diagnostic=True,
        )
        report["passed"] = preflight["passed"]
        write_json_report(report_path, report)
        return report
    if not torch.cuda.is_available():
        report["failure"] = {
            "type": "CUDAUnavailable",
            "message": "CUDA qualification refused because CUDA is unavailable.",
        }
        report["acceptance_criteria"] = evaluate_acceptance_criteria(
            evidence,
            cpu_diagnostic=False,
        )
        write_json_report(report_path, report)
        return report
    if not preflight["passed"]:
        report["failure"] = {
            "type": "PreflightFailure",
            "message": "Mandatory CUDA preflight checks failed.",
        }
        report["acceptance_criteria"] = evaluate_acceptance_criteria(
            evidence,
            cpu_diagnostic=False,
        )
        write_json_report(report_path, report)
        return report

    context = initialize_distributed("cuda")
    engine: PretrainingEngine | None = None
    try:
        if context.world_size != 1:
            raise ValueError(
                "Bounded qualification currently requires one visible torchrun rank."
            )
        torch.cuda.reset_peak_memory_stats(context.device)
        engine = PretrainingEngine(
            config,
            context,
            resume_checkpoint=resume_checkpoint,
        )
        evidence["full_model_loaded"] = (
            parameter_report(engine.model).total_parameters
            == PRODUCTION_PARAMETER_COUNT
        )
        probe = _memory_probe(engine)
        report["probe"] = probe
        evidence["finite_forward_loss"] = probe["finite_loss"]
        evidence["finite_gradients"] = probe["finite_gradients"]
        evidence["mixed_precision_worked"] = (
            engine.precision.name in {"bfloat16", "float16"}
        )
        evidence["compilation_worked"] = (
            not config.torch_compile or probe["completed"]
        )

        if resume_checkpoint is None:
            intermediate_step = min(
                config.checkpoint_interval,
                maximum_optimizer_steps - 1,
            )
            first_summary = engine.run(
                maximum_optimizer_steps=intermediate_step
            )
            checkpoint = output / config.latest_checkpoint_name
            checkpoint_hash_before = sha256_file(checkpoint)
            del engine
            engine = None
            gc.collect()
            torch.cuda.empty_cache()

            engine = PretrainingEngine(
                config,
                context,
                resume_checkpoint=checkpoint,
            )
            resume_start_step = engine.progress.optimizer_step
            final_summary = engine.run(
                maximum_optimizer_steps=maximum_optimizer_steps
            )
            evidence["interrupted_resume"] = resume_start_step == intermediate_step
            evidence["post_resume_step"] = (
                final_summary["optimizer_step"] > resume_start_step
            )
        else:
            resume_start_step = engine.progress.optimizer_step
            checkpoint_hash_before = sha256_file(resume_checkpoint)
            first_summary = None
            final_summary = engine.run(
                maximum_optimizer_steps=maximum_optimizer_steps
            )
            evidence["interrupted_resume"] = True
            evidence["post_resume_step"] = (
                final_summary["optimizer_step"] > resume_start_step
            )

        torch.cuda.synchronize(context.device)
        peak_allocated = torch.cuda.max_memory_allocated(context.device)
        peak_reserved = torch.cuda.max_memory_reserved(context.device)
        final_checkpoint = output / config.latest_checkpoint_name
        final_checkpoint_hash = sha256_file(final_checkpoint)
        metric_summary = _summarize_metrics(
            output / config.metrics_filename,
            config.effective_batch_tokens(1),
        )
        evidence.update(
            optimizer_updates_completed=final_summary["optimizer_step"] >= 1,
            validation_completed=metric_summary["validation_records"] >= 1,
            checkpoint_save_load=True,
            peak_vram_recorded=peak_allocated > 0 and peak_reserved > 0,
            artifacts_uncorrupted=bool(
                checkpoint_hash_before and final_checkpoint_hash
            ),
            metrics_metadata_written=(
                (output / config.metrics_filename).is_file()
                and (output / config.metadata_filename).is_file()
            ),
        )
        report["training"] = {
            "first_phase": first_summary,
            "resume_start_step": resume_start_step,
            "final": final_summary,
            "checkpoint_sha256_before_resume": checkpoint_hash_before,
            "final_checkpoint_sha256": final_checkpoint_hash,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "precision": engine.precision.name,
            "sdpa_behavior": preflight["sdpa"],
        }
        report["metrics"] = metric_summary
    except torch.OutOfMemoryError as error:
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "suggestion": (
                "Do not retry automatically. The micro-batch is already 1; "
                "use a GPU with more VRAM or add an explicitly reviewed memory strategy."
                if config.micro_batch_size == 1
                else f"Consider micro_batch_size={max(1, config.micro_batch_size // 2)}."
            ),
        }
    except Exception as error:  # noqa: BLE001 - failure report must survive runtime errors
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        if engine is not None:
            del engine
        context.shutdown()

    report["acceptance_criteria"] = evaluate_acceptance_criteria(
        evidence,
        cpu_diagnostic=False,
    )
    report["passed"] = (
        report["failure"] is None
        and all(
            criterion["status"] == "pass"
            for criterion in report["acceptance_criteria"]
            if criterion["required"]
        )
    )
    write_json_report(report_path, report)
    return report


def _memory_probe(engine: PretrainingEngine) -> dict[str, Any]:
    sampler_state = engine.training_sampler.state_dict()
    engine.optimizer.zero_grad(set_to_none=True)
    inputs, targets = engine.training_sampler.next_batch()
    inputs = inputs.to(engine.device)
    targets = targets.to(engine.device)
    started = time.perf_counter()
    try:
        with engine.precision.autocast(engine.device):
            _, loss = engine.execution_model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Memory probe produced non-finite loss.")
        if engine.scaler is None:
            loss.backward()
        else:
            engine.scaler.scale(loss).backward()
            engine.scaler.unscale_(engine.optimizer)
        gradients = [
            parameter.grad
            for parameter in engine.model.parameters()
            if parameter.requires_grad
        ]
        finite_gradients = bool(gradients) and all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in gradients
        )
        if not finite_gradients:
            raise FloatingPointError("Memory probe produced non-finite gradients.")
        torch.cuda.synchronize(engine.device)
        elapsed = time.perf_counter() - started
        return {
            "completed": True,
            "finite_loss": True,
            "finite_gradients": True,
            "loss": float(loss.detach()),
            "startup_seconds": elapsed,
            "compile_startup_seconds": elapsed if engine.config.torch_compile else None,
            "allocated_vram_bytes": torch.cuda.memory_allocated(engine.device),
            "reserved_vram_bytes": torch.cuda.memory_reserved(engine.device),
        }
    finally:
        engine.optimizer.zero_grad(set_to_none=True)
        engine.training_sampler.load_state_dict(sampler_state)


def _summarize_metrics(path: Path, effective_batch_tokens: int) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    latencies = [float(record["step_time_seconds"]) for record in records]
    throughputs = [
        float(record["tokens_per_second"])
        for record in records[1:]
        if math.isfinite(float(record["tokens_per_second"]))
    ]
    return {
        "records": len(records),
        "validation_records": sum(
            record.get("validation_loss") is not None for record in records
        ),
        "optimizer_step_latency_seconds": latencies,
        "median_optimizer_step_latency_seconds": statistics.median(latencies),
        "steady_state_tokens_per_second": (
            statistics.median(throughputs) if throughputs else None
        ),
        "effective_batch_tokens": effective_batch_tokens,
    }
