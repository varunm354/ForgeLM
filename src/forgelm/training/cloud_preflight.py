"""Offline environment and artifact checks for CUDA qualification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from forgelm.data.token_shards import sha256_file
from forgelm.models.gpt import GPTConfig, GPTLanguageModel, parameter_report
from forgelm.training.pretrain_config import PrecisionName, PretrainConfig
from forgelm.training.pretrain_runtime import (
    select_precision,
    validate_training_compatibility,
)

CLOUD_PREFLIGHT_FORMAT = "forgelm.cuda_environment_preflight"
CLOUD_PREFLIGHT_VERSION = 1
PRODUCTION_MODEL_PARAMETERS = 49_295_872
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


def run_cloud_preflight(
    training_config: PretrainConfig,
    *,
    cpu_diagnostic: bool,
    precision_override: PrecisionName | None = None,
    output_directory: Path | None = None,
    expected_model_parameters: int | None = PRODUCTION_MODEL_PARAMETERS,
) -> dict[str, Any]:
    """Return a secret-free report without requiring network access."""
    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        passed: bool | None,
        detail: str,
        *,
        required: bool = True,
    ) -> None:
        checks.append(
            {
                "name": name,
                "required": required,
                "status": (
                    "pass" if passed is True else "fail" if passed is False else "deferred"
                ),
                "detail": detail,
            }
        )

    model_config = GPTConfig.load(training_config.model_config_path)
    manifest, manifest_identity = validate_training_compatibility(
        training_config.manifest_path,
        model_config,
        training_config,
    )
    check("configuration_compatibility", True, "Model, training, and manifest agree.")

    model = GPTLanguageModel(model_config)
    parameters = parameter_report(model)
    expected_ok = (
        expected_model_parameters is None
        or parameters.total_parameters == expected_model_parameters
    )
    check(
        "model_parameter_count",
        expected_ok,
        f"Loaded {parameters.total_parameters:,} unique trainable parameters."
        + (
            ""
            if expected_model_parameters is None
            else f" Expected {expected_model_parameters:,}."
        ),
    )
    del model

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    if cpu_diagnostic:
        check(
            "cuda_detected",
            None,
            "CPU diagnostic mode explicitly defers mandatory CUDA detection.",
            required=False,
        )
    else:
        check(
            "cuda_detected",
            cuda_available,
            "CUDA is available." if cuda_available else "CUDA is unavailable.",
        )
        check(
            "visible_gpu_count",
            gpu_count >= 1,
            f"{gpu_count} visible CUDA device(s).",
        )

    device = torch.device("cuda", 0) if cuda_available else torch.device("cpu")
    requested_precision = precision_override or training_config.precision
    try:
        precision = select_precision(requested_precision, device)
    except ValueError as error:
        precision_name = None
        check("precision_selection", False, str(error), required=not cpu_diagnostic)
    else:
        precision_name = precision.name
        precision_ok = (
            precision.name in {"bfloat16", "float16"}
            if not cpu_diagnostic
            else precision.name == "float32" or cuda_available
        )
        check(
            "precision_selection",
            precision_ok,
            f"Requested {requested_precision}; selected {precision.name}.",
            required=not cpu_diagnostic,
        )

    gpu: dict[str, Any] = {
        "name": None,
        "compute_capability": None,
        "total_vram_bytes": None,
        "initially_free_vram_bytes": None,
        "bf16_supported": False,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            total_vram_bytes=properties.total_memory,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        except (RuntimeError, TypeError):
            pass
        else:
            gpu["initially_free_vram_bytes"] = free_bytes
            gpu["total_vram_bytes"] = total_bytes

    cudnn_available = bool(torch.backends.cudnn.is_available())
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    distributed_available = torch.distributed.is_available()
    nccl_available = (
        bool(torch.distributed.is_nccl_available())
        if distributed_available
        else False
    )
    compile_available = callable(getattr(torch, "compile", None))
    check(
        "torch_compile_available",
        compile_available,
        "torch.compile is callable." if compile_available else "torch.compile unavailable.",
        required=False,
    )
    check(
        "nccl_available",
        nccl_available if cuda_available else None,
        "NCCL is available."
        if nccl_available
        else "NCCL is unavailable or no CUDA device is active.",
        required=False,
    )

    disk_path = _nearest_existing_path(
        output_directory or training_config.output_directory
    )
    disk = shutil.disk_usage(disk_path)
    check(
        "free_disk_space",
        disk.free > 0,
        f"{disk.free:,} bytes free at {disk_path}.",
    )

    sdpa = {
        "api_available": callable(
            getattr(torch.nn.functional, "scaled_dot_product_attention", None)
        ),
        "automatic_dispatch": True,
        "flash_dispatch_setting_enabled": _safe_backend_flag("flash_sdp_enabled"),
        "memory_efficient_dispatch_setting_enabled": _safe_backend_flag(
            "mem_efficient_sdp_enabled"
        ),
        "math_dispatch_setting_enabled": _safe_backend_flag("math_sdp_enabled"),
        "cuda_kernel_eligibility": (
            "not probed without representative CUDA tensors"
            if cuda_available
            else "not applicable on CPU"
        ),
        "selected_backend": (
            "automatic CUDA dispatch; exact kernel determined at operation runtime"
            if cuda_available
            else "CPU math implementation"
        ),
    }
    check(
        "sdpa_api_available",
        sdpa["api_available"],
        str(sdpa["selected_backend"]),
    )

    fixture = manifest.get("qualification_fixture")
    fixture_identity = (
        fixture.get("identity") if isinstance(fixture, dict) else None
    )
    report = {
        "format": CLOUD_PREFLIGHT_FORMAT,
        "version": CLOUD_PREFLIGHT_VERSION,
        "mode": "cpu_diagnostic" if cpu_diagnostic else "cuda_required",
        "passed": all(
            item["status"] == "pass"
            for item in checks
            if item["required"]
        ),
        "checks": checks,
        "environment": {
            "operating_system": platform.platform(),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_available": cudnn_available,
            "cudnn_version": cudnn_version,
            "visible_gpu_count": gpu_count,
            "distributed_available": distributed_available,
            "nccl_available": nccl_available,
            "torch_compile_available": compile_available,
        },
        "gpu": gpu,
        "sdpa": sdpa,
        "model": {
            "parameter_count": parameters.total_parameters,
            "trainable_parameter_count": parameters.trainable_parameters,
            "parameter_memory_bytes": {
                "float32": parameters.memory.float32_bytes,
                "float16": parameters.memory.float16_bytes,
                "bfloat16": parameters.memory.bfloat16_bytes,
            },
            "sequence_length": training_config.sequence_length,
            "selected_precision": precision_name,
        },
        "identities": {
            "model_configuration_sha256": sha256_file(
                training_config.model_config_path
            ),
            "training_configuration_sha256": _configuration_hash(training_config),
            "manifest_sha256": manifest_identity,
            "tokenizer_or_fixture_fingerprint": manifest["tokenizer"]["fingerprint"],
            "qualification_fixture_identity": fixture_identity,
        },
        "storage": {
            "path": str(disk_path),
            "free_bytes": disk.free,
            "total_bytes": disk.total,
        },
    }
    return redact_sensitive_fields(report)


def redact_sensitive_fields(value: Any) -> Any:
    """Recursively redact credential-shaped fields before serialization."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in SENSITIVE_KEYS
                or normalized.endswith(
                    (
                        "_password",
                        "_secret",
                        "_api_key",
                        "_credential",
                        "_access_token",
                    )
                )
            ):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_sensitive_fields(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_fields(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_fields(item) for item in value]
    return value


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    """Atomically write a strict machine-readable report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                redact_sensitive_fields(report),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _configuration_hash(config: PretrainConfig) -> str:
    canonical = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_backend_flag(name: str) -> bool | None:
    function = getattr(torch.backends.cuda, name, None)
    if not callable(function):
        return None
    try:
        return bool(function())
    except RuntimeError:
        return None


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
