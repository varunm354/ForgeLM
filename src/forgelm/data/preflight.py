"""Read-only validation for large pretraining-data preparation."""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from forgelm.data.pretraining_config import (
    UINT16_VOCABULARY_LIMIT,
    PretrainingDataConfig,
)
from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer

IMMUTABLE_REVISION_LENGTH = 40
PREFLIGHT_FORMAT = "forgelm.pretraining_data_preflight"
PREFLIGHT_VERSION = 1
REQUIRED_IMPORTS = ("datasets", "numpy", "tokenizers", "torch", "yaml")


def is_immutable_dataset_revision(revision: str) -> bool:
    """Return whether a revision is a lowercase 40-character Git commit."""
    return (
        isinstance(revision, str)
        and len(revision) == IMMUTABLE_REVISION_LENGTH
        and all(character in "0123456789abcdef" for character in revision)
    )


def run_data_preflight(
    config: PretrainingDataConfig,
    *,
    source_factory: Callable[[], Iterable[str | Mapping[str, Any]]] | None,
    local_source: bool,
    force: bool,
    check_connectivity: bool = True,
    document_limit: int = 2,
    connectivity_probe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Inspect configuration and a bounded source sample without writing files."""
    if document_limit < 1:
        raise ValueError("document_limit must be positive.")
    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool | None,
        detail: str,
        *,
        required: bool = True,
        category: str = "configuration",
    ) -> None:
        checks.append(
            {
                "name": name,
                "required": required,
                "category": category,
                "status": (
                    "pass" if passed is True else "fail" if passed is False else "skipped"
                ),
                "detail": detail,
            }
        )

    add("configuration_parsed", True, "Strict YAML configuration parsed.")
    add(
        "streaming_enabled",
        config.streaming is True,
        "Source processing is explicitly configured as streaming.",
    )
    immutable = is_immutable_dataset_revision(config.dataset_revision)
    add(
        "immutable_dataset_revision",
        immutable or local_source,
        (
            f"Pinned source revision: {config.dataset_revision}."
            if immutable
            else "Local source override does not use a Hugging Face revision."
            if local_source
            else "Remote sources require a lowercase 40-character commit revision."
        ),
    )
    add(
        "uint16_vocabulary",
        config.tokenizer_vocab_size <= UINT16_VOCABULARY_LIMIT,
        f"Vocabulary size {config.tokenizer_vocab_size:,}; uint16 limit "
        f"{UINT16_VOCABULARY_LIMIT:,}.",
    )

    train_shards = math.ceil(config.target_tokens / config.tokens_per_shard)
    validation_shards = math.ceil(
        config.validation_target_tokens / config.tokens_per_shard
    )
    token_bytes = (
        config.target_tokens + config.validation_target_tokens
    ) * 2
    staging_headroom = config.tokens_per_shard * 2
    estimated_minimum_bytes = (
        token_bytes + staging_headroom + config.tokenizer_training_bytes
    )
    add(
        "storage_estimate",
        True,
        f"Expected {train_shards} training and {validation_shards} validation "
        f"shards; raw token bytes approximately {token_bytes:,}.",
    )

    disk_root = _nearest_existing_path(config.output_directory)
    try:
        disk = shutil.disk_usage(disk_root)
    except OSError as error:
        add("free_disk_space", False, f"Could not inspect {disk_root}: {error}")
        free_bytes: int | None = None
    else:
        free_bytes = disk.free
        add(
            "free_disk_space",
            free_bytes >= estimated_minimum_bytes,
            f"{free_bytes:,} bytes free at {disk_root}; conservative preparation "
            f"minimum {estimated_minimum_bytes:,} bytes.",
        )

    paths_accessible = all(
        os.access(_nearest_existing_path(path), os.W_OK)
        for path in (config.output_directory, config.tokenizer_path.parent)
    )
    add(
        "output_paths",
        paths_accessible,
        "Output and tokenizer paths resolve under writable existing ancestors."
        if paths_accessible
        else "An output or tokenizer path has no writable existing ancestor.",
    )

    staging = config.output_directory.with_name(
        f".{config.output_directory.name}.tmp"
    )
    existing = [path for path in (config.output_directory, staging) if path.exists()]
    overwrite_ok = not existing or force
    add(
        "overwrite_protection",
        overwrite_ok,
        (
            "No completed or staging output exists."
            if not existing
            else "Existing output would be replaced only because --force was supplied; "
            "preflight did not modify it."
            if force
            else "Existing output or staging path blocks preparation: "
            + ", ".join(str(path) for path in existing)
        ),
    )

    missing_imports = [
        name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None
    ]
    add(
        "required_python_imports",
        not missing_imports,
        "All required imports are discoverable."
        if not missing_imports
        else f"Missing imports: {', '.join(missing_imports)}.",
    )

    tokenizer_status: dict[str, Any] = {
        "path": str(config.tokenizer_path),
        "exists": config.tokenizer_path.is_file(),
        "fingerprint": None,
        "vocabulary_size": None,
    }
    if config.tokenizer_path.is_file():
        try:
            tokenizer = ScalableByteLevelTokenizer.load(
                config.tokenizer_path,
                end_of_document_token=config.end_of_document_token,
                expected_vocab_size=config.tokenizer_vocab_size,
            )
        except (OSError, TypeError, ValueError) as error:
            add("tokenizer_status", False, f"Tokenizer validation failed: {error}")
        else:
            tokenizer_status["fingerprint"] = tokenizer.fingerprint()
            tokenizer_status["vocabulary_size"] = tokenizer.vocab_size
            add(
                "tokenizer_status",
                True,
                f"Existing tokenizer fingerprint: {tokenizer.fingerprint()}.",
            )
    else:
        add(
            "tokenizer_status",
            None,
            "Production tokenizer does not exist yet; preparation would train it "
            "from production training documents.",
            required=False,
        )

    connectivity: dict[str, Any] = {
        "checked": False,
        "available": None,
        "error": None,
    }
    source_probe: dict[str, Any] = {
        "attempted": False,
        "documents_yielded": 0,
        "error": None,
    }
    if local_source:
        connectivity.update(checked=False, available=None)
        add(
            "huggingface_connectivity",
            None,
            "Skipped for local source override.",
            required=False,
            category="connectivity",
        )
        _probe_source(source_factory, document_limit, source_probe, add)
    elif not check_connectivity:
        add(
            "huggingface_connectivity",
            None,
            "Connectivity check explicitly skipped.",
            required=False,
            category="connectivity",
        )
        add(
            "bounded_source_probe",
            None,
            "Source probe skipped with connectivity check.",
            required=False,
            category="source",
        )
    else:
        connectivity["checked"] = True
        probe = connectivity_probe or _default_connectivity_probe
        try:
            probe()
        except Exception as error:  # noqa: BLE001 - connectivity adapters vary
            connectivity["available"] = False
            connectivity["error"] = f"{type(error).__name__}: {error}"
            add(
                "huggingface_connectivity",
                False,
                connectivity["error"],
                required=False,
                category="connectivity",
            )
            add(
                "bounded_source_probe",
                None,
                "Skipped because Hugging Face connectivity was unavailable.",
                required=False,
                category="source",
            )
        else:
            connectivity["available"] = True
            add(
                "huggingface_connectivity",
                True,
                "Hugging Face endpoint was reachable.",
                required=False,
                category="connectivity",
            )
            _probe_source(source_factory, document_limit, source_probe, add)

    required_passed = all(
        check["status"] == "pass"
        for check in checks
        if check["required"]
    )
    return {
        "format": PREFLIGHT_FORMAT,
        "version": PREFLIGHT_VERSION,
        "mode": "read_only",
        "passed": required_passed,
        "checks": checks,
        "estimates": {
            "training_shards": train_shards,
            "validation_shards": validation_shards,
            "raw_token_storage_bytes": token_bytes,
            "conservative_minimum_free_bytes": estimated_minimum_bytes,
            "free_bytes": free_bytes,
        },
        "tokenizer": tokenizer_status,
        "connectivity": connectivity,
        "source_probe": source_probe,
    }


def _probe_source(
    source_factory: Callable[[], Iterable[str | Mapping[str, Any]]] | None,
    limit: int,
    result: dict[str, Any],
    add: Callable[..., None],
) -> None:
    if source_factory is None:
        result["error"] = "No source factory was supplied."
        add(
            "bounded_source_probe",
            False,
            result["error"],
            required=False,
            category="source",
        )
        return
    result["attempted"] = True
    try:
        iterator = iter(source_factory())
        for _ in range(limit):
            try:
                next(iterator)
            except StopIteration:
                break
            result["documents_yielded"] += 1
    except Exception as error:  # noqa: BLE001 - Datasets raises backend-specific errors
        result["error"] = f"{type(error).__name__}: {error}"
        add(
            "bounded_source_probe",
            False,
            "Exact configured source/revision probe failed without fallback: "
            + result["error"],
            required=False,
            category="source",
        )
    else:
        yielded = result["documents_yielded"]
        add(
            "bounded_source_probe",
            yielded > 0,
            f"Exact configured source/revision yielded {yielded} bounded documents.",
            required=False,
            category="source",
        )


def _default_connectivity_probe() -> None:
    request = urllib.request.Request(
        "https://huggingface.co/",
        method="HEAD",
        headers={"User-Agent": "ForgeLM-data-preflight/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            return
    except urllib.error.HTTPError as error:
        if 200 <= error.code < 500:
            return
        raise


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
