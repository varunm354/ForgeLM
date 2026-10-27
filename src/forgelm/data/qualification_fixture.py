"""Deterministic synthetic token shards for bounded CUDA qualification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from forgelm.data.token_shards import (
    FORMAT_NAME,
    FORMAT_VERSION,
    BinaryTokenShardWriter,
    validate_manifest,
)

FIXTURE_FORMAT = "forgelm.cuda_qualification_fixture"
FIXTURE_VERSION = 1
SYNTHETIC_WARNING = (
    "SYNTHETIC QUALIFICATION DATA: never use for model-quality claims."
)


def fixture_fingerprint(
    *,
    seed: int,
    vocabulary_size: int,
    train_tokens: int,
    validation_tokens: int,
    tokens_per_shard: int,
) -> str:
    """Return the stable identity of fixture-generation inputs."""
    identity = {
        "format": FIXTURE_FORMAT,
        "version": FIXTURE_VERSION,
        "seed": seed,
        "vocabulary_size": vocabulary_size,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "tokens_per_shard": tokens_per_shard,
        "warning": SYNTHETIC_WARNING,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_qualification_fixture(
    output_directory: str | Path,
    *,
    seed: int = 4_004,
    vocabulary_size: int = 16_384,
    train_tokens: int = 65_536,
    validation_tokens: int = 8_192,
    tokens_per_shard: int = 32_768,
    force: bool = False,
) -> dict[str, Any]:
    """Atomically create small manifest-compatible synthetic token shards."""
    for value, name in (
        (seed, "seed"),
        (vocabulary_size, "vocabulary_size"),
        (train_tokens, "train_tokens"),
        (validation_tokens, "validation_tokens"),
        (tokens_per_shard, "tokens_per_shard"),
    ):
        minimum = 0 if name == "seed" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if vocabulary_size > 65_536:
        raise ValueError("vocabulary_size must fit uint16.")

    output = Path(output_directory).expanduser().resolve()
    staging = output.with_name(f".{output.name}.tmp")
    existing = [path for path in (output, staging) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite qualification fixture or staging path: "
            + ", ".join(str(path) for path in existing)
        )
    if force:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    train_writer = BinaryTokenShardWriter(
        staging,
        split="train",
        tokens_per_shard=tokens_per_shard,
        vocabulary_size=vocabulary_size,
    )
    validation_writer = BinaryTokenShardWriter(
        staging,
        split="validation",
        tokens_per_shard=tokens_per_shard,
        vocabulary_size=vocabulary_size,
    )
    fingerprint = fixture_fingerprint(
        seed=seed,
        vocabulary_size=vocabulary_size,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        tokens_per_shard=tokens_per_shard,
    )
    try:
        _write_tokens(
            train_writer,
            count=train_tokens,
            seed=seed,
            vocabulary_size=vocabulary_size,
        )
        _write_tokens(
            validation_writer,
            count=validation_tokens,
            seed=seed + 1,
            vocabulary_size=vocabulary_size,
        )
        train_shards = train_writer.finish()
        validation_shards = validation_writer.finish()
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "source": {
                "type": "synthetic_cuda_qualification",
                "dataset_name": "local/forgelm-cuda-qualification-synthetic",
                "dataset_config": "deterministic-token-pattern",
                "dataset_split": "train-and-validation",
                "dataset_revision": f"fixture-v{FIXTURE_VERSION}",
                "text_field": "not_applicable_tokens_generated_locally",
                "synthetic": True,
                "warning": SYNTHETIC_WARNING,
            },
            "creation_configuration": {
                "fixture_format": FIXTURE_FORMAT,
                "fixture_version": FIXTURE_VERSION,
                "seed": seed,
                "train_tokens": train_tokens,
                "validation_tokens": validation_tokens,
                "tokens_per_shard": tokens_per_shard,
                "output_dtype": "uint16",
                "network_access": False,
                "atomic_publish": True,
                "synthetic": True,
                "warning": SYNTHETIC_WARNING,
            },
            "tokenizer": {
                "path": "not_applicable_synthetic_fixture",
                "fingerprint": fingerprint,
                "vocabulary_size": vocabulary_size,
                "end_of_document_token": "<|endoftext|>",
                "end_of_document_id": vocabulary_size - 1,
                "identity_type": "synthetic_fixture_not_production_tokenizer",
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
                "train_tokens": train_tokens,
                "validation_tokens": validation_tokens,
            },
            "shards": {
                "train": train_shards,
                "validation": validation_shards,
            },
            "qualification_fixture": {
                "identity": fingerprint,
                "synthetic": True,
                "warning": SYNTHETIC_WARNING,
                "quality_claims_permitted": False,
            },
        }
        validate_manifest(manifest)
        _write_json_atomic(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        train_writer.abort(remove_finalized=True)
        validation_writer.abort(remove_finalized=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_tokens(
    writer: BinaryTokenShardWriter,
    *,
    count: int,
    seed: int,
    vocabulary_size: int,
) -> None:
    generator = np.random.default_rng(seed)
    remaining = count
    chunk_size = min(16_384, writer.tokens_per_shard)
    while remaining:
        size = min(chunk_size, remaining)
        values = generator.integers(
            0,
            vocabulary_size - 1,
            size=size,
            dtype=np.uint16,
        )
        writer.append(values.astype(np.int64).tolist())
        remaining -= size


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
