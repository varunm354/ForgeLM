"""Staged, restartable production-data artifacts for Phase 4E-A."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer
from forgelm.data.streaming_corpus import StreamingDocumentPipeline
from forgelm.data.token_shards import (
    FORMAT_NAME,
    FORMAT_VERSION,
    LITTLE_ENDIAN_UINT16,
    BinaryTokenShardWriter,
    load_manifest,
    sha256_file,
    validate_manifest,
)

TOKENIZER_CORPUS_FORMAT = "forgelm.production_tokenizer_corpus"
TOKENIZER_CORPUS_VERSION = 1
TOKENIZER_CORPUS_MAGIC = b"FORGELM_TOKENIZER_CORPUS_V1\n"
TOKENIZER_BUNDLE_FORMAT = "forgelm.production_tokenizer"
TOKENIZER_BUNDLE_VERSION = 1
TOKENIZER_APPROVAL_FORMAT = "forgelm.production_tokenizer_approval"
TOKENIZER_APPROVAL_VERSION = 1
SHARD_PROGRESS_FORMAT = "forgelm.production_shard_progress"
SHARD_PROGRESS_VERSION = 1
PRODUCTION_LOCK_FORMAT = "forgelm.production_data_lock"
PRODUCTION_LOCK_VERSION = 1
VERIFICATION_FORMAT = "forgelm.production_data_verification"
VERIFICATION_VERSION = 1
PARTITION_ALGORITHM = "sha256(seed_nul_document_sha256)_first64_threshold_v1"
DEDUPLICATION_ALGORITHM = "exact_sha256_after_filtering_v1"

Record = str | Mapping[str, Any]
SourceFactory = Callable[[], Iterable[Record]]


class BuildInterrupted(RuntimeError):
    """Raised by bounded tests after safely persisting resumable progress."""


@dataclass(frozen=True)
class ProductionPaths:
    """Persistent artifact layout rooted outside the repository."""

    root: Path
    source_state: Path
    tokenizer_corpus: Path
    tokenizer: Path
    shards: Path
    manifests: Path
    progress: Path
    logs: Path
    verification: Path

    @classmethod
    def from_root(cls, root: str | Path) -> ProductionPaths:
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            source_state=resolved / "source_state",
            tokenizer_corpus=resolved / "tokenizer_corpus",
            tokenizer=resolved / "tokenizer",
            shards=resolved / "shards",
            manifests=resolved / "manifests",
            progress=resolved / "progress",
            logs=resolved / "logs",
            verification=resolved / "verification",
        )

    @property
    def corpus_path(self) -> Path:
        return self.tokenizer_corpus / "train-only-corpus.bin"

    @property
    def corpus_manifest_path(self) -> Path:
        return self.tokenizer_corpus / "manifest.json"

    @property
    def tokenizer_path(self) -> Path:
        return self.tokenizer / "tokenizer.json"

    @property
    def tokenizer_manifest_path(self) -> Path:
        return self.tokenizer / "manifest.json"

    @property
    def tokenizer_approval_path(self) -> Path:
        return self.tokenizer / "approval.json"

    @property
    def shard_manifest_path(self) -> Path:
        return self.shards / "manifest.json"

    @property
    def shard_progress_path(self) -> Path:
        return self.progress / "shard-build.json"

    @property
    def production_lock_path(self) -> Path:
        return self.manifests / "production-data-lock.json"

    @property
    def verification_report_path(self) -> Path:
        return self.verification / "production-data-verification.json"


def production_configuration(config: PretrainingDataConfig) -> dict[str, Any]:
    """Return the path-independent configuration contract."""
    return {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "dataset_revision": config.dataset_revision,
        "streaming": config.streaming,
        "text_field": config.text_field,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "minimum_characters": config.minimum_characters,
        "maximum_characters": config.maximum_characters,
        "normalize_line_endings": config.normalize_line_endings,
        "tokenizer_vocab_size": config.tokenizer_vocab_size,
        "tokenizer_training_bytes": config.tokenizer_training_bytes,
        "target_tokens": config.target_tokens,
        "validation_target_tokens": config.validation_target_tokens,
        "tokens_per_shard": config.tokens_per_shard,
        "output_dtype": config.output_dtype,
        "end_of_document_token": config.end_of_document_token,
    }


def production_configuration_identity(config: PretrainingDataConfig) -> str:
    """Hash every source, partition, filtering, tokenizer, and shard setting."""
    return _canonical_identity(production_configuration(config))


def default_source_metadata(config: PretrainingDataConfig) -> dict[str, Any]:
    """Return the configured immutable source identity."""
    return {
        "type": "huggingface_streaming",
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "dataset_revision": config.dataset_revision,
        "streaming": config.streaming,
        "text_field": config.text_field,
    }


def partition_contract(config: PretrainingDataConfig) -> dict[str, Any]:
    """Describe the split and exact-deduplication algorithms."""
    return {
        "algorithm": PARTITION_ALGORITHM,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "applied_before_tokenizer_corpus_collection": True,
        "deduplication": DEDUPLICATION_ALGORITHM,
    }


def filtering_contract(config: PretrainingDataConfig) -> dict[str, Any]:
    """Describe filtering before document identity and partitioning."""
    return {
        "text_field": config.text_field,
        "minimum_characters": config.minimum_characters,
        "maximum_characters": config.maximum_characters,
        "normalize_line_endings": config.normalize_line_endings,
        "reject_blank": True,
    }


def _shard_creation_configuration(
    config: PretrainingDataConfig,
) -> dict[str, Any]:
    return {
        **production_configuration(config),
        "configuration_identity": production_configuration_identity(config),
        "partition_algorithm": PARTITION_ALGORITHM,
        "deduplication_algorithm": DEDUPLICATION_ALGORITHM,
        "restart_policy": "replay_from_start_and_verify_committed_prefix_v1",
    }


def build_tokenizer_stage(
    config: PretrainingDataConfig,
    source_factory: SourceFactory,
    production_root: str | Path,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create the train-only corpus and candidate tokenizer, one atomic artifact at a time."""
    paths = ProductionPaths.from_root(production_root)
    source = dict(source_metadata or default_source_metadata(config))
    corpus_manifest = _build_or_load_tokenizer_corpus(
        config,
        source_factory,
        paths,
        source,
        force=force,
    )
    if paths.tokenizer.is_dir():
        return validate_tokenizer_bundle(config, paths.root)

    staging = paths.root / ".tokenizer.tmp"
    _prepare_staging_directory(staging, force=force)
    try:
        tokenizer = ScalableByteLevelTokenizer.train_from_iterator(
            iter_tokenizer_corpus(paths.corpus_path),
            vocab_size=config.tokenizer_vocab_size,
            end_of_document_token=config.end_of_document_token,
            length=corpus_manifest["corpus"]["documents"],
        )
        tokenizer_path = staging / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        artifact_fingerprint = tokenizer.fingerprint()
        manifest: dict[str, Any] = {
            "format": TOKENIZER_BUNDLE_FORMAT,
            "version": TOKENIZER_BUNDLE_VERSION,
            "status": "candidate_not_production_pinned",
            "configuration_identity": production_configuration_identity(config),
            "source": source,
            "partition": partition_contract(config),
            "filtering": filtering_contract(config),
            "corpus": {
                "manifest_identity": corpus_manifest["identity"],
                "sha256": corpus_manifest["corpus"]["sha256"],
                "utf8_bytes": corpus_manifest["corpus"]["utf8_bytes"],
                "documents": corpus_manifest["corpus"]["documents"],
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "file_sha256": sha256_file(tokenizer_path),
                "fingerprint": artifact_fingerprint,
                "vocabulary_size": tokenizer.vocab_size,
                "configured_vocabulary_size": config.tokenizer_vocab_size,
                "required_special_tokens": [config.end_of_document_token],
                "special_token_ids": {
                    config.end_of_document_token: tokenizer.end_of_document_id
                },
            },
        }
        manifest["identity"] = _canonical_identity(manifest)
        _write_json_exclusive(staging / "manifest.json", manifest)
        os.replace(staging, paths.tokenizer)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_tokenizer_bundle(config, paths.root)


def iter_tokenizer_corpus(path: str | Path) -> Iterator[str]:
    """Read the deterministic length-prefixed train-only tokenizer corpus."""
    source = Path(path)
    try:
        handle = source.open("rb")
    except OSError as error:
        raise ValueError(f"Could not open tokenizer corpus: {source}") from error
    with handle:
        if handle.read(len(TOKENIZER_CORPUS_MAGIC)) != TOKENIZER_CORPUS_MAGIC:
            raise ValueError("Unsupported or corrupted tokenizer corpus header.")
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError("Tokenizer corpus has a partial record length.")
            length = struct.unpack(">Q", length_bytes)[0]
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError("Tokenizer corpus has a partial text record.")
            try:
                yield payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("Tokenizer corpus contains invalid UTF-8.") from error


def validate_tokenizer_bundle(
    config: PretrainingDataConfig,
    production_root: str | Path,
) -> dict[str, Any]:
    """Validate corpus identity, tokenizer structure, file hash, and fingerprint."""
    paths = ProductionPaths.from_root(production_root)
    corpus_manifest = _load_json(paths.corpus_manifest_path, "tokenizer corpus manifest")
    _validate_corpus_manifest(config, paths, corpus_manifest)
    manifest = _load_json(paths.tokenizer_manifest_path, "tokenizer manifest")
    if (
        manifest.get("format") != TOKENIZER_BUNDLE_FORMAT
        or manifest.get("version") != TOKENIZER_BUNDLE_VERSION
    ):
        raise ValueError("Unsupported production tokenizer manifest.")
    identity = manifest.get("identity")
    if identity != _identity_without_field(manifest, "identity"):
        raise ValueError("Production tokenizer manifest identity mismatch.")
    if manifest.get("configuration_identity") != production_configuration_identity(
        config
    ):
        raise ValueError("Production tokenizer configuration identity mismatch.")
    if manifest.get("corpus", {}).get("manifest_identity") != corpus_manifest.get(
        "identity"
    ):
        raise ValueError("Production tokenizer corpus identity mismatch.")
    tokenizer_metadata = manifest.get("tokenizer")
    if not isinstance(tokenizer_metadata, dict):
        raise TypeError("Production tokenizer metadata must be an object.")
    if sha256_file(paths.tokenizer_path) != tokenizer_metadata.get("file_sha256"):
        raise ValueError("Production tokenizer file checksum mismatch.")
    tokenizer = ScalableByteLevelTokenizer.load(
        paths.tokenizer_path,
        end_of_document_token=config.end_of_document_token,
        expected_vocab_size=config.tokenizer_vocab_size,
    )
    if tokenizer.fingerprint() != tokenizer_metadata.get("fingerprint"):
        raise ValueError("Production tokenizer fingerprint mismatch.")
    expected_special_tokens = [config.end_of_document_token]
    if tokenizer_metadata.get("required_special_tokens") != expected_special_tokens:
        raise ValueError("Production tokenizer required special tokens mismatch.")
    if tokenizer_metadata.get("special_token_ids") != {
        config.end_of_document_token: tokenizer.end_of_document_id
    }:
        raise ValueError("Production tokenizer special-token IDs mismatch.")
    return manifest


def approve_tokenizer(
    config: PretrainingDataConfig,
    production_root: str | Path,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Create the explicit human-review boundary required before sharding."""
    paths = ProductionPaths.from_root(production_root)
    manifest = validate_tokenizer_bundle(config, paths.root)
    actual = manifest["tokenizer"]["fingerprint"]
    if expected_fingerprint != actual:
        raise ValueError(
            "Reviewed tokenizer fingerprint does not match the candidate artifact."
        )
    approval = {
        "format": TOKENIZER_APPROVAL_FORMAT,
        "version": TOKENIZER_APPROVAL_VERSION,
        "status": "approved_for_shard_build_not_model_training",
        "configuration_identity": production_configuration_identity(config),
        "tokenizer_manifest_identity": manifest["identity"],
        "tokenizer_fingerprint": actual,
        "special_token_ids": manifest["tokenizer"]["special_token_ids"],
    }
    approval["identity"] = _canonical_identity(approval)
    if paths.tokenizer_approval_path.exists():
        existing = _load_json(paths.tokenizer_approval_path, "tokenizer approval")
        if existing != approval:
            raise FileExistsError("A different tokenizer approval already exists.")
        return existing
    _write_json_exclusive(paths.tokenizer_approval_path, approval)
    return approval


def build_shard_stage(
    config: PretrainingDataConfig,
    source_factory: SourceFactory,
    production_root: str | Path,
    *,
    source_metadata: Mapping[str, Any] | None = None,
    resume: bool = False,
    force: bool = False,
    interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Build exact token targets with replay-verified, shard-boundary resumption."""
    if resume and force:
        raise ValueError("--resume and --force are mutually exclusive.")
    if interrupt_after_documents is not None and interrupt_after_documents < 1:
        raise ValueError("interrupt_after_documents must be positive.")
    paths = ProductionPaths.from_root(production_root)
    source = dict(source_metadata or default_source_metadata(config))
    tokenizer_manifest = validate_tokenizer_bundle(config, paths.root)
    if source != tokenizer_manifest.get("source"):
        raise ValueError(
            "Shard source identity does not match the tokenizer corpus source."
        )
    approval = _load_and_validate_approval(config, paths, tokenizer_manifest)
    if paths.production_lock_path.exists() or paths.shard_manifest_path.exists():
        raise FileExistsError(
            "Completed production data is immutable; use a new production root."
        )

    if force:
        shutil.rmtree(paths.shards, ignore_errors=True)
        paths.shard_progress_path.unlink(missing_ok=True)
    paths.shards.mkdir(parents=True, exist_ok=True)
    paths.progress.mkdir(parents=True, exist_ok=True)

    progress_exists = paths.shard_progress_path.is_file()
    existing_files = list(paths.shards.glob("*.bin"))
    if (progress_exists or existing_files) and not resume:
        raise FileExistsError(
            "A partial shard build exists; use --resume or --force explicitly."
        )

    build_identity = _canonical_identity(
        {
            "configuration_identity": production_configuration_identity(config),
            "source": source,
            "tokenizer_manifest_identity": tokenizer_manifest["identity"],
            "tokenizer_approval_identity": approval["identity"],
        }
    )
    existing = {"train": [], "validation": []}
    if progress_exists:
        progress = _load_json(paths.shard_progress_path, "shard progress")
        _validate_progress(progress, build_identity)
        existing = progress["shards"]
        _remove_partial_shards(paths.shards)
    elif existing_files:
        raise ValueError("Shard files exist without versioned progress metadata.")
    for split in ("train", "validation"):
        existing[split] = _recover_untracked_shards(
            paths.shards,
            split=split,
            tracked=existing[split],
            tokens_per_shard=config.tokens_per_shard,
            vocabulary_size=tokenizer_manifest["tokenizer"]["vocabulary_size"],
        )

    tokenizer = ScalableByteLevelTokenizer.load(
        paths.tokenizer_path,
        end_of_document_token=config.end_of_document_token,
        expected_vocab_size=config.tokenizer_vocab_size,
    )
    if existing["train"] or existing["validation"]:
        _verify_replayed_prefix(
            config,
            source_factory,
            tokenizer,
            paths.shards,
            existing,
        )

    train_writer: BinaryTokenShardWriter
    validation_writer: BinaryTokenShardWriter

    def persist_progress(_entries: list[dict[str, Any]]) -> None:
        _write_progress(
            paths.shard_progress_path,
            build_identity=build_identity,
            status="building",
            train_shards=train_writer.shards,
            validation_shards=validation_writer.shards,
        )

    train_writer = BinaryTokenShardWriter(
        paths.shards,
        split="train",
        tokens_per_shard=config.tokens_per_shard,
        vocabulary_size=tokenizer.vocab_size,
        existing_shards=existing["train"],
        on_shard_finalized=persist_progress,
    )
    validation_writer = BinaryTokenShardWriter(
        paths.shards,
        split="validation",
        tokens_per_shard=config.tokens_per_shard,
        vocabulary_size=tokenizer.vocab_size,
        existing_shards=existing["validation"],
        on_shard_finalized=persist_progress,
    )
    _write_progress(
        paths.shard_progress_path,
        build_identity=build_identity,
        status="building",
        train_shards=train_writer.shards,
        validation_shards=validation_writer.shards,
    )
    targets = {
        "train": config.target_tokens,
        "validation": config.validation_target_tokens,
    }
    writers = {"train": train_writer, "validation": validation_writer}
    committed = {
        "train": _PrefixVerifier(paths.shards, existing["train"]),
        "validation": _PrefixVerifier(paths.shards, existing["validation"]),
    }
    logical_tokens = {"train": 0, "validation": 0}
    written_documents = {"train": 0, "validation": 0}
    pipeline = _make_pipeline(config)
    processed_documents = 0
    try:
        for document in pipeline.process(source_factory()):
            encoded = tokenizer.encode(document.text)
            encoded.append(tokenizer.end_of_document_id)
            split = document.split
            take = min(
                len(encoded),
                targets[split] - logical_tokens[split],
            )
            if take > 0:
                output_tokens = encoded[:take]
                written_documents[split] += 1
                logical_tokens[split] += take
                unwritten = committed[split].consume(output_tokens)
                writers[split].append(unwritten)
            processed_documents += 1
            if (
                interrupt_after_documents is not None
                and processed_documents >= interrupt_after_documents
            ):
                raise BuildInterrupted(
                    "Simulated interruption after a safe document boundary."
                )
            if all(logical_tokens[name] == targets[name] for name in targets):
                break

        for split, target in targets.items():
            if logical_tokens[split] != target:
                raise ValueError(
                    f"Source stream ended after {logical_tokens[split]:,} {split} "
                    f"tokens; {target:,} were requested."
                )
            if not committed[split].complete:
                raise ValueError(
                    f"Source replay ended before committed {split} tokens were verified."
                )

        train_shards = train_writer.finish()
        validation_shards = validation_writer.finish()
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "source": source,
            "creation_configuration": _shard_creation_configuration(config),
            "tokenizer": {
                "path": "../tokenizer/tokenizer.json",
                "fingerprint": tokenizer.fingerprint(),
                "manifest_identity": tokenizer_manifest["identity"],
                "approval_identity": approval["identity"],
                "vocabulary_size": tokenizer.vocab_size,
                "end_of_document_token": tokenizer.end_of_document_token,
                "end_of_document_id": tokenizer.end_of_document_id,
            },
            "storage": {"dtype": "uint16", "byte_order": "little"},
            "counts": {
                **pipeline.statistics.to_dict(),
                "train_documents_written": written_documents["train"],
                "validation_documents_written": written_documents["validation"],
                "train_tokens": train_writer.total_tokens,
                "validation_tokens": validation_writer.total_tokens,
                "tokenized_storage_bytes": (
                    train_writer.total_tokens + validation_writer.total_tokens
                )
                * LITTLE_ENDIAN_UINT16.itemsize,
            },
            "shards": {
                "train": train_shards,
                "validation": validation_shards,
            },
        }
        validate_manifest(manifest)
        _write_json_exclusive(paths.shard_manifest_path, manifest)
        _write_progress(
            paths.shard_progress_path,
            build_identity=build_identity,
            status="completed_pending_verification",
            train_shards=train_shards,
            validation_shards=validation_shards,
            manifest_sha256=sha256_file(paths.shard_manifest_path),
        )
        return manifest
    except Exception as error:
        train_writer.abort(remove_finalized=False)
        validation_writer.abort(remove_finalized=False)
        _write_progress(
            paths.shard_progress_path,
            build_identity=build_identity,
            status="interrupted",
            train_shards=train_writer.shards,
            validation_shards=validation_writer.shards,
            error_type=type(error).__name__,
        )
        raise


def verify_production_stage(
    config: PretrainingDataConfig,
    production_root: str | Path,
    *,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all artifacts and atomically publish the final production-data lock."""
    paths = ProductionPaths.from_root(production_root)
    tokenizer_manifest = validate_tokenizer_bundle(config, paths.root)
    approval = _load_and_validate_approval(config, paths, tokenizer_manifest)
    manifest = load_manifest(paths.shard_manifest_path)
    corpus_manifest = _load_json(paths.corpus_manifest_path, "corpus manifest")
    _validate_corpus_manifest(config, paths, corpus_manifest)
    if (
        manifest["source"] != tokenizer_manifest.get("source")
        or manifest["source"] != corpus_manifest.get("source")
    ):
        raise ValueError(
            "Corpus, tokenizer, and shard source identities do not match."
        )
    if source_metadata is not None:
        if manifest["source"] != dict(source_metadata):
            raise ValueError(
                "Final shard source identity does not match configuration."
            )
    else:
        expected_source = default_source_metadata(config)
        for name, expected in expected_source.items():
            if (
                name == "type"
                and config.dataset_name != "HuggingFaceFW/fineweb-edu"
            ):
                continue
            if manifest["source"].get(name) != expected:
                raise ValueError(
                    f"Final shard source field {name!r} does not match configuration."
                )
    if manifest["creation_configuration"] != _shard_creation_configuration(
        config
    ):
        raise ValueError("Final shard creation configuration is incompatible.")
    if manifest["counts"]["train_tokens"] != config.target_tokens:
        raise ValueError("Final training token count does not match configuration.")
    if (
        manifest["counts"]["validation_tokens"]
        != config.validation_target_tokens
    ):
        raise ValueError("Final validation token count does not match configuration.")
    if manifest["tokenizer"]["fingerprint"] != tokenizer_manifest["tokenizer"][
        "fingerprint"
    ]:
        raise ValueError("Shard tokenizer fingerprint does not match approved tokenizer.")
    if manifest["tokenizer"].get("manifest_identity") != tokenizer_manifest.get(
        "identity"
    ):
        raise ValueError("Shard tokenizer manifest identity mismatch.")
    if manifest["tokenizer"].get("approval_identity") != approval.get("identity"):
        raise ValueError("Shard tokenizer approval identity mismatch.")
    _verify_shard_files(paths.shard_manifest_path, manifest)
    manifest_sha256 = sha256_file(paths.shard_manifest_path)

    lock: dict[str, Any] = {
        "format": PRODUCTION_LOCK_FORMAT,
        "version": PRODUCTION_LOCK_VERSION,
        "status": "complete_verified_not_training_approval",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source": manifest["source"],
        "partition": partition_contract(config),
        "filtering": filtering_contract(config),
        "configuration": production_configuration(config),
        "configuration_identity": production_configuration_identity(config),
        "tokenizer": {
            "path": str(paths.tokenizer_path),
            "fingerprint": tokenizer_manifest["tokenizer"]["fingerprint"],
            "manifest_identity": tokenizer_manifest["identity"],
            "approval_identity": approval["identity"],
            "vocabulary_size": tokenizer_manifest["tokenizer"]["vocabulary_size"],
            "special_token_ids": tokenizer_manifest["tokenizer"][
                "special_token_ids"
            ],
        },
        "tokenizer_corpus": {
            "path": str(paths.corpus_path),
            "manifest_identity": corpus_manifest["identity"],
            "sha256": corpus_manifest["corpus"]["sha256"],
            "utf8_bytes": corpus_manifest["corpus"]["utf8_bytes"],
            "documents": corpus_manifest["corpus"]["documents"],
        },
        "shard_manifest": {
            "path": str(paths.shard_manifest_path),
            "sha256": manifest_sha256,
        },
        "counts": {
            "train_tokens": manifest["counts"]["train_tokens"],
            "validation_tokens": manifest["counts"]["validation_tokens"],
            "train_shards": len(manifest["shards"]["train"]),
            "validation_shards": len(manifest["shards"]["validation"]),
        },
        "shards": {
            split: [
                {
                    **entry,
                    "path": str(paths.shards / entry["name"]),
                }
                for entry in manifest["shards"][split]
            ]
            for split in ("train", "validation")
        },
        "code": {
            "git_commit": _git_commit(),
            "configuration_identity": production_configuration_identity(config),
        },
    }
    lock["identity"] = _canonical_identity(lock)
    if paths.production_lock_path.exists():
        existing = _load_json(paths.production_lock_path, "production data lock")
        validate_production_data_lock(
            paths.production_lock_path,
            manifest_path=paths.shard_manifest_path,
            expected_tokenizer_fingerprint=tokenizer_manifest["tokenizer"][
                "fingerprint"
            ],
        )
        if not paths.verification_report_path.is_file():
            raise ValueError(
                "Production data lock exists without its verification report."
            )
        report = _load_json(
            paths.verification_report_path,
            "production data verification report",
        )
        _validate_verification_report(report)
        if report.get("production_lock") != existing:
            raise ValueError(
                "Production data verification report and lock do not match."
            )
        return existing

    paths.manifests.mkdir(parents=True, exist_ok=True)
    paths.verification.mkdir(parents=True, exist_ok=True)
    if paths.verification_report_path.exists():
        report = _load_json(
            paths.verification_report_path,
            "production data verification report",
        )
        _validate_verification_report(report)
        reported_lock = report.get("production_lock")
        if not isinstance(reported_lock, dict) or _lock_contract(
            reported_lock
        ) != _lock_contract(lock):
            raise ValueError(
                "Existing verification report is incompatible with current artifacts."
            )
        if reported_lock.get("identity") != _identity_without_field(
            reported_lock,
            "identity",
        ):
            raise ValueError("Reported production data lock identity mismatch.")
        lock = reported_lock
    else:
        report = {
            "format": VERIFICATION_FORMAT,
            "version": VERIFICATION_VERSION,
            "passed": True,
            "production_lock_identity": lock["identity"],
            "manifest_sha256": manifest_sha256,
            "verified_shards": sum(
                len(manifest["shards"][split])
                for split in ("train", "validation")
            ),
            "verified_bytes": sum(
                entry["byte_size"]
                for split in ("train", "validation")
                for entry in manifest["shards"][split]
            ),
            "training_started": False,
            "production_lock": lock,
        }
        _write_json_exclusive(paths.verification_report_path, report)
    _write_json_exclusive(paths.production_lock_path, lock)
    return lock


def validate_production_data_lock(
    lock_path: str | Path,
    *,
    manifest_path: str | Path,
    expected_tokenizer_fingerprint: str,
) -> dict[str, Any]:
    """Validate the immutable lock before any real-data training engine starts."""
    lock = _load_json(Path(lock_path), "production data lock")
    if (
        lock.get("format") != PRODUCTION_LOCK_FORMAT
        or lock.get("version") != PRODUCTION_LOCK_VERSION
        or lock.get("status") != "complete_verified_not_training_approval"
    ):
        raise ValueError("Unsupported or incomplete production data lock.")
    if lock.get("identity") != _identity_without_field(lock, "identity"):
        raise ValueError("Production data lock identity mismatch.")
    manifest_source = Path(manifest_path)
    if sha256_file(manifest_source) != lock.get("shard_manifest", {}).get("sha256"):
        raise ValueError("Production data lock manifest checksum mismatch.")
    manifest = load_manifest(manifest_source)
    if (
        lock.get("tokenizer", {}).get("fingerprint")
        != expected_tokenizer_fingerprint
        or manifest["tokenizer"]["fingerprint"] != expected_tokenizer_fingerprint
    ):
        raise ValueError("Production data lock tokenizer fingerprint mismatch.")
    if lock.get("source") != manifest["source"]:
        raise ValueError("Production data lock source identity mismatch.")
    if lock.get("counts", {}).get("train_tokens") != manifest["counts"]["train_tokens"]:
        raise ValueError("Production data lock training token count mismatch.")
    if lock.get("counts", {}).get("validation_tokens") != manifest["counts"][
        "validation_tokens"
    ]:
        raise ValueError("Production data lock validation token count mismatch.")
    locked_shards = lock.get("shards")
    if not isinstance(locked_shards, dict):
        raise TypeError("Production data lock shard metadata must be an object.")
    for split in ("train", "validation"):
        entries = locked_shards.get(split)
        if not isinstance(entries, list) or len(entries) != len(
            manifest["shards"][split]
        ):
            raise ValueError("Production data lock shard count mismatch.")
        for locked_entry, manifest_entry in zip(
            entries,
            manifest["shards"][split],
            strict=True,
        ):
            if not isinstance(locked_entry, dict):
                raise TypeError("Production data lock shard entry must be an object.")
            without_path = {
                name: value
                for name, value in locked_entry.items()
                if name != "path"
            }
            if without_path != manifest_entry:
                raise ValueError("Production data lock shard metadata mismatch.")
            expected_path = manifest_source.parent / manifest_entry["name"]
            if Path(locked_entry.get("path", "")).resolve() != expected_path.resolve():
                raise ValueError("Production data lock shard path mismatch.")
    _verify_shard_files(manifest_source, manifest)
    return lock


def _build_or_load_tokenizer_corpus(
    config: PretrainingDataConfig,
    source_factory: SourceFactory,
    paths: ProductionPaths,
    source: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    if paths.tokenizer_corpus.is_dir():
        manifest = _load_json(paths.corpus_manifest_path, "tokenizer corpus manifest")
        _validate_corpus_manifest(config, paths, manifest)
        return manifest
    staging = paths.root / ".tokenizer_corpus.tmp"
    _prepare_staging_directory(staging, force=force)
    corpus_path = staging / "train-only-corpus.bin"
    pipeline = _make_pipeline(config)
    corpus_bytes = 0
    documents_written = 0
    deterministically_capped = False
    try:
        with corpus_path.open("xb") as handle:
            handle.write(TOKENIZER_CORPUS_MAGIC)
            for document in pipeline.process(source_factory()):
                if document.split != "train":
                    continue
                remaining = config.tokenizer_training_bytes - corpus_bytes
                if remaining <= 0:
                    break
                payload = document.text.encode("utf-8")
                if len(payload) > remaining:
                    text = payload[:remaining].decode("utf-8", errors="ignore")
                    payload = text.encode("utf-8")
                    deterministically_capped = True
                if payload:
                    handle.write(struct.pack(">Q", len(payload)))
                    handle.write(payload)
                    corpus_bytes += len(payload)
                    documents_written += 1
                if corpus_bytes >= config.tokenizer_training_bytes or (
                    deterministically_capped
                ):
                    break
            handle.flush()
            os.fsync(handle.fileno())
        if corpus_bytes == 0:
            raise ValueError("No train-partition text was available for tokenization.")
        if (
            corpus_bytes < config.tokenizer_training_bytes
            and not deterministically_capped
        ):
            raise ValueError(
                "Source ended before the configured tokenizer byte target: "
                f"{corpus_bytes:,} of {config.tokenizer_training_bytes:,} bytes."
            )
        manifest: dict[str, Any] = {
            "format": TOKENIZER_CORPUS_FORMAT,
            "version": TOKENIZER_CORPUS_VERSION,
            "configuration_identity": production_configuration_identity(config),
            "source": source,
            "partition": partition_contract(config),
            "filtering": filtering_contract(config),
            "statistics": pipeline.statistics.to_dict(),
            "corpus": {
                "path": "train-only-corpus.bin",
                "format": "uint64_be_length_prefixed_utf8_v1",
                "sha256": sha256_file(corpus_path),
                "file_bytes": corpus_path.stat().st_size,
                "utf8_bytes": corpus_bytes,
                "target_utf8_bytes": config.tokenizer_training_bytes,
                "deterministically_capped": deterministically_capped,
                "cap_policy": "truncate_final_train_document_at_utf8_boundary_v1",
                "documents": documents_written,
                "tokenizer_input_utf8_bytes": corpus_bytes,
            },
        }
        manifest["identity"] = _canonical_identity(manifest)
        _write_json_exclusive(staging / "manifest.json", manifest)
        os.replace(staging, paths.tokenizer_corpus)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_corpus_manifest(
    config: PretrainingDataConfig,
    paths: ProductionPaths,
    manifest: Mapping[str, Any],
) -> None:
    if (
        manifest.get("format") != TOKENIZER_CORPUS_FORMAT
        or manifest.get("version") != TOKENIZER_CORPUS_VERSION
    ):
        raise ValueError("Unsupported tokenizer corpus manifest.")
    if manifest.get("identity") != _identity_without_field(manifest, "identity"):
        raise ValueError("Tokenizer corpus manifest identity mismatch.")
    if manifest.get("configuration_identity") != production_configuration_identity(
        config
    ):
        raise ValueError("Tokenizer corpus configuration identity mismatch.")
    corpus = manifest.get("corpus")
    if not isinstance(corpus, Mapping):
        raise TypeError("Tokenizer corpus metadata must be an object.")
    corpus_bytes = corpus.get("utf8_bytes")
    if (
        not isinstance(corpus_bytes, int)
        or isinstance(corpus_bytes, bool)
        or not 0 < corpus_bytes <= config.tokenizer_training_bytes
        or (
            corpus_bytes < config.tokenizer_training_bytes
            and corpus.get("deterministically_capped") is not True
        )
    ):
        raise ValueError("Tokenizer corpus byte target or cap policy mismatch.")
    if sha256_file(paths.corpus_path) != corpus.get("sha256"):
        raise ValueError("Tokenizer corpus checksum mismatch.")
    observed_documents = 0
    observed_bytes = 0
    for text in iter_tokenizer_corpus(paths.corpus_path):
        observed_documents += 1
        observed_bytes += len(text.encode("utf-8"))
    if observed_documents != corpus.get("documents"):
        raise ValueError("Tokenizer corpus document count mismatch.")
    if observed_bytes != corpus.get("utf8_bytes"):
        raise ValueError("Tokenizer corpus UTF-8 byte count mismatch.")


def _load_and_validate_approval(
    config: PretrainingDataConfig,
    paths: ProductionPaths,
    tokenizer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    approval = _load_json(paths.tokenizer_approval_path, "tokenizer approval")
    if (
        approval.get("format") != TOKENIZER_APPROVAL_FORMAT
        or approval.get("version") != TOKENIZER_APPROVAL_VERSION
    ):
        raise ValueError("Unsupported tokenizer approval.")
    if approval.get("identity") != _identity_without_field(approval, "identity"):
        raise ValueError("Tokenizer approval identity mismatch.")
    if approval.get("configuration_identity") != production_configuration_identity(
        config
    ):
        raise ValueError("Tokenizer approval configuration identity mismatch.")
    if approval.get("tokenizer_manifest_identity") != tokenizer_manifest.get(
        "identity"
    ):
        raise ValueError("Tokenizer approval manifest identity mismatch.")
    if approval.get("tokenizer_fingerprint") != tokenizer_manifest.get(
        "tokenizer", {}
    ).get("fingerprint"):
        raise ValueError("Tokenizer approval fingerprint mismatch.")
    return approval


def _verify_replayed_prefix(
    config: PretrainingDataConfig,
    source_factory: SourceFactory,
    tokenizer: ScalableByteLevelTokenizer,
    shard_directory: Path,
    existing: Mapping[str, list[dict[str, Any]]],
) -> None:
    verifiers = {
        split: _PrefixVerifier(shard_directory, existing[split])
        for split in ("train", "validation")
    }
    pipeline = _make_pipeline(config)
    for document in pipeline.process(source_factory()):
        verifier = verifiers[document.split]
        if verifier.complete:
            continue
        encoded = tokenizer.encode(document.text)
        encoded.append(tokenizer.end_of_document_id)
        verifier.consume(encoded)
        if all(item.complete for item in verifiers.values()):
            return
    incomplete = [split for split, verifier in verifiers.items() if not verifier.complete]
    raise ValueError(
        "Pinned source replay did not reproduce committed shard prefixes: "
        + ", ".join(incomplete)
    )


class _PrefixVerifier:
    def __init__(
        self,
        shard_directory: Path,
        entries: list[dict[str, Any]],
    ) -> None:
        self._arrays = [
            np.memmap(
                shard_directory / entry["name"],
                dtype=LITTLE_ENDIAN_UINT16,
                mode="r",
            )
            for entry in entries
        ]
        self._shard_index = 0
        self._offset = 0
        self._remaining = sum(entry["token_count"] for entry in entries)

    @property
    def complete(self) -> bool:
        return self._remaining == 0

    def consume(self, token_ids: list[int]) -> list[int]:
        consumed = 0
        while consumed < len(token_ids) and self._remaining:
            array = self._arrays[self._shard_index]
            count = min(len(token_ids) - consumed, len(array) - self._offset)
            expected = np.asarray(
                array[self._offset : self._offset + count],
                dtype=np.int64,
            )
            actual = np.asarray(token_ids[consumed : consumed + count], dtype=np.int64)
            if not np.array_equal(expected, actual):
                raise ValueError(
                    "Pinned source replay differs from committed shard contents."
                )
            consumed += count
            self._offset += count
            self._remaining -= count
            if self._offset == len(array):
                self._shard_index += 1
                self._offset = 0
        return token_ids[consumed:]


def _write_progress(
    path: Path,
    *,
    build_identity: str,
    status: str,
    train_shards: list[dict[str, Any]],
    validation_shards: list[dict[str, Any]],
    manifest_sha256: str | None = None,
    error_type: str | None = None,
) -> None:
    progress = {
        "format": SHARD_PROGRESS_FORMAT,
        "version": SHARD_PROGRESS_VERSION,
        "build_identity": build_identity,
        "status": status,
        "restart_policy": "replay_from_start_and_verify_committed_prefix_v1",
        "shards": {
            "train": list(train_shards),
            "validation": list(validation_shards),
        },
        "tokens": {
            "train": sum(entry["token_count"] for entry in train_shards),
            "validation": sum(
                entry["token_count"] for entry in validation_shards
            ),
        },
        "manifest_sha256": manifest_sha256,
        "error_type": error_type,
    }
    _write_json_replace(path, progress)


def _validate_progress(progress: Mapping[str, Any], build_identity: str) -> None:
    if (
        progress.get("format") != SHARD_PROGRESS_FORMAT
        or progress.get("version") != SHARD_PROGRESS_VERSION
    ):
        raise ValueError("Unsupported shard progress format.")
    if progress.get("build_identity") != build_identity:
        raise ValueError(
            "Shard progress is incompatible with source, tokenizer, or configuration."
        )
    shards = progress.get("shards")
    if not isinstance(shards, dict) or any(
        not isinstance(shards.get(split), list)
        for split in ("train", "validation")
    ):
        raise ValueError("Shard progress metadata is malformed.")


def _verify_shard_files(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    for split in ("train", "validation"):
        for entry in manifest["shards"][split]:
            path = manifest_path.parent / entry["name"]
            if path.stat().st_size != entry["byte_size"]:
                raise ValueError(f"Shard byte size mismatch: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {path}")
            values = np.memmap(path, dtype=LITTLE_ENDIAN_UINT16, mode="r")
            if len(values) != entry["token_count"]:
                raise ValueError(f"Shard token count mismatch: {path}")
            if int(values.max()) != entry["maximum_token_id"]:
                raise ValueError(f"Shard maximum token ID mismatch: {path}")


def _make_pipeline(config: PretrainingDataConfig) -> StreamingDocumentPipeline:
    return StreamingDocumentPipeline(
        text_field=config.text_field,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        minimum_characters=config.minimum_characters,
        maximum_characters=config.maximum_characters,
        normalize_line_endings=config.normalize_line_endings,
    )


def _prepare_staging_directory(path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Partial staged artifact exists: {path}. Use --force to restart "
                "only this incomplete stage."
            )
        shutil.rmtree(path)
    path.mkdir()


def _remove_partial_shards(directory: Path) -> None:
    for path in directory.glob(".*.tmp"):
        path.unlink()


def _recover_untracked_shards(
    directory: Path,
    *,
    split: str,
    tracked: list[dict[str, Any]],
    tokens_per_shard: int,
    vocabulary_size: int,
) -> list[dict[str, Any]]:
    recovered = [dict(entry) for entry in tracked]
    files = sorted(directory.glob(f"{split}-*.bin"))
    tracked_names = [entry["name"] for entry in recovered]
    if [path.name for path in files[: len(tracked_names)]] != tracked_names:
        raise ValueError(f"Finalized {split} shards do not match progress metadata.")
    for index, path in enumerate(files[len(recovered) :], start=len(recovered)):
        expected_name = f"{split}-{index:05d}.bin"
        if path.name != expected_name:
            raise ValueError(f"Untracked {split} shard names are not contiguous.")
        byte_size = path.stat().st_size
        if byte_size <= 0 or byte_size % LITTLE_ENDIAN_UINT16.itemsize:
            raise ValueError(f"Untracked shard has an invalid byte size: {path}")
        token_count = byte_size // LITTLE_ENDIAN_UINT16.itemsize
        if token_count > tokens_per_shard:
            raise ValueError(f"Untracked shard exceeds configured size: {path}")
        if recovered and recovered[-1]["token_count"] != tokens_per_shard:
            raise ValueError(f"Untracked shard follows a partial shard: {path}")
        values = np.memmap(path, dtype=LITTLE_ENDIAN_UINT16, mode="r")
        maximum = int(values.max())
        if maximum >= vocabulary_size:
            raise ValueError(f"Untracked shard token ID exceeds vocabulary: {path}")
        recovered.append(
            {
                "name": path.name,
                "token_count": token_count,
                "byte_size": byte_size,
                "sha256": sha256_file(path),
                "maximum_token_id": maximum,
            }
        )
    return recovered


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {description}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description.capitalize()} must be a JSON object.")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite finalized artifact: {path}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_identity(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _identity_without_field(value: Mapping[str, Any], field: str) -> str:
    return _canonical_identity(
        {name: item for name, item in value.items() if name != field}
    )


def _lock_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: item
        for name, item in value.items()
        if name not in {"completed_at_utc", "identity"}
    }


def _validate_verification_report(report: Mapping[str, Any]) -> None:
    if (
        report.get("format") != VERIFICATION_FORMAT
        or report.get("version") != VERIFICATION_VERSION
        or report.get("passed") is not True
        or report.get("training_started") is not False
    ):
        raise ValueError("Production data verification report is incomplete.")
    lock = report.get("production_lock")
    if not isinstance(lock, dict):
        raise TypeError("Verification report production lock must be an object.")
    if lock.get("identity") != _identity_without_field(lock, "identity"):
        raise ValueError("Verification report production lock identity mismatch.")
    if report.get("production_lock_identity") != lock["identity"]:
        raise ValueError("Verification report lock reference mismatch.")
    if report.get("manifest_sha256") != lock.get("shard_manifest", {}).get(
        "sha256"
    ):
        raise ValueError("Verification report manifest reference mismatch.")
    expected_shards = sum(
        len(lock.get("shards", {}).get(split, []))
        for split in ("train", "validation")
    )
    expected_bytes = sum(
        entry.get("byte_size", 0)
        for split in ("train", "validation")
        for entry in lock.get("shards", {}).get(split, [])
        if isinstance(entry, dict)
    )
    if report.get("verified_shards") != expected_shards:
        raise ValueError("Verification report shard count mismatch.")
    if report.get("verified_bytes") != expected_bytes:
        raise ValueError("Verification report byte count mismatch.")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
