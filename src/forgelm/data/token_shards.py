"""Atomic packed-token shard writing and memory-mapped random batching."""

import bisect
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer
from forgelm.data.streaming_corpus import StreamingDocumentPipeline

FORMAT_NAME = "forgelm.packed_token_shards"
FORMAT_VERSION = 1
UINT16_CAPACITY = 1 << 16
LITTLE_ENDIAN_UINT16 = np.dtype("<u2")
ShardSplit = Literal["train", "validation"]


class BinaryTokenShardWriter:
    """Write one split into fixed-size atomic little-endian uint16 shards."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        split: ShardSplit,
        tokens_per_shard: int,
        vocabulary_size: int,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")
        _validate_positive_integer(tokens_per_shard, "tokens_per_shard")
        _validate_uint16_vocabulary(vocabulary_size)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.tokens_per_shard = tokens_per_shard
        self.vocabulary_size = vocabulary_size
        self.shards: list[dict[str, Any]] = []
        self.total_tokens = 0
        self._file: Any = None
        self._temporary_path: Path | None = None
        self._final_path: Path | None = None
        self._shard_tokens = 0
        self._maximum_token_id = -1
        self._checksum: Any = None
        self._finished = False

    def append(self, token_ids: Sequence[int]) -> None:
        """Append IDs, rolling over exactly at the configured token threshold."""
        if self._finished:
            raise RuntimeError("Cannot append after shard writer finalization.")
        if not token_ids:
            return
        for token_id in token_ids:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.vocabulary_size
            ):
                raise ValueError(
                    f"Token ID {token_id!r} is outside vocabulary size "
                    f"{self.vocabulary_size}."
                )

        offset = 0
        while offset < len(token_ids):
            if self._file is None:
                self._open_shard()
            available = self.tokens_per_shard - self._shard_tokens
            count = min(available, len(token_ids) - offset)
            chunk = np.asarray(token_ids[offset : offset + count], dtype="<u2")
            payload = chunk.tobytes(order="C")
            self._file.write(payload)
            self._checksum.update(payload)
            self._shard_tokens += count
            self.total_tokens += count
            self._maximum_token_id = max(
                self._maximum_token_id,
                int(chunk.max()),
            )
            offset += count
            if self._shard_tokens == self.tokens_per_shard:
                self._finalize_current_shard()

    def finish(self) -> list[dict[str, Any]]:
        """Finalize a non-empty remainder and return immutable shard metadata."""
        if not self._finished:
            if self._file is not None:
                self._finalize_current_shard()
            self._finished = True
        return list(self.shards)

    def abort(self, *, remove_finalized: bool = False) -> None:
        """Close and remove a partial file, optionally removing finalized outputs."""
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
        if remove_finalized:
            for shard in self.shards:
                (self.output_directory / shard["name"]).unlink(missing_ok=True)
            self.shards.clear()

    def _open_shard(self) -> None:
        index = len(self.shards)
        name = f"{self.split}-{index:05d}.bin"
        final_path = self.output_directory / name
        temporary_path = self.output_directory / f".{name}.tmp"
        if final_path.exists() or temporary_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing or partial shard: {name}"
            )
        self._file = temporary_path.open("xb")
        self._temporary_path = temporary_path
        self._final_path = final_path
        self._shard_tokens = 0
        self._maximum_token_id = -1
        self._checksum = hashlib.sha256()

    def _finalize_current_shard(self) -> None:
        if (
            self._file is None
            or self._temporary_path is None
            or self._final_path is None
            or self._shard_tokens == 0
        ):
            raise RuntimeError("Cannot finalize an empty or unopened shard.")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None
        byte_size = self._temporary_path.stat().st_size
        expected_size = self._shard_tokens * LITTLE_ENDIAN_UINT16.itemsize
        if byte_size != expected_size:
            raise RuntimeError(
                f"Shard byte size {byte_size} does not match {expected_size}."
            )
        os.replace(self._temporary_path, self._final_path)
        self.shards.append(
            {
                "name": self._final_path.name,
                "token_count": self._shard_tokens,
                "byte_size": byte_size,
                "sha256": self._checksum.hexdigest(),
                "maximum_token_id": self._maximum_token_id,
            }
        )
        self._temporary_path = None
        self._final_path = None
        self._shard_tokens = 0
        self._maximum_token_id = -1
        self._checksum = None


def build_token_shards(
    records: Iterable[str | Mapping[str, Any]],
    *,
    pipeline: StreamingDocumentPipeline,
    tokenizer: ScalableByteLevelTokenizer,
    output_directory: str | Path,
    target_tokens: int,
    validation_target_tokens: int,
    tokens_per_shard: int,
    source_metadata: Mapping[str, Any],
    creation_configuration: Mapping[str, Any],
    tokenizer_path: str | Path,
    manifest_filename: str = "manifest.json",
) -> dict[str, Any]:
    """Tokenize one filtered stream and write a complete versioned manifest."""
    _validate_positive_integer(target_tokens, "target_tokens")
    _validate_positive_integer(validation_target_tokens, "validation_target_tokens")
    _validate_positive_integer(tokens_per_shard, "tokens_per_shard")
    _validate_uint16_vocabulary(tokenizer.vocab_size)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    train_writer = BinaryTokenShardWriter(
        output,
        split="train",
        tokens_per_shard=tokens_per_shard,
        vocabulary_size=tokenizer.vocab_size,
    )
    validation_writer = BinaryTokenShardWriter(
        output,
        split="validation",
        tokens_per_shard=tokens_per_shard,
        vocabulary_size=tokenizer.vocab_size,
    )
    written_documents = {"train": 0, "validation": 0}

    try:
        for document in pipeline.process(records):
            encoded = tokenizer.encode(document.text)
            encoded.append(tokenizer.end_of_document_id)
            if document.split == "train":
                remaining = target_tokens - train_writer.total_tokens
                if remaining > 0:
                    train_writer.append(encoded[:remaining])
                    written_documents["train"] += 1
            else:
                remaining = (
                    validation_target_tokens - validation_writer.total_tokens
                )
                if remaining > 0:
                    validation_writer.append(encoded[:remaining])
                    written_documents["validation"] += 1

            if train_writer.total_tokens == target_tokens:
                break

        if train_writer.total_tokens < target_tokens:
            statistics = pipeline.statistics.to_dict()
            raise ValueError(
                "Source stream ended after "
                f"{train_writer.total_tokens:,} training tokens; "
                f"{target_tokens:,} were requested. "
                f"Accepted {statistics['documents_accepted']:,} documents; "
                f"rejection counts: {statistics['rejection_reasons']}."
            )
        if validation_writer.total_tokens == 0:
            raise ValueError(
                "No validation tokens were produced before the training budget "
                "was reached. Increase the source size or validation fraction."
            )

        train_shards = train_writer.finish()
        validation_shards = validation_writer.finish()
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "source": dict(source_metadata),
            "creation_configuration": dict(creation_configuration),
            "tokenizer": {
                "path": str(tokenizer_path),
                "fingerprint": tokenizer.fingerprint(),
                "vocabulary_size": tokenizer.vocab_size,
                "end_of_document_token": tokenizer.end_of_document_token,
                "end_of_document_id": tokenizer.end_of_document_id,
            },
            "storage": {
                "dtype": "uint16",
                "byte_order": "little",
            },
            "counts": {
                **pipeline.statistics.to_dict(),
                "train_documents_written": written_documents["train"],
                "validation_documents_written": written_documents["validation"],
                "train_tokens": train_writer.total_tokens,
                "validation_tokens": validation_writer.total_tokens,
            },
            "shards": {
                "train": train_shards,
                "validation": validation_shards,
            },
        }
        validate_manifest(manifest)
        _write_json_atomic(output / manifest_filename, manifest)
        return manifest
    except Exception:
        train_writer.abort(remove_finalized=True)
        validation_writer.abort(remove_finalized=True)
        raise


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a shard manifest."""
    source = Path(path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load token-shard manifest: {source}") from error
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Any) -> None:
    """Validate schema, counts, storage compatibility, and shard metadata."""
    if not isinstance(manifest, dict):
        raise TypeError("Token-shard manifest must be an object.")
    if (
        manifest.get("format") != FORMAT_NAME
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("Unsupported token-shard manifest format or version.")
    for key in (
        "source",
        "creation_configuration",
        "tokenizer",
        "storage",
        "counts",
        "shards",
    ):
        if not isinstance(manifest.get(key), dict):
            raise TypeError(f"Manifest field {key!r} must be an object.")

    source = manifest["source"]
    for name in (
        "dataset_name",
        "dataset_config",
        "dataset_split",
        "dataset_revision",
        "text_field",
    ):
        if not isinstance(source.get(name), str) or not source[name]:
            raise ValueError(f"Manifest source field {name!r} is missing.")

    storage = manifest["storage"]
    if storage.get("dtype") != "uint16" or storage.get("byte_order") != "little":
        raise ValueError("Only little-endian uint16 shards are supported.")
    tokenizer = manifest["tokenizer"]
    vocabulary_size = tokenizer.get("vocabulary_size")
    _validate_uint16_vocabulary(vocabulary_size)
    eod_id = tokenizer.get("end_of_document_id")
    if (
        not isinstance(eod_id, int)
        or isinstance(eod_id, bool)
        or not 0 <= eod_id < vocabulary_size
    ):
        raise ValueError("Manifest EOD ID is outside the tokenizer vocabulary.")
    for name in ("path", "end_of_document_token"):
        if not isinstance(tokenizer.get(name), str) or not tokenizer[name]:
            raise ValueError(f"Manifest tokenizer field {name!r} is missing.")
    fingerprint = tokenizer.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Manifest tokenizer fingerprint must be SHA-256.")

    counts = manifest["counts"]
    shards = manifest["shards"]
    for split in ("train", "validation"):
        entries = shards.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest must contain at least one {split} shard.")
        total = 0
        for index, entry in enumerate(entries):
            _validate_shard_entry(entry, vocabulary_size)
            expected_name = f"{split}-{index:05d}.bin"
            if entry["name"] != expected_name:
                raise ValueError(
                    f"Manifest {split} shard names are not deterministic."
                )
            total += entry["token_count"]
        if counts.get(f"{split}_tokens") != total:
            raise ValueError(f"Manifest {split} token count is inconsistent.")
    required_counts = (
        "documents_seen",
        "documents_accepted",
        "documents_rejected",
        "documents_duplicated",
        "total_raw_utf8_bytes",
    )
    for name in required_counts:
        value = counts.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Manifest count {name!r} must be non-negative.")
    if (
        counts["documents_seen"]
        != counts["documents_accepted"]
        + counts["documents_rejected"]
        + counts["documents_duplicated"]
    ):
        raise ValueError("Manifest document counters are inconsistent.")


class MemoryMappedShardDataset:
    """Random next-token batches sampled within validated memory-mapped shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: ShardSplit,
        block_size: int,
        expected_vocabulary_size: int | None = None,
        expected_tokenizer_fingerprint: str | None = None,
        verify_checksums: bool = False,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")
        _validate_positive_integer(block_size, "block_size")
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        tokenizer = self.manifest["tokenizer"]
        if (
            expected_vocabulary_size is not None
            and tokenizer["vocabulary_size"] != expected_vocabulary_size
        ):
            raise ValueError("Manifest tokenizer vocabulary is incompatible.")
        if (
            expected_tokenizer_fingerprint is not None
            and tokenizer["fingerprint"] != expected_tokenizer_fingerprint
        ):
            raise ValueError("Manifest tokenizer fingerprint is incompatible.")

        self.split = split
        self.block_size = block_size
        self._shards: list[np.memmap] = []
        self._cumulative_windows: list[int] = []
        cumulative = 0
        for entry in self.manifest["shards"][split]:
            path = self.manifest_path.parent / entry["name"]
            if not path.is_file():
                raise ValueError(f"Shard is missing: {path}")
            actual_size = path.stat().st_size
            if actual_size != entry["byte_size"]:
                raise ValueError(f"Shard byte size mismatch: {path}")
            if verify_checksums and sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {path}")
            if entry["token_count"] <= block_size:
                raise ValueError(
                    f"Shard {entry['name']} is too short for block_size + 1."
                )
            shard = np.memmap(path, dtype=LITTLE_ENDIAN_UINT16, mode="r")
            if len(shard) != entry["token_count"]:
                raise ValueError(f"Shard token count mismatch: {path}")
            self._shards.append(shard)
            cumulative += entry["token_count"] - block_size
            self._cumulative_windows.append(cumulative)
        self.total_valid_windows = cumulative

    def get_batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a deterministic batch without crossing shard boundaries."""
        _validate_positive_integer(batch_size, "batch_size")
        global_starts = torch.randint(
            0,
            self.total_valid_windows,
            (batch_size,),
            generator=generator,
        ).tolist()
        inputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for global_start in global_starts:
            shard_index = bisect.bisect_right(
                self._cumulative_windows,
                global_start,
            )
            prior = (
                0 if shard_index == 0 else self._cumulative_windows[shard_index - 1]
            )
            local_start = global_start - prior
            sequence = np.asarray(
                self._shards[shard_index][
                    local_start : local_start + self.block_size + 1
                ],
                dtype=np.int64,
            ).copy()
            tensor = torch.from_numpy(sequence)
            inputs.append(tensor[:-1])
            targets.append(tensor[1:])
        return torch.stack(inputs), torch.stack(targets)


def sha256_file(path: str | Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shard_entry(entry: Any, vocabulary_size: int) -> None:
    if not isinstance(entry, dict):
        raise TypeError("Shard metadata must be an object.")
    name = entry.get("name")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Shard name must be a relative file name.")
    for field in ("token_count", "byte_size"):
        value = entry.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Shard {field} must be a positive integer.")
    if entry["byte_size"] != entry["token_count"] * LITTLE_ENDIAN_UINT16.itemsize:
        raise ValueError("Shard byte size is inconsistent with token count.")
    checksum = entry.get("sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("Shard checksum must be lowercase SHA-256 hexadecimal.")
    maximum = entry.get("maximum_token_id")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not 0 <= maximum < vocabulary_size
    ):
        raise ValueError("Shard maximum token ID is outside the vocabulary.")


def _validate_uint16_vocabulary(vocabulary_size: Any) -> None:
    if (
        not isinstance(vocabulary_size, int)
        or isinstance(vocabulary_size, bool)
        or not 1 <= vocabulary_size <= UINT16_CAPACITY
    ):
        raise ValueError("uint16 requires vocabulary_size between 1 and 65536.")


def _validate_positive_integer(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing or partial manifest: {path}"
        )
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
