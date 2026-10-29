"""Deterministic filtering, deduplication, and splitting for streamed documents."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DocumentSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class CorpusDocument:
    """A validated unique document with a stable identity and split."""

    text: str
    split: DocumentSplit
    sha256: str
    utf8_bytes: int


@dataclass
class CorpusStatistics:
    """Mutable counters accumulated without retaining document contents."""

    documents_seen: int = 0
    documents_accepted: int = 0
    documents_rejected: int = 0
    documents_duplicated: int = 0
    train_documents: int = 0
    validation_documents: int = 0
    total_raw_utf8_bytes: int = 0
    total_accepted_utf8_bytes: int = 0
    train_accepted_utf8_bytes: int = 0
    validation_accepted_utf8_bytes: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable counters with deterministic reason ordering."""
        return {
            "documents_seen": self.documents_seen,
            "documents_accepted": self.documents_accepted,
            "documents_rejected": self.documents_rejected,
            "documents_duplicated": self.documents_duplicated,
            "train_documents": self.train_documents,
            "validation_documents": self.validation_documents,
            "total_raw_utf8_bytes": self.total_raw_utf8_bytes,
            "total_accepted_utf8_bytes": self.total_accepted_utf8_bytes,
            "train_accepted_utf8_bytes": self.train_accepted_utf8_bytes,
            "validation_accepted_utf8_bytes": self.validation_accepted_utf8_bytes,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
        }


class StreamingDocumentPipeline:
    """Process records in one pass while storing only exact-deduplication hashes."""

    def __init__(
        self,
        *,
        text_field: str,
        seed: int,
        validation_fraction: float,
        minimum_characters: int,
        maximum_characters: int,
        normalize_line_endings: bool = False,
    ) -> None:
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if (
            not isinstance(validation_fraction, (int, float))
            or isinstance(validation_fraction, bool)
            or not 0.0 < validation_fraction < 1.0
        ):
            raise ValueError("validation_fraction must be between 0 and 1.")
        for value, name in (
            (minimum_characters, "minimum_characters"),
            (maximum_characters, "maximum_characters"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if minimum_characters > maximum_characters:
            raise ValueError(
                "minimum_characters cannot exceed maximum_characters."
            )
        if not isinstance(normalize_line_endings, bool):
            raise TypeError("normalize_line_endings must be a boolean.")

        self.text_field = text_field
        self.seed = seed
        self.validation_fraction = float(validation_fraction)
        self.minimum_characters = minimum_characters
        self.maximum_characters = maximum_characters
        self.normalize_line_endings = normalize_line_endings
        self.statistics = CorpusStatistics()
        self._document_hashes: set[bytes] = set()

    def process(
        self,
        records: Iterable[str | Mapping[str, Any]],
    ) -> Iterator[CorpusDocument]:
        """Yield valid, unique records from strings or mapping-like dataset rows."""
        iterator = iter(records)
        try:
            for record in iterator:
                self.statistics.documents_seen += 1
                text, reason = self._extract_text(record)
                if reason is not None:
                    self._reject(reason)
                    continue
                if text is None:
                    raise RuntimeError("Validated text unexpectedly missing.")

                self.statistics.total_raw_utf8_bytes += len(text.encode("utf-8"))
                if self.normalize_line_endings:
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                if not text.strip():
                    self._reject("empty")
                    continue
                if len(text) < self.minimum_characters:
                    self._reject("too_short")
                    continue
                if len(text) > self.maximum_characters:
                    self._reject("too_long")
                    continue

                identity = hashlib.sha256(text.encode("utf-8")).digest()
                if identity in self._document_hashes:
                    self.statistics.documents_duplicated += 1
                    continue
                self._document_hashes.add(identity)

                split = assign_document_split(
                    identity,
                    seed=self.seed,
                    validation_fraction=self.validation_fraction,
                )
                accepted_utf8_bytes = len(text.encode("utf-8"))
                self.statistics.documents_accepted += 1
                self.statistics.total_accepted_utf8_bytes += accepted_utf8_bytes
                if split == "train":
                    self.statistics.train_documents += 1
                    self.statistics.train_accepted_utf8_bytes += accepted_utf8_bytes
                else:
                    self.statistics.validation_documents += 1
                    self.statistics.validation_accepted_utf8_bytes += accepted_utf8_bytes
                yield CorpusDocument(
                    text=text,
                    split=split,
                    sha256=identity.hex(),
                    utf8_bytes=accepted_utf8_bytes,
                )
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def _extract_text(
        self,
        record: str | Mapping[str, Any],
    ) -> tuple[str | None, str | None]:
        if isinstance(record, str):
            return record, None
        if not isinstance(record, Mapping):
            return None, "invalid_record"
        if self.text_field not in record:
            return None, "missing_field"
        text = record[self.text_field]
        if not isinstance(text, str):
            return None, "non_string"
        return text, None

    def _reject(self, reason: str) -> None:
        self.statistics.documents_rejected += 1
        self.statistics.rejection_reasons[reason] += 1


def assign_document_split(
    document_identity: bytes | str,
    *,
    seed: int,
    validation_fraction: float,
) -> DocumentSplit:
    """Assign a document using SHA-256, independent of process and stream order."""
    if isinstance(document_identity, str):
        try:
            identity = bytes.fromhex(document_identity)
        except ValueError as error:
            raise ValueError("document_identity must be SHA-256 hexadecimal.") from error
    elif isinstance(document_identity, bytes):
        identity = document_identity
    else:
        raise TypeError("document_identity must be bytes or hexadecimal text.")
    if len(identity) != hashlib.sha256().digest_size:
        raise ValueError("document_identity must contain a SHA-256 digest.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    seeded_digest = hashlib.sha256(
        str(seed).encode("ascii") + b"\0" + identity
    ).digest()
    bucket = int.from_bytes(seeded_digest[:8], byteorder="big")
    threshold = int(validation_fraction * (1 << 64))
    return "validation" if bucket < threshold else "train"


def collect_tokenizer_training_documents(
    records: Iterable[str | Mapping[str, Any]],
    pipeline: StreamingDocumentPipeline,
    *,
    byte_budget: int,
) -> tuple[list[str], int]:
    """Collect a bounded train-only sample, truncating only the final UTF-8 document."""
    if not isinstance(byte_budget, int) or isinstance(byte_budget, bool) or byte_budget < 1:
        raise ValueError("byte_budget must be a positive integer.")

    documents: list[str] = []
    collected_bytes = 0
    with closing(pipeline.process(records)) as processed:
        for document in processed:
            if document.split != "train":
                continue
            remaining = byte_budget - collected_bytes
            if remaining <= 0:
                break
            encoded = document.text.encode("utf-8")
            if len(encoded) > remaining:
                truncated = encoded[:remaining].decode("utf-8", errors="ignore")
                if truncated:
                    documents.append(truncated)
                    collected_bytes += len(truncated.encode("utf-8"))
                break
            documents.append(document.text)
            collected_bytes += len(encoded)
            if collected_bytes == byte_budget:
                break
    return documents, collected_bytes


def load_huggingface_stream(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    revision: str,
) -> Iterable[Mapping[str, Any]]:
    """Open a Hugging Face iterable dataset without downloading it in full."""
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        revision=revision,
        streaming=True,
    )


def iter_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Stream JSON objects from a UTF-8 JSONL file with line-specific errors."""
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not open local JSONL source: {source}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {source}:{line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON at {source}:{line_number}: {error.msg}."
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"JSONL record at {source}:{line_number} must be an object."
                )
            yield record
