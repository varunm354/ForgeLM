"""Tests for deterministic streamed-document preparation."""

import hashlib
from pathlib import Path

import pytest

from forgelm.data.streaming_corpus import (
    StreamingDocumentPipeline,
    assign_document_split,
    collect_tokenizer_training_documents,
    iter_jsonl,
)


def _identity(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _pipeline(**overrides: object) -> StreamingDocumentPipeline:
    arguments = {
        "text_field": "text",
        "seed": 1337,
        "validation_fraction": 0.25,
        "minimum_characters": 5,
        "maximum_characters": 100,
        "normalize_line_endings": False,
    }
    arguments.update(overrides)
    return StreamingDocumentPipeline(**arguments)  # type: ignore[arg-type]


def test_hash_split_is_deterministic_and_not_python_hash() -> None:
    identity = _identity("stable synthetic document")

    first = assign_document_split(
        identity,
        seed=1337,
        validation_fraction=0.25,
    )
    second = assign_document_split(
        identity.hex(),
        seed=1337,
        validation_fraction=0.25,
    )

    assert first == second


def test_split_assignment_is_independent_of_iteration_order() -> None:
    records = [{"text": f"synthetic document number {index}"} for index in range(30)]

    forward = {
        document.sha256: document.split
        for document in _pipeline().process(records)
    }
    backward = {
        document.sha256: document.split
        for document in _pipeline().process(reversed(records))
    }

    assert forward == backward
    assert set(forward.values()) == {"train", "validation"}


def test_exact_duplicates_are_rejected_after_normal_filtering() -> None:
    pipeline = _pipeline()
    documents = list(
        pipeline.process(
            [
                {"text": "same synthetic text"},
                {"text": "same synthetic text"},
                {"text": "different synthetic text"},
            ]
        )
    )

    assert len(documents) == 2
    assert pipeline.statistics.documents_duplicated == 1
    assert pipeline.statistics.documents_accepted == 2


def test_invalid_documents_are_counted_by_reason() -> None:
    pipeline = _pipeline(maximum_characters=20)

    documents = list(
        pipeline.process(
            [
                {"missing": "field"},
                {"text": None},
                {"text": "   "},
                {"text": "tiny"},
                {"text": "x" * 21},
                42,
                {"text": "valid text"},
            ]
        )
    )

    assert [document.text for document in documents] == ["valid text"]
    assert pipeline.statistics.documents_seen == 7
    assert pipeline.statistics.documents_rejected == 6
    assert pipeline.statistics.rejection_reasons == {
        "missing_field": 1,
        "non_string": 1,
        "empty": 1,
        "too_short": 1,
        "too_long": 1,
        "invalid_record": 1,
    }


def test_line_endings_change_only_when_enabled() -> None:
    text = "line one\r\nline two\rline three"

    unchanged = next(_pipeline().process([text]))
    normalized = next(
        _pipeline(normalize_line_endings=True).process([text])
    )

    assert unchanged.text == text
    assert normalized.text == "line one\nline two\nline three"


def test_tokenizer_sample_contains_training_documents_only() -> None:
    records = [{"text": f"synthetic sample document {index}"} for index in range(50)]
    expected_train = {
        record["text"]
        for record in records
        if assign_document_split(
            _identity(record["text"]),
            seed=1337,
            validation_fraction=0.4,
        )
        == "train"
    }
    expected_validation = {record["text"] for record in records} - expected_train
    pipeline = _pipeline(validation_fraction=0.4)

    documents, byte_count = collect_tokenizer_training_documents(
        records,
        pipeline,
        byte_budget=10_000,
    )

    assert set(documents) == expected_train
    assert set(documents).isdisjoint(expected_validation)
    assert byte_count == sum(len(document.encode("utf-8")) for document in documents)


def test_jsonl_reader_reports_malformed_record(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"text": "valid"}\nnot-json\n', encoding="utf-8")

    iterator = iter_jsonl(source)
    assert next(iterator) == {"text": "valid"}
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        next(iterator)
