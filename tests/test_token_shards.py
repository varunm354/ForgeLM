"""Tests for packed token shards, manifests, and memory-mapped batching."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer
from forgelm.data.streaming_corpus import (
    StreamingDocumentPipeline,
    assign_document_split,
)
from forgelm.data.token_shards import (
    BinaryTokenShardWriter,
    MemoryMappedShardDataset,
    build_token_shards,
    load_manifest,
    sha256_file,
)

EOD = "<|endoftext|>"


def _tokenizer() -> ScalableByteLevelTokenizer:
    documents = [
        "synthetic packed shard training alpha beta gamma " * 10,
        "synthetic packed shard training delta epsilon zeta " * 10,
    ]
    return ScalableByteLevelTokenizer.train_from_iterator(
        documents,
        vocab_size=280,
        end_of_document_token=EOD,
        length=len(documents),
    )


def _documents_by_split(
    *,
    seed: int,
    validation_fraction: float,
    train_count: int,
    validation_count: int,
) -> tuple[list[str], list[str]]:
    train: list[str] = []
    validation: list[str] = []
    index = 0
    while len(train) < train_count or len(validation) < validation_count:
        text = (
            f"synthetic packed document {index:03d} with enough repeated words "
            "for deterministic testing"
        )
        identity = hashlib.sha256(text.encode("utf-8")).digest()
        split = assign_document_split(
            identity,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        destination = train if split == "train" else validation
        needed = train_count if split == "train" else validation_count
        if len(destination) < needed:
            destination.append(text)
        index += 1
    return train, validation


@pytest.fixture
def built_shards(tmp_path: Path) -> dict[str, Any]:
    tokenizer = _tokenizer()
    seed = 17
    fraction = 0.4
    train, validation = _documents_by_split(
        seed=seed,
        validation_fraction=fraction,
        train_count=4,
        validation_count=2,
    )
    expected_train = [
        token_id
        for text in train
        for token_id in [*tokenizer.encode(text), tokenizer.end_of_document_id]
    ]
    expected_validation = [
        token_id
        for text in validation
        for token_id in [*tokenizer.encode(text), tokenizer.end_of_document_id]
    ]
    records = [
        {"text": validation[0]},
        {"text": validation[1]},
        {"text": train[0]},
        {"text": train[0]},
        {"text": "x"},
        *({"text": text} for text in train[1:]),
    ]
    pipeline = StreamingDocumentPipeline(
        text_field="text",
        seed=seed,
        validation_fraction=fraction,
        minimum_characters=5,
        maximum_characters=200,
    )
    output = tmp_path / "shards"
    manifest = build_token_shards(
        records,
        pipeline=pipeline,
        tokenizer=tokenizer,
        output_directory=output,
        target_tokens=len(expected_train),
        validation_target_tokens=len(expected_validation),
        tokens_per_shard=max(len(expected_train), len(expected_validation)),
        source_metadata={
            "dataset_name": "local/synthetic",
            "dataset_config": "offline",
            "dataset_split": "train",
            "dataset_revision": "fixture-v1",
            "text_field": "text",
        },
        creation_configuration={"seed": seed},
        tokenizer_path=tmp_path / "tokenizer.json",
    )
    return {
        "tokenizer": tokenizer,
        "manifest": manifest,
        "manifest_path": output / "manifest.json",
        "output": output,
        "expected_train": expected_train,
        "expected_validation": expected_validation,
    }


def test_writer_rollover_preserves_every_token_and_is_atomic(
    tmp_path: Path,
) -> None:
    writer = BinaryTokenShardWriter(
        tmp_path,
        split="train",
        tokens_per_shard=4,
        vocabulary_size=20,
    )
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    writer.append(expected[:3])
    assert not (tmp_path / "train-00000.bin").exists()
    assert (tmp_path / ".train-00000.bin.tmp").exists()
    writer.append(expected[3:])
    metadata = writer.finish()

    actual = [
        int(token_id)
        for entry in metadata
        for token_id in np.memmap(
            tmp_path / entry["name"],
            dtype="<u2",
            mode="r",
        )
    ]
    assert [entry["token_count"] for entry in metadata] == [4, 4, 3]
    assert actual == expected
    assert not list(tmp_path.glob(".*.tmp"))
    assert all(
        sha256_file(tmp_path / entry["name"]) == entry["sha256"]
        for entry in metadata
    )


def test_writer_does_not_create_empty_shards(tmp_path: Path) -> None:
    writer = BinaryTokenShardWriter(
        tmp_path,
        split="validation",
        tokens_per_shard=4,
        vocabulary_size=20,
    )

    assert writer.finish() == []
    assert list(tmp_path.iterdir()) == []


def test_uint16_vocabulary_and_token_limits_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="65536"):
        BinaryTokenShardWriter(
            tmp_path,
            split="train",
            tokens_per_shard=4,
            vocabulary_size=65_537,
        )

    writer = BinaryTokenShardWriter(
        tmp_path,
        split="train",
        tokens_per_shard=4,
        vocabulary_size=65_536,
    )
    with pytest.raises(ValueError, match="outside vocabulary"):
        writer.append([65_536])


def test_eod_placement_manifest_counts_and_checksums(
    built_shards: dict[str, Any],
) -> None:
    manifest = load_manifest(built_shards["manifest_path"])
    train_entry = manifest["shards"]["train"][0]
    validation_entry = manifest["shards"]["validation"][0]
    train_ids = np.memmap(
        built_shards["output"] / train_entry["name"],
        dtype="<u2",
        mode="r",
    ).tolist()

    assert train_ids == built_shards["expected_train"]
    assert manifest["format_version"] == 1
    assert manifest["storage"] == {"dtype": "uint16", "byte_order": "little"}
    assert manifest["counts"]["documents_seen"] == 8
    assert manifest["counts"]["documents_accepted"] == 6
    assert manifest["counts"]["documents_rejected"] == 1
    assert manifest["counts"]["documents_duplicated"] == 1
    assert manifest["counts"]["train_tokens"] == len(
        built_shards["expected_train"]
    )
    assert manifest["counts"]["validation_tokens"] == len(
        built_shards["expected_validation"]
    )
    for entry in (train_entry, validation_entry):
        assert sha256_file(built_shards["output"] / entry["name"]) == entry["sha256"]


def test_memory_mapped_batches_are_deterministic_and_shifted(
    built_shards: dict[str, Any],
) -> None:
    tokenizer = built_shards["tokenizer"]
    dataset = MemoryMappedShardDataset(
        built_shards["manifest_path"],
        split="train",
        block_size=8,
        expected_vocabulary_size=tokenizer.vocab_size,
        expected_tokenizer_fingerprint=tokenizer.fingerprint(),
        verify_checksums=True,
    )

    first = dataset.get_batch(
        4,
        generator=torch.Generator().manual_seed(99),
    )
    second = dataset.get_batch(
        4,
        generator=torch.Generator().manual_seed(99),
    )

    assert isinstance(dataset._shards[0], np.memmap)
    assert first[0].shape == first[1].shape == (4, 8)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[0][:, 1:], first[1][:, :-1])


def test_checksum_verification_detects_corruption(
    built_shards: dict[str, Any],
) -> None:
    entry = built_shards["manifest"]["shards"]["train"][0]
    path = built_shards["output"] / entry["name"]
    contents = bytearray(path.read_bytes())
    contents[0] ^= 1
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="checksum mismatch"):
        MemoryMappedShardDataset(
            built_shards["manifest_path"],
            split="train",
            block_size=8,
            verify_checksums=True,
        )


def test_reader_rejects_shard_too_short_for_requested_block(
    built_shards: dict[str, Any],
) -> None:
    token_count = built_shards["manifest"]["shards"]["train"][0]["token_count"]

    with pytest.raises(ValueError, match="too short"):
        MemoryMappedShardDataset(
            built_shards["manifest_path"],
            split="train",
            block_size=token_count,
        )
