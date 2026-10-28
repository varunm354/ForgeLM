"""Offline Phase 4E-A tokenizer, resumable-shard, and lock tests."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.production_data import (
    BuildInterrupted,
    ProductionPaths,
    approve_tokenizer,
    build_shard_stage,
    build_tokenizer_stage,
    default_source_metadata,
    iter_tokenizer_corpus,
    validate_production_data_lock,
    validate_tokenizer_bundle,
    verify_production_stage,
)
from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer
from forgelm.data.streaming_corpus import (
    StreamingDocumentPipeline,
    assign_document_split,
)
from forgelm.data.token_shards import load_manifest, sha256_file
from forgelm.models.gpt import GPTConfig
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_runtime import validate_training_compatibility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs/pretraining_data_smoke.yaml"
EOD = "<|endoftext|>"


def _records(count: int = 100) -> list[dict[str, str]]:
    return [
        {
            "text": (
                f"Offline production fixture document {index:03d} contains "
                "deterministic alpha beta gamma patterns for tokenizer and "
                "restart testing."
            )
        }
        for index in range(count)
    ]


def _split(text: str, config: PretrainingDataConfig) -> str:
    identity = hashlib.sha256(text.encode("utf-8")).digest()
    return assign_document_split(
        identity,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
    )


def _config(tmp_path: Path) -> tuple[PretrainingDataConfig, list[dict[str, str]]]:
    records = _records()
    base = PretrainingDataConfig.load(
        SMOKE_CONFIG,
        project_root=PROJECT_ROOT,
    ).with_overrides(
        tokenizer_vocab_size=280,
        target_tokens=250,
        validation_target_tokens=110,
        tokens_per_shard=64,
        minimum_characters=10,
        maximum_characters=500,
        output_directory=tmp_path / "unused-output",
        tokenizer_path=tmp_path / "unused-tokenizer.json",
    )
    training_documents = [
        record["text"]
        for record in records
        if _split(record["text"], base) == "train"
    ]
    exact_budget = sum(
        len(text.encode("utf-8")) for text in training_documents[:8]
    )
    return base.with_overrides(tokenizer_training_bytes=exact_budget), records


def _pipeline(config: PretrainingDataConfig) -> StreamingDocumentPipeline:
    return StreamingDocumentPipeline(
        text_field=config.text_field,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        minimum_characters=config.minimum_characters,
        maximum_characters=config.maximum_characters,
        normalize_line_endings=config.normalize_line_endings,
    )


def _prepare_approved_tokenizer(
    config: PretrainingDataConfig,
    records: list[dict[str, str]],
    root: Path,
) -> dict[str, Any]:
    manifest = build_tokenizer_stage(config, lambda: iter(records), root)
    approve_tokenizer(
        config,
        root,
        expected_fingerprint=manifest["tokenizer"]["fingerprint"],
    )
    return manifest


def _all_tokens(root: Path, split: str) -> list[int]:
    paths = ProductionPaths.from_root(root)
    manifest = load_manifest(paths.shard_manifest_path)
    return [
        int(token_id)
        for entry in manifest["shards"][split]
        for token_id in np.memmap(
            paths.shards / entry["name"],
            dtype="<u2",
            mode="r",
        )
    ]


def _expected_tokens(
    config: PretrainingDataConfig,
    records: list[dict[str, str]],
    tokenizer: ScalableByteLevelTokenizer,
) -> dict[str, list[int]]:
    targets = {
        "train": config.target_tokens,
        "validation": config.validation_target_tokens,
    }
    expected: dict[str, list[int]] = {"train": [], "validation": []}
    for document in _pipeline(config).process(records):
        destination = expected[document.split]
        remaining = targets[document.split] - len(destination)
        if remaining > 0:
            destination.extend(
                [*tokenizer.encode(document.text), tokenizer.end_of_document_id][
                    :remaining
                ]
            )
        if all(len(expected[split]) == targets[split] for split in targets):
            break
    return expected


def test_train_only_corpus_is_deterministic_and_has_no_validation_leakage(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = build_tokenizer_stage(config, lambda: iter(records), first_root)
    second = build_tokenizer_stage(config, lambda: iter(records), second_root)
    corpus_documents = list(
        iter_tokenizer_corpus(ProductionPaths.from_root(first_root).corpus_path)
    )
    train_documents = {
        record["text"]
        for record in records
        if _split(record["text"], config) == "train"
    }
    validation_documents = {
        record["text"]
        for record in records
        if _split(record["text"], config) == "validation"
    }

    assert set(corpus_documents) <= train_documents
    assert set(corpus_documents).isdisjoint(validation_documents)
    assert sum(len(text.encode("utf-8")) for text in corpus_documents) == (
        config.tokenizer_training_bytes
    )
    assert first["tokenizer"]["fingerprint"] == second["tokenizer"]["fingerprint"]
    first_corpus = json.loads(
        ProductionPaths.from_root(first_root).corpus_manifest_path.read_text()
    )
    second_corpus = json.loads(
        ProductionPaths.from_root(second_root).corpus_manifest_path.read_text()
    )
    assert first_corpus["corpus"]["sha256"] == second_corpus["corpus"]["sha256"]
    assert first_corpus["identity"] == second_corpus["identity"]


def test_tokenizer_bundle_rejects_mutation_and_wrong_approval(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    manifest = build_tokenizer_stage(config, lambda: iter(records), root)
    paths = ProductionPaths.from_root(root)

    with pytest.raises(ValueError, match="does not match"):
        approve_tokenizer(
            config,
            root,
            expected_fingerprint="0" * 64,
        )

    paths.tokenizer_path.write_text(
        paths.tokenizer_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_tokenizer_bundle(config, root)

    assert manifest["status"] == "candidate_not_production_pinned"
    assert not paths.tokenizer_approval_path.exists()


def test_shards_preserve_split_eod_accounting_and_final_remainders(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    _prepare_approved_tokenizer(config, records, root)

    manifest = build_shard_stage(config, lambda: iter(records), root)
    paths = ProductionPaths.from_root(root)
    tokenizer = ScalableByteLevelTokenizer.load(
        paths.tokenizer_path,
        end_of_document_token=EOD,
        expected_vocab_size=config.tokenizer_vocab_size,
    )
    expected = _expected_tokens(config, records, tokenizer)

    assert _all_tokens(root, "train") == expected["train"]
    assert _all_tokens(root, "validation") == expected["validation"]
    assert tokenizer.end_of_document_id in expected["train"]
    assert tokenizer.end_of_document_id in expected["validation"]
    assert [entry["token_count"] for entry in manifest["shards"]["train"]] == [
        64,
        64,
        64,
        58,
    ]
    assert [
        entry["token_count"] for entry in manifest["shards"]["validation"]
    ] == [64, 46]
    assert manifest["counts"]["train_tokens"] == 250
    assert manifest["counts"]["validation_tokens"] == 110
    assert manifest["counts"]["tokenized_storage_bytes"] == 720
    assert not paths.production_lock_path.exists()


def test_interrupted_build_resumes_without_rewriting_or_changing_output(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    resumed_root = tmp_path / "resumed"
    reference_root = tmp_path / "reference"
    _prepare_approved_tokenizer(config, records, resumed_root)
    _prepare_approved_tokenizer(config, records, reference_root)

    with pytest.raises(BuildInterrupted):
        build_shard_stage(
            config,
            lambda: iter(records),
            resumed_root,
            interrupt_after_documents=6,
        )
    resumed_paths = ProductionPaths.from_root(resumed_root)
    progress = json.loads(resumed_paths.shard_progress_path.read_text())
    committed_before = {
        path.name: sha256_file(path) for path in resumed_paths.shards.glob("*.bin")
    }

    assert progress["version"] == 1
    assert progress["status"] == "interrupted"
    assert committed_before
    assert not list(resumed_paths.shards.glob(".*.tmp"))
    orphan_split = next(
        split for split in ("train", "validation") if progress["shards"][split]
    )
    progress["shards"][orphan_split].pop()
    progress["tokens"][orphan_split] = sum(
        entry["token_count"] for entry in progress["shards"][orphan_split]
    )
    resumed_paths.shard_progress_path.write_text(
        json.dumps(progress),
        encoding="utf-8",
    )

    resumed = build_shard_stage(
        config,
        lambda: iter(records),
        resumed_root,
        resume=True,
    )
    reference = build_shard_stage(
        config,
        lambda: iter(records),
        reference_root,
    )

    assert resumed == reference
    assert all(
        sha256_file(resumed_paths.shards / name) == checksum
        for name, checksum in committed_before.items()
    )
    for split in ("train", "validation"):
        assert [
            entry["sha256"] for entry in resumed["shards"][split]
        ] == [entry["sha256"] for entry in reference["shards"][split]]


def test_resume_rejects_changed_configuration_and_source_replay(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    _prepare_approved_tokenizer(config, records, root)
    with pytest.raises(BuildInterrupted):
        build_shard_stage(
            config,
            lambda: iter(records),
            root,
            interrupt_after_documents=6,
        )

    changed = config.with_overrides(seed=config.seed + 1)
    with pytest.raises(ValueError, match="configuration identity"):
        build_shard_stage(
            changed,
            lambda: iter(records),
            root,
            resume=True,
        )

    reordered = list(reversed(records))
    with pytest.raises(ValueError, match="replay"):
        build_shard_stage(
            config,
            lambda: iter(reordered),
            root,
            resume=True,
        )


def test_shard_build_rejects_source_different_from_tokenizer_corpus(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    _prepare_approved_tokenizer(config, records, root)
    changed_source = {
        **default_source_metadata(config),
        "type": "local_jsonl_fixture",
        "local_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="tokenizer corpus source"):
        build_shard_stage(
            config,
            lambda: iter(records),
            root,
            source_metadata=changed_source,
        )

    fineweb = config.with_overrides(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_revision="924804708136147342bc2390cd29abb1edaf0c67",
    )
    local_root = tmp_path / "fineweb-local-fixture"
    local_source = {
        **default_source_metadata(fineweb),
        "type": "local_jsonl_fixture",
        "local_path": "/offline/fixture.jsonl",
        "local_byte_size": 123,
        "local_sha256": "1" * 64,
    }
    candidate = build_tokenizer_stage(
        fineweb,
        lambda: iter(records),
        local_root,
        source_metadata=local_source,
    )
    approve_tokenizer(
        fineweb,
        local_root,
        expected_fingerprint=candidate["tokenizer"]["fingerprint"],
    )
    build_shard_stage(
        fineweb,
        lambda: iter(records),
        local_root,
        source_metadata=local_source,
    )
    with pytest.raises(ValueError, match="source field 'type'"):
        verify_production_stage(fineweb, local_root)


def test_verification_publishes_lock_and_detects_corruption(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    tokenizer_manifest = _prepare_approved_tokenizer(config, records, root)
    build_shard_stage(config, lambda: iter(records), root)
    paths = ProductionPaths.from_root(root)

    lock = verify_production_stage(config, root)

    assert lock["status"] == "complete_verified_not_training_approval"
    assert lock["counts"] == {
        "train_tokens": 250,
        "validation_tokens": 110,
        "train_shards": 4,
        "validation_shards": 2,
    }
    assert lock["tokenizer"]["fingerprint"] == tokenizer_manifest["tokenizer"][
        "fingerprint"
    ]
    assert validate_production_data_lock(
        paths.production_lock_path,
        manifest_path=paths.shard_manifest_path,
        expected_tokenizer_fingerprint=lock["tokenizer"]["fingerprint"],
    ) == lock
    original_report = paths.verification_report_path.read_text(encoding="utf-8")
    contradictory_report = json.loads(original_report)
    contradictory_report["verified_bytes"] += 1
    paths.verification_report_path.write_text(
        json.dumps(contradictory_report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="byte count mismatch"):
        verify_production_stage(config, root)
    paths.verification_report_path.write_text(original_report, encoding="utf-8")
    paths.verification_report_path.unlink()
    with pytest.raises(ValueError, match="without its verification report"):
        verify_production_stage(config, root)
    report = {
        "format": "forgelm.production_data_verification",
        "version": 1,
        "passed": True,
        "training_started": False,
        "production_lock": lock,
    }
    paths.verification_report_path.write_text(json.dumps(report), encoding="utf-8")
    shard_after_lock = next(paths.shards.glob("train-*.bin"))
    locked_contents = bytearray(shard_after_lock.read_bytes())
    locked_contents[0] ^= 1
    shard_after_lock.write_bytes(locked_contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_production_data_lock(
            paths.production_lock_path,
            manifest_path=paths.shard_manifest_path,
            expected_tokenizer_fingerprint=lock["tokenizer"]["fingerprint"],
        )

    corrupt_root = tmp_path / "corrupt"
    corrupt_manifest = _prepare_approved_tokenizer(config, records, corrupt_root)
    build_shard_stage(config, lambda: iter(records), corrupt_root)
    corrupt_paths = ProductionPaths.from_root(corrupt_root)
    shard = next(corrupt_paths.shards.glob("train-*.bin"))
    contents = bytearray(shard.read_bytes())
    contents[0] ^= 1
    shard.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_production_stage(config, corrupt_root)
    assert not corrupt_paths.production_lock_path.exists()
    assert corrupt_manifest["tokenizer"]["vocabulary_size"] == 280


def test_verification_rejects_altered_creation_contract(tmp_path: Path) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    _prepare_approved_tokenizer(config, records, root)
    build_shard_stage(config, lambda: iter(records), root)
    paths = ProductionPaths.from_root(root)
    manifest = json.loads(paths.shard_manifest_path.read_text())
    manifest["creation_configuration"]["seed"] += 1
    paths.shard_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="creation configuration"):
        verify_production_stage(config, root)


def test_each_stage_stops_before_the_next_expensive_boundary(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    root = tmp_path / "production"
    paths = ProductionPaths.from_root(root)

    tokenizer_manifest = build_tokenizer_stage(
        config,
        lambda: iter(records),
        root,
    )
    assert paths.tokenizer_manifest_path.is_file()
    assert not paths.tokenizer_approval_path.exists()
    assert not paths.shards.exists()

    approve_tokenizer(
        config,
        root,
        expected_fingerprint=tokenizer_manifest["tokenizer"]["fingerprint"],
    )
    assert paths.tokenizer_approval_path.is_file()
    assert not paths.shards.exists()

    build_shard_stage(config, lambda: iter(records), root)
    assert paths.shard_manifest_path.is_file()
    assert not paths.production_lock_path.exists()

    verify_production_stage(config, root)
    assert paths.production_lock_path.is_file()
    with pytest.raises(FileExistsError, match="immutable"):
        build_shard_stage(
            config,
            lambda: iter(records),
            root,
            force=True,
        )


def test_fineweb_training_requires_and_validates_production_lock(
    tmp_path: Path,
) -> None:
    config, records = _config(tmp_path)
    config = config.with_overrides(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train",
        dataset_revision="924804708136147342bc2390cd29abb1edaf0c67",
    )
    root = tmp_path / "production"
    tokenizer_manifest = _prepare_approved_tokenizer(config, records, root)
    build_shard_stage(config, lambda: iter(records), root)
    verify_production_stage(config, root)
    paths = ProductionPaths.from_root(root)
    model_config = GPTConfig(
        vocab_size=280,
        block_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        feed_forward_hidden_dim=32,
        rope_max_sequence_length=8,
    )
    values = {
        "model_config_path": tmp_path / "model.yaml",
        "manifest_path": paths.shard_manifest_path,
        "expected_tokenizer_fingerprint": tokenizer_manifest["tokenizer"][
            "fingerprint"
        ],
        "seed": 1,
        "micro_batch_size": 1,
        "sequence_length": 8,
        "gradient_accumulation_steps": 1,
        "target_training_tokens": 16,
        "learning_rate": 1e-3,
        "minimum_learning_rate": 1e-4,
        "warmup_steps": 1,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "epsilon": 1e-8,
        "gradient_clipping": 1.0,
        "logging_interval": 1,
        "validation_interval": 1,
        "validation_batches": 1,
        "checkpoint_interval": 1,
        "precision": "auto",
        "torch_compile": False,
        "compile_allow_fallback": False,
        "output_directory": tmp_path / "training",
        "latest_checkpoint_name": "latest.pt",
        "best_checkpoint_name": "best.pt",
        "metrics_filename": "metrics.jsonl",
        "metadata_filename": "metadata.json",
        "verify_checksums": True,
        "fixed_validation_batches": True,
    }
    unlocked = PretrainConfig(**values)
    with pytest.raises(ValueError, match="production data lock"):
        validate_training_compatibility(
            paths.shard_manifest_path,
            model_config,
            unlocked,
        )

    locked = PretrainConfig(
        **values,
        production_data_lock_path=paths.production_lock_path,
    )
    manifest, identity = validate_training_compatibility(
        paths.shard_manifest_path,
        model_config,
        locked,
    )

    assert manifest["source"]["dataset_revision"] == config.dataset_revision
    assert identity == sha256_file(paths.shard_manifest_path)
