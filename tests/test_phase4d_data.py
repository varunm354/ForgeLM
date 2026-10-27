"""Offline tests for Phase 4D data pinning, preflight, and fixture creation."""

import subprocess
import sys
from pathlib import Path

import pytest

from forgelm.data.preflight import (
    is_immutable_dataset_revision,
    run_data_preflight,
)
from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.qualification_fixture import create_qualification_fixture
from forgelm.data.token_shards import MemoryMappedShardDataset, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = PROJECT_ROOT / "configs/pretraining_data.yaml"
SMOKE_CONFIG = PROJECT_ROOT / "configs/pretraining_data_smoke.yaml"
FIXTURE_JSONL = PROJECT_ROOT / "tests/fixtures/pretraining_documents.jsonl"
PREPARE_SCRIPT = PROJECT_ROOT / "scripts/prepare_pretraining_data.py"
PINNED_REVISION = "924804708136147342bc2390cd29abb1edaf0c67"


def test_production_source_uses_exact_immutable_revision() -> None:
    config = PretrainingDataConfig.load(
        PRODUCTION_CONFIG,
        project_root=PROJECT_ROOT,
    )

    assert config.dataset_name == "HuggingFaceFW/fineweb-edu"
    assert config.dataset_config == "sample-10BT"
    assert config.dataset_split == "train"
    assert config.dataset_revision == PINNED_REVISION
    assert config.streaming is True
    assert is_immutable_dataset_revision(config.dataset_revision)
    assert not is_immutable_dataset_revision("main")
    assert not is_immutable_dataset_revision(PINNED_REVISION[:-1])


def test_read_only_preflight_does_not_create_or_replace_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    config = PretrainingDataConfig.load(
        SMOKE_CONFIG,
        project_root=PROJECT_ROOT,
    ).with_overrides(
        output_directory=output,
        tokenizer_path=tmp_path / "tokenizer.json",
    )

    report = run_data_preflight(
        config,
        source_factory=lambda: iter([{"text": "bounded local document"}]),
        local_source=True,
        force=True,
        document_limit=1,
    )

    assert report["passed"]
    assert report["mode"] == "read_only"
    assert report["source_probe"]["documents_yielded"] == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not config.tokenizer_path.exists()


def test_connectivity_failure_is_separate_from_configuration_failure(
    tmp_path: Path,
) -> None:
    config = PretrainingDataConfig.load(
        PRODUCTION_CONFIG,
        project_root=PROJECT_ROOT,
    ).with_overrides(
        output_directory=tmp_path / "output",
        tokenizer_path=tmp_path / "tokenizer.json",
    )

    def unavailable() -> None:
        raise ConnectionError("offline test")

    report = run_data_preflight(
        config,
        source_factory=lambda: pytest.fail("source must not be opened"),
        local_source=False,
        force=False,
        connectivity_probe=unavailable,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["passed"]
    assert checks["configuration_parsed"]["status"] == "pass"
    assert checks["immutable_dataset_revision"]["status"] == "pass"
    assert checks["huggingface_connectivity"]["status"] == "fail"
    assert checks["bounded_source_probe"]["status"] == "skipped"
    assert report["source_probe"]["attempted"] is False


def test_revision_incompatibility_is_reported_without_main_fallback(
    tmp_path: Path,
) -> None:
    config = PretrainingDataConfig.load(
        PRODUCTION_CONFIG,
        project_root=PROJECT_ROOT,
    ).with_overrides(
        output_directory=tmp_path / "output",
        tokenizer_path=tmp_path / "tokenizer.json",
    )

    def incompatible_source() -> object:
        raise ValueError(f"revision {PINNED_REVISION} is unavailable")

    report = run_data_preflight(
        config,
        source_factory=incompatible_source,  # type: ignore[arg-type]
        local_source=False,
        force=False,
        connectivity_probe=lambda: None,
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["huggingface_connectivity"]["status"] == "pass"
    assert checks["bounded_source_probe"]["status"] == "fail"
    assert PINNED_REVISION in checks["bounded_source_probe"]["detail"]
    assert "fallback" in checks["bounded_source_probe"]["detail"]


def test_preflight_cli_with_local_source_creates_nothing(tmp_path: Path) -> None:
    output = tmp_path / "never-created"
    tokenizer = tmp_path / "never-created-tokenizer.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--config",
            str(SMOKE_CONFIG),
            "--local-jsonl",
            str(FIXTURE_JSONL),
            "--output-dir",
            str(output),
            "--tokenizer-path",
            str(tokenizer),
            "--preflight-only",
            "--preflight-documents",
            "2",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"mode": "read_only"' in completed.stdout
    assert not output.exists()
    assert not tokenizer.exists()


def test_qualification_fixture_is_deterministic_checksummed_and_compatible(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    options = {
        "seed": 77,
        "vocabulary_size": 16_384,
        "train_tokens": 4_096,
        "validation_tokens": 2_048,
        "tokens_per_shard": 2_048,
    }

    first_manifest = create_qualification_fixture(first, **options)
    second_manifest = create_qualification_fixture(second, **options)

    assert first_manifest == second_manifest
    assert first_manifest["source"]["synthetic"] is True
    assert first_manifest["qualification_fixture"]["quality_claims_permitted"] is False
    for split in ("train", "validation"):
        for entry in first_manifest["shards"][split]:
            assert sha256_file(first / entry["name"]) == entry["sha256"]
            assert sha256_file(second / entry["name"]) == entry["sha256"]
    dataset = MemoryMappedShardDataset(
        first / "manifest.json",
        split="train",
        block_size=1_024,
        expected_vocabulary_size=16_384,
        expected_tokenizer_fingerprint=first_manifest["tokenizer"]["fingerprint"],
        verify_checksums=True,
    )
    inputs, targets = dataset.get_batch(1)
    assert inputs.shape == targets.shape == (1, 1_024)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_qualification_fixture(first, **options)
