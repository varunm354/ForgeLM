"""Offline integration tests for pretraining-data configuration and CLI."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer
from forgelm.data.token_shards import MemoryMappedShardDataset, load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs/pretraining_data_smoke.yaml"
FIXTURE = PROJECT_ROOT / "tests/fixtures/pretraining_documents.jsonl"
SCRIPT = PROJECT_ROOT / "scripts/prepare_pretraining_data.py"


def test_configuration_loads_paths_and_validates_uint16_limit() -> None:
    config = PretrainingDataConfig.load(
        SMOKE_CONFIG,
        project_root=PROJECT_ROOT,
    )

    assert config.output_directory.is_absolute()
    assert config.tokenizer_path.is_absolute()
    assert config.tokenizer_vocab_size == 320
    with pytest.raises(ValueError, match="between 257 and 65536"):
        config.with_overrides(tokenizer_vocab_size=65_537)


def test_offline_cli_builds_reloadable_shards_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    tokenizer_path = tmp_path / "tokenizer.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(SMOKE_CONFIG),
        "--local-jsonl",
        str(FIXTURE),
        "--output-dir",
        str(output),
        "--tokenizer-path",
        str(tokenizer_path),
        "--retrain-tokenizer",
        "--force",
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = output / "manifest.json"
    manifest = load_manifest(manifest_path)
    tokenizer = ScalableByteLevelTokenizer.load(
        tokenizer_path,
        end_of_document_token="<|endoftext|>",
        expected_vocab_size=320,
    )
    dataset = MemoryMappedShardDataset(
        manifest_path,
        split="train",
        block_size=16,
        expected_vocabulary_size=tokenizer.vocab_size,
        expected_tokenizer_fingerprint=tokenizer.fingerprint(),
        verify_checksums=True,
    )
    inputs, targets = dataset.get_batch(
        2,
        generator=torch.Generator().manual_seed(1337),
    )

    assert "tokenizer vocabulary: 320" in completed.stdout
    assert manifest["counts"]["train_tokens"] == 512
    assert manifest["counts"]["validation_tokens"] == 192
    assert len(manifest["shards"]["train"]) == 4
    assert len(manifest["shards"]["validation"]) == 2
    assert inputs.shape == targets.shape == (2, 16)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])

    refused = subprocess.run(
        command[:-2],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "Refusing to overwrite" in refused.stderr
