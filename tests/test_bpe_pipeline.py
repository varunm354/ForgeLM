"""Tests for BPE artifacts and Transformer integration."""

from pathlib import Path

import pytest
import torch

from forgelm.data.bpe_artifact import load_or_train_bpe_tokenizer
from forgelm.data.dataset import train_validation_text_split
from forgelm.models.transformer_lm import TransformerLanguageModel
from forgelm.training.checkpoint import load_checkpoint, save_checkpoint


def test_tokenizer_training_uses_only_training_text(tmp_path: Path) -> None:
    training_text, validation_text = train_validation_text_split(
        "aaaazzzz",
        train_fraction=0.5,
    )

    tokenizer, source = load_or_train_bpe_tokenizer(
        training_text,
        tmp_path / "tokenizer.json",
        target_vocab_size=257,
    )

    assert source == "trained"
    assert validation_text == "zzzz"
    assert tokenizer.merges == [(ord("a"), ord("a"))]
    assert (ord("z"), ord("z")) not in tokenizer.merges


def test_missing_artifact_is_trained_saved_and_deterministic(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first, first_source = load_or_train_bpe_tokenizer(
        "abcabc",
        first_path,
        target_vocab_size=258,
    )
    second, second_source = load_or_train_bpe_tokenizer(
        "abcabc",
        second_path,
        target_vocab_size=258,
    )

    assert first_source == second_source == "trained"
    assert first_path.exists()
    assert second_path.exists()
    assert first.fingerprint() == second.fingerprint()


def test_existing_artifact_is_loaded_without_retraining(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    trained, _ = load_or_train_bpe_tokenizer(
        "abcabc",
        path,
        target_vocab_size=258,
    )

    loaded, source = load_or_train_bpe_tokenizer(
        "xyxyxy",
        path,
        target_vocab_size=258,
    )

    assert source == "loaded"
    assert loaded.fingerprint() == trained.fingerprint()


def test_retrain_replaces_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    original, _ = load_or_train_bpe_tokenizer(
        "abcabc",
        path,
        target_vocab_size=258,
    )

    replacement, source = load_or_train_bpe_tokenizer(
        "xyxyxy",
        path,
        target_vocab_size=258,
        retrain=True,
    )

    assert source == "trained"
    assert replacement.fingerprint() != original.fingerprint()
    reloaded, _ = load_or_train_bpe_tokenizer(
        "",
        path,
        target_vocab_size=258,
    )
    assert reloaded.fingerprint() == replacement.fingerprint()


def test_loaded_tokenizer_must_match_model_vocabulary(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    load_or_train_bpe_tokenizer(
        "abab",
        path,
        target_vocab_size=257,
    )

    with pytest.raises(ValueError, match="does not match configured size"):
        load_or_train_bpe_tokenizer(
            "abcabc",
            path,
            target_vocab_size=258,
        )


def test_bpe_transformer_forward_loss_and_generation(tmp_path: Path) -> None:
    tokenizer, _ = load_or_train_bpe_tokenizer(
        "abcabc",
        tmp_path / "tokenizer.json",
        target_vocab_size=258,
    )
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        block_size=8,
        num_heads=3,
        num_layers=1,
    )
    token_ids = torch.tensor([tokenizer.encode("abcabc")])
    targets = torch.roll(token_ids, shifts=-1, dims=1)

    logits, loss = model(token_ids, targets)

    assert logits.shape == (1, token_ids.shape[1], tokenizer.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)

    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[ord("a")] = 100
    model.eval()
    prefix = torch.tensor([tokenizer.encode("a")])
    generated = model.generate(prefix, max_new_tokens=3, top_k=1)
    assert tokenizer.decode(generated[0].tolist()) == "aaaa"


def test_bpe_checkpoint_metadata_and_compatibility(tmp_path: Path) -> None:
    tokenizer, _ = load_or_train_bpe_tokenizer(
        "abcabc",
        tmp_path / "tokenizer.json",
        target_vocab_size=258,
    )
    fingerprint = tokenizer.fingerprint()
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        block_size=8,
        num_heads=3,
        num_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "checkpoint.pt"
    metadata = {
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": fingerprint,
        "tokenizer_path": "tokenizer.json",
        "model_config": {"vocab_size": tokenizer.vocab_size},
        "step": 4,
        "validation_loss": 2.5,
    }
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=4,
        config={"tokenizer_vocab_size": tokenizer.vocab_size},
        metrics={"validation": {"loss": 2.5}},
        metadata=metadata,
    )

    restored_model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        block_size=8,
        num_heads=3,
        num_layers=1,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        restored_model,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_fingerprint=fingerprint,
    )

    assert checkpoint["metadata"] == metadata

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_checkpoint(
            checkpoint_path,
            restored_model,
            tokenizer_vocab_size=tokenizer.vocab_size,
            tokenizer_fingerprint="different",
        )

    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_checkpoint(
            checkpoint_path,
            restored_model,
            tokenizer_vocab_size=tokenizer.vocab_size + 1,
            tokenizer_fingerprint=fingerprint,
        )
