"""Tests for the Rust-backed production byte-level BPE wrapper."""

from pathlib import Path

from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer

EOD = "<|endoftext|>"


def _train(vocab_size: int = 280) -> ScalableByteLevelTokenizer:
    documents = [
        "synthetic alpha beta gamma delta " * 8,
        "synthetic epsilon zeta eta theta " * 8,
        "byte fallback training text with repeated patterns " * 8,
    ]
    return ScalableByteLevelTokenizer.train_from_iterator(
        documents,
        vocab_size=vocab_size,
        end_of_document_token=EOD,
        length=len(documents),
    )


def test_unicode_round_trip_has_no_unknown_token_requirement() -> None:
    tokenizer = _train()
    text = "naïve café — 你好 🌍\nΚαλημέρα"

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids)


def test_literal_eod_text_round_trips_and_special_token_is_explicit() -> None:
    tokenizer = _train()
    text = f"before {EOD} after"

    assert tokenizer.end_of_document_id == tokenizer.encode(EOD)[0]
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_load_preserves_encoding_and_stable_fingerprint(
    tmp_path: Path,
) -> None:
    tokenizer = _train()
    path = tmp_path / "nested" / "tokenizer.json"

    tokenizer.save(path)
    loaded = ScalableByteLevelTokenizer.load(
        path,
        end_of_document_token=EOD,
        expected_vocab_size=tokenizer.vocab_size,
    )

    assert loaded.fingerprint() == tokenizer.fingerprint()
    assert loaded.encode("unseen text 🌍") == tokenizer.encode("unseen text 🌍")
    assert loaded.decode(loaded.encode("unseen text 🌍")) == "unseen text 🌍"


def test_training_and_batch_encoding_are_deterministic() -> None:
    first = _train()
    second = _train()
    texts = ["alpha beta", "gamma delta", "café"]

    assert first.fingerprint() == second.fingerprint()
    assert first.encode_batch(texts) == [first.encode(text) for text in texts]
    assert first.encode_batch(texts) == second.encode_batch(texts)
