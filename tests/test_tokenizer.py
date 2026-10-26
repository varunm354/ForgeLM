"""Tests for character-level tokenization."""

import json
from pathlib import Path

import pytest

from forgelm.data.tokenizer import CharTokenizer


def test_encode_decode_round_trip() -> None:
    tokenizer = CharTokenizer("hello\n")
    text = "hello\nhello"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocabulary_is_sorted_and_deterministic() -> None:
    first = CharTokenizer("caba")
    second = CharTokenizer("abcc")

    assert first.vocab_size == 3
    assert first.encode("abc") == [0, 1, 2]
    assert first.encode("cab") == second.encode("cab")


def test_encode_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer("abc")

    with pytest.raises(ValueError, match="Unknown character: 'z'"):
        tokenizer.encode("abz")


def test_decode_rejects_invalid_token_id() -> None:
    tokenizer = CharTokenizer("abc")

    with pytest.raises(ValueError, match="Invalid token ID"):
        tokenizer.decode([3])


def test_token_byte_length_handles_ascii_and_unicode() -> None:
    tokenizer = CharTokenizer("aé🌍")
    token_ids = tokenizer.encode("aé🌍")

    assert [tokenizer.token_byte_length(token_id) for token_id in token_ids] == [
        1,
        2,
        4,
    ]

    with pytest.raises(ValueError, match="Invalid token ID"):
        tokenizer.token_byte_length(tokenizer.vocab_size)


def test_save_and_load_preserve_vocabulary(tmp_path: Path) -> None:
    tokenizer = CharTokenizer("b\na")
    path = tmp_path / "nested" / "tokenizer.json"

    tokenizer.save(path)
    loaded = CharTokenizer.load(path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "characters": ["\n", "a", "b"]
    }
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("ab\n") == tokenizer.encode("ab\n")
    assert loaded.decode([0, 1, 2]) == "\nab"


def test_tokenizer_rejects_empty_text_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty text"):
        CharTokenizer("")

    path = tmp_path / "tokenizer.json"
    path.write_text('{"characters": ["aa"]}', encoding="utf-8")

    with pytest.raises(ValueError, match="single-character"):
        CharTokenizer.load(path)
