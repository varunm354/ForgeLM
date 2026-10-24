"""Tests for deterministic byte-level BPE tokenization."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from forgelm.data.bpe_tokenizer import BytePairTokenizer


@pytest.mark.parametrize("target_vocab_size", [0, 255, True])
def test_target_vocabulary_size_is_validated(
    target_vocab_size: int,
) -> None:
    with pytest.raises(ValueError, match="at least 256"):
        BytePairTokenizer(target_vocab_size)


def test_empty_training_corpus_keeps_base_vocabulary() -> None:
    tokenizer = BytePairTokenizer(300)

    tokenizer.train("")

    assert tokenizer.vocab_size == 256
    assert tokenizer.num_merges == 0


def test_training_is_deterministic() -> None:
    first = BytePairTokenizer(264)
    second = BytePairTokenizer(264)

    first.train("banana bandana banana")
    second.train("banana bandana banana")

    assert first.merges == second.merges
    assert first.encode("banana") == second.encode("banana")


def test_ties_use_lexicographic_token_id_order() -> None:
    tokenizer = BytePairTokenizer(257)

    tokenizer.train("abac")

    assert tokenizer.merges == [(ord("a"), ord("b"))]


def test_most_frequent_pair_is_selected() -> None:
    tokenizer = BytePairTokenizer(257)

    tokenizer.train("abcab")

    assert tokenizer.merges[0] == (ord("a"), ord("b"))


def test_learned_token_ids_are_sequential() -> None:
    tokenizer = BytePairTokenizer(258)

    tokenizer.train("abcabc")

    assert tokenizer.vocab_size == 258
    assert tokenizer.token_bytes(256) == b"ab"
    assert tokenizer.token_bytes(257) == b"abc"


def test_encoding_applies_merges_in_learned_order() -> None:
    tokenizer = BytePairTokenizer(258)
    tokenizer.train("abcabc")

    assert tokenizer.merges == [(ord("a"), ord("b")), (256, ord("c"))]
    assert tokenizer.encode("abc") == [257]


def test_repeated_adjacent_pairs_merge_left_to_right() -> None:
    tokenizer = BytePairTokenizer(257)
    tokenizer.train("aaaa")

    assert tokenizer.merges == [(ord("a"), ord("a"))]
    assert tokenizer.encode("aaaa") == [256, 256]


def test_merges_do_not_cross_pre_tokenization_boundaries() -> None:
    tokenizer = BytePairTokenizer(300)

    tokenizer.train("a a")

    assert tokenizer.num_merges == 0
    assert tokenizer.encode("a a") == [ord("a"), ord(" "), ord("a")]


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "line one\n\n  line two\t",
        "naïve café — 你好 🌍",
    ],
)
def test_round_trip(text: str) -> None:
    tokenizer = BytePairTokenizer(280)
    tokenizer.train(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unseen_unicode_uses_bytes_without_unknown_tokens() -> None:
    tokenizer = BytePairTokenizer(260)
    tokenizer.train("plain ascii training text")
    unseen = "Καλημέρα 🌍"

    token_ids = tokenizer.encode(unseen)

    assert all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids)
    assert tokenizer.decode(token_ids) == unseen


def test_empty_string_encode_and_decode() -> None:
    tokenizer = BytePairTokenizer()
    tokenizer.train("training text")

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


@pytest.mark.parametrize("token_ids", [[-1], [256], [True]])
def test_decode_rejects_invalid_token_ids(token_ids: list[int]) -> None:
    tokenizer = BytePairTokenizer(256)

    with pytest.raises(ValueError, match="Invalid token ID"):
        tokenizer.decode(token_ids)


def test_decode_rejects_bytes_that_are_not_valid_utf8() -> None:
    tokenizer = BytePairTokenizer(256)

    with pytest.raises(ValueError, match="valid UTF-8"):
        tokenizer.decode([255])


def test_training_stops_when_no_pairs_remain() -> None:
    tokenizer = BytePairTokenizer(1_000)

    tokenizer.train("x")

    assert tokenizer.vocab_size == 256


def test_json_save_load_preserves_encoding(tmp_path: Path) -> None:
    tokenizer = BytePairTokenizer(270)
    tokenizer.train("banana bandana\nbanana")
    path = tmp_path / "nested" / "bpe.json"

    tokenizer.save(path)
    loaded = BytePairTokenizer.load(path)

    assert loaded.target_vocab_size == tokenizer.target_vocab_size
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.merges == tokenizer.merges
    assert loaded.encode("banana unseen 🌍") == tokenizer.encode(
        "banana unseen 🌍"
    )


def test_saved_json_records_required_metadata(tmp_path: Path) -> None:
    tokenizer = BytePairTokenizer(257)
    tokenizer.train("abab")
    path = tmp_path / "bpe.json"

    tokenizer.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["format"] == "forgelm.byte_pair_tokenizer"
    assert data["version"] == 1
    assert data["base_vocabulary"]["size"] == 256
    assert data["effective_vocab_size"] == 257
    assert data["pre_tokenization"]
    assert data["merges"][0]["token_id"] == 256


def _make_out_of_order(data: dict[str, Any]) -> None:
    data["merges"][0]["token_id"] = 257


def _make_duplicate(data: dict[str, Any]) -> None:
    data["merges"][1]["left"] = data["merges"][0]["left"]
    data["merges"][1]["right"] = data["merges"][0]["right"]


def _make_inconsistent(data: dict[str, Any]) -> None:
    data["merges"][0]["bytes_hex"] = "00"


@pytest.mark.parametrize(
    "corrupt",
    [_make_out_of_order, _make_duplicate, _make_inconsistent],
)
def test_load_rejects_malformed_merge_data(
    tmp_path: Path,
    corrupt: Callable[[dict[str, Any]], None],
) -> None:
    tokenizer = BytePairTokenizer(258)
    tokenizer.train("abcabc")
    path = tmp_path / "bpe.json"
    tokenizer.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    corrupt(data)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        BytePairTokenizer.load(path)
