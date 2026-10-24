"""Deterministic byte-level byte-pair encoding."""

import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

BASE_VOCAB_SIZE = 256
FORMAT_NAME = "forgelm.byte_pair_tokenizer"
FORMAT_VERSION = 1
PRE_TOKENIZATION_POLICY = r"regex:\s+|\S+; merges do not cross chunk boundaries"
_CHUNK_PATTERN = re.compile(r"\s+|\S+")


class BytePairTokenizer:
    """Learn and apply deterministic merges over UTF-8 bytes."""

    def __init__(self, target_vocab_size: int = 512) -> None:
        if (
            not isinstance(target_vocab_size, int)
            or isinstance(target_vocab_size, bool)
            or target_vocab_size < BASE_VOCAB_SIZE
        ):
            raise ValueError("target_vocab_size must be at least 256.")

        self.target_vocab_size = target_vocab_size
        self.merges: list[tuple[int, int]] = []
        self._reset_token_bytes()

    @property
    def vocab_size(self) -> int:
        """Return the effective vocabulary size."""
        return BASE_VOCAB_SIZE + len(self.merges)

    @property
    def num_merges(self) -> int:
        """Return the number of learned merges."""
        return len(self.merges)

    def train(self, text: str) -> None:
        """Learn frequent adjacent-token merges from text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        self.merges = []
        self._reset_token_bytes()
        sequence_counts = self._initial_sequence_counts(text)

        while self.vocab_size < self.target_vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in sequence_counts.items():
                for pair in pairwise(sequence):
                    pair_counts[pair] += frequency

            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (
                    -pair_counts[pair],
                    pair[0],
                    pair[1],
                ),
            )
            new_token_id = self.vocab_size
            self.merges.append(best_pair)
            self._token_bytes[new_token_id] = (
                self._token_bytes[best_pair[0]]
                + self._token_bytes[best_pair[1]]
            )

            merged_counts: Counter[tuple[int, ...]] = Counter()
            for sequence, frequency in sequence_counts.items():
                merged = tuple(_merge_pair(sequence, best_pair, new_token_id))
                merged_counts[merged] += frequency
            sequence_counts = merged_counts

    def encode(self, text: str) -> list[int]:
        """Encode arbitrary UTF-8 text without an unknown token."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        merge_ranks = {
            pair: (rank, BASE_VOCAB_SIZE + rank)
            for rank, pair in enumerate(self.merges)
        }
        encoded_chunks: dict[str, tuple[int, ...]] = {}
        encoded: list[int] = []
        for chunk in _pre_tokenize(text):
            chunk_tokens = encoded_chunks.get(chunk)
            if chunk_tokens is None:
                chunk_tokens = tuple(
                    _encode_chunk(chunk.encode("utf-8"), merge_ranks)
                )
                encoded_chunks[chunk] = chunk_tokens
            encoded.extend(chunk_tokens)
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into their exact UTF-8 text."""
        byte_parts: list[bytes] = []
        for token_id in token_ids:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id not in self._token_bytes
            ):
                raise ValueError(f"Invalid token ID: {token_id!r}")
            byte_parts.append(self._token_bytes[token_id])

        try:
            return b"".join(byte_parts).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Token IDs do not form valid UTF-8 text.") from error

    def token_bytes(self, token_id: int) -> bytes:
        """Return the byte sequence represented by a token."""
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id not in self._token_bytes
        ):
            raise ValueError(f"Invalid token ID: {token_id!r}")
        return self._token_bytes[token_id]

    def save(self, path: str | Path) -> None:
        """Save vocabulary metadata and ordered merges as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        merge_data = []
        for rank, (left, right) in enumerate(self.merges):
            token_id = BASE_VOCAB_SIZE + rank
            merge_data.append(
                {
                    "token_id": token_id,
                    "left": left,
                    "right": right,
                    "bytes_hex": self._token_bytes[token_id].hex(),
                }
            )

        data = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "base_vocabulary": {
                "type": "all_byte_values",
                "size": BASE_VOCAB_SIZE,
                "token_id_range": [0, BASE_VOCAB_SIZE - 1],
            },
            "pre_tokenization": PRE_TOKENIZATION_POLICY,
            "target_vocab_size": self.target_vocab_size,
            "effective_vocab_size": self.vocab_size,
            "merges": merge_data,
        }
        destination.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and validate a saved byte-level BPE vocabulary."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load BPE tokenizer from {source}.") from error
        return cls._from_data(data)

    @classmethod
    def _from_data(cls, data: Any) -> Self:
        if not isinstance(data, dict):
            raise TypeError("BPE tokenizer JSON must contain an object.")
        if data.get("format") != FORMAT_NAME or data.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported BPE tokenizer format or version.")
        if data.get("base_vocabulary") != {
            "type": "all_byte_values",
            "size": BASE_VOCAB_SIZE,
            "token_id_range": [0, BASE_VOCAB_SIZE - 1],
        }:
            raise ValueError("Invalid base vocabulary description.")
        if data.get("pre_tokenization") != PRE_TOKENIZATION_POLICY:
            raise ValueError("Unsupported pre-tokenization policy.")

        target_vocab_size = data.get("target_vocab_size")
        try:
            tokenizer = cls(target_vocab_size)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid target vocabulary size.") from error

        merges = data.get("merges")
        if not isinstance(merges, list):
            raise TypeError("merges must be an ordered list.")
        if len(merges) > target_vocab_size - BASE_VOCAB_SIZE:
            raise ValueError("Merge count exceeds the target vocabulary size.")

        seen_pairs: set[tuple[int, int]] = set()
        for rank, merge in enumerate(merges):
            expected_token_id = BASE_VOCAB_SIZE + rank
            if not isinstance(merge, dict):
                raise TypeError("Each merge must be an object.")
            token_id = merge.get("token_id")
            left = merge.get("left")
            right = merge.get("right")
            bytes_hex = merge.get("bytes_hex")
            if token_id != expected_token_id:
                raise ValueError("Merge token IDs must be sequential and ordered.")
            if not _is_prior_token_id(left, expected_token_id) or not _is_prior_token_id(
                right,
                expected_token_id,
            ):
                raise ValueError("Merge inputs must reference earlier token IDs.")
            pair = (left, right)
            if pair in seen_pairs:
                raise ValueError("Duplicate merge pair.")
            seen_pairs.add(pair)

            expected_bytes = (
                tokenizer._token_bytes[left] + tokenizer._token_bytes[right]
            )
            if not isinstance(bytes_hex, str):
                raise TypeError("Merge byte sequences must be hexadecimal strings.")
            try:
                saved_bytes = bytes.fromhex(bytes_hex)
            except ValueError as error:
                raise ValueError("Invalid hexadecimal merge byte sequence.") from error
            if saved_bytes != expected_bytes:
                raise ValueError("Inconsistent merge byte sequence.")

            tokenizer.merges.append(pair)
            tokenizer._token_bytes[token_id] = expected_bytes

        if data.get("effective_vocab_size") != tokenizer.vocab_size:
            raise ValueError("Inconsistent effective vocabulary size.")
        return tokenizer

    def _initial_sequence_counts(
        self,
        text: str,
    ) -> Counter[tuple[int, ...]]:
        chunk_frequencies = Counter(_pre_tokenize(text))
        sequence_counts: Counter[tuple[int, ...]] = Counter()
        for chunk, frequency in chunk_frequencies.items():
            sequence_counts[tuple(chunk.encode("utf-8"))] += frequency
        return sequence_counts

    def _reset_token_bytes(self) -> None:
        self._token_bytes = {
            token_id: bytes([token_id])
            for token_id in range(BASE_VOCAB_SIZE)
        }


def _pre_tokenize(text: str) -> list[str]:
    chunks = _CHUNK_PATTERN.findall(text)
    if "".join(chunks) != text:
        raise RuntimeError("Pre-tokenization did not preserve the input text.")
    return chunks


def _merge_pair(
    sequence: tuple[int, ...] | list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


def _encode_chunk(
    chunk: bytes,
    merge_ranks: dict[tuple[int, int], tuple[int, int]],
) -> list[int]:
    token_ids = list(chunk)
    while len(token_ids) > 1:
        selected_pair: tuple[int, int] | None = None
        selected_rank: int | None = None
        selected_token_id = -1
        for pair in pairwise(token_ids):
            ranked_merge = merge_ranks.get(pair)
            if ranked_merge is not None and (
                selected_rank is None or ranked_merge[0] < selected_rank
            ):
                selected_rank, selected_token_id = ranked_merge
                selected_pair = pair
        if selected_pair is None:
            break
        token_ids = _merge_pair(token_ids, selected_pair, selected_token_id)
    return token_ids


def _is_prior_token_id(value: Any, upper_bound: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < upper_bound
    )
