"""Character-level tokenization."""

import json
from pathlib import Path
from typing import Self


class CharTokenizer:
    """Map characters to deterministic integer token IDs."""

    def __init__(self, text: str) -> None:
        """Build a vocabulary from the sorted unique characters in text."""
        if not text:
            raise ValueError("Cannot build a vocabulary from empty text.")
        self._set_vocabulary(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Return the number of characters in the vocabulary."""
        return len(self._characters)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        try:
            return [self._char_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        characters: list[str] = []
        for token_id in token_ids:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.vocab_size
            ):
                raise ValueError(f"Invalid token ID: {token_id!r}")
            characters.append(self._characters[token_id])
        return "".join(characters)

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"characters": self._characters}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a vocabulary from JSON."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load tokenizer vocabulary from {source}.") from error

        characters = data.get("characters") if isinstance(data, dict) else None
        if (
            not isinstance(characters, list)
            or not characters
            or any(not isinstance(character, str) or len(character) != 1 for character in characters)
            or len(set(characters)) != len(characters)
        ):
            raise ValueError("Tokenizer JSON must contain unique, single-character entries.")

        tokenizer = cls.__new__(cls)
        tokenizer._set_vocabulary(characters)
        return tokenizer

    def _set_vocabulary(self, characters: list[str]) -> None:
        self._characters = list(characters)
        self._char_to_id = {
            character: token_id for token_id, character in enumerate(self._characters)
        }
