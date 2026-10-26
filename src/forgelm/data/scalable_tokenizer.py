"""Rust-backed byte-level BPE for large-corpus preparation.

ForgeLM keeps its small from-scratch ``BytePairTokenizer`` for teaching the
algorithm. This wrapper is a separate production path backed by Hugging Face
Tokenizers, whose Rust trainer and encoder scale to large iterators.
"""

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

MINIMUM_VOCABULARY_SIZE = 257


class ScalableByteLevelTokenizer:
    """Validated byte-level BPE tokenizer with stable artifact identity."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        end_of_document_token: str,
        configured_vocab_size: int,
    ) -> None:
        self._tokenizer = tokenizer
        self.end_of_document_token = end_of_document_token
        self.configured_vocab_size = configured_vocab_size
        self._validate()

    @classmethod
    def train_from_iterator(
        cls,
        documents: Iterable[str],
        *,
        vocab_size: int,
        end_of_document_token: str,
        length: int | None = None,
    ) -> Self:
        """Train deterministic byte-level BPE from a restartable or streaming input."""
        _validate_training_options(vocab_size, end_of_document_token)
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=False,
            special_tokens=[end_of_document_token],
            initial_alphabet=ByteLevel.alphabet(),
        )

        def checked_documents() -> Iterable[str]:
            for document in documents:
                if not isinstance(document, str):
                    raise TypeError("Tokenizer training documents must be strings.")
                yield document

        tokenizer.train_from_iterator(
            checked_documents(),
            trainer=trainer,
            length=length,
        )
        trained = cls(
            tokenizer,
            end_of_document_token=end_of_document_token,
            configured_vocab_size=vocab_size,
        )
        if trained.vocab_size != vocab_size:
            raise ValueError(
                f"Tokenizer training produced {trained.vocab_size} tokens, "
                f"but {vocab_size} were requested. Supply more training text."
            )
        return trained

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size including the EOD special token."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def end_of_document_id(self) -> int:
        """Return the validated EOD token ID."""
        token_id = self._tokenizer.token_to_id(self.end_of_document_token)
        if token_id is None:
            raise RuntimeError("The validated EOD token is no longer present.")
        return token_id

    def encode(self, text: str) -> list[int]:
        """Encode valid Unicode text with complete byte fallback."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode a batch while preserving input ordering."""
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All batch items must be strings.")
        return [
            encoding.ids
            for encoding in self._tokenizer.encode_batch(
                list(texts),
                add_special_tokens=False,
            )
        ]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode IDs exactly, including literal occurrences of the EOD token."""
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in token_ids
        ):
            raise TypeError("token_ids must contain integers.")
        try:
            return self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
            )
        except Exception as error:
            raise ValueError("Could not decode token IDs.") from error

    def fingerprint(self) -> str:
        """Return SHA-256 over a canonical representation of the tokenizer JSON."""
        canonical = json.dumps(
            json.loads(self._tokenizer.to_str()),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        """Atomically save the native Tokenizers JSON artifact."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists():
            raise FileExistsError(
                f"Refusing to overwrite partial tokenizer file: {temporary}"
            )
        try:
            temporary.write_text(self._tokenizer.to_str(pretty=True) + "\n", "utf-8")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        end_of_document_token: str,
        expected_vocab_size: int | None = None,
    ) -> Self:
        """Load a native JSON artifact and validate its byte-level contract."""
        source = Path(path)
        try:
            tokenizer = Tokenizer.from_file(str(source))
        except Exception as error:
            raise ValueError(f"Could not load scalable tokenizer from {source}.") from error
        configured_size = (
            tokenizer.get_vocab_size(with_added_tokens=True)
            if expected_vocab_size is None
            else expected_vocab_size
        )
        loaded = cls(
            tokenizer,
            end_of_document_token=end_of_document_token,
            configured_vocab_size=configured_size,
        )
        if expected_vocab_size is not None and loaded.vocab_size != expected_vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size {loaded.vocab_size} does not match "
                f"configured size {expected_vocab_size}."
            )
        return loaded

    def _validate(self) -> None:
        _validate_training_options(
            self.configured_vocab_size,
            self.end_of_document_token,
        )
        data: dict[str, Any] = json.loads(self._tokenizer.to_str())
        model = data.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError("Tokenizer model must be BPE.")
        if model.get("unk_token") is not None:
            raise ValueError("Byte-level tokenizer must not define an unknown token.")

        pre_tokenizer = data.get("pre_tokenizer")
        if (
            not isinstance(pre_tokenizer, dict)
            or pre_tokenizer.get("type") != "ByteLevel"
            or pre_tokenizer.get("add_prefix_space") is not False
        ):
            raise ValueError("Tokenizer must use ByteLevel pre-tokenization.")
        decoder = data.get("decoder")
        if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
            raise ValueError("Tokenizer must use ByteLevel decoding.")

        vocabulary = self._tokenizer.get_vocab(with_added_tokens=True)
        missing_alphabet = set(ByteLevel.alphabet()) - vocabulary.keys()
        if missing_alphabet:
            raise ValueError("Tokenizer vocabulary does not contain every byte value.")
        if self.vocab_size != self.configured_vocab_size:
            raise ValueError(
                "Tokenizer effective vocabulary does not match its configured size."
            )

        matching_tokens = [
            token
            for token in data.get("added_tokens", [])
            if token.get("content") == self.end_of_document_token
        ]
        if len(matching_tokens) != 1 or matching_tokens[0].get("special") is not True:
            raise ValueError("EOD token must exist exactly once as a special token.")


def _validate_training_options(vocab_size: int, end_of_document_token: str) -> None:
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < MINIMUM_VOCABULARY_SIZE
    ):
        raise ValueError("vocab_size must be an integer of at least 257.")
    if not isinstance(end_of_document_token, str) or not end_of_document_token:
        raise ValueError("end_of_document_token must be a non-empty string.")
