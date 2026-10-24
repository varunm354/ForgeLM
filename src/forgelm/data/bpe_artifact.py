"""Loading and deterministic training for BPE tokenizer artifacts."""

from pathlib import Path
from typing import Literal

from forgelm.data.bpe_tokenizer import BytePairTokenizer

TokenizerSource = Literal["trained", "loaded"]


def load_or_train_bpe_tokenizer(
    training_text: str,
    path: str | Path,
    *,
    target_vocab_size: int,
    retrain: bool = False,
) -> tuple[BytePairTokenizer, TokenizerSource]:
    """Load a compatible tokenizer or train and save a replacement."""
    artifact_path = Path(path)
    if artifact_path.exists() and not retrain:
        tokenizer = BytePairTokenizer.load(artifact_path)
        _validate_vocabulary(tokenizer, target_vocab_size)
        return tokenizer, "loaded"

    tokenizer = BytePairTokenizer(target_vocab_size)
    tokenizer.train(training_text)
    _validate_vocabulary(tokenizer, target_vocab_size)
    tokenizer.save(artifact_path)
    return tokenizer, "trained"


def _validate_vocabulary(
    tokenizer: BytePairTokenizer,
    expected_vocab_size: int,
) -> None:
    if tokenizer.target_vocab_size != expected_vocab_size:
        raise ValueError(
            "Tokenizer target vocabulary size "
            f"{tokenizer.target_vocab_size} does not match configured size "
            f"{expected_vocab_size}."
        )
    if tokenizer.vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer achieved vocabulary size {tokenizer.vocab_size} does "
            f"not match model vocabulary size {expected_vocab_size}."
        )
