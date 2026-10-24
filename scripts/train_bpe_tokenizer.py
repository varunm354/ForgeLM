"""Train and inspect ForgeLM's byte-level BPE tokenizer."""

import argparse
import time
from pathlib import Path

from forgelm.data.bpe_tokenizer import BASE_VOCAB_SIZE, BytePairTokenizer
from forgelm.data.tokenizer import CharTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/tiny_shakespeare.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/tokenizers/tiny_shakespeare_bpe.json"


def parse_args() -> argparse.Namespace:
    """Parse tokenizer training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="UTF-8 training corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination tokenizer JSON",
    )
    parser.add_argument(
        "--vocab-size",
        type=_vocabulary_size,
        default=512,
        help="target vocabulary size (minimum: 256)",
    )
    return parser.parse_args()


def _vocabulary_size(value: str) -> int:
    parsed = int(value)
    if parsed < BASE_VOCAB_SIZE:
        raise argparse.ArgumentTypeError("vocabulary size must be at least 256")
    return parsed


def readable_bytes(value: bytes) -> str:
    """Represent token bytes as UTF-8 text when possible, otherwise hexadecimal."""
    try:
        return repr(value.decode("utf-8"))
    except UnicodeDecodeError:
        return f"0x{value.hex()}"


def main() -> int:
    """Train, save, and summarize a byte-level BPE tokenizer."""
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    byte_count = len(text.encode("utf-8"))
    character_token_count = len(CharTokenizer(text).encode(text))

    tokenizer = BytePairTokenizer(args.vocab_size)
    started = time.perf_counter()
    tokenizer.train(text)
    training_seconds = time.perf_counter() - started
    bpe_tokens = tokenizer.encode(text)
    tokenizer.save(output_path)

    bpe_token_count = len(bpe_tokens)
    bytes_per_token = byte_count / bpe_token_count if bpe_token_count else 0.0
    reduction = (
        (character_token_count - bpe_token_count)
        / character_token_count
        * 100
        if character_token_count
        else 0.0
    )
    round_trip = tokenizer.decode(bpe_tokens) == text

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"corpus bytes: {byte_count:,}")
    print(f"corpus characters: {len(text):,}")
    print(f"requested vocabulary size: {args.vocab_size:,}")
    print(f"achieved vocabulary size: {tokenizer.vocab_size:,}")
    print(f"learned merges: {tokenizer.num_merges:,}")
    print(f"character tokens: {character_token_count:,}")
    print(f"BPE tokens: {bpe_token_count:,}")
    print(f"bytes per BPE token: {bytes_per_token:.4f}")
    print(f"token-count reduction vs characters: {reduction:.2f}%")
    print(f"training runtime: {training_seconds:.2f} seconds")
    print(f"round-trip verified: {round_trip}")
    print("\nFirst 20 learned merges:")
    for rank, (left, right) in enumerate(tokenizer.merges[:20]):
        token_id = BASE_VOCAB_SIZE + rank
        print(
            f"{token_id}: "
            f"{left} {readable_bytes(tokenizer.token_bytes(left))} + "
            f"{right} {readable_bytes(tokenizer.token_bytes(right))} -> "
            f"{readable_bytes(tokenizer.token_bytes(token_id))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
