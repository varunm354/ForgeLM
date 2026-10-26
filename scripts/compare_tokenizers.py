"""Compare character and BPE Transformers with exact bits per byte."""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from forgelm.data.bpe_tokenizer import BytePairTokenizer
from forgelm.data.dataset import (
    train_validation_split,
    train_validation_text_split,
)
from forgelm.data.tokenizer import CharTokenizer
from forgelm.models.transformer_lm import TransformerLanguageModel
from forgelm.training.checkpoint import load_checkpoint
from forgelm.training.evaluate import (
    ByteNormalizedEvaluation,
    evaluate_full_corpus_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/tiny_shakespeare.txt"
CHAR_CONFIG_PATH = PROJECT_ROOT / "configs/transformer.yaml"
BPE_CONFIG_PATH = PROJECT_ROOT / "configs/transformer_bpe.yaml"
CHAR_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/transformer/best.pt"
BPE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/transformer_bpe/best.pt"


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="evaluation device (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=32,
        help="number of contiguous windows per evaluation batch",
    )
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return config


def require_file(path: Path, description: str) -> None:
    """Raise a useful error when a benchmark input is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def select_device(name: str) -> torch.device:
    """Validate and return an explicitly selected evaluation device."""
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return torch.device(name)


def build_model(config: dict[str, Any], vocab_size: int) -> TransformerLanguageModel:
    """Construct a Transformer from its saved training configuration."""
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        block_size=config["block_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )


def parameter_count(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def print_metrics(
    name: str,
    model: torch.nn.Module,
    validation_token_count: int,
    result: ByteNormalizedEvaluation,
) -> None:
    """Print one model's deterministic byte-normalized metrics."""
    print(name)
    print(f"  parameters: {parameter_count(model):,}")
    print(f"  validation tokens: {validation_token_count:,}")
    print(f"  evaluated target tokens: {result.evaluated_target_tokens:,}")
    print(f"  evaluated target bytes: {result.evaluated_target_bytes:,}")
    print(f"  mean token loss: {result.mean_token_loss:.9f}")
    print(f"  token perplexity: {result.token_perplexity:.9f}")
    print(
        "  total negative log-likelihood: "
        f"{result.total_negative_log_likelihood:.9f}"
    )
    print(f"  bits per byte: {result.bits_per_byte:.9f}")


def main() -> int:
    """Load both checkpoints and compare exact validation bits per byte."""
    args = parse_args()
    char_config = load_yaml(CHAR_CONFIG_PATH)
    bpe_config = load_yaml(BPE_CONFIG_PATH)
    tokenizer_path = PROJECT_ROOT / bpe_config["tokenizer_path"]
    for path, description in (
        (DATA_PATH, "Tiny Shakespeare dataset"),
        (CHAR_CHECKPOINT_PATH, "character Transformer checkpoint"),
        (BPE_CHECKPOINT_PATH, "BPE Transformer checkpoint"),
        (tokenizer_path, "train-only BPE tokenizer"),
    ):
        require_file(path, description)

    raw_text = DATA_PATH.read_text(encoding="utf-8")
    _, raw_validation_text = train_validation_text_split(raw_text, 0.9)

    char_tokenizer = CharTokenizer(raw_text)
    all_char_tokens = torch.tensor(
        char_tokenizer.encode(raw_text),
        dtype=torch.long,
    )
    _, char_validation_tokens = train_validation_split(all_char_tokens, 0.9)

    bpe_tokenizer = BytePairTokenizer.load(tokenizer_path)
    if bpe_tokenizer.vocab_size != bpe_config["tokenizer_vocab_size"]:
        raise ValueError(
            "BPE tokenizer vocabulary does not match transformer_bpe.yaml."
        )
    bpe_validation_tokens = torch.tensor(
        bpe_tokenizer.encode(raw_validation_text),
        dtype=torch.long,
    )
    tokenizer_fingerprint = bpe_tokenizer.fingerprint()

    char_validation_matches = (
        char_tokenizer.decode(char_validation_tokens.tolist())
        == raw_validation_text
    )
    bpe_validation_matches = (
        bpe_tokenizer.decode(bpe_validation_tokens.tolist())
        == raw_validation_text
    )
    same_raw_validation_text = char_validation_matches and bpe_validation_matches
    if not same_raw_validation_text:
        raise ValueError("Tokenizers did not reconstruct the same validation text.")

    device = select_device(args.device)
    char_model = build_model(char_config, char_tokenizer.vocab_size)
    bpe_model = build_model(bpe_config, bpe_tokenizer.vocab_size)
    char_checkpoint = load_checkpoint(
        CHAR_CHECKPOINT_PATH,
        char_model,
        device=device,
    )
    bpe_checkpoint = load_checkpoint(
        BPE_CHECKPOINT_PATH,
        bpe_model,
        device=device,
        tokenizer_vocab_size=bpe_tokenizer.vocab_size,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )

    char_result = evaluate_full_corpus_bytes(
        char_model,
        char_validation_tokens,
        char_tokenizer.token_byte_length,
        block_size=char_config["block_size"],
        batch_size=args.batch_size,
    )
    bpe_result = evaluate_full_corpus_bytes(
        bpe_model,
        bpe_validation_tokens,
        bpe_tokenizer.token_byte_length,
        block_size=bpe_config["block_size"],
        batch_size=args.batch_size,
    )

    bpb_difference = bpe_result.bits_per_byte - char_result.bits_per_byte
    bpb_change_percent = (
        (char_result.bits_per_byte - bpe_result.bits_per_byte)
        / char_result.bits_per_byte
        * 100
    )
    byte_count_difference = (
        bpe_result.evaluated_target_bytes - char_result.evaluated_target_bytes
    )
    change_label = "improvement" if bpb_change_percent >= 0 else "regression"

    print("Exact full-corpus tokenizer-normalized Transformer benchmark")
    print(f"device: {device}")
    print(f"same raw validation text: {same_raw_validation_text}")
    print(
        f"character checkpoint loaded: step {char_checkpoint['step']} "
        f"({CHAR_CHECKPOINT_PATH})"
    )
    print(
        f"BPE checkpoint loaded: step {bpe_checkpoint['step']} "
        f"({BPE_CHECKPOINT_PATH})"
    )
    print(f"BPE tokenizer fingerprint: {tokenizer_fingerprint}")
    print()
    print_metrics(
        "Character Transformer",
        char_model,
        len(char_validation_tokens),
        char_result,
    )
    print()
    print_metrics(
        "BPE Transformer",
        bpe_model,
        len(bpe_validation_tokens),
        bpe_result,
    )
    print()
    print(f"BPE minus character bits per byte: {bpb_difference:+.9f}")
    print(f"BPE bits-per-byte {change_label}: {abs(bpb_change_percent):.4f}%")
    print(f"scored-byte-count difference (BPE - character): {byte_count_difference:+d}")
    if byte_count_difference:
        print(
            "The scored-byte difference comes from each tokenizer's first "
            "unscored context token."
        )
    else:
        print(
            "Both initial unscored context tokens represent one byte, so "
            "the scored byte counts match."
        )
    print("Token perplexity is not comparable across different tokenizers.")
    print(
        "Context resets at non-overlapping 128-token window boundaries "
        "for both models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
