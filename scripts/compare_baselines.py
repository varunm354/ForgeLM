"""Compare best Bigram and Transformer checkpoints on the full validation set."""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from forgelm.data.dataset import train_validation_split
from forgelm.data.tokenizer import CharTokenizer
from forgelm.models.bigram import BigramLanguageModel
from forgelm.models.transformer_lm import TransformerLanguageModel
from forgelm.training.checkpoint import load_checkpoint
from forgelm.training.evaluate import evaluate_full_corpus
from forgelm.utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/tiny_shakespeare.txt"
BIGRAM_CONFIG_PATH = PROJECT_ROOT / "configs/bigram.yaml"
TRANSFORMER_CONFIG_PATH = PROJECT_ROOT / "configs/transformer.yaml"
BIGRAM_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/bigram/best.pt"
TRANSFORMER_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/transformer/best.pt"


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="evaluation device (default: auto)",
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


def select_device(name: str) -> torch.device:
    """Resolve an automatic or explicitly requested evaluation device."""
    if name == "auto":
        return get_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return torch.device(name)


def require_file(path: Path, description: str) -> None:
    """Raise a useful error when a benchmark input is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def parameter_count(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def main() -> int:
    """Load both checkpoints and run deterministic full-corpus evaluation."""
    args = parse_args()
    require_file(DATA_PATH, "Tiny Shakespeare dataset")
    require_file(BIGRAM_CHECKPOINT_PATH, "Bigram checkpoint")
    require_file(TRANSFORMER_CHECKPOINT_PATH, "Transformer checkpoint")

    device = select_device(args.device)
    bigram_config = load_yaml(BIGRAM_CONFIG_PATH)
    transformer_config = load_yaml(TRANSFORMER_CONFIG_PATH)

    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    _, validation_tokens = train_validation_split(tokens, 0.9)

    bigram = BigramLanguageModel(tokenizer.vocab_size)
    transformer = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=transformer_config["embedding_dim"],
        block_size=transformer_config["block_size"],
        num_heads=transformer_config["num_heads"],
        num_layers=transformer_config["num_layers"],
        dropout=transformer_config["dropout"],
    )

    bigram_checkpoint = load_checkpoint(
        BIGRAM_CHECKPOINT_PATH,
        bigram,
        device=device,
    )
    transformer_checkpoint = load_checkpoint(
        TRANSFORMER_CHECKPOINT_PATH,
        transformer,
        device=device,
    )

    bigram_metrics = evaluate_full_corpus(
        bigram,
        validation_tokens,
        block_size=bigram_config["block_size"],
        batch_size=args.batch_size,
    )
    transformer_metrics = evaluate_full_corpus(
        transformer,
        validation_tokens,
        block_size=transformer_config["block_size"],
        batch_size=args.batch_size,
    )

    loss_reduction = (
        (bigram_metrics["loss"] - transformer_metrics["loss"])
        / bigram_metrics["loss"]
        * 100
    )
    perplexity_reduction = (
        (bigram_metrics["perplexity"] - transformer_metrics["perplexity"])
        / bigram_metrics["perplexity"]
        * 100
    )

    print("Deterministic full-corpus validation benchmark")
    print("Each of the 111,539 validation targets is evaluated exactly once.")
    print("These are not the sampled metrics reported during training.")
    print(f"device: {device}")
    print(
        f"Bigram checkpoint loaded: step {bigram_checkpoint['step']} "
        f"({BIGRAM_CHECKPOINT_PATH})"
    )
    print(
        f"Transformer checkpoint loaded: step {transformer_checkpoint['step']} "
        f"({TRANSFORMER_CHECKPOINT_PATH})"
    )
    print()
    print(f"Bigram parameters: {parameter_count(bigram):,}")
    print(f"Bigram validation loss: {bigram_metrics['loss']:.6f}")
    print(f"Bigram validation perplexity: {bigram_metrics['perplexity']:.6f}")
    print()
    print(f"Transformer parameters: {parameter_count(transformer):,}")
    print(f"Transformer validation loss: {transformer_metrics['loss']:.6f}")
    print(
        "Transformer validation perplexity: "
        f"{transformer_metrics['perplexity']:.6f}"
    )
    print()
    print(f"Validation loss reduction: {loss_reduction:.2f}%")
    print(f"Validation perplexity reduction: {perplexity_reduction:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
