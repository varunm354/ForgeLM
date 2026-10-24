"""Train the decoder-only Transformer with train-only byte-level BPE."""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from forgelm.data.bpe_artifact import load_or_train_bpe_tokenizer
from forgelm.data.dataset import get_batch, train_validation_text_split
from forgelm.models.transformer_lm import TransformerLanguageModel
from forgelm.training.checkpoint import load_checkpoint, save_checkpoint
from forgelm.training.evaluate import estimate_loss
from forgelm.utils.device import get_device
from forgelm.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/transformer_bpe.yaml"
DATA_PATH = PROJECT_ROOT / "data/raw/tiny_shakespeare.txt"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/transformer_bpe/best.pt"


def load_config() -> dict[str, Any]:
    """Load and validate the BPE Transformer configuration."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {CONFIG_PATH}")

    integer_fields = (
        "seed",
        "tokenizer_vocab_size",
        "batch_size",
        "block_size",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "max_steps",
        "evaluation_interval",
        "evaluation_batches",
        "generation_length",
        "generation_top_k",
    )
    for name in integer_fields:
        value = config.get(name)
        minimum = 0 if name == "generation_length" else 1
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )

    if config["tokenizer_vocab_size"] < 256:
        raise ValueError("tokenizer_vocab_size must be at least 256.")
    if config["embedding_dim"] % config["num_heads"] != 0:
        raise ValueError("embedding_dim must be divisible by num_heads.")
    if config["generation_top_k"] > config["tokenizer_vocab_size"]:
        raise ValueError("generation_top_k cannot exceed tokenizer_vocab_size.")
    tokenizer_path = config.get("tokenizer_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("tokenizer_path must be a non-empty string.")

    _validate_number(config, "dropout", minimum=0.0, maximum=1.0)
    _validate_number(config, "learning_rate", minimum=0.0, exclusive_minimum=True)
    _validate_number(config, "weight_decay", minimum=0.0)
    _validate_number(
        config,
        "generation_temperature",
        minimum=0.0,
        exclusive_minimum=True,
    )
    return config


def _validate_number(
    config: dict[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> None:
    value = config.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number.")
    below_minimum = value <= minimum if exclusive_minimum else value < minimum
    if below_minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: {value}.")


def select_device(name: str) -> torch.device:
    """Resolve an automatic or explicitly requested training device."""
    if name == "auto":
        return get_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    """Parse training and tokenizer options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="override the configured number of optimization steps",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="training device (default: auto)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="override the configured BPE tokenizer artifact",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="replace any existing tokenizer using training text only",
    )
    return parser.parse_args()


def main() -> int:
    """Train, checkpoint, and sample from the BPE Transformer."""
    args = parse_args()
    config = load_config()
    is_smoke_test = (
        args.max_steps is not None and args.max_steps < config["max_steps"]
    )
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be a positive integer.")
        config["max_steps"] = args.max_steps

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run "
            "`uv run python scripts/download_tiny_shakespeare.py` first."
        )

    configured_tokenizer_path = PROJECT_ROOT / config["tokenizer_path"]
    tokenizer_path = (
        args.tokenizer_path.resolve()
        if args.tokenizer_path is not None
        else configured_tokenizer_path
    )

    raw_text = DATA_PATH.read_text(encoding="utf-8")
    training_text, validation_text = train_validation_text_split(raw_text, 0.9)
    tokenizer, tokenizer_source = load_or_train_bpe_tokenizer(
        training_text,
        tokenizer_path,
        target_vocab_size=config["tokenizer_vocab_size"],
        retrain=args.retrain_tokenizer,
    )
    tokenizer_fingerprint = tokenizer.fingerprint()
    train_tokens = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
    validation_tokens = torch.tensor(
        tokenizer.encode(validation_text),
        dtype=torch.long,
    )

    seed = config["seed"]
    set_seed(seed)
    device = select_device(args.device)
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        block_size=config["block_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    training_generator = torch.Generator().manual_seed(seed)
    evaluation_generator = torch.Generator().manual_seed(seed + 1)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if is_smoke_test:
        print(f"run: smoke test ({config['max_steps']} optimization steps)")
    print(f"device: {device}")
    print(f"tokenizer: {tokenizer_source} ({tokenizer_path})")
    print(f"tokenizer vocabulary size: {tokenizer.vocab_size:,}")
    print(f"tokenizer fingerprint: {tokenizer_fingerprint}")
    print(f"trainable parameters: {parameter_count:,}")
    print(f"training BPE tokens: {len(train_tokens):,}")
    print(f"validation BPE tokens: {len(validation_tokens):,}")
    print(
        "warning: BPE token-level perplexity is not directly comparable "
        "with character-level perplexity."
    )

    best_validation_loss = float("inf")
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": config["embedding_dim"],
        "block_size": config["block_size"],
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
    }

    def evaluate_and_save(step: int) -> None:
        nonlocal best_validation_loss
        metrics = estimate_loss(
            model,
            train_tokens,
            validation_tokens,
            batch_size=config["batch_size"],
            block_size=config["block_size"],
            evaluation_batches=config["evaluation_batches"],
            generator=evaluation_generator,
        )
        train_metrics = metrics["train"]
        validation_metrics = metrics["validation"]
        print(
            f"step {step:5d} | "
            f"train token loss {train_metrics['loss']:.4f}, "
            f"perplexity {train_metrics['perplexity']:.2f} | "
            f"validation token loss {validation_metrics['loss']:.4f}, "
            f"perplexity {validation_metrics['perplexity']:.2f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                CHECKPOINT_PATH,
                model,
                optimizer,
                step=step,
                config=config,
                metrics=metrics,
                metadata={
                    "model_config": model_config,
                    "tokenizer_fingerprint": tokenizer_fingerprint,
                    "tokenizer_vocab_size": tokenizer.vocab_size,
                    "tokenizer_path": str(tokenizer_path),
                    "step": step,
                    "validation_loss": validation_metrics["loss"],
                },
            )
            print(f"saved best checkpoint: {CHECKPOINT_PATH}")

    max_steps = config["max_steps"]
    for step in range(max_steps):
        if step % config["evaluation_interval"] == 0:
            evaluate_and_save(step)

        inputs, targets = get_batch(
            train_tokens,
            batch_size=config["batch_size"],
            block_size=config["block_size"],
            generator=training_generator,
        )
        _, loss = model(inputs.to(device), targets.to(device))
        if loss is None:
            raise RuntimeError("The model did not return a training loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    evaluate_and_save(max_steps)
    load_checkpoint(
        CHECKPOINT_PATH,
        model,
        device=device,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    model.eval()

    prompt = torch.tensor(
        [tokenizer.encode("\n")],
        dtype=torch.long,
        device=device,
    )
    generated = model.generate(
        prompt,
        config["generation_length"],
        temperature=config["generation_temperature"],
        top_k=config["generation_top_k"],
    )
    generated_ids = generated[0].cpu().tolist()
    try:
        sample = tokenizer.decode(generated_ids)
    except ValueError:
        generated_bytes = b"".join(
            tokenizer.token_bytes(token_id) for token_id in generated_ids
        )
        sample = generated_bytes.decode("utf-8", errors="replace")
        print(
            "warning: generated bytes were not valid UTF-8; "
            "replacement characters are shown."
        )
    print("\nGenerated sample:\n")
    print(sample)
    if is_smoke_test:
        print("\nSmoke test complete; this is not a full training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
