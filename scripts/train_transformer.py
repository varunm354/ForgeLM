"""Train the character-level decoder-only Transformer."""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from forgelm.data.dataset import get_batch, train_validation_split
from forgelm.data.tokenizer import CharTokenizer
from forgelm.models.transformer_lm import TransformerLanguageModel
from forgelm.training.checkpoint import load_checkpoint, save_checkpoint
from forgelm.training.evaluate import estimate_loss
from forgelm.utils.device import get_device
from forgelm.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/transformer.yaml"
DATA_PATH = PROJECT_ROOT / "data/raw/tiny_shakespeare.txt"
CHECKPOINT_DIRECTORY = PROJECT_ROOT / "checkpoints/transformer"
CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "best.pt"
TOKENIZER_PATH = CHECKPOINT_DIRECTORY / "tokenizer.json"


def load_config() -> dict[str, Any]:
    """Load and validate the Transformer training configuration."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {CONFIG_PATH}")

    integer_fields = (
        "seed",
        "batch_size",
        "block_size",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "max_steps",
        "evaluation_interval",
        "evaluation_batches",
        "generation_length",
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

    if config["embedding_dim"] % config["num_heads"] != 0:
        raise ValueError("embedding_dim must be divisible by num_heads.")

    _validate_number(config, "dropout", minimum=0.0, maximum=1.0)
    _validate_number(config, "learning_rate", minimum=0.0, exclusive_minimum=True)
    _validate_number(config, "weight_decay", minimum=0.0)
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
        interval = "greater than" if exclusive_minimum else "at least"
        message = f"{name} must be {interval} {minimum}"
        if maximum is not None:
            message += f" and at most {maximum}"
        raise ValueError(f"{message}.")


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
    """Parse command-line training overrides."""
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
    return parser.parse_args()


def main() -> int:
    """Train, checkpoint, and sample from the Transformer."""
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

    seed = config["seed"]
    set_seed(seed)
    device = select_device(args.device)

    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_tokens, validation_tokens = train_validation_split(tokens, 0.9)

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
    print(f"vocabulary size: {tokenizer.vocab_size}")
    print(f"trainable parameters: {parameter_count:,}")
    print(f"training tokens: {len(train_tokens):,}")
    print(f"validation tokens: {len(validation_tokens):,}")

    best_validation_loss = float("inf")

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
            f"train loss {train_metrics['loss']:.4f}, "
            f"perplexity {train_metrics['perplexity']:.2f} | "
            f"validation loss {validation_metrics['loss']:.4f}, "
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
            )
            tokenizer.save(TOKENIZER_PATH)
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
    load_checkpoint(CHECKPOINT_PATH, model, device=device)
    model.eval()

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(prompt, config["generation_length"])
    sample = tokenizer.decode(generated[0].cpu().tolist())
    print("\nGenerated sample:\n")
    print(sample)
    if is_smoke_test:
        print("\nSmoke test complete; this is not a full training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
