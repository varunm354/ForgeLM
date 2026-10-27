"""Inspect and CPU-smoke-test the production GPT architecture."""

import argparse
from pathlib import Path

import torch

from forgelm.models.gpt import GPTConfig, GPTLanguageModel, parameter_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gpt_50m.yaml"


def parse_args() -> argparse.Namespace:
    """Parse the architecture configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Print exact model properties and run bounded CPU checks."""
    args = parse_args()
    config_path = args.config.resolve()
    config = GPTConfig.load(config_path)
    torch.manual_seed(1337)
    model = GPTLanguageModel(config)
    report = parameter_report(model)

    print(f"config: {config_path}")
    print("architecture:")
    for name, value in config.to_dict().items():
        print(f"  {name}: {value}")
    print(f"total parameters: {report.total_parameters:,}")
    print(f"trainable parameters: {report.trainable_parameters:,}")
    print(f"shared parameter references: {report.shared_parameter_references}")
    print(
        "float32 parameter memory: "
        f"{report.memory.float32_bytes:,} bytes "
        f"({_mebibytes(report.memory.float32_bytes):.2f} MiB)"
    )
    print(
        "float16 parameter memory: "
        f"{report.memory.float16_bytes:,} bytes "
        f"({_mebibytes(report.memory.float16_bytes):.2f} MiB)"
    )
    print(
        "bfloat16 parameter memory: "
        f"{report.memory.bfloat16_bytes:,} bytes "
        f"({_mebibytes(report.memory.bfloat16_bytes):.2f} MiB)"
    )
    tied = (
        model.lm_head.weight is model.token_embedding.weight
        and model.lm_head.weight.untyped_storage().data_ptr()
        == model.token_embedding.weight.untyped_storage().data_ptr()
    )
    print(f"tied token/output weights verified: {tied}")

    model.train()
    sequence_length = min(8, config.block_size)
    inputs = torch.randint(
        0,
        config.vocab_size,
        (1, sequence_length),
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        config.vocab_size,
        inputs.shape,
        dtype=torch.long,
    )
    logits, loss = model(inputs, targets)
    if loss is None:
        raise RuntimeError("Expected a training loss.")
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"input shape: {tuple(inputs.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss shape: {tuple(loss.shape)}")
    print(f"finite loss: {bool(torch.isfinite(loss))}")
    print(f"finite backward gradients: {finite_gradients}")

    model.zero_grad(set_to_none=True)
    model.eval()
    prefix = inputs[:, : min(4, sequence_length)]
    generated = model.generate(prefix, max_new_tokens=3, top_k=1)
    print(f"generation prefix shape: {tuple(prefix.shape)}")
    print(f"generated shape: {tuple(generated.shape)}")
    print(f"generation preserved prefix: {torch.equal(generated[:, : prefix.shape[1]], prefix)}")
    return 0


def _mebibytes(byte_count: int) -> float:
    return byte_count / (1024**2)


if __name__ == "__main__":
    raise SystemExit(main())
