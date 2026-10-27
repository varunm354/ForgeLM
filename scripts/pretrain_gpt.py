"""Pretrain the production GPT on Phase 4A memory-mapped token shards."""

import argparse
from pathlib import Path

from forgelm.models.gpt import parameter_report
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.pretrain_engine import PretrainingEngine
from forgelm.training.pretrain_runtime import initialize_distributed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pretrain_gpt_50m.yaml"


def parse_args() -> argparse.Namespace:
    """Parse cloud and smoke-run overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights-only", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override torch.compile configuration",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    return parser.parse_args()


def main() -> int:
    """Initialize runtime state, train, checkpoint, and shut down cleanly."""
    args = parse_args()
    if args.resume is not None and args.weights_only is not None:
        raise ValueError("--resume and --weights-only are mutually exclusive.")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive.")

    config = PretrainConfig.load(
        args.config.resolve(),
        project_root=PROJECT_ROOT,
    )
    overrides = {}
    if args.manifest is not None:
        overrides["manifest_path"] = args.manifest.resolve()
    if args.output_dir is not None:
        overrides["output_directory"] = args.output_dir.resolve()
    if args.compile is not None:
        overrides["torch_compile"] = args.compile
    if args.precision is not None:
        overrides["precision"] = args.precision
    if overrides:
        config = config.with_overrides(**overrides)

    resume = args.resume.resolve() if args.resume is not None else None
    weights_only = (
        args.weights_only.resolve() if args.weights_only is not None else None
    )
    if resume is None and config.output_directory.exists() and any(
        config.output_directory.iterdir()
    ):
        raise FileExistsError(
            f"Fresh output directory is not empty: {config.output_directory}. "
            "Choose another directory or resume explicitly."
        )

    distributed = initialize_distributed(args.device)
    try:
        engine = PretrainingEngine(
            config,
            distributed,
            resume_checkpoint=resume,
            weights_only_checkpoint=weights_only,
        )
        engine.install_signal_handlers()
        if distributed.is_rank_zero:
            report = parameter_report(engine.model)
            print(f"model parameters: {report.total_parameters:,}")
            print(f"device: {engine.device}")
            print(f"precision: {engine.precision.name}")
            print(f"world size: {distributed.world_size}")
            print(f"micro-batch size: {config.micro_batch_size}")
            print(
                "gradient accumulation steps: "
                f"{config.gradient_accumulation_steps}"
            )
            print(
                "effective global batch tokens: "
                f"{engine.effective_batch_tokens:,} = "
                f"{config.micro_batch_size} × {config.sequence_length} × "
                f"{config.gradient_accumulation_steps} × "
                f"{distributed.world_size}"
            )
            print(f"target training tokens: {config.target_training_tokens:,}")
            print(f"calculated optimizer steps: {engine.total_optimizer_steps:,}")
            print(f"manifest identity: {engine.manifest_identity}")
            print(
                f"tokenizer fingerprint: {engine.tokenizer_fingerprint}"
            )
            print(f"resume status: {engine.resume_status}")
            print(f"torch.compile enabled: {config.torch_compile}")
            print(f"fused AdamW enabled: {engine.optimizer_fused}")

        summary = engine.run(
            maximum_optimizer_steps=args.max_optimizer_steps,
        )
        if distributed.is_rank_zero:
            print("training boundary complete")
            for name, value in summary.items():
                print(f"{name}: {value}")
        return 0
    finally:
        distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
