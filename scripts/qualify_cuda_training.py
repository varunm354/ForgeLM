"""Run a bounded full-model CUDA qualification with forced checkpoint resume."""

import argparse
import json
from pathlib import Path

from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.qualification import (
    MAX_QUALIFICATION_STEPS,
    run_qualification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pretrain_gpt_cuda_qualification.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=MAX_QUALIFICATION_STEPS,
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--cpu-diagnostic", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PretrainConfig.load(
        args.config.resolve(),
        project_root=PROJECT_ROOT,
    )
    overrides = {}
    if args.manifest is not None:
        overrides["manifest_path"] = args.manifest.resolve()
    if args.output_dir is not None:
        overrides["output_directory"] = args.output_dir.resolve()
    if args.precision is not None:
        overrides["precision"] = args.precision
    if args.compile is not None:
        overrides["torch_compile"] = args.compile
    if overrides:
        config = config.with_overrides(**overrides)

    report = run_qualification(
        config,
        maximum_optimizer_steps=args.max_optimizer_steps,
        cpu_diagnostic=args.cpu_diagnostic,
        resume_checkpoint=(
            args.resume_checkpoint.resolve()
            if args.resume_checkpoint is not None
            else None
        ),
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
