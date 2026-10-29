"""Run or verify the bounded Phase 4E-B production-data CUDA canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.training.pretrain_config import PretrainConfig
from forgelm.training.production_canary import (
    MAX_CANARY_OPTIMIZER_STEPS,
    run_production_canary,
    verify_canary_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = (
    PROJECT_ROOT / "configs/pretrain_gpt_production_canary.yaml"
)
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs/pretraining_data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the bounded CUDA canary")
    run.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    run.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=MAX_CANARY_OPTIMIZER_STEPS,
    )
    run.add_argument("--resume-checkpoint", type=Path)
    run.add_argument(
        "--expected-production-lock-identity",
        required=True,
    )

    verify = subparsers.add_parser(
        "verify",
        help="re-hash and verify an existing passing report",
    )
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument(
        "--expected-production-lock-identity",
        required=True,
    )
    verify.add_argument(
        "--expected-tokenizer-fingerprint",
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "verify":
        result = verify_canary_report(
            args.report,
            expected_production_lock_identity=(
                args.expected_production_lock_identity
            ),
            expected_tokenizer_fingerprint=args.expected_tokenizer_fingerprint,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    config = PretrainConfig.load(
        args.config.resolve(),
        project_root=PROJECT_ROOT,
    )
    if args.output_dir is not None:
        config = config.with_overrides(
            output_directory=args.output_dir.resolve(),
        )
    data_config = PretrainingDataConfig.load(
        args.data_config.resolve(),
        project_root=PROJECT_ROOT,
    )
    report = run_production_canary(
        config,
        data_config,
        expected_production_lock_identity=(
            args.expected_production_lock_identity
        ),
        maximum_optimizer_steps=args.max_optimizer_steps,
        resume_checkpoint=(
            args.resume_checkpoint.resolve()
            if args.resume_checkpoint is not None
            else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
