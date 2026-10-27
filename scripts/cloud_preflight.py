"""Report whether a host and ForgeLM artifacts are ready for CUDA qualification."""

import argparse
import json
from pathlib import Path

from forgelm.training.cloud_preflight import run_cloud_preflight, write_json_report
from forgelm.training.pretrain_config import PretrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pretrain_gpt_cuda_qualification.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--cpu-diagnostic", action="store_true")
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
    if overrides:
        config = config.with_overrides(**overrides)

    report = run_cloud_preflight(
        config,
        cpu_diagnostic=args.cpu_diagnostic,
        precision_override=args.precision,
        output_directory=args.output_dir,
    )
    if args.json_output is not None:
        write_json_report(args.json_output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
