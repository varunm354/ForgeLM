"""Create small deterministic synthetic shards for CUDA qualification."""

import argparse
import json
from pathlib import Path

from forgelm.data.qualification_fixture import create_qualification_fixture
from forgelm.data.token_shards import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/cuda_qualification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=4_004)
    parser.add_argument("--train-tokens", type=int, default=65_536)
    parser.add_argument("--validation-tokens", type=int, default=8_192)
    parser.add_argument("--tokens-per-shard", type=int, default=32_768)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = create_qualification_fixture(
        args.output_dir,
        seed=args.seed,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        tokens_per_shard=args.tokens_per_shard,
        force=args.force,
    )
    output = args.output_dir.resolve()
    summary = {
        "output_directory": str(output),
        "manifest": str(output / "manifest.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "fixture_identity": manifest["qualification_fixture"]["identity"],
        "synthetic_warning": manifest["qualification_fixture"]["warning"],
        "train_tokens": manifest["counts"]["train_tokens"],
        "validation_tokens": manifest["counts"]["validation_tokens"],
        "train_shards": len(manifest["shards"]["train"]),
        "validation_shards": len(manifest["shards"]["validation"]),
        "total_bytes": sum(
            entry["byte_size"]
            for split in ("train", "validation")
            for entry in manifest["shards"][split]
        ),
        "checksums": {
            entry["name"]: entry["sha256"]
            for split in ("train", "validation")
            for entry in manifest["shards"][split]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
