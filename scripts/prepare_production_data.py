"""Run one explicit Phase 4E-A production-data stage without cascading."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from forgelm.data.preflight import (
    is_immutable_dataset_revision,
    run_data_preflight,
)
from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.production_data import (
    ProductionPaths,
    approve_tokenizer,
    build_shard_stage,
    build_tokenizer_stage,
    default_source_metadata,
    verify_production_stage,
)
from forgelm.data.streaming_corpus import iter_jsonl, load_huggingface_stream
from forgelm.data.token_shards import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pretraining_data.yaml"
Record = str | Mapping[str, Any]
SourceFactory = Callable[[], Iterable[Record]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("probe", "tokenizer", "pin-tokenizer", "shards", "verify"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-jsonl", type=Path)
    parser.add_argument("--preflight-documents", type=_positive_integer, default=2)
    parser.add_argument("--skip-connectivity-check", action="store_true")
    parser.add_argument("--expected-tokenizer-fingerprint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PretrainingDataConfig.load(
        args.config.resolve(),
        project_root=PROJECT_ROOT,
    )
    paths = ProductionPaths.from_root(args.root)
    if (
        args.expected_tokenizer_fingerprint is not None
        and args.stage != "pin-tokenizer"
    ):
        raise ValueError(
            "--expected-tokenizer-fingerprint is valid only for pin-tokenizer."
        )
    if args.resume and args.stage != "shards":
        raise ValueError("--resume is valid only for the shards stage.")
    if args.force and args.stage not in {"probe", "tokenizer", "shards"}:
        raise ValueError(f"Stage {args.stage!r} does not accept --force.")
    source_required = args.stage in {"probe", "tokenizer", "shards"}
    if not source_required and args.local_jsonl is not None:
        raise ValueError(f"Stage {args.stage!r} does not read a source.")
    if source_required and args.local_jsonl is None and not is_immutable_dataset_revision(
        config.dataset_revision
    ):
        raise ValueError(
            "Production stages require the configured immutable dataset revision; "
            "there is no fallback to main."
        )
    source_factory: SourceFactory | None = None
    source_metadata: dict[str, Any] | None = None
    if source_required:
        source_factory, source_metadata = _source(config, args.local_jsonl)

    if args.stage == "probe":
        if source_factory is None:
            raise RuntimeError("Source factory was not initialized.")
        preflight_config = config.with_overrides(
            output_directory=paths.shards,
            tokenizer_path=paths.tokenizer_path,
        )
        report = run_data_preflight(
            preflight_config,
            source_factory=source_factory,
            local_source=args.local_jsonl is not None,
            force=args.force,
            check_connectivity=not args.skip_connectivity_check,
            document_limit=args.preflight_documents,
            persistent_root=paths.root,
        )
        _print_json(report)
        return 0 if report["passed"] else 2

    if args.stage == "tokenizer":
        _reject_irrelevant_flags(args, allowed={"force"})
        if source_factory is None or source_metadata is None:
            raise RuntimeError("Source factory was not initialized.")
        manifest = build_tokenizer_stage(
            config,
            source_factory,
            paths.root,
            source_metadata=source_metadata,
            force=args.force,
        )
        _print_json(
            {
                "stage": "tokenizer",
                "status": manifest["status"],
                "tokenizer_fingerprint": manifest["tokenizer"]["fingerprint"],
                "tokenizer_identity": manifest["identity"],
                "next_stage": "pin-tokenizer",
                "automatic_next_stage_started": False,
            }
        )
        return 0

    if args.stage == "pin-tokenizer":
        _reject_irrelevant_flags(args, allowed=set())
        if args.expected_tokenizer_fingerprint is None:
            raise ValueError(
                "--expected-tokenizer-fingerprint is required after human review."
            )
        approval = approve_tokenizer(
            config,
            paths.root,
            expected_fingerprint=args.expected_tokenizer_fingerprint,
        )
        _print_json(
            {
                "stage": "pin-tokenizer",
                "status": approval["status"],
                "approval_identity": approval["identity"],
                "next_stage": "shards",
                "automatic_next_stage_started": False,
            }
        )
        return 0

    if args.stage == "shards":
        if source_factory is None or source_metadata is None:
            raise RuntimeError("Source factory was not initialized.")
        manifest = build_shard_stage(
            config,
            source_factory,
            paths.root,
            source_metadata=source_metadata,
            resume=args.resume,
            force=args.force,
        )
        _print_json(
            {
                "stage": "shards",
                "train_tokens": manifest["counts"]["train_tokens"],
                "validation_tokens": manifest["counts"]["validation_tokens"],
                "train_shards": len(manifest["shards"]["train"]),
                "validation_shards": len(manifest["shards"]["validation"]),
                "next_stage": "verify",
                "automatic_next_stage_started": False,
            }
        )
        return 0

    _reject_irrelevant_flags(args, allowed=set())
    lock = verify_production_stage(
        config,
        paths.root,
    )
    _print_json(
        {
            "stage": "verify",
            "status": lock["status"],
            "production_lock_identity": lock["identity"],
            "training_started": False,
            "automatic_next_stage_started": False,
        }
    )
    return 0


def _source(
    config: PretrainingDataConfig,
    local_jsonl: Path | None,
) -> tuple[SourceFactory, dict[str, Any]]:
    if local_jsonl is None:

        def factory() -> Iterable[Record]:
            return load_huggingface_stream(
                dataset_name=config.dataset_name,
                dataset_config=config.dataset_config,
                split=config.dataset_split,
                revision=config.dataset_revision,
            )

        return factory, default_source_metadata(config)

    source = local_jsonl.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Local JSONL source not found: {source}")

    def local_factory() -> Iterable[Record]:
        return iter_jsonl(source)

    return local_factory, {
        **default_source_metadata(config),
        "type": "local_jsonl_fixture",
        "local_path": str(source),
        "local_byte_size": source.stat().st_size,
        "local_sha256": sha256_file(source),
    }


def _reject_irrelevant_flags(
    args: argparse.Namespace,
    *,
    allowed: set[str],
) -> None:
    supplied = {
        "force": args.force,
        "resume": args.resume,
    }
    invalid = sorted(name for name, enabled in supplied.items() if enabled and name not in allowed)
    if invalid:
        raise ValueError(
            f"Stage {args.stage!r} does not accept: "
            + ", ".join(f"--{name}" for name in invalid)
        )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
