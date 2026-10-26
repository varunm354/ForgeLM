"""Prepare deterministic packed pretraining shards from streaming or local data."""

import argparse
import itertools
import shutil
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from forgelm.data.pretraining_config import PretrainingDataConfig
from forgelm.data.scalable_tokenizer import ScalableByteLevelTokenizer
from forgelm.data.streaming_corpus import (
    StreamingDocumentPipeline,
    collect_tokenizer_training_documents,
    iter_jsonl,
    load_huggingface_stream,
)
from forgelm.data.token_shards import build_token_shards

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/pretraining_data.yaml"
Record = str | Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse source, output, and bounded-work overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--local-jsonl",
        type=Path,
        help="use local JSONL instead of the configured Hugging Face stream",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-tokens", type=_positive_integer)
    parser.add_argument("--maximum-documents", type=_positive_integer)
    parser.add_argument("--tokenizer-vocab-size", type=_positive_integer)
    parser.add_argument("--tokenizer-training-bytes", type=_positive_integer)
    parser.add_argument("--tokens-per-shard", type=_positive_integer)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="replace an existing tokenizer using the train split only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing output and stale staging directories",
    )
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def make_pipeline(config: PretrainingDataConfig) -> StreamingDocumentPipeline:
    """Construct a fresh stateful filtering pass."""
    return StreamingDocumentPipeline(
        text_field=config.text_field,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        minimum_characters=config.minimum_characters,
        maximum_characters=config.maximum_characters,
        normalize_line_endings=config.normalize_line_endings,
    )


def main() -> int:
    """Train/load a tokenizer, build shards, and publish the directory atomically."""
    args = parse_args()
    config = PretrainingDataConfig.load(
        args.config.resolve(),
        project_root=PROJECT_ROOT,
    )
    config = _apply_overrides(config, args)
    source_factory, source_metadata = _make_source_factory(
        config,
        local_jsonl=args.local_jsonl,
        maximum_documents=args.maximum_documents,
    )
    _prepare_destination(config.output_directory, force=args.force)

    if config.dataset_revision == "main" and args.local_jsonl is None:
        print(
            "warning: dataset_revision is 'main'; pin an immutable revision "
            "for the final run"
        )
    started = time.perf_counter()
    tokenizer, tokenizer_source, tokenizer_bytes = _load_or_train_tokenizer(
        config,
        source_factory,
        retrain=args.retrain_tokenizer,
    )
    print(f"tokenizer: {tokenizer_source} ({config.tokenizer_path})")
    print(f"tokenizer vocabulary: {tokenizer.vocab_size:,}")
    print(f"tokenizer fingerprint: {tokenizer.fingerprint()}")
    if tokenizer_source == "trained":
        print(f"tokenizer training bytes: {tokenizer_bytes:,}")

    output = config.output_directory
    staging = output.with_name(f".{output.name}.tmp")
    staging.mkdir(parents=True)
    try:
        print(f"streaming and writing staged shards: {staging}")
        pipeline = make_pipeline(config)
        manifest = build_token_shards(
            source_factory(),
            pipeline=pipeline,
            tokenizer=tokenizer,
            output_directory=staging,
            target_tokens=config.target_tokens,
            validation_target_tokens=config.validation_target_tokens,
            tokens_per_shard=config.tokens_per_shard,
            source_metadata=source_metadata,
            creation_configuration=_creation_configuration(config),
            tokenizer_path=config.tokenizer_path,
            manifest_filename=config.manifest_filename,
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    elapsed = time.perf_counter() - started
    counts = manifest["counts"]
    print(f"output: {output}")
    print(f"manifest: {output / config.manifest_filename}")
    print(f"documents seen: {counts['documents_seen']:,}")
    print(f"documents accepted: {counts['documents_accepted']:,}")
    print(f"documents rejected: {counts['documents_rejected']:,}")
    print(f"documents duplicated: {counts['documents_duplicated']:,}")
    print(f"training tokens: {counts['train_tokens']:,}")
    print(f"validation tokens: {counts['validation_tokens']:,}")
    print(f"training shards: {len(manifest['shards']['train']):,}")
    print(f"validation shards: {len(manifest['shards']['validation']):,}")
    print(f"elapsed seconds: {elapsed:.2f}")
    return 0


def _apply_overrides(
    config: PretrainingDataConfig,
    args: argparse.Namespace,
) -> PretrainingDataConfig:
    overrides: dict[str, Any] = {}
    for argument, field in (
        (args.target_tokens, "target_tokens"),
        (args.tokenizer_vocab_size, "tokenizer_vocab_size"),
        (args.tokenizer_training_bytes, "tokenizer_training_bytes"),
        (args.tokens_per_shard, "tokens_per_shard"),
        (args.dataset_revision, "dataset_revision"),
    ):
        if argument is not None:
            overrides[field] = argument
    if args.output_dir is not None:
        overrides["output_directory"] = args.output_dir.resolve()
    if args.tokenizer_path is not None:
        overrides["tokenizer_path"] = args.tokenizer_path.resolve()
    return config.with_overrides(**overrides) if overrides else config


def _make_source_factory(
    config: PretrainingDataConfig,
    *,
    local_jsonl: Path | None,
    maximum_documents: int | None,
) -> tuple[Callable[[], Iterable[Record]], dict[str, Any]]:
    if local_jsonl is not None:
        source = local_jsonl.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Local JSONL source not found: {source}")

        def unbounded_factory() -> Iterable[Record]:
            return iter_jsonl(source)

        source_type = "local_jsonl"
        local_path: str | None = str(source)
    else:

        def unbounded_factory() -> Iterable[Record]:
            return load_huggingface_stream(
                dataset_name=config.dataset_name,
                dataset_config=config.dataset_config,
                split=config.dataset_split,
                revision=config.dataset_revision,
            )

        source_type = "huggingface_streaming"
        local_path = None

    def factory() -> Iterable[Record]:
        records = unbounded_factory()
        if maximum_documents is None:
            return records
        return itertools.islice(records, maximum_documents)

    metadata = {
        "type": source_type,
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "dataset_revision": config.dataset_revision,
        "text_field": config.text_field,
        "local_path": local_path,
        "maximum_documents": maximum_documents,
    }
    return factory, metadata


def _load_or_train_tokenizer(
    config: PretrainingDataConfig,
    source_factory: Callable[[], Iterable[Record]],
    *,
    retrain: bool,
) -> tuple[ScalableByteLevelTokenizer, str, int]:
    path = config.tokenizer_path
    if path.exists() and not retrain:
        tokenizer = ScalableByteLevelTokenizer.load(
            path,
            end_of_document_token=config.end_of_document_token,
            expected_vocab_size=config.tokenizer_vocab_size,
        )
        return tokenizer, "loaded", 0

    print("collecting bounded train-split tokenizer sample")
    sample_pipeline = make_pipeline(config)
    documents, byte_count = collect_tokenizer_training_documents(
        source_factory(),
        sample_pipeline,
        byte_budget=config.tokenizer_training_bytes,
    )
    if not documents:
        reasons = sample_pipeline.statistics.to_dict()["rejection_reasons"]
        raise ValueError(
            "No training documents were available for tokenizer fitting. "
            f"Check text_field and filters; rejection counts: {reasons}"
        )
    tokenizer = ScalableByteLevelTokenizer.train_from_iterator(
        documents,
        vocab_size=config.tokenizer_vocab_size,
        end_of_document_token=config.end_of_document_token,
        length=len(documents),
    )
    tokenizer.save(path)
    return tokenizer, "trained", byte_count


def _prepare_destination(output: Path, *, force: bool) -> None:
    staging = output.with_name(f".{output.name}.tmp")
    existing = [path for path in (output, staging) if path.exists()]
    if existing and not force:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output or partial staging path: {paths}. "
            "Use --force to replace it."
        )
    if force:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _creation_configuration(config: PretrainingDataConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "minimum_characters": config.minimum_characters,
        "maximum_characters": config.maximum_characters,
        "normalize_line_endings": config.normalize_line_endings,
        "tokenizer_vocab_size": config.tokenizer_vocab_size,
        "tokenizer_training_bytes": config.tokenizer_training_bytes,
        "target_tokens": config.target_tokens,
        "validation_target_tokens": config.validation_target_tokens,
        "tokens_per_shard": config.tokens_per_shard,
        "output_dtype": config.output_dtype,
        "end_of_document_token": config.end_of_document_token,
    }


if __name__ == "__main__":
    raise SystemExit(main())
