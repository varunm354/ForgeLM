"""Validated configuration for production GPT pretraining."""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self

import yaml

PrecisionName = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class PretrainConfig:
    """All reproducibility-critical settings for one pretraining run."""

    model_config_path: Path
    manifest_path: Path
    expected_tokenizer_fingerprint: str | None
    seed: int
    micro_batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    target_training_tokens: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    gradient_clipping: float
    logging_interval: int
    validation_interval: int
    validation_batches: int
    checkpoint_interval: int
    precision: PrecisionName
    torch_compile: bool
    compile_allow_fallback: bool
    output_directory: Path
    latest_checkpoint_name: str
    best_checkpoint_name: str
    metrics_filename: str
    metadata_filename: str
    verify_checksums: bool
    fixed_validation_batches: bool
    production_data_lock_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "micro_batch_size",
            "sequence_length",
            "gradient_accumulation_steps",
            "target_training_tokens",
            "logging_interval",
            "validation_interval",
            "validation_batches",
            "checkpoint_interval",
        ):
            _positive_integer(getattr(self, name), name)
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
        if (
            not isinstance(self.warmup_steps, int)
            or isinstance(self.warmup_steps, bool)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be a non-negative integer.")
        _positive_number(self.learning_rate, "learning_rate")
        _positive_number(self.minimum_learning_rate, "minimum_learning_rate")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                "minimum_learning_rate cannot exceed learning_rate."
            )
        _nonnegative_number(self.weight_decay, "weight_decay")
        _positive_number(self.epsilon, "epsilon")
        _positive_number(self.gradient_clipping, "gradient_clipping")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= value < 1.0
                for value in self.betas
            )
        ):
            raise ValueError("betas must contain two values in [0, 1).")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported precision setting.")
        for name in (
            "torch_compile",
            "compile_allow_fallback",
            "verify_checksums",
            "fixed_validation_batches",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.expected_tokenizer_fingerprint is not None and (
            len(self.expected_tokenizer_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_tokenizer_fingerprint
            )
        ):
            raise ValueError(
                "expected_tokenizer_fingerprint must be SHA-256 hexadecimal."
            )
        if self.production_data_lock_path is not None and not isinstance(
            self.production_data_lock_path,
            Path,
        ):
            raise TypeError("production_data_lock_path must be a path or null.")
        for name in (
            "latest_checkpoint_name",
            "best_checkpoint_name",
            "metrics_filename",
            "metadata_filename",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or Path(value).name != value
            ):
                raise ValueError(f"{name} must be a file name.")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        project_root: Path | None = None,
    ) -> Self:
        """Load strict YAML and resolve paths relative to the project root."""
        source = Path(path)
        try:
            values = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(
                f"Could not load pretraining configuration: {source}"
            ) from error
        if not isinstance(values, dict):
            raise TypeError("Pretraining configuration must be a YAML mapping.")
        allowed = set(cls.__dataclass_fields__)
        optional = {"production_data_lock_path"}
        missing = sorted(allowed - optional - values.keys())
        unknown = sorted(values.keys() - allowed)
        if missing:
            raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(unknown)}")
        root = project_root if project_root is not None else source.parent
        converted = dict(values)
        converted.setdefault("production_data_lock_path", None)
        for name in ("model_config_path", "manifest_path", "output_directory"):
            converted[name] = _resolve_path(converted[name], root, name)
        if converted["production_data_lock_path"] is not None:
            converted["production_data_lock_path"] = _resolve_path(
                converted["production_data_lock_path"],
                root,
                "production_data_lock_path",
            )
        betas = converted["betas"]
        if not isinstance(betas, list):
            raise TypeError("betas must be a YAML list.")
        converted["betas"] = tuple(betas)
        return cls(**converted)

    def with_overrides(self, **overrides: Any) -> Self:
        """Return a validated copy containing command-line overrides."""
        return replace(self, **overrides)

    def effective_batch_tokens(self, world_size: int) -> int:
        """Return globally processed tokens per optimizer update."""
        _positive_integer(world_size, "world_size")
        return (
            self.micro_batch_size
            * self.sequence_length
            * self.gradient_accumulation_steps
            * world_size
        )

    def optimizer_steps(self, world_size: int) -> int:
        """Return updates needed to meet or exceed the token target."""
        return math.ceil(
            self.target_training_tokens / self.effective_batch_tokens(world_size)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        return {
            field: (
                str(value)
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for field in self.__dataclass_fields__
            if (value := getattr(self, field)) is not None
        } | (
            {"expected_tokenizer_fingerprint": None}
            if self.expected_tokenizer_fingerprint is None
            else {}
        )


def _resolve_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _positive_integer(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _positive_number(value: Any, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


def _nonnegative_number(value: Any, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number.")
