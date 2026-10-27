"""Validated configuration for scalable pretraining-data preparation."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import yaml

UINT16_VOCABULARY_LIMIT = 1 << 16


@dataclass(frozen=True)
class PretrainingDataConfig:
    """Configuration shared by the streaming preparation pipeline."""

    dataset_name: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    streaming: bool
    text_field: str
    seed: int
    validation_fraction: float
    minimum_characters: int
    maximum_characters: int
    normalize_line_endings: bool
    tokenizer_vocab_size: int
    tokenizer_training_bytes: int
    target_tokens: int
    validation_target_tokens: int
    tokens_per_shard: int
    output_dtype: str
    end_of_document_token: str
    output_directory: Path
    tokenizer_path: Path
    manifest_filename: str

    @classmethod
    def load(cls, path: str | Path, *, project_root: Path | None = None) -> Self:
        """Load a YAML mapping, resolve output paths, and validate every field."""
        source = Path(path)
        try:
            values = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"Could not load configuration from {source}.") from error
        if not isinstance(values, dict):
            raise TypeError("Pretraining configuration must be a YAML mapping.")

        root = project_root if project_root is not None else source.parent
        return cls.from_mapping(values, project_root=root)

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any],
        *,
        project_root: Path = Path("."),
    ) -> Self:
        """Construct and validate a configuration from parsed values."""
        required = {
            "dataset_name",
            "dataset_config",
            "dataset_split",
            "dataset_revision",
            "streaming",
            "text_field",
            "seed",
            "validation_fraction",
            "minimum_characters",
            "maximum_characters",
            "normalize_line_endings",
            "tokenizer_vocab_size",
            "tokenizer_training_bytes",
            "target_tokens",
            "validation_target_tokens",
            "tokens_per_shard",
            "output_dtype",
            "end_of_document_token",
            "output_directory",
            "tokenizer_path",
            "manifest_filename",
        }
        missing = sorted(required - values.keys())
        unknown = sorted(values.keys() - required)
        if missing:
            raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(unknown)}")

        string_fields = (
            "dataset_name",
            "dataset_config",
            "dataset_split",
            "dataset_revision",
            "text_field",
            "output_dtype",
            "end_of_document_token",
            "manifest_filename",
        )
        for name in string_fields:
            value = values[name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")

        integer_fields = (
            "seed",
            "minimum_characters",
            "maximum_characters",
            "tokenizer_vocab_size",
            "tokenizer_training_bytes",
            "target_tokens",
            "validation_target_tokens",
            "tokens_per_shard",
        )
        for name in integer_fields:
            value = values[name]
            minimum = 0 if name == "seed" else 1
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")

        fraction = values["validation_fraction"]
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not isinstance(values["normalize_line_endings"], bool):
            raise TypeError("normalize_line_endings must be a boolean.")
        if values["streaming"] is not True:
            raise ValueError("streaming must be true for bounded source processing.")
        if values["minimum_characters"] > values["maximum_characters"]:
            raise ValueError(
                "minimum_characters cannot exceed maximum_characters."
            )
        if values["output_dtype"] != "uint16":
            raise ValueError("Only output_dtype: uint16 is currently supported.")
        if not 257 <= values["tokenizer_vocab_size"] <= UINT16_VOCABULARY_LIMIT:
            raise ValueError(
                "tokenizer_vocab_size must be between 257 and 65536 for uint16."
            )

        output_directory = _resolve_path(
            values["output_directory"],
            project_root,
            "output_directory",
        )
        tokenizer_path = _resolve_path(
            values["tokenizer_path"],
            project_root,
            "tokenizer_path",
        )
        manifest_filename = values["manifest_filename"]
        if Path(manifest_filename).name != manifest_filename:
            raise ValueError("manifest_filename must be a file name, not a path.")

        return cls(
            dataset_name=values["dataset_name"],
            dataset_config=values["dataset_config"],
            dataset_split=values["dataset_split"],
            dataset_revision=values["dataset_revision"],
            streaming=values["streaming"],
            text_field=values["text_field"],
            seed=values["seed"],
            validation_fraction=float(fraction),
            minimum_characters=values["minimum_characters"],
            maximum_characters=values["maximum_characters"],
            normalize_line_endings=values["normalize_line_endings"],
            tokenizer_vocab_size=values["tokenizer_vocab_size"],
            tokenizer_training_bytes=values["tokenizer_training_bytes"],
            target_tokens=values["target_tokens"],
            validation_target_tokens=values["validation_target_tokens"],
            tokens_per_shard=values["tokens_per_shard"],
            output_dtype=values["output_dtype"],
            end_of_document_token=values["end_of_document_token"],
            output_directory=output_directory,
            tokenizer_path=tokenizer_path,
            manifest_filename=manifest_filename,
        )

    def with_overrides(self, **overrides: Any) -> Self:
        """Return a validated copy containing CLI overrides."""
        updated = replace(self, **overrides)
        updated.validate()
        return updated

    def validate(self) -> None:
        """Revalidate an instance after programmatic replacement."""
        values = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"output_directory", "tokenizer_path"}
        }
        values["output_directory"] = str(self.output_directory)
        values["tokenizer_path"] = str(self.tokenizer_path)
        self.from_mapping(values)


def _resolve_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()
