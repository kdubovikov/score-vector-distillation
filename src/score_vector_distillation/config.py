"""Typed configuration loading with canonical hashing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is malformed."""


class ConfigKind(StrEnum):
    """Top-level configuration roles understood by the public repository."""

    EXECUTABLE_RUN = "executable_run"
    SWEEP_INDEX = "sweep_index"


class DistillationLoss(StrEnum):
    """Losses implemented by :class:`ScoreVectorStudent`."""

    CENTERED_MSE = "centered_mse"
    FORWARD_KL_TAU_0P1 = "forward_kl_tau_0p1"
    POSITIVE_NEGATIVE_MARGIN_MSE = "positive_negative_margin_mse"
    UNIT_CENTERED_MSE = "unit_centered_mse"
    RANK_WEIGHTED_PAIR_MSE = "rank_weighted_pair_mse"
    LABEL_ONLY_CROSS_ENTROPY = "label_only_cross_entropy"
    LABEL_ONLY_INBATCH_CROSS_ENTROPY = "label_only_inbatch_cross_entropy"


class SchedulerKind(StrEnum):
    """Learning-rate schedules implemented by the public trainer."""

    COSINE = "cosine"


class ParameterDType(StrEnum):
    """Trainable-parameter dtypes implemented by the public trainer."""

    FLOAT32 = "float32"


class MixedPrecision(StrEnum):
    """Accelerate mixed-precision modes accepted by the trainer."""

    NONE = "none"
    FLOAT16 = "fp16"
    BFLOAT16 = "bf16"


class RetrievalMetric(StrEnum):
    """Full-corpus retrieval metrics emitted by the evaluation pipeline."""

    NDCG = "NDCG"
    RECALL = "Recall"
    PRECISION = "P"
    MAP = "MAP"


class RetrievalSimilarity(StrEnum):
    """Similarity functions implemented by full-corpus evaluation."""

    NORMALIZED_COSINE = "normalized_cosine"


class EvaluationAggregation(StrEnum):
    """How paper results aggregate independently evaluated datasets."""

    DATASET_MACRO = "dataset_macro"


_METRIC_PATTERN = re.compile(r"^(NDCG|Recall|P|MAP)@([1-9][0-9]*)$")
EnumT = TypeVar("EnumT", bound=StrEnum)


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}.")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{context} keys must be strings.")
    return dict(value)


def _validate_keys(
    values: Mapping[str, Any],
    *,
    context: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required.difference(values))
    unknown = sorted(set(values).difference(required | optional))
    if missing:
        raise ConfigError(f"{context} is missing required fields: {missing}.")
    if unknown:
        raise ConfigError(f"{context} contains unsupported fields: {unknown}.")


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string.")
    return value


def _integer(value: object, *, context: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{context} must be an integer greater than or equal to {minimum}.")
    return value


def _number(
    value: object,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{context} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{context} must be greater than or equal to {minimum}.")
    if maximum is not None and (result > maximum if maximum_inclusive else result >= maximum):
        qualifier = "less than or equal to" if maximum_inclusive else "less than"
        raise ConfigError(f"{context} must be {qualifier} {maximum}.")
    return result


def _boolean(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be a boolean.")
    return value


def _enum_value(enum_type: type[EnumT], value: object, *, context: str) -> EnumT:
    if not isinstance(value, str):
        raise ConfigError(f"{context} must be one of {[item.value for item in enum_type]}.")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ConfigError(f"{context} must be one of {[item.value for item in enum_type]}; got {value!r}.") from error


@dataclass(frozen=True)
class MetricSpec:
    """One named full-corpus retrieval metric at an explicit cutoff."""

    name: RetrievalMetric
    cutoff: int

    @classmethod
    def parse(cls, value: object, *, context: str) -> Self:
        label = _string(value, context=context)
        match = _METRIC_PATTERN.fullmatch(label)
        if match is None:
            raise ConfigError(f"{context} must have the form NDCG@10, Recall@100, P@10, or MAP@10.")
        return cls(name=RetrievalMetric(match.group(1)), cutoff=int(match.group(2)))

    @property
    def label(self) -> str:
        """Return the metric key used in evaluation artifacts."""

        return f"{self.name.value}@{self.cutoff}"


@dataclass(frozen=True)
class LoraConfig:
    """Typed LoRA adapter settings from ``training.lora``."""

    rank: int
    alpha: int
    dropout: float
    target_modules: str | tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object, *, context: str = "training.lora") -> Self:
        values = _mapping(value, context=context)
        _validate_keys(
            values,
            context=context,
            required=frozenset({"rank", "alpha", "dropout", "target_modules"}),
        )
        raw_targets = values["target_modules"]
        if isinstance(raw_targets, str):
            targets: str | tuple[str, ...] = _string(raw_targets, context=f"{context}.target_modules")
        elif isinstance(raw_targets, list) and raw_targets:
            targets = tuple(
                _string(item, context=f"{context}.target_modules[{index}]") for index, item in enumerate(raw_targets)
            )
        else:
            raise ConfigError(f"{context}.target_modules must be a non-empty string or list of strings.")
        return cls(
            rank=_integer(values["rank"], context=f"{context}.rank"),
            alpha=_integer(values["alpha"], context=f"{context}.alpha"),
            dropout=_number(
                values["dropout"],
                context=f"{context}.dropout",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
            target_modules=targets,
        )


@dataclass(frozen=True)
class CandidateCheckpointingConfig:
    """Checkpoint policy for scoring each candidate row."""

    document_chunk_size: int
    preserve_rng_state: bool
    encode_query_once: bool

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str = "training.candidate_checkpointing",
    ) -> Self:
        values = _mapping(value, context=context)
        _validate_keys(
            values,
            context=context,
            required=frozenset({"document_chunk_size", "preserve_rng_state", "encode_query_once"}),
        )
        settings = cls(
            document_chunk_size=_integer(
                values["document_chunk_size"],
                context=f"{context}.document_chunk_size",
                minimum=0,
            ),
            preserve_rng_state=_boolean(values["preserve_rng_state"], context=f"{context}.preserve_rng_state"),
            encode_query_once=_boolean(values["encode_query_once"], context=f"{context}.encode_query_once"),
        )
        if not settings.encode_query_once:
            raise ConfigError("Paper runs require training.candidate_checkpointing.encode_query_once=true.")
        if not settings.preserve_rng_state:
            raise ConfigError("Paper runs require training.candidate_checkpointing.preserve_rng_state=true.")
        return settings

    def realized_document_chunk_size(self, *, candidate_count: int) -> int:
        """Resolve zero to one checkpoint spanning the complete candidate row."""

        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        return self.document_chunk_size or candidate_count


@dataclass(frozen=True)
class TrainingConfig:
    """Complete executable optimizer and batching protocol for one run."""

    loss: DistillationLoss
    epochs: int
    seeds: tuple[int, ...]
    per_device_batch_size: int
    gradient_accumulation_steps: int
    reference_world_size: int
    effective_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    scheduler: SchedulerKind
    max_gradient_norm: float
    parameter_dtype: ParameterDType
    mixed_precision: MixedPrecision
    backbone_gradient_checkpointing: bool
    lora: LoraConfig
    candidate_checkpointing: CandidateCheckpointingConfig
    eval_batch_size: int = 1
    loss_temperature: float | None = None

    @classmethod
    def from_mapping(cls, value: object, *, context: str = "training") -> Self:
        values = _mapping(value, context=context)
        required = frozenset(
            {
                "loss",
                "epochs",
                "seeds",
                "per_device_batch_size",
                "gradient_accumulation_steps",
                "reference_world_size",
                "effective_batch_size",
                "learning_rate",
                "weight_decay",
                "warmup_ratio",
                "scheduler",
                "max_gradient_norm",
                "parameter_dtype",
                "mixed_precision",
                "backbone_gradient_checkpointing",
                "lora",
                "candidate_checkpointing",
            }
        )
        _validate_keys(
            values,
            context=context,
            required=required,
            optional=frozenset({"eval_batch_size", "loss_temperature"}),
        )
        raw_seeds = values["seeds"]
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ConfigError(f"{context}.seeds must be a non-empty list of integers.")
        seeds = tuple(
            _integer(seed, context=f"{context}.seeds[{index}]", minimum=0) for index, seed in enumerate(raw_seeds)
        )
        settings = cls(
            loss=_enum_value(DistillationLoss, values["loss"], context=f"{context}.loss"),
            epochs=_integer(values["epochs"], context=f"{context}.epochs"),
            seeds=seeds,
            per_device_batch_size=_integer(
                values["per_device_batch_size"],
                context=f"{context}.per_device_batch_size",
            ),
            gradient_accumulation_steps=_integer(
                values["gradient_accumulation_steps"],
                context=f"{context}.gradient_accumulation_steps",
            ),
            reference_world_size=_integer(
                values["reference_world_size"],
                context=f"{context}.reference_world_size",
            ),
            effective_batch_size=_integer(
                values["effective_batch_size"],
                context=f"{context}.effective_batch_size",
            ),
            learning_rate=_number(values["learning_rate"], context=f"{context}.learning_rate", minimum=0.0),
            weight_decay=_number(values["weight_decay"], context=f"{context}.weight_decay", minimum=0.0),
            warmup_ratio=_number(
                values["warmup_ratio"],
                context=f"{context}.warmup_ratio",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
            scheduler=_enum_value(SchedulerKind, values["scheduler"], context=f"{context}.scheduler"),
            max_gradient_norm=_number(
                values["max_gradient_norm"],
                context=f"{context}.max_gradient_norm",
                minimum=0.0,
            ),
            parameter_dtype=_enum_value(
                ParameterDType,
                values["parameter_dtype"],
                context=f"{context}.parameter_dtype",
            ),
            mixed_precision=_enum_value(
                MixedPrecision,
                values["mixed_precision"],
                context=f"{context}.mixed_precision",
            ),
            backbone_gradient_checkpointing=_boolean(
                values["backbone_gradient_checkpointing"],
                context=f"{context}.backbone_gradient_checkpointing",
            ),
            lora=LoraConfig.from_mapping(values["lora"], context=f"{context}.lora"),
            candidate_checkpointing=CandidateCheckpointingConfig.from_mapping(
                values["candidate_checkpointing"],
                context=f"{context}.candidate_checkpointing",
            ),
            eval_batch_size=_integer(
                values.get("eval_batch_size", 1),
                context=f"{context}.eval_batch_size",
            ),
            loss_temperature=(
                None
                if values.get("loss_temperature") is None
                else _number(
                    values["loss_temperature"],
                    context=f"{context}.loss_temperature",
                    minimum=0.0,
                    maximum_inclusive=False,
                )
            ),
        )
        raw_loss = values["loss"]
        if raw_loss == DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY and settings.loss_temperature is None:
            raise ConfigError(
                f"{context}.loss_temperature is required when {context}.loss="
                f"{DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY.value!r}."
            )
        if raw_loss != DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY and settings.loss_temperature is not None:
            raise ConfigError(
                f"{context}.loss_temperature is only supported when {context}.loss="
                f"{DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY.value!r}."
            )
        expected_batch = (
            settings.per_device_batch_size * settings.gradient_accumulation_steps * settings.reference_world_size
        )
        if settings.effective_batch_size != expected_batch:
            raise ConfigError(
                f"{context}.effective_batch_size={settings.effective_batch_size} does not match "
                f"per_device_batch_size * gradient_accumulation_steps * reference_world_size={expected_batch}."
            )
        if settings.learning_rate <= 0.0:
            raise ConfigError(f"{context}.learning_rate must be positive.")
        if settings.max_gradient_norm <= 0.0:
            raise ConfigError(f"{context}.max_gradient_norm must be positive.")
        return settings


@dataclass(frozen=True)
class EvaluationConfig:
    """Protocol for separate full-corpus retrieval evaluation.

    This does not configure the trainer's per-epoch candidate-row validation
    loss.  It is a typed contract for the independent retrieval pipeline and
    paper-level aggregation after each dataset has been evaluated.
    """

    primary_metric: MetricSpec
    metric_k_values: tuple[int, ...]
    similarity: RetrievalSimilarity
    aggregation: EvaluationAggregation
    top_k: int

    @classmethod
    def from_mapping(cls, value: object, *, context: str = "evaluation") -> Self:
        values = _mapping(value, context=context)
        _validate_keys(
            values,
            context=context,
            required=frozenset({"primary_metric", "metric_k_values", "similarity", "aggregation", "top_k"}),
        )
        raw_cutoffs = values["metric_k_values"]
        if not isinstance(raw_cutoffs, list) or not raw_cutoffs:
            raise ConfigError(f"{context}.metric_k_values must be a non-empty sorted list of positive integers.")
        cutoffs = tuple(
            _integer(value, context=f"{context}.metric_k_values[{index}]") for index, value in enumerate(raw_cutoffs)
        )
        if cutoffs != tuple(sorted(set(cutoffs))):
            raise ConfigError(f"{context}.metric_k_values must be sorted and contain no duplicates.")
        settings = cls(
            primary_metric=MetricSpec.parse(values["primary_metric"], context=f"{context}.primary_metric"),
            metric_k_values=cutoffs,
            similarity=_enum_value(
                RetrievalSimilarity,
                values["similarity"],
                context=f"{context}.similarity",
            ),
            aggregation=_enum_value(
                EvaluationAggregation,
                values["aggregation"],
                context=f"{context}.aggregation",
            ),
            top_k=_integer(values["top_k"], context=f"{context}.top_k"),
        )
        if settings.primary_metric.cutoff not in settings.metric_k_values:
            raise ConfigError(f"{context}.primary_metric cutoff must be included in metric_k_values.")
        if settings.top_k < max(settings.metric_k_values):
            raise ConfigError(f"{context}.top_k must cover the largest metric_k_values cutoff.")
        return settings


@dataclass(frozen=True)
class ReferenceHardwareConfig:
    """Reference topology recorded alongside an executable paper run."""

    accelerator: str
    count: int

    @classmethod
    def from_mapping(cls, value: object, *, context: str = "reference_hardware") -> Self:
        values = _mapping(value, context=context)
        _validate_keys(values, context=context, required=frozenset({"accelerator", "count"}))
        return cls(
            accelerator=_string(values["accelerator"], context=f"{context}.accelerator"),
            count=_integer(values["count"], context=f"{context}.count"),
        )


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _assert_mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping, got {type(value).__name__}.")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class ResolvedConfig:
    """A parsed configuration paired with its stable identity."""

    path: Path
    data: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class ExecutableRunConfig:
    """Strict, discoverable view of one fully inherited training run.

    ``resolved`` retains the canonical merged mapping and its SHA-256 identity;
    parsing this typed view never rewrites or re-hashes configuration data.
    """

    resolved: ResolvedConfig
    schema_version: int
    name: str
    data_config: Path
    student_config: Path
    teacher_configs: tuple[Path, ...]
    teacher_weights: dict[str, float]
    training: TrainingConfig
    evaluation: EvaluationConfig
    reference_hardware: ReferenceHardwareConfig

    @classmethod
    def from_resolved(cls, resolved: ResolvedConfig) -> Self:
        values = resolved.data
        raw_kind = values.get("config_kind")
        if raw_kind == ConfigKind.SWEEP_INDEX:
            raise ConfigError(
                f"{resolved.path} is a sweep_index, not an executable run. "
                "Execute one of the concrete control run configs it indexes."
            )
        kind = _enum_value(ConfigKind, raw_kind, context="config_kind")
        if kind is not ConfigKind.EXECUTABLE_RUN:  # pragma: no cover - enum currently has two roles
            raise ConfigError(f"{resolved.path} is not an executable run.")
        _validate_keys(
            values,
            context="run config",
            required=frozenset(
                {
                    "schema_version",
                    "config_kind",
                    "name",
                    "data_config",
                    "student_config",
                    "teacher_configs",
                    "teacher_weights",
                    "training",
                    "evaluation",
                    "reference_hardware",
                }
            ),
        )
        schema_version = _integer(values["schema_version"], context="schema_version")
        if schema_version != 1:
            raise ConfigError(f"Unsupported executable run schema_version={schema_version}; expected 1.")

        def relative_yaml_path(value: object, *, context: str) -> Path:
            reference = _string(value, context=context)
            path = Path(reference)
            if path.is_absolute():
                raise ConfigError(f"{context} must be relative to the run configuration.")
            resolved_path = (resolved.path.parent / path).resolve()
            if resolved_path.suffix.lower() not in {".yaml", ".yml"}:
                raise ConfigError(f"{context} must reference a YAML file.")
            return resolved_path

        raw_teachers = values["teacher_configs"]
        if not isinstance(raw_teachers, list) or not raw_teachers:
            raise ConfigError("teacher_configs must be a non-empty list of relative YAML paths.")
        teacher_configs = tuple(
            relative_yaml_path(value, context=f"teacher_configs[{index}]") for index, value in enumerate(raw_teachers)
        )
        raw_weights = _mapping(values["teacher_weights"], context="teacher_weights")
        if not raw_weights:
            raise ConfigError("teacher_weights must not be empty.")
        teacher_weights = {
            _string(label, context="teacher_weights label"): _number(
                weight,
                context=f"teacher_weights.{label}",
                minimum=0.0,
            )
            for label, weight in raw_weights.items()
        }
        if not any(weight > 0.0 for weight in teacher_weights.values()):
            raise ConfigError("teacher_weights must contain at least one positive weight.")
        training = TrainingConfig.from_mapping(values["training"])
        evaluation = EvaluationConfig.from_mapping(values["evaluation"])
        reference_hardware = ReferenceHardwareConfig.from_mapping(values["reference_hardware"])
        if reference_hardware.count != training.reference_world_size:
            raise ConfigError(
                "reference_hardware.count must match training.reference_world_size; "
                f"got {reference_hardware.count} and {training.reference_world_size}."
            )
        return cls(
            resolved=resolved,
            schema_version=schema_version,
            name=_string(values["name"], context="name"),
            data_config=relative_yaml_path(values["data_config"], context="data_config"),
            student_config=relative_yaml_path(values["student_config"], context="student_config"),
            teacher_configs=teacher_configs,
            teacher_weights=teacher_weights,
            training=training,
            evaluation=evaluation,
            reference_hardware=reference_hardware,
        )

    @property
    def path(self) -> Path:
        """Path of the leaf run configuration used for inheritance."""

        return self.resolved.path

    @property
    def sha256(self) -> str:
        """Canonical hash of the raw, fully inherited executable mapping."""

        return self.resolved.sha256


def load_config(path: str | Path) -> ResolvedConfig:
    """Load a YAML mapping and compute its canonical SHA-256 digest."""

    resolved_path = Path(path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data = _assert_mapping(raw, context=str(resolved_path))
    return ResolvedConfig(path=resolved_path, data=data, sha256=content_hash(data))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {str(key): value for key, value in base.items()}
    for key, value in override.items():
        normalized_key = str(key)
        existing = merged.get(normalized_key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[normalized_key] = _deep_merge(existing, value)
        else:
            merged[normalized_key] = value
    return merged


def resolve_run_config(path: str | Path, *, _seen: frozenset[Path] = frozenset()) -> ResolvedConfig:
    """Resolve ``base_run`` inheritance and hash the executable configuration."""

    current = load_config(path)
    if current.path in _seen:
        chain = ", ".join(str(value) for value in (*_seen, current.path))
        raise ConfigError(f"Circular base_run configuration chain: {chain}")
    base_value = current.data.get("base_run")
    if base_value is None:
        return current
    if not isinstance(base_value, str) or not base_value.strip():
        raise ConfigError("base_run must be a non-empty YAML path.")
    base = resolve_run_config(
        current.path.parent / base_value,
        _seen=_seen | {current.path},
    )
    override = {key: value for key, value in current.data.items() if key != "base_run"}
    merged = _deep_merge(base.data, override)
    return ResolvedConfig(path=current.path, data=merged, sha256=content_hash(merged))


def load_executable_run_config(path: str | Path) -> ExecutableRunConfig:
    """Resolve inheritance, validate every executable field, and return typed settings."""

    return ExecutableRunConfig.from_resolved(resolve_run_config(path))
