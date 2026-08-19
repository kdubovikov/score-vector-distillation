"""Load and validate snapshot registries without importing a benchmark SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from score_vector_distillation.data.schema import (
    CandidateConstruction,
    DatasetSpec,
    DatasetSplit,
    PanelConfig,
    SnapshotSpec,
    ValidationIssue,
)


class RegistryValidationError(ValueError):
    """Raised when a dataset registry is unsafe or internally inconsistent."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(f"dataset registry validation failed: {details}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a list")
    return value


def _text(mapping: Mapping[str, object], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path}.{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, path: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be an integer")
    return value


def _optional_integer(mapping: Mapping[str, object], key: str, path: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be an integer or null")
    return value


def _nonnegative_integer(mapping: Mapping[str, object], key: str, path: str, *, default: int = 0) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{path}.{key} must be a non-negative integer")
    return value


def _boolean(mapping: Mapping[str, object], key: str, path: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be a boolean")
    return value


def _parse_split(value: object, path: str) -> DatasetSplit:
    payload = _mapping(value, path)
    return DatasetSplit(
        dataset=_text(payload, "dataset", path),
        split=_text(payload, "split", path),
        max_rows=_optional_integer(payload, "max_rows", path),
        expected_rows=_optional_integer(payload, "expected_rows", path),
    )


def _parse_dataset(value: object, path: str) -> DatasetSpec:
    payload = _mapping(value, path)
    snapshot_payload = _mapping(payload.get("snapshot"), f"{path}.snapshot")
    raw_sha256 = snapshot_payload.get("sha256")
    if raw_sha256 is not None and not isinstance(raw_sha256, str):
        raise TypeError(f"{path}.snapshot.sha256 must be a string or null")
    raw_upstream_md5 = snapshot_payload.get("upstream_md5")
    if raw_upstream_md5 is not None and not isinstance(raw_upstream_md5, str):
        raise TypeError(f"{path}.snapshot.upstream_md5 must be a string or null")
    return DatasetSpec(
        name=_text(payload, "name", path),
        snapshot=SnapshotSpec(
            url=_text(snapshot_payload, "url", f"{path}.snapshot"),
            archive_name=_text(snapshot_payload, "archive_name", f"{path}.snapshot"),
            sha256=raw_sha256,
            upstream_md5=raw_upstream_md5,
        ),
        expected_missing_qrel_documents=_nonnegative_integer(
            payload,
            "expected_missing_qrel_documents",
            path,
        ),
    )


def _parse_panel(payload: Mapping[str, object]) -> PanelConfig:
    provenance = _mapping(payload.get("provenance"), "provenance")
    citation = _mapping(provenance.get("citation"), "provenance.citation")
    row_construction = _mapping(payload.get("row_construction"), "row_construction")
    candidate_payload = _mapping(row_construction.get("candidates"), "row_construction.candidates")
    datasets = tuple(
        _parse_dataset(value, f"datasets[{index}]")
        for index, value in enumerate(_sequence(payload.get("datasets"), "datasets"))
    )
    train = tuple(
        _parse_split(value, f"row_construction.train[{index}]")
        for index, value in enumerate(_sequence(row_construction.get("train"), "row_construction.train"))
    )
    row_eval = tuple(
        _parse_split(value, f"row_construction.eval[{index}]")
        for index, value in enumerate(_sequence(row_construction.get("eval"), "row_construction.eval"))
    )
    full_corpus_eval = tuple(
        _parse_split(value, f"full_corpus_evaluation[{index}]")
        for index, value in enumerate(_sequence(payload.get("full_corpus_evaluation"), "full_corpus_evaluation"))
    )
    return PanelConfig(
        schema_version=_integer(payload, "schema_version", "root"),
        provider=_text(provenance, "provider", "provenance"),
        citation_title=_text(citation, "title", "provenance.citation"),
        citation_url=_text(citation, "url", "provenance.citation"),
        datasets=datasets,
        train=train,
        row_eval=row_eval,
        full_corpus_eval=full_corpus_eval,
        candidates=CandidateConstruction(
            candidate_count=_integer(candidate_payload, "count", "row_construction.candidates"),
            positive_index=_integer(candidate_payload, "positive_index", "row_construction.candidates"),
            seed=_integer(candidate_payload, "seed", "row_construction.candidates"),
            tfidf_max_features=_integer(
                candidate_payload,
                "tfidf_max_features",
                "row_construction.candidates",
            ),
            tfidf_top_pool=_integer(candidate_payload, "tfidf_top_pool", "row_construction.candidates"),
            query_batch_size=_integer(candidate_payload, "query_batch_size", "row_construction.candidates"),
            require_disjoint_query_ids=_boolean(
                candidate_payload,
                "require_disjoint_query_ids",
                "row_construction.candidates",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class DatasetRegistry:
    """Validated access to a paper dataset panel."""

    panel: PanelConfig

    def dataset(self, name: str) -> DatasetSpec:
        """Return a dataset by its publication-facing name."""

        for dataset in self.panel.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(f"unknown dataset: {name}")

    def validation_issues(self) -> tuple[ValidationIssue, ...]:
        """Return errors and explicit unresolved-pin warnings."""

        issues: list[ValidationIssue] = []
        if self.panel.schema_version != 1:
            issues.append(ValidationIssue("schema_version", "only schema version 1 is supported", "error", "schema"))
        names = [dataset.name for dataset in self.panel.datasets]
        if len(names) != len(set(names)):
            issues.append(ValidationIssue("datasets", "dataset names must be unique", "error", "duplicate_dataset"))
        known = set(names)
        for group_name, group in (
            ("row_construction.train", self.panel.train),
            ("row_construction.eval", self.panel.row_eval),
            ("full_corpus_evaluation", self.panel.full_corpus_eval),
        ):
            for index, split in enumerate(group):
                if split.dataset not in known:
                    issues.append(
                        ValidationIssue(
                            f"{group_name}[{index}].dataset",
                            f"references unknown dataset {split.dataset!r}",
                            "error",
                            "unknown_dataset",
                        )
                    )
        if self.panel.candidates.require_disjoint_query_ids:
            train_sources = {(item.dataset, item.split) for item in self.panel.train}
            eval_sources = {(item.dataset, item.split) for item in self.panel.row_eval}
            for dataset_name, split_name in sorted(train_sources & eval_sources):
                issues.append(
                    ValidationIssue(
                        "row_construction.eval",
                        f"{dataset_name}:{split_name} is also a training source; "
                        "query disjointness cannot be guaranteed",
                        "error",
                        "shared_source_split",
                    )
                )
        for index, dataset_spec in enumerate(self.panel.datasets):
            issues.extend(dataset_spec.snapshot.validation_issues(path=f"datasets[{index}].snapshot"))
        return tuple(issues)

    def validate(self, *, require_resolved_pins: bool = False) -> None:
        """Raise for invalid configuration and, optionally, every unresolved pin."""

        blocking = tuple(
            issue
            for issue in self.validation_issues()
            if issue.severity == "error" or (require_resolved_pins and issue.code == "unresolved_pin")
        )
        if blocking:
            raise RegistryValidationError(blocking)


def load_registry(path: str | Path, *, require_resolved_pins: bool = False) -> DatasetRegistry:
    """Load YAML, validate references, and preserve unresolved-pin warnings."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    registry = DatasetRegistry(_parse_panel(_mapping(raw, "root")))
    registry.validate(require_resolved_pins=require_resolved_pins)
    return registry
