"""Typed schemas shared by the public data pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

IssueSeverity = Literal["error", "warning"]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class DataSchemaError(ValueError):
    """Raised when a data artifact violates the public schema."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable configuration validation finding."""

    path: str
    message: str
    severity: IssueSeverity
    code: str


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    """One public dataset archive with an optional immutable checksum."""

    url: str
    archive_name: str
    sha256: str | None
    upstream_md5: str | None = None

    def validation_issues(self, *, path: str) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not self.url.strip():
            issues.append(ValidationIssue(f"{path}.url", "must not be empty", "error", "empty_url"))
        if not self.archive_name.strip() or Path(self.archive_name).name != self.archive_name:
            issues.append(
                ValidationIssue(
                    f"{path}.archive_name",
                    "must be a non-empty filename without directory components",
                    "error",
                    "invalid_archive_name",
                )
            )
        normalized_hash = self.sha256.lower() if self.sha256 is not None else None
        if normalized_hash is not None and _SHA256_PATTERN.fullmatch(normalized_hash) is None:
            issues.append(
                ValidationIssue(
                    f"{path}.sha256",
                    "must be exactly 64 hexadecimal characters",
                    "error",
                    "invalid_sha256",
                )
            )
        if normalized_hash is None:
            issues.append(
                ValidationIssue(
                    path,
                    "snapshot has no SHA-256; exact reproduction is not pinned",
                    "warning",
                    "unresolved_pin",
                )
            )
        normalized_md5 = self.upstream_md5.lower() if self.upstream_md5 is not None else None
        if normalized_md5 is not None and _MD5_PATTERN.fullmatch(normalized_md5) is None:
            issues.append(
                ValidationIssue(
                    f"{path}.upstream_md5",
                    "must be exactly 32 hexadecimal characters",
                    "error",
                    "invalid_upstream_md5",
                )
            )
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Public name and snapshot metadata for one retrieval dataset."""

    name: str
    snapshot: SnapshotSpec
    expected_missing_qrel_documents: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.expected_missing_qrel_documents, int)
            or isinstance(self.expected_missing_qrel_documents, bool)
            or self.expected_missing_qrel_documents < 0
        ):
            raise DataSchemaError("expected_missing_qrel_documents must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A named split used for row construction or full-corpus evaluation."""

    dataset: str
    split: str
    max_rows: int | None = None
    expected_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.dataset.strip() or not self.split.strip():
            raise DataSchemaError("dataset and split must not be empty")
        if self.max_rows is not None and self.max_rows <= 0:
            raise DataSchemaError("max_rows must be positive when provided")
        if self.expected_rows is not None and self.expected_rows <= 0:
            raise DataSchemaError("expected_rows must be positive when provided")


@dataclass(frozen=True, slots=True)
class CandidateConstruction:
    """Deterministic candidate-row construction settings."""

    candidate_count: int
    positive_index: int
    seed: int
    tfidf_max_features: int
    tfidf_top_pool: int
    query_batch_size: int
    require_disjoint_query_ids: bool

    def __post_init__(self) -> None:
        if self.candidate_count < 2:
            raise DataSchemaError("candidate_count must be at least two")
        if self.positive_index != 0:
            raise DataSchemaError("the public row schema requires positive_index == 0")
        if self.tfidf_max_features <= 0 or self.tfidf_top_pool <= 0 or self.query_batch_size <= 0:
            raise DataSchemaError("TF-IDF limits and query_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class PanelConfig:
    """The paper's dataset panel and row-construction protocol."""

    schema_version: int
    provider: str
    citation_title: str
    citation_url: str
    datasets: tuple[DatasetSpec, ...]
    train: tuple[DatasetSplit, ...]
    row_eval: tuple[DatasetSplit, ...]
    full_corpus_eval: tuple[DatasetSplit, ...]
    candidates: CandidateConstruction


@dataclass(frozen=True, slots=True)
class Document:
    """A retrieval document loaded directly from a snapshot."""

    document_id: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}".strip() if self.title.strip() else self.text.strip()


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval query loaded directly from a snapshot."""

    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Qrel:
    """A graded query-document relevance judgment."""

    query_id: str
    document_id: str
    relevance: float


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    """A validated in-memory view of one dataset split."""

    name: str
    split: str
    documents: dict[str, Document]
    queries: dict[str, Query]
    qrels: tuple[Qrel, ...]
    missing_qrel_document_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """One score-vector row with nested candidate IDs and texts.

    Candidate text can be empty when an immutable public snapshot contains an
    addressable document with no title or body.  Candidate IDs remain the
    stable identity and must still be unique within the row.
    """

    row_id: str
    split: str
    dataset: str
    source_split: str
    query_id: str
    query_text: str
    positive_id: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    negative_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.candidate_ids) < 2:
            raise DataSchemaError("a candidate row must contain one positive and at least one negative")
        if len(self.candidate_ids) != len(self.candidate_texts):
            raise DataSchemaError("candidate_ids and candidate_texts must have equal lengths")
        if len(self.negative_sources) != len(self.candidate_ids) - 1:
            raise DataSchemaError("negative_sources must describe every non-positive candidate")
        if self.candidate_ids[0] != self.positive_id:
            raise DataSchemaError("positive_id must be candidate_ids[0]")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise DataSchemaError("candidate IDs must be unique within a row")

    @property
    def positive_index(self) -> int:
        """The fixed positive position in the public schema."""

        return 0

    def to_dict(self) -> dict[str, object]:
        """Return the stable nested serialization contract used by scoring."""

        return {
            "row_id": self.row_id,
            "split": self.split,
            "dataset": self.dataset,
            "source_split": self.source_split,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "positive_id": self.positive_id,
            "candidate_ids": list(self.candidate_ids),
            "candidate_texts": list(self.candidate_texts),
            "negative_sources": list(self.negative_sources),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CandidateRow:
        """Validate and decode one serialized candidate row."""

        def required_text(key: str) -> str:
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                raise DataSchemaError(f"{key} must be a non-empty string")
            return value

        def required_text_list(key: str) -> tuple[str, ...]:
            value = payload.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise DataSchemaError(f"{key} must be a list of strings")
            return tuple(value)

        return cls(
            row_id=required_text("row_id"),
            split=required_text("split"),
            dataset=required_text("dataset"),
            source_split=required_text("source_split"),
            query_id=required_text("query_id"),
            query_text=required_text("query_text"),
            positive_id=required_text("positive_id"),
            candidate_ids=required_text_list("candidate_ids"),
            candidate_texts=required_text_list("candidate_texts"),
            negative_sources=required_text_list("negative_sources"),
        )
