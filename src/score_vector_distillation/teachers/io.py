"""Strict teacher-score ingestion and aligned training-table materialization."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from score_vector_distillation.data.candidates import read_candidate_rows_jsonl
from score_vector_distillation.teachers.fusion import fuse_score_rows, normalize_weights


class TeacherTableError(ValueError):
    """Raised when teacher scores do not satisfy the aligned table contract."""


@dataclass(frozen=True, slots=True)
class TeacherScoreTable:
    """One validated, fixed-width score table keyed by candidate-row ID."""

    model_label: str
    scores: Mapping[str, tuple[float, ...]]
    width: int


@dataclass(frozen=True, slots=True)
class FusedTrainingArtifact:
    """Summary of one atomically written nested training table."""

    path: Path
    row_count: int
    candidate_count: int
    teacher_labels: tuple[str, ...]


def _validated_expected_ids(expected_row_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_row_ids is None:
        return None
    materialized = tuple(expected_row_ids)
    if not materialized or any(not row_id for row_id in materialized):
        raise TeacherTableError("expected_row_ids must contain non-empty row IDs")
    if len(materialized) != len(set(materialized)):
        raise TeacherTableError("expected_row_ids contains duplicates")
    return materialized


def read_teacher_score_table(
    path: str | Path,
    *,
    expected_model_label: str,
    expected_row_ids: Sequence[str] | None = None,
    expected_width: int | None = None,
) -> TeacherScoreTable:
    """Read a score Parquet with strict label, ID, width, and value checks."""

    if not expected_model_label:
        raise TeacherTableError("expected_model_label must not be empty")
    if expected_width is not None and expected_width <= 0:
        raise TeacherTableError("expected_width must be positive")
    expected_ids = _validated_expected_ids(expected_row_ids)
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        frame = pd.read_parquet(source)
    except Exception as error:
        raise TeacherTableError(f"cannot read teacher score table {source.name}: {error}") from error

    required = {"row_id", "model_label", "scores"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise TeacherTableError(f"teacher score table is missing columns: {missing}")
    if frame.empty:
        raise TeacherTableError("teacher score table must contain at least one row")

    parsed: dict[str, tuple[float, ...]] = {}
    width: int | None = expected_width
    for index, (raw_row_id, raw_model_label, raw_vector) in enumerate(
        zip(frame["row_id"], frame["model_label"], frame["scores"], strict=True)
    ):
        if not isinstance(raw_row_id, str) or not raw_row_id:
            raise TeacherTableError(f"row {index} has an invalid row_id")
        if raw_row_id in parsed:
            raise TeacherTableError(f"duplicate row_id in teacher score table: {raw_row_id!r}")
        if not isinstance(raw_model_label, str) or raw_model_label != expected_model_label:
            raise TeacherTableError(
                f"model-label mismatch at row {raw_row_id!r}: "
                f"expected {expected_model_label!r}, got {raw_model_label!r}"
            )
        try:
            vector = np.asarray(raw_vector, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TeacherTableError(f"row {raw_row_id!r} has non-numeric scores") from error
        if vector.ndim != 1 or vector.size == 0:
            raise TeacherTableError(f"row {raw_row_id!r} scores must be a non-empty vector")
        if not np.isfinite(vector).all():
            raise TeacherTableError(f"row {raw_row_id!r} scores contain non-finite values")
        current_width = int(vector.size)
        if width is None:
            width = current_width
        elif current_width != width:
            raise TeacherTableError(
                f"score-width mismatch at row {raw_row_id!r}: expected {width}, got {current_width}"
            )
        parsed[raw_row_id] = tuple(float(value) for value in vector)

    assert width is not None
    if expected_ids is not None and set(parsed) != set(expected_ids):
        missing_ids = sorted(set(expected_ids) - set(parsed))[:5]
        extra_ids = sorted(set(parsed) - set(expected_ids))[:5]
        raise TeacherTableError(f"row-ID mismatch: missing={missing_ids}, extra={extra_ids}")
    return TeacherScoreTable(model_label=expected_model_label, scores=parsed, width=width)


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.parquet",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fuse_teacher_tables_to_training_parquet(
    candidate_rows_path: str | Path,
    teacher_score_paths: Mapping[str, str | Path],
    weights: Mapping[str, float],
    output_path: str | Path,
) -> FusedTrainingArtifact:
    """Fuse aligned teacher scores and join them to candidate rows.

    Candidate JSONL order is the canonical output order. Teacher Parquet rows
    may be stored in any order, but their row-ID sets must match exactly.
    """

    rows = read_candidate_rows_jsonl(candidate_rows_path)
    if not rows:
        raise TeacherTableError("candidate row artifact must contain at least one row")
    row_ids = tuple(row.row_id for row in rows)
    if len(row_ids) != len(set(row_ids)):
        raise TeacherTableError("candidate row artifact contains duplicate row IDs")
    widths = {len(row.candidate_ids) for row in rows}
    if len(widths) != 1:
        raise TeacherTableError(f"candidate rows have inconsistent widths: {sorted(widths)}")
    candidate_count = widths.pop()

    score_labels = set(teacher_score_paths)
    weight_labels = set(weights)
    if not score_labels:
        raise TeacherTableError("at least one teacher score table is required")
    if score_labels != weight_labels:
        missing_weights = sorted(score_labels - weight_labels)
        missing_tables = sorted(weight_labels - score_labels)
        raise TeacherTableError(
            f"teacher/weight labels differ: missing_weights={missing_weights}, missing_tables={missing_tables}"
        )

    labels = tuple(sorted(score_labels))
    tables = {
        label: read_teacher_score_table(
            teacher_score_paths[label],
            expected_model_label=label,
            expected_row_ids=row_ids,
            expected_width=candidate_count,
        )
        for label in labels
    }
    normalized_weights = normalize_weights({label: float(weights[label]) for label in labels})
    fused = fuse_score_rows({label: tables[label].scores for label in labels}, normalized_weights)
    if set(fused) != set(row_ids):
        raise TeacherTableError("internal fusion alignment failure")

    records = [
        {
            "row_id": row.row_id,
            "split": row.split,
            "dataset": row.dataset,
            "source_split": row.source_split,
            "query_id": row.query_id,
            "query_text": row.query_text,
            "positive_id": row.positive_id,
            "positive_index": row.positive_index,
            "candidate_count": len(row.candidate_ids),
            "candidate_ids": list(row.candidate_ids),
            "candidate_texts": list(row.candidate_texts),
            "negative_sources": list(row.negative_sources),
            "teacher_scores": fused[row.row_id],
        }
        for row in rows
    ]
    destination = Path(output_path)
    _write_parquet_atomic(pd.DataFrame.from_records(records), destination)
    return FusedTrainingArtifact(
        path=destination,
        row_count=len(rows),
        candidate_count=candidate_count,
        teacher_labels=labels,
    )
