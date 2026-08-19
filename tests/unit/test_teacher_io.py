from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from score_vector_distillation.data.candidates import write_candidate_rows_jsonl
from score_vector_distillation.data.schema import CandidateRow
from score_vector_distillation.teachers.io import (
    TeacherTableError,
    fuse_teacher_tables_to_training_parquet,
    read_teacher_score_table,
)
from score_vector_distillation.teachers.scoring import TeacherScores, write_score_table


def _candidate_rows() -> tuple[CandidateRow, ...]:
    return (
        CandidateRow(
            row_id="r1",
            split="train",
            dataset="Toy",
            source_split="train",
            query_id="Toy:q1",
            query_text="first query",
            positive_id="Toy:d1",
            candidate_ids=("Toy:d1", "Toy:d2"),
            candidate_texts=("first positive", "first negative"),
            negative_sources=("tfidf",),
        ),
        CandidateRow(
            row_id="r2",
            split="train",
            dataset="Toy",
            source_split="train",
            query_id="Toy:q2",
            query_text="second query",
            positive_id="Toy:d3",
            candidate_ids=("Toy:d3", "Toy:d4"),
            candidate_texts=("second positive", "second negative"),
            negative_sources=("random_fallback",),
        ),
    )


def _write_scores(
    path: Path,
    label: str,
    rows: list[tuple[str, list[float]]],
) -> None:
    records = [{"row_id": row_id, "model_label": label, "scores": scores} for row_id, scores in rows]
    pd.DataFrame(records).to_parquet(path, index=False)


def test_fuse_teacher_tables_joins_by_row_id_and_preserves_nested_metadata(tmp_path: Path) -> None:
    candidates = tmp_path / "train.candidates.jsonl"
    teacher_a = tmp_path / "teacher-a.parquet"
    teacher_b = tmp_path / "teacher-b.parquet"
    destination = tmp_path / "training.parquet"
    write_candidate_rows_jsonl(_candidate_rows(), candidates)
    _write_scores(
        teacher_a,
        "teacher-a",
        [("r1", [1.0, 3.0]), ("r2", [2.0, 4.0])],
    )
    _write_scores(teacher_b, "teacher-b", [("r2", [6.0, 8.0]), ("r1", [5.0, 7.0])])

    artifact = fuse_teacher_tables_to_training_parquet(
        candidates,
        {"teacher-b": teacher_b, "teacher-a": teacher_a},
        {"teacher-a": 0.25, "teacher-b": 0.75},
        destination,
    )

    assert artifact.row_count == 2
    assert artifact.candidate_count == 2
    assert artifact.teacher_labels == ("teacher-a", "teacher-b")
    frame = pd.read_parquet(destination)
    assert list(frame["row_id"]) == ["r1", "r2"]
    assert list(frame["query_id"]) == ["Toy:q1", "Toy:q2"]
    assert list(frame["positive_index"]) == [0, 0]
    assert list(frame.iloc[0]["candidate_ids"]) == ["Toy:d1", "Toy:d2"]
    np.testing.assert_allclose(frame.iloc[0]["teacher_scores"], [4.0, 6.0])
    np.testing.assert_allclose(frame.iloc[1]["teacher_scores"], [5.0, 7.0])
    assert not destination.with_name("training.parquet.manifest.json").exists()
    assert not list(tmp_path.glob(".training.parquet.*.tmp.parquet"))


def test_read_teacher_score_table_rejects_label_width_and_finiteness_errors(tmp_path: Path) -> None:
    path = tmp_path / "teacher.parquet"
    _write_scores(path, "wrong-label", [("r1", [1.0, 2.0])])
    with pytest.raises(TeacherTableError, match="model-label mismatch"):
        read_teacher_score_table(path, expected_model_label="teacher", expected_width=2)

    _write_scores(path, "teacher", [("r1", [1.0])])
    with pytest.raises(TeacherTableError, match="score-width mismatch"):
        read_teacher_score_table(path, expected_model_label="teacher", expected_width=2)

    _write_scores(path, "teacher", [("r1", [1.0, float("nan")])])
    with pytest.raises(TeacherTableError, match="non-finite"):
        read_teacher_score_table(path, expected_model_label="teacher", expected_width=2)


def test_public_score_writer_emits_only_the_fusion_columns(tmp_path: Path) -> None:
    path = tmp_path / "teacher.parquet"

    write_score_table(
        [TeacherScores(row_id="r1", scores=(0.5, -0.2))],
        path,
        model_label="teacher",
    )
    table = read_teacher_score_table(path, expected_model_label="teacher", expected_width=2)

    assert set(pd.read_parquet(path).columns) == {"row_id", "model_label", "scores"}
    assert table.scores["r1"] == (0.5, -0.2)


def test_fusion_requires_exact_teacher_row_set_and_does_not_publish(tmp_path: Path) -> None:
    candidates = tmp_path / "train.candidates.jsonl"
    teacher = tmp_path / "teacher.parquet"
    destination = tmp_path / "training.parquet"
    write_candidate_rows_jsonl(_candidate_rows(), candidates)
    _write_scores(teacher, "teacher", [("r1", [1.0, 2.0])])

    with pytest.raises(TeacherTableError, match="row-ID mismatch"):
        fuse_teacher_tables_to_training_parquet(
            candidates,
            {"teacher": teacher},
            {"teacher": 1.0},
            destination,
        )

    assert not destination.exists()
    assert not destination.with_name("training.parquet.manifest.json").exists()
