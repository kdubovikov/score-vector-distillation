from __future__ import annotations

from pathlib import Path

import pytest

from score_vector_distillation.data.candidates import write_candidate_rows_jsonl
from score_vector_distillation.data.schema import CandidateRow
from score_vector_distillation.teachers.io import fuse_teacher_tables_to_training_parquet
from score_vector_distillation.teachers.scoring import TeacherScores, write_score_table
from score_vector_distillation.training.dataset import DistillationDataset


def test_score_fuse_and_training_dataset_share_one_simple_table_contract(tmp_path: Path) -> None:
    candidates = tmp_path / "train.candidates.jsonl"
    scores = tmp_path / "teacher.parquet"
    training = tmp_path / "train.parquet"
    row = CandidateRow(
        row_id="train-Toy-train-00000000",
        split="train",
        dataset="Toy",
        source_split="train",
        query_id="Toy:q1",
        query_text="query",
        positive_id="Toy:d1",
        candidate_ids=("Toy:d1", "Toy:d2"),
        candidate_texts=("", "negative"),
        negative_sources=("tfidf",),
    )
    write_candidate_rows_jsonl([row], candidates)
    write_score_table(
        [TeacherScores(row_id=row.row_id, scores=(0.7, 0.2))],
        scores,
        model_label="teacher",
    )
    fuse_teacher_tables_to_training_parquet(
        candidates,
        {"teacher": scores},
        {"teacher": 1.0},
        training,
    )
    dataset = DistillationDataset.from_parquet(training)

    assert dataset[0].candidate_texts == ("", "negative")
    assert dataset[0].teacher_scores == pytest.approx((0.7, 0.2))
    assert not training.with_name("train.parquet.manifest.json").exists()
