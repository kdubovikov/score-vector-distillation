"""Validated training rows for score-vector distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DistillationRow:
    row_id: str
    query_text: str
    candidate_texts: tuple[str, ...]
    teacher_scores: tuple[float, ...]
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row_id must not be empty.")
        if not self.query_text:
            raise ValueError(f"query_text must not be empty for row {self.row_id!r}.")
        if not self.candidate_texts:
            raise ValueError(f"candidate_texts must not be empty for row {self.row_id!r}.")
        if len(self.candidate_texts) != len(self.teacher_scores):
            raise ValueError(
                f"Candidate/score length mismatch for {self.row_id!r}: "
                f"{len(self.candidate_texts)} != {len(self.teacher_scores)}"
            )
        if len(self.candidate_ids) != len(self.candidate_texts):
            raise ValueError(
                f"Candidate ID/text length mismatch for {self.row_id!r}: "
                f"{len(self.candidate_ids)} != {len(self.candidate_texts)}"
            )
        if not np.isfinite(np.asarray(self.teacher_scores, dtype=np.float64)).all():
            raise ValueError(f"teacher_scores contain non-finite values for row {self.row_id!r}.")


class DistillationDataset(Dataset[DistillationRow]):
    """In-memory validated rows loaded from the public nested Parquet schema."""

    def __init__(self, rows: list[DistillationRow]) -> None:
        if not rows:
            raise ValueError("DistillationDataset requires at least one row.")
        widths = {len(row.candidate_texts) for row in rows}
        if len(widths) != 1:
            raise ValueError(f"Rows must have one candidate width per run, got {sorted(widths)}")
        self.rows = rows

    @classmethod
    def from_parquet(cls, path: str | Path, *, max_rows: int | None = None) -> DistillationDataset:
        frame = pd.read_parquet(path)
        required = {"row_id", "query_text", "candidate_texts", "teacher_scores", "candidate_ids"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required training columns: {missing}")
        if max_rows is not None:
            frame = frame.iloc[:max_rows]
        records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
        rows: list[DistillationRow] = []
        for record in records:
            candidate_values = record["candidate_texts"]
            teacher_values = record["teacher_scores"]
            candidate_id_values = record["candidate_ids"]
            if not isinstance(candidate_values, (list, tuple, np.ndarray)):
                raise ValueError(f"candidate_texts must be a sequence for row {record['row_id']!r}.")
            if not isinstance(teacher_values, (list, tuple, np.ndarray)):
                raise ValueError(f"teacher_scores must be a sequence for row {record['row_id']!r}.")
            if not isinstance(candidate_id_values, (list, tuple, np.ndarray)):
                raise ValueError(f"candidate_ids must be a sequence for row {record['row_id']!r}.")
            rows.append(
                DistillationRow(
                    row_id=str(record["row_id"]),
                    query_text=str(record["query_text"]),
                    candidate_texts=tuple(str(value) for value in candidate_values),
                    teacher_scores=tuple(float(value) for value in teacher_values),
                    candidate_ids=tuple(str(value) for value in candidate_id_values),
                )
            )
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DistillationRow:
        return self.rows[index]


def collate_distillation_rows(rows: list[DistillationRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "row_ids": [row.row_id for row in rows],
        "query_texts": [row.query_text for row in rows],
        "candidate_texts": [list(row.candidate_texts) for row in rows],
        "candidate_ids": [list(row.candidate_ids) for row in rows],
        "teacher_scores": torch.tensor([row.teacher_scores for row in rows], dtype=torch.float32),
    }
