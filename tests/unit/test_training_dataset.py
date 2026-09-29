from pathlib import Path

import pandas as pd
import pytest

from score_vector_distillation.training.dataset import DistillationDataset, DistillationRow


def test_distillation_row_rejects_score_mismatch() -> None:
    with pytest.raises(ValueError, match="Candidate/score length mismatch"):
        DistillationRow("row", "query", ("positive", "negative"), (1.0,), ("pos-id", "neg-id"))


def test_distillation_row_rejects_id_mismatch() -> None:
    with pytest.raises(ValueError, match="Candidate ID/text length mismatch"):
        DistillationRow("row", "query", ("positive", "negative"), (1.0, 0.5), ("pos-id",))


def test_nested_parquet_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame(
        [
            {
                "row_id": "row-1",
                "query_text": "query",
                "candidate_texts": ["positive", "negative"],
                "teacher_scores": [0.7, 0.1],
                "candidate_ids": ["pos-id", "neg-id"],
            }
        ]
    ).to_parquet(path, index=False)

    dataset = DistillationDataset.from_parquet(path)

    assert len(dataset) == 1
    assert dataset[0].candidate_texts == ("positive", "negative")
