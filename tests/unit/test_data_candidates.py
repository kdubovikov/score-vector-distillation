from pathlib import Path

import pytest

from score_vector_distillation.data.candidates import (
    CandidateConstructionError,
    assert_query_disjoint,
    build_candidate_rows,
    read_candidate_rows_jsonl,
    write_candidate_rows_jsonl,
)
from score_vector_distillation.data.loaders import load_retrieval_dataset
from score_vector_distillation.data.schema import CandidateRow

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "retrieval" / "toy"


def _rows(split: str, logical_split: str) -> tuple[CandidateRow, ...]:
    dataset = load_retrieval_dataset(FIXTURE_ROOT, name="Toy", split=split)
    return build_candidate_rows(
        dataset,
        logical_split=logical_split,
        candidate_count=4,
        seed=42,
        tfidf_max_features=100,
        tfidf_top_pool=4,
        query_batch_size=1,
    )


def test_candidate_rows_are_deterministic_nested_and_positive_first() -> None:
    first = _rows("train", "train")
    second = _rows("train", "train")

    assert first == second
    assert len(first) == 2
    assert first[1].positive_id == "Toy:d2"
    assert first[1].candidate_ids[0] == first[1].positive_id
    assert "Toy:d3" not in first[1].candidate_ids[1:]
    assert first[0].positive_index == 0
    assert set(first[0].to_dict()) == {
        "row_id",
        "split",
        "dataset",
        "source_split",
        "query_id",
        "query_text",
        "positive_id",
        "candidate_ids",
        "candidate_texts",
        "negative_sources",
    }
    assert isinstance(first[0].to_dict()["candidate_ids"], list)
    assert not any(key.startswith("negative_") and key != "negative_sources" for key in first[0].to_dict())


def test_source_stable_cap_and_train_eval_disjointness() -> None:
    train = _rows("train", "train")
    evaluation = _rows("test", "eval")
    dataset = load_retrieval_dataset(FIXTURE_ROOT, name="Toy", split="test")

    capped_first = build_candidate_rows(dataset, logical_split="eval", candidate_count=4, seed=7, max_rows=1)
    capped_second = build_candidate_rows(dataset, logical_split="eval", candidate_count=4, seed=7, max_rows=1)

    assert capped_first == capped_second
    assert len(capped_first) == 1
    assert_query_disjoint(train, evaluation)
    with pytest.raises(CandidateConstructionError, match="overlap"):
        assert_query_disjoint(train, train)


def test_candidate_jsonl_round_trip_preserves_schema(tmp_path: Path) -> None:
    rows = _rows("test", "eval")
    destination = tmp_path / "rows.jsonl"

    write_candidate_rows_jsonl(rows, destination)

    assert read_candidate_rows_jsonl(destination) == rows


def test_candidate_jsonl_round_trip_preserves_empty_document_text(tmp_path: Path) -> None:
    row = CandidateRow(
        row_id="empty-document-row",
        split="train",
        dataset="FiQA",
        source_split="train",
        query_id="FiQA:q1",
        query_text="non-empty query",
        positive_id="FiQA:d-empty",
        candidate_ids=("FiQA:d-empty", "FiQA:d2"),
        candidate_texts=("", "non-empty negative"),
        negative_sources=("tfidf",),
    )
    destination = tmp_path / "rows.jsonl"

    write_candidate_rows_jsonl((row,), destination)

    assert read_candidate_rows_jsonl(destination) == (row,)
