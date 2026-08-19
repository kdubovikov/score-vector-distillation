from pathlib import Path

import pytest

from score_vector_distillation.data.loaders import SnapshotFormatError, load_retrieval_dataset

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "retrieval" / "toy"


def test_load_snapshot_directly_without_benchmark_package() -> None:
    dataset = load_retrieval_dataset(FIXTURE_ROOT, name="Toy", split="train")

    assert dataset.name == "Toy"
    assert dataset.split == "train"
    assert len(dataset.documents) == 6
    assert len(dataset.queries) == 4
    assert [(qrel.query_id, qrel.document_id, qrel.relevance) for qrel in dataset.qrels] == [
        ("q1", "d1", 1.0),
        ("q2", "d2", 2.0),
        ("q2", "d3", 1.0),
    ]
    assert dataset.documents["d1"].searchable_text == "Alpha study\nalpha evidence about calibrated retrieval"


def test_loader_rejects_qrels_referencing_missing_records(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"document"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nmissing\td1\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotFormatError, match="missing snapshot records"):
        load_retrieval_dataset(tmp_path, name="Toy", split="test")


def test_loader_preserves_addressable_empty_documents(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d-empty","title":"","text":""}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td-empty\t1\n",
        encoding="utf-8",
    )

    dataset = load_retrieval_dataset(tmp_path, name="Toy", split="test")

    assert dataset.documents["d-empty"].searchable_text == ""


def test_loader_allows_an_exact_pinned_count_of_missing_qrel_documents(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"document"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td-missing\t1\n",
        encoding="utf-8",
    )

    dataset = load_retrieval_dataset(
        tmp_path,
        name="Toy",
        split="test",
        expected_missing_qrel_documents=1,
    )

    assert dataset.missing_qrel_document_ids == ("d-missing",)
    assert dataset.qrels[0].document_id == "d-missing"


def test_loader_rejects_a_missing_document_count_that_differs_from_the_pin(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text('{"_id":"d1","text":"document"}\n', encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td-missing\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotFormatError, match="expected_missing_documents=2"):
        load_retrieval_dataset(
            tmp_path,
            name="Toy",
            split="test",
            expected_missing_qrel_documents=2,
        )
