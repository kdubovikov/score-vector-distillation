"""Focused tests for end-to-end full-corpus evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy import typing as npt

from score_vector_distillation.data.schema import Document, Qrel, Query, RetrievalDataset
from score_vector_distillation.evaluation.pipeline import evaluate_retrieval_model


def _toy_dataset() -> RetrievalDataset:
    return RetrievalDataset(
        name="toy-panel",
        split="test",
        documents={
            "d-c": Document("d-c", "", "PRIVATE_DOCUMENT_GAMMA"),
            "d-a": Document("d-a", "", "PRIVATE_DOCUMENT_ALPHA"),
            "d-b": Document("d-b", "", "PRIVATE_DOCUMENT_BETA"),
        },
        queries={
            "q-unused": Query("q-unused", "PRIVATE_QUERY_UNUSED"),
            "q-2": Query("q-2", "PRIVATE_QUERY_BETA"),
            "q-1": Query("q-1", "PRIVATE_QUERY_ALPHA"),
        },
        qrels=(Qrel("q-2", "d-b", 1.0), Qrel("q-1", "d-a", 1.0)),
    )


def _encoders() -> tuple[
    Any,
    Any,
    list[tuple[str, ...]],
    list[tuple[str, ...]],
]:
    vectors = {
        "PRIVATE_QUERY_ALPHA": (1.0, 0.0),
        "PRIVATE_QUERY_BETA": (0.0, 1.0),
        "PRIVATE_DOCUMENT_ALPHA": (1.0, 0.0),
        "PRIVATE_DOCUMENT_BETA": (0.0, 1.0),
        "PRIVATE_DOCUMENT_GAMMA": (-1.0, 0.0),
    }
    query_calls: list[tuple[str, ...]] = []
    document_calls: list[tuple[str, ...]] = []

    def encode_queries(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        query_calls.append(tuple(texts))
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    def encode_documents(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        document_calls.append(tuple(texts))
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    return encode_queries, encode_documents, query_calls, document_calls


def test_pipeline_encodes_once_streams_documents_and_writes_safe_artifacts(tmp_path: Path) -> None:
    encode_queries, encode_documents, query_calls, document_calls = _encoders()

    result = evaluate_retrieval_model(
        _toy_dataset(),
        encode_queries=encode_queries,
        encode_documents=encode_documents,
        model_identity={"model_id": "public/toy-encoder", "revision": "revision-1"},
        snapshot_sha256="a" * 64,
        run_config={"name": "toy-run"},
        runtime={"attention_implementation": "eager"},
        top_k=2,
        metric_k_values=(2, 1),
        document_chunk_size=2,
        query_chunk_size=1,
        output_dir=tmp_path,
    )

    assert query_calls == [("PRIVATE_QUERY_ALPHA", "PRIVATE_QUERY_BETA")]
    assert document_calls == [
        ("PRIVATE_DOCUMENT_ALPHA", "PRIVATE_DOCUMENT_BETA"),
        ("PRIVATE_DOCUMENT_GAMMA",),
    ]
    assert list(result.rankings) == ["q-1", "q-2"]
    assert list(result.rankings["q-1"]) == ["d-a", "d-b"]
    assert list(result.rankings["q-2"]) == ["d-b", "d-a"]
    assert result.metrics["NDCG@1"] == pytest.approx(1.0)
    assert result.metrics["Recall@2"] == pytest.approx(1.0)
    assert [row["query_id"] for row in result.per_query] == ["q-1", "q-2"]

    assert set(result.artifacts) == {"rankings.json", "metrics.json", "per_query.json", "run.json"}
    combined_output = "\n".join(path.read_text(encoding="utf-8") for path in result.artifacts.values())
    assert "PRIVATE_QUERY" not in combined_output
    assert "PRIVATE_DOCUMENT" not in combined_output
    assert str(tmp_path) not in combined_output
    assert not list(tmp_path.glob(".*.tmp"))

    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["dataset"]["document_count"] == 3
    assert run["dataset"]["evaluated_query_count"] == 2
    assert run["dataset"]["query_document_id_overlap_count"] == 0
    assert run["dataset"]["snapshot_sha256"] == "a" * 64
    assert run["model"] == {"model_id": "public/toy-encoder", "revision": "revision-1"}
    assert run["config"] == {"name": "toy-run"}
    assert run["evaluation"]["exclude_identical_ids"] is True
    assert run["runtime"]["attention_implementation"] == "eager"


def test_pipeline_encodes_addressable_empty_document_text() -> None:
    dataset = RetrievalDataset(
        name="empty-document-panel",
        split="test",
        documents={"d-empty": Document("d-empty", "", "")},
        queries={"q1": Query("q1", "non-empty query")},
        qrels=(Qrel("q1", "d-empty", 1.0),),
    )
    document_calls: list[tuple[str, ...]] = []

    def encode_queries(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return np.ones((len(texts), 2), dtype=np.float32)

    def encode_documents(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        document_calls.append(tuple(texts))
        return np.ones((len(texts), 2), dtype=np.float32)

    result = evaluate_retrieval_model(
        dataset,
        encode_queries=encode_queries,
        encode_documents=encode_documents,
        model_identity={"model_id": "public/toy"},
        top_k=1,
        metric_k_values=(1,),
    )

    assert document_calls == [("",)]
    assert result.metrics["NDCG@1"] == pytest.approx(1.0)


def test_pipeline_preserves_declared_missing_relevant_documents_in_metrics(tmp_path: Path) -> None:
    dataset = RetrievalDataset(
        name="missing-document-panel",
        split="test",
        documents={"d-present": Document("d-present", "", "present document")},
        queries={"q1": Query("q1", "query")},
        qrels=(Qrel("q1", "d-missing", 1.0),),
        missing_qrel_document_ids=("d-missing",),
    )

    def encode(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return np.ones((len(texts), 2), dtype=np.float32)

    result = evaluate_retrieval_model(
        dataset,
        encode_queries=encode,
        encode_documents=encode,
        model_identity={"model_id": "public/toy"},
        top_k=1,
        metric_k_values=(1,),
        output_dir=tmp_path,
    )

    assert result.metrics["NDCG@1"] == 0.0
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["dataset"]["missing_qrel_document_count"] == 1
    assert run["dataset"]["missing_qrel_document_ids"] == ["d-missing"]


def test_pipeline_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for destination in (first, second):
        encode_queries, encode_documents, _, _ = _encoders()
        evaluate_retrieval_model(
            _toy_dataset(),
            encode_queries=encode_queries,
            encode_documents=encode_documents,
            model_identity={"model_id": "public/toy-encoder", "revision": "revision-1"},
            top_k=2,
            metric_k_values=(1, 2),
            document_chunk_size=2,
            output_dir=destination,
        )

    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_pipeline_requires_retrieval_depth_to_cover_metrics() -> None:
    encode_queries, encode_documents, _, _ = _encoders()
    with pytest.raises(ValueError, match="largest metric cutoff"):
        evaluate_retrieval_model(
            _toy_dataset(),
            encode_queries=encode_queries,
            encode_documents=encode_documents,
            model_identity={"model_id": "public/toy"},
            top_k=1,
            metric_k_values=(1, 2),
        )


def test_pipeline_rejects_encoder_dimension_mismatch() -> None:
    encode_queries, _, _, _ = _encoders()

    def wrong_document_dimension(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return np.ones((len(texts), 3), dtype=np.float32)

    with pytest.raises(ValueError, match="dimension mismatch"):
        evaluate_retrieval_model(
            _toy_dataset(),
            encode_queries=encode_queries,
            encode_documents=wrong_document_dimension,
            model_identity={"model_id": "public/toy"},
            top_k=1,
            metric_k_values=(1,),
        )


@pytest.mark.parametrize(
    "identity",
    [
        {"model_id": "/private/checkpoints/model"},
        {"access_token": "should-never-be-written"},
    ],
)
def test_pipeline_rejects_non_public_model_identity(identity: dict[str, str]) -> None:
    encode_queries, encode_documents, _, _ = _encoders()
    with pytest.raises(ValueError, match=r"local path|credential-like"):
        evaluate_retrieval_model(
            _toy_dataset(),
            encode_queries=encode_queries,
            encode_documents=encode_documents,
            model_identity=identity,
            top_k=1,
            metric_k_values=(1,),
        )
