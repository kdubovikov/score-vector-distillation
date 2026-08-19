"""Tests for memory-bounded normalized-cosine retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy import typing as npt

from score_vector_distillation.evaluation.retrieval import (
    EmbeddingChunk,
    load_embedding_chunk,
    select_top_k,
    streaming_top_k,
)


def test_streaming_top_k_normalizes_and_breaks_ties_by_document_id() -> None:
    query_ids = ["q-x", "q-y"]
    queries = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    chunks = [
        EmbeddingChunk(
            document_ids=("d-z", "d-b"),
            embeddings=np.asarray([[0.1, 0.0], [1.0, 1.0]], dtype=np.float32),
        ),
        EmbeddingChunk(
            document_ids=("d-y", "d-a"),
            embeddings=np.asarray([[0.0, 9.0], [1.0, 1.0]], dtype=np.float32),
        ),
    ]

    results = streaming_top_k(query_ids, queries, chunks, top_k=3, query_chunk_size=1)

    # Magnitude cannot change cosine rank, and exact ties use ascending IDs.
    assert list(results["q-x"]) == ["d-z", "d-a", "d-b"]
    assert list(results["q-y"]) == ["d-y", "d-a", "d-b"]
    assert results["q-x"]["d-z"] == pytest.approx(1.0)
    assert results["q-y"]["d-y"] == pytest.approx(1.0)


def test_streaming_top_k_matches_dense_normalized_cosine() -> None:
    generator = np.random.default_rng(17)
    queries = generator.normal(size=(7, 11)).astype(np.float32)
    documents = generator.normal(size=(29, 11)).astype(np.float32)
    query_ids = [f"q-{index:02d}" for index in range(len(queries))]
    document_ids = [f"d-{index:02d}" for index in range(len(documents))]
    chunks = [
        EmbeddingChunk(document_ids=document_ids[:8], embeddings=documents[:8]),
        EmbeddingChunk(document_ids=document_ids[8:23], embeddings=documents[8:23]),
        EmbeddingChunk(document_ids=document_ids[23:], embeddings=documents[23:]),
    ]

    actual = streaming_top_k(query_ids, queries, chunks, top_k=6, query_chunk_size=3)
    normalized_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    normalized_documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    dense_scores = normalized_queries @ normalized_documents.T

    for query_index, query_id in enumerate(query_ids):
        expected_ids, expected_scores = select_top_k(document_ids, dense_scores[query_index], top_k=6)
        assert tuple(actual[query_id]) == expected_ids
        np.testing.assert_allclose(list(actual[query_id].values()), expected_scores, rtol=1e-6, atol=1e-7)


def test_streaming_top_k_excludes_query_document_identity_before_selection() -> None:
    results = streaming_top_k(
        ["shared-id"],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [
            EmbeddingChunk(
                document_ids=("shared-id", "other-a", "other-b"),
                embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
            )
        ],
        top_k=2,
    )

    assert list(results["shared-id"]) == ["other-a", "other-b"]


def test_load_embedding_chunk_uses_safe_memory_mapping(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.txt"
    embeddings_path = tmp_path / "embeddings.npy"
    ids_path.write_text("d-1\nd-2\n", encoding="utf-8")
    np.save(embeddings_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16), allow_pickle=False)

    chunk = load_embedding_chunk(ids_path, embeddings_path)

    assert chunk.document_ids == ("d-1", "d-2")
    assert isinstance(chunk.embeddings, np.memmap)
    assert chunk.embeddings.dtype == np.float16


@pytest.mark.parametrize(
    ("query_ids", "queries", "chunks", "match"),
    [
        (
            ["q", "q"],
            np.ones((2, 2), dtype=np.float32),
            [EmbeddingChunk(["d"], np.ones((1, 2), dtype=np.float32))],
            "query_ids must be unique",
        ),
        (
            ["q"],
            np.ones((1, 3), dtype=np.float32),
            [EmbeddingChunk(["d"], np.ones((1, 2), dtype=np.float32))],
            "dimension mismatch",
        ),
        (
            ["q"],
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            [EmbeddingChunk(["d"], np.ones((1, 2), dtype=np.float32))],
            "zero-norm",
        ),
        (
            ["q"],
            np.ones((1, 2), dtype=np.float32),
            [EmbeddingChunk(["d"], np.asarray([[np.nan, 0.0]], dtype=np.float32))],
            "finite",
        ),
        (
            ["q"],
            np.ones((1, 2), dtype=np.float32),
            [
                EmbeddingChunk(["d"], np.ones((1, 2), dtype=np.float32)),
                EmbeddingChunk(["d"], np.ones((1, 2), dtype=np.float32)),
            ],
            "unique across chunks",
        ),
    ],
)
def test_streaming_top_k_rejects_invalid_inputs(
    query_ids: list[str],
    queries: npt.NDArray[np.float32],
    chunks: list[EmbeddingChunk],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        streaming_top_k(query_ids, queries, chunks, top_k=1)


def test_streaming_top_k_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        streaming_top_k(["q"], np.ones((1, 2), dtype=np.float32), [], top_k=1)
