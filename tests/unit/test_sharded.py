"""Tests for shard-local top-k merging."""

from __future__ import annotations

import numpy as np
import pytest

from score_vector_distillation.evaluation.retrieval import EmbeddingChunk, streaming_top_k
from score_vector_distillation.evaluation.sharded import merge_sharded_top_k, shard_local_top_k


def test_sharded_local_top_k_merge_matches_monolithic_retrieval() -> None:
    generator = np.random.default_rng(23)
    query_ids = ["q-2", "q-1", "q-3"]
    queries = generator.normal(size=(3, 5)).astype(np.float32)
    documents = generator.normal(size=(17, 5)).astype(np.float32)
    document_ids = [f"d-{index:02d}" for index in range(17)]
    shard_chunks = [
        [EmbeddingChunk(document_ids[:5], documents[:5])],
        [EmbeddingChunk(document_ids[5:12], documents[5:12])],
        [EmbeddingChunk(document_ids[12:], documents[12:])],
    ]

    local = [shard_local_top_k(query_ids, queries, chunks, top_k=4, query_chunk_size=2) for chunks in shard_chunks]
    actual = merge_sharded_top_k(local, top_k=4, expected_query_ids=query_ids)
    expected = streaming_top_k(
        query_ids,
        queries,
        [chunk for chunks in shard_chunks for chunk in chunks],
        top_k=4,
        query_chunk_size=2,
    )

    assert actual.keys() == expected.keys()
    for query_id in query_ids:
        assert tuple(actual[query_id]) == tuple(expected[query_id])
        np.testing.assert_allclose(
            list(actual[query_id].values()),
            list(expected[query_id].values()),
            rtol=1e-6,
            atol=1e-7,
        )


def test_merge_is_independent_of_shard_order_and_breaks_ties_by_id() -> None:
    first = {"q": {"d-z": 1.0, "d-b": 0.5}}
    second = {"q": {"d-y": 1.0, "d-a": 0.5}}

    forward = merge_sharded_top_k([first, second], top_k=3)
    reverse = merge_sharded_top_k([second, first], top_k=3)

    assert list(forward["q"]) == ["d-y", "d-z", "d-a"]
    assert forward == reverse


def test_merge_rejects_query_mismatch_document_overlap_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="query-ID mismatch"):
        merge_sharded_top_k([{"q1": {"d1": 1.0}}, {"q2": {"d2": 1.0}}], top_k=1)

    with pytest.raises(ValueError, match="overlap across shards"):
        merge_sharded_top_k([{"q": {"d": 1.0}}, {"q": {"d": 0.5}}], top_k=1)

    with pytest.raises(ValueError, match="non-finite"):
        merge_sharded_top_k([{"q": {"d": float("nan")}}], top_k=1)


def test_merge_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_sharded_top_k([], top_k=1)

    with pytest.raises(ValueError, match="positive"):
        merge_sharded_top_k([{"q": {"d": 1.0}}], top_k=0)
