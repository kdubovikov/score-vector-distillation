"""Shard-local retrieval and exact deterministic top-k merging."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy import typing as npt

from .retrieval import EmbeddingChunk, select_top_k, streaming_top_k

__all__ = ["merge_sharded_top_k", "shard_local_top_k"]

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def shard_local_top_k(
    query_ids: Sequence[str],
    query_embeddings: npt.NDArray[Any],
    document_chunks: Iterable[EmbeddingChunk],
    *,
    top_k: int,
    query_chunk_size: int = 128,
    normalization_eps: float = 1e-12,
) -> dict[str, dict[str, float]]:
    """Compute one shard's local top-k using the full-corpus scoring contract."""

    return streaming_top_k(
        query_ids,
        query_embeddings,
        document_chunks,
        top_k=top_k,
        query_chunk_size=query_chunk_size,
        normalization_eps=normalization_eps,
    )


def _validated_query_ids(query_ids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(query_ids)
    if not values:
        raise ValueError("expected_query_ids must not be empty.")
    invalid = [index for index, value in enumerate(values) if not isinstance(value, str) or not value.strip()]
    if invalid:
        raise ValueError(f"expected_query_ids must contain non-empty strings; invalid indices={invalid[:5]}.")
    if len(set(values)) != len(values):
        raise ValueError("expected_query_ids must be unique.")
    return values


def _validated_local_ranking(
    ranking: Mapping[str, float],
    *,
    shard_index: int,
    query_id: str,
) -> tuple[tuple[str, ...], FloatArray]:
    document_ids = tuple(ranking)
    if not document_ids:
        raise ValueError(f"Shard {shard_index} has an empty ranking for query {query_id!r}.")
    invalid = [doc_id for doc_id in document_ids if not isinstance(doc_id, str) or not doc_id.strip()]
    if invalid:
        raise ValueError(f"Shard {shard_index} has invalid document IDs for query {query_id!r}: {invalid[:5]}.")
    scores = np.asarray(tuple(ranking.values()), dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"Shard {shard_index} has non-finite scores for query {query_id!r}.")
    return document_ids, scores


def merge_sharded_top_k(
    shards: Iterable[Mapping[str, Mapping[str, float]]],
    *,
    top_k: int,
    expected_query_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Merge disjoint shard-local rankings into one exact global top-k.

    Each shard must provide the same query set.  Local rankings must retain at
    least ``min(top_k, shard_document_count)`` documents; under that standard
    shard-local top-k contract, selecting top-k from their union is exact.
    Duplicate candidate IDs across shards are rejected because they indicate
    an invalid corpus partition.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}.")
    shard_list = tuple(shards)
    if not shard_list:
        raise ValueError("shards must contain at least one local ranking.")

    if expected_query_ids is None:
        first_query_ids = tuple(shard_list[0])
        query_ids = tuple(sorted(_validated_query_ids(first_query_ids)))
    else:
        query_ids = _validated_query_ids(expected_query_ids)
    expected = set(query_ids)

    accumulated_ids: dict[str, list[str]] = {query_id: [] for query_id in query_ids}
    accumulated_scores: dict[str, list[float]] = {query_id: [] for query_id in query_ids}
    seen_ids: dict[str, set[str]] = {query_id: set() for query_id in query_ids}

    for shard_index, shard in enumerate(shard_list):
        actual = set(shard)
        if actual != expected:
            missing = sorted(expected.difference(actual))[:5]
            extra = sorted(actual.difference(expected))[:5]
            raise ValueError(f"Shard {shard_index} query-ID mismatch: missing={missing}, extra={extra}.")
        for query_id in query_ids:
            document_ids, scores = _validated_local_ranking(
                shard[query_id],
                shard_index=shard_index,
                query_id=query_id,
            )
            duplicates = sorted(seen_ids[query_id].intersection(document_ids))
            if duplicates:
                raise ValueError(f"Document IDs overlap across shards for query {query_id!r}: {duplicates[:5]}.")
            seen_ids[query_id].update(document_ids)
            accumulated_ids[query_id].extend(document_ids)
            accumulated_scores[query_id].extend(float(value) for value in scores)

    merged: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        document_ids, scores = select_top_k(
            accumulated_ids[query_id],
            accumulated_scores[query_id],
            top_k=top_k,
        )
        merged[query_id] = {document_id: float(score) for document_id, score in zip(document_ids, scores, strict=True)}
    return merged
