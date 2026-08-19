"""Dependency-light candidate-row and full-corpus retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def candidate_metrics(score_rows: Sequence[Sequence[float]], *, k: int = 10) -> dict[str, float | int]:
    """Evaluate rows whose relevant document is fixed at index zero."""

    if k <= 0:
        raise ValueError("k must be positive.")
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    recalls: list[float] = []
    for row in score_rows:
        scores = np.asarray(row, dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError("Each candidate row must be a non-empty finite vector.")
        rank = 1 + int(np.sum(scores[1:] > scores[0]))
        reciprocal_ranks.append(1.0 / rank if rank <= k else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
        recalls.append(float(rank <= k))
    return {
        f"MRR@{k}": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "query_count": len(reciprocal_ranks),
    }


def _ranked_documents(scores: Mapping[str, float], k: int) -> list[str]:
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))[:k]]


def retrieval_metrics(
    qrels: Mapping[str, Mapping[str, int | float]],
    results: Mapping[str, Mapping[str, float]],
    *,
    k_values: Sequence[int] = (10, 100),
    require_all_queries: bool = True,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Compute query-macro retrieval metrics and return per-query rows."""

    expected = set(qrels)
    actual = set(results)
    if require_all_queries and expected != actual:
        missing = sorted(expected.difference(actual))[:5]
        extra = sorted(actual.difference(expected))[:5]
        raise ValueError(f"Query-ID mismatch: missing={missing}, extra={extra}")
    query_ids = sorted(expected.intersection(actual))
    ks = sorted({int(value) for value in k_values})
    if not ks or ks[0] <= 0:
        raise ValueError("k_values must contain positive integers.")

    per_query: list[dict[str, Any]] = []
    for query_id in query_ids:
        relevant = {str(doc_id): float(value) for doc_id, value in qrels[query_id].items() if float(value) > 0.0}
        if not relevant:
            continue
        ranked_all = _ranked_documents(results[query_id], max(ks))
        row: dict[str, Any] = {"query_id": query_id}
        for k in ks:
            ranked = ranked_all[:k]
            gains = [relevant.get(doc_id, 0.0) for doc_id in ranked]
            dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
            ideal = sorted(relevant.values(), reverse=True)[:k]
            idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
            hits = [index for index, doc_id in enumerate(ranked, start=1) if doc_id in relevant]
            precision_sum = sum(position / rank for position, rank in enumerate(hits, start=1))
            row[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
            row[f"Recall@{k}"] = len(hits) / len(relevant)
            row[f"P@{k}"] = len(hits) / k
            row[f"MAP@{k}"] = precision_sum / min(len(relevant), k)
        per_query.append(row)

    metrics: dict[str, float | int] = {"query_count": len(per_query)}
    for k in ks:
        for name in ("NDCG", "Recall", "P", "MAP"):
            key = f"{name}@{k}"
            metrics[key] = float(np.mean([float(row[key]) for row in per_query])) if per_query else 0.0
    return metrics, per_query
