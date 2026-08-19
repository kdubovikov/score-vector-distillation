"""Deterministic TF-IDF candidate-row construction."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from score_vector_distillation.data.schema import CandidateRow, Qrel, RetrievalDataset


class CandidateConstructionError(RuntimeError):
    """Raised when a requested candidate row cannot be constructed safely."""


def _stable_seed(base_seed: int, *source_parts: str) -> int:
    payload = "\0".join((str(base_seed), *source_parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _select_query_ids(
    query_ids: Sequence[str],
    *,
    max_rows: int | None,
    seed: int,
    dataset: str,
    split: str,
) -> list[str]:
    ordered = sorted(query_ids)
    if max_rows is None or len(ordered) <= max_rows:
        return ordered
    rng = random.Random(_stable_seed(seed, dataset, split, "query-cap"))
    return sorted(rng.sample(ordered, k=max_rows))


def _top_candidate_indices(
    scores: NDArray[np.float32],
    *,
    doc_ids: Sequence[str],
    top_pool: int,
) -> list[int]:
    """Return deterministic top indices, including deterministic boundary ties."""

    count = min(top_pool, int(scores.size))
    if count <= 0:
        return []
    if count == int(scores.size):
        candidates = list(range(int(scores.size)))
    else:
        threshold = float(np.partition(scores, int(scores.size) - count)[int(scores.size) - count])
        above = np.flatnonzero(scores > threshold).tolist()
        tied = np.flatnonzero(scores == threshold).tolist()
        tied.sort(key=lambda index: doc_ids[index])
        candidates = above + tied[: count - len(above)]
    candidates.sort(key=lambda index: (-float(scores[index]), doc_ids[index]))
    return candidates


def _group_positive_qrels(dataset: RetrievalDataset) -> dict[str, list[Qrel]]:
    grouped: dict[str, list[Qrel]] = defaultdict(list)
    for qrel in dataset.qrels:
        if qrel.relevance > 0 and qrel.query_id in dataset.queries and qrel.document_id in dataset.documents:
            grouped[qrel.query_id].append(qrel)
    for qrels in grouped.values():
        qrels.sort(key=lambda item: (-item.relevance, item.document_id))
    return grouped


def build_candidate_rows(
    dataset: RetrievalDataset,
    *,
    logical_split: str,
    candidate_count: int = 32,
    seed: int = 42,
    max_rows: int | None = None,
    tfidf_max_features: int = 200_000,
    tfidf_top_pool: int = 200,
    query_batch_size: int = 256,
) -> tuple[CandidateRow, ...]:
    """Build positive-first rows using deterministic TF-IDF negatives.

    The positive is the highest-grade judged document, with document ID as the
    deterministic tie-breaker. All judged positives are excluded from negative
    selection. Random fallback uses a query-local source-stable seed, so adding
    another dataset or changing query iteration order cannot alter a row.
    """

    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")
    if tfidf_max_features <= 0 or tfidf_top_pool <= 0 or query_batch_size <= 0:
        raise ValueError("TF-IDF limits and query_batch_size must be positive")
    if len(dataset.documents) < candidate_count:
        raise CandidateConstructionError(
            f"{dataset.name}:{dataset.split} has {len(dataset.documents)} documents, fewer than C{candidate_count}"
        )

    grouped_qrels = _group_positive_qrels(dataset)
    query_ids = _select_query_ids(
        tuple(grouped_qrels),
        max_rows=max_rows,
        seed=seed,
        dataset=dataset.name,
        split=dataset.split,
    )
    if not query_ids:
        raise CandidateConstructionError(f"{dataset.name}:{dataset.split} has no queries with positive qrels")

    doc_ids = sorted(dataset.documents)
    doc_texts = [dataset.documents[document_id].searchable_text for document_id in doc_ids]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=tfidf_max_features,
        dtype=np.float32,
    )
    try:
        doc_matrix = vectorizer.fit_transform(doc_texts)
    except ValueError as error:
        raise CandidateConstructionError(f"cannot fit TF-IDF for {dataset.name}:{dataset.split}: {error}") from error
    negative_count = candidate_count - 1
    top_pool = min(len(doc_ids), max(tfidf_top_pool, negative_count * 4))
    rows: list[CandidateRow] = []
    for start in range(0, len(query_ids), query_batch_size):
        batch_ids = query_ids[start : start + query_batch_size]
        query_matrix = vectorizer.transform([dataset.queries[query_id].text for query_id in batch_ids])
        score_matrix = np.asarray((query_matrix @ doc_matrix.T).toarray(), dtype=np.float32)
        for batch_offset, query_id in enumerate(batch_ids):
            positives = grouped_qrels[query_id]
            positive_id = positives[0].document_id
            positive_ids = {qrel.document_id for qrel in positives}
            selected: list[str] = []
            negative_sources: list[str] = []
            ranked_indices = _top_candidate_indices(
                score_matrix[batch_offset],
                doc_ids=doc_ids,
                top_pool=top_pool,
            )
            for document_index in ranked_indices:
                document_id = doc_ids[document_index]
                if document_id in positive_ids:
                    continue
                selected.append(document_id)
                negative_sources.append("tfidf")
                if len(selected) == negative_count:
                    break
            if len(selected) < negative_count:
                remaining = [
                    document_id
                    for document_id in doc_ids
                    if document_id not in positive_ids and document_id not in selected
                ]
                rng = random.Random(_stable_seed(seed, dataset.name, dataset.split, query_id, "fallback"))
                rng.shuffle(remaining)
                needed = negative_count - len(selected)
                selected.extend(remaining[:needed])
                negative_sources.extend(["random_fallback"] * min(needed, len(remaining)))
            if len(selected) != negative_count:
                raise CandidateConstructionError(
                    f"cannot construct C{candidate_count} for {dataset.name}:{dataset.split}:{query_id}"
                )

            candidate_local_ids = [positive_id, *selected]
            rows.append(
                CandidateRow(
                    row_id=f"{logical_split}-{dataset.name}-{dataset.split}-{len(rows):08d}",
                    split=logical_split,
                    dataset=dataset.name,
                    source_split=dataset.split,
                    query_id=f"{dataset.name}:{query_id}",
                    query_text=dataset.queries[query_id].text,
                    positive_id=f"{dataset.name}:{positive_id}",
                    candidate_ids=tuple(f"{dataset.name}:{document_id}" for document_id in candidate_local_ids),
                    candidate_texts=tuple(
                        dataset.documents[document_id].searchable_text for document_id in candidate_local_ids
                    ),
                    negative_sources=tuple(negative_sources),
                )
            )
    return tuple(rows)


def assert_query_disjoint(train_rows: Iterable[CandidateRow], eval_rows: Iterable[CandidateRow]) -> None:
    """Fail if any namespaced query ID appears in both row partitions."""

    train_query_ids = {row.query_id for row in train_rows}
    eval_query_ids = {row.query_id for row in eval_rows}
    overlap = sorted(train_query_ids & eval_query_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise CandidateConstructionError(f"train/eval row query IDs overlap ({len(overlap)} total; first: {preview})")


def write_candidate_rows_jsonl(rows: Iterable[CandidateRow], path: str | Path) -> Path:
    """Write rows atomically using the stable nested JSON schema."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_candidate_rows_jsonl(path: str | Path) -> tuple[CandidateRow, ...]:
    """Read and validate nested candidate rows."""

    rows: list[CandidateRow] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise CandidateConstructionError(f"invalid JSON at {source}:{line_number}: {error}") from error
            if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
                raise CandidateConstructionError(f"expected an object at {source}:{line_number}")
            try:
                rows.append(CandidateRow.from_dict(payload))
            except (TypeError, ValueError) as error:
                raise CandidateConstructionError(f"invalid row at {source}:{line_number}: {error}") from error
    return tuple(rows)
