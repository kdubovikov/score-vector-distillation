"""End-to-end, dependency-neutral full-corpus retrieval evaluation.

The pipeline deliberately accepts text encoder callbacks instead of a model
class.  This keeps snapshot loading, embedding inference, retrieval, and metric
calculation independently testable while preserving a small public API.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy import typing as npt

from score_vector_distillation.data.schema import RetrievalDataset

from .metrics import retrieval_metrics
from .retrieval import EmbeddingChunk, streaming_top_k

PublicIdentityValue: TypeAlias = str | int | float | bool | None
TextEncoder: TypeAlias = Callable[[Sequence[str]], npt.NDArray[Any]]

_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|credential|password|secret|token)", re.IGNORECASE)

__all__ = ["RetrievalEvaluation", "evaluate_retrieval_model"]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """In-memory evaluation results and optional persisted artifact paths."""

    rankings: dict[str, dict[str, float]]
    metrics: dict[str, float | int]
    per_query: tuple[dict[str, object], ...]
    artifacts: dict[str, Path] = field(default_factory=dict)


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: object) -> None:
    """Write one complete JSON document and atomically publish it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_public_text(value: str, *, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be empty.")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a single-line public identifier.")
    if Path(value).is_absolute() or value.lower().startswith("file:"):
        raise ValueError(f"{label} must not contain a local path.")
    return value


def _validate_identity(
    identity: Mapping[str, PublicIdentityValue],
    *,
    label: str,
) -> dict[str, PublicIdentityValue]:
    if not identity:
        raise ValueError(f"{label} must not be empty.")
    validated: dict[str, PublicIdentityValue] = {}
    for key in sorted(identity):
        _validate_public_text(key, label=f"{label} key")
        if _SENSITIVE_KEY.search(key):
            raise ValueError(f"{label} must not contain credential-like key {key!r}.")
        value = identity[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label}[{key!r}] must be finite.")
        if isinstance(value, str):
            _validate_public_text(value, label=f"{label}[{key!r}]")
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise TypeError(f"{label}[{key!r}] must be a JSON scalar; got {type(value).__name__}.")
        validated[key] = value
    return validated


def _validate_dataset(
    dataset: RetrievalDataset,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, dict[str, float]]]:
    """Validate retrieval-dataset referential integrity."""

    _validate_public_text(dataset.name, label="dataset.name")
    _validate_public_text(dataset.split, label="dataset.split")
    if not dataset.documents:
        raise ValueError("dataset.documents must not be empty.")
    if not dataset.queries:
        raise ValueError("dataset.queries must not be empty.")
    if not dataset.qrels:
        raise ValueError("dataset.qrels must not be empty.")

    document_ids = tuple(sorted(dataset.documents))
    query_ids_all = tuple(sorted(dataset.queries))

    for document_id in document_ids:
        document = dataset.documents[document_id]
        _validate_public_text(document_id, label="document ID")
        if document.document_id != document_id:
            raise ValueError(f"Document mapping key {document_id!r} does not match its record ID.")
        # Public retrieval snapshots can contain addressable documents whose
        # title and body are both empty (FiQA is one example).  Their IDs and
        # qrels are still part of the benchmark, so preserve the empty string
        # and let the encoder apply its normal empty-input behavior.

    for query_id in query_ids_all:
        query = dataset.queries[query_id]
        _validate_public_text(query_id, label="query ID")
        if query.query_id != query_id:
            raise ValueError(f"Query mapping key {query_id!r} does not match its record ID.")
        if not query.text.strip():
            raise ValueError(f"Query {query_id!r} has empty text.")

    qrels: dict[str, dict[str, float]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    canonical_qrels = sorted(dataset.qrels, key=lambda row: (row.query_id, row.document_id))
    declared_missing_documents = tuple(sorted(dataset.missing_qrel_document_ids))
    if len(declared_missing_documents) != len(set(declared_missing_documents)):
        raise ValueError("dataset.missing_qrel_document_ids must not contain duplicates.")
    observed_missing_documents = tuple(
        sorted({qrel.document_id for qrel in canonical_qrels if qrel.document_id not in dataset.documents})
    )
    if observed_missing_documents != declared_missing_documents:
        raise ValueError(
            "Qrel missing-document declarations do not match the snapshot: "
            f"declared={declared_missing_documents[:5]}, observed={observed_missing_documents[:5]}."
        )
    for qrel in canonical_qrels:
        pair = (qrel.query_id, qrel.document_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate qrel for query-document pair {pair!r}.")
        seen_pairs.add(pair)
        if qrel.query_id not in dataset.queries:
            raise ValueError(f"Qrel references unknown query {qrel.query_id!r}.")
        relevance = float(qrel.relevance)
        if not math.isfinite(relevance):
            raise ValueError(f"Qrel {pair!r} has non-finite relevance.")
        qrels.setdefault(qrel.query_id, {})[qrel.document_id] = relevance

    evaluated_query_ids = tuple(sorted(qrels))
    without_positive = [
        query_id
        for query_id in evaluated_query_ids
        if not any(relevance > 0.0 for relevance in qrels[query_id].values())
    ]
    if without_positive:
        raise ValueError(f"Every evaluated query must have a positive qrel; missing for {without_positive[:5]}.")
    return evaluated_query_ids, document_ids, qrels


def _validate_embeddings(
    values: object,
    *,
    label: str,
    expected_rows: int,
    expected_dimension: int | None = None,
) -> npt.NDArray[Any]:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{label} must return a NumPy array; got {type(values).__name__}.")
    if values.ndim != 2:
        raise ValueError(f"{label} must return a two-dimensional matrix; got shape={values.shape}.")
    if values.shape[0] != expected_rows:
        raise ValueError(f"{label} row count mismatch; expected {expected_rows}, got {values.shape[0]}.")
    if values.shape[1] == 0:
        raise ValueError(f"{label} must return a non-empty embedding dimension.")
    if expected_dimension is not None and values.shape[1] != expected_dimension:
        raise ValueError(f"{label} dimension mismatch; expected {expected_dimension}, got {values.shape[1]}.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{label} must return floating-point values; got dtype={values.dtype}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must return only finite values.")
    return values


def _validated_cutoffs(metric_k_values: Sequence[int]) -> tuple[int, ...]:
    if not metric_k_values:
        raise ValueError("metric_k_values must not be empty.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in metric_k_values):
        raise ValueError("metric_k_values must contain positive integers.")
    return tuple(sorted(set(metric_k_values)))


def evaluate_retrieval_model(
    dataset: RetrievalDataset,
    *,
    encode_queries: TextEncoder,
    encode_documents: TextEncoder,
    model_identity: Mapping[str, PublicIdentityValue],
    top_k: int = 100,
    metric_k_values: Sequence[int] = (10, 100),
    document_chunk_size: int = 1_024,
    query_chunk_size: int = 128,
    output_dir: str | Path | None = None,
    snapshot_sha256: str | None = None,
    run_config: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> RetrievalEvaluation:
    """Encode and evaluate one validated full-corpus retrieval split.

    Queries with judgments in ``dataset.qrels`` are encoded exactly once.
    Documents are sorted by ID and encoded in bounded chunks.  The scorer uses
    normalized cosine similarity and retains only ``top_k`` documents per
    query. Persisted rankings never contain query/document text or local paths;
    ``run.json`` records the selected public model and runtime.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if isinstance(document_chunk_size, bool) or not isinstance(document_chunk_size, int) or document_chunk_size <= 0:
        raise ValueError("document_chunk_size must be a positive integer.")
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer.")
    cutoffs = _validated_cutoffs(metric_k_values)
    if top_k < max(cutoffs):
        raise ValueError(f"top_k ({top_k}) must be at least the largest metric cutoff ({max(cutoffs)}).")
    if snapshot_sha256 is not None and (
        len(snapshot_sha256) != 64 or any(character not in "0123456789abcdef" for character in snapshot_sha256.lower())
    ):
        raise ValueError("snapshot_sha256 must be a SHA-256 hexadecimal digest.")

    public_model_identity = _validate_identity(model_identity, label="model_identity")
    query_ids, document_ids, qrels = _validate_dataset(dataset)

    query_texts = tuple(dataset.queries[query_id].text for query_id in query_ids)
    query_embeddings = _validate_embeddings(
        encode_queries(query_texts),
        label="encode_queries",
        expected_rows=len(query_ids),
    )
    embedding_dimension = int(query_embeddings.shape[1])

    def document_chunks() -> Iterator[EmbeddingChunk]:
        for start in range(0, len(document_ids), document_chunk_size):
            chunk_ids = document_ids[start : start + document_chunk_size]
            texts = tuple(dataset.documents[document_id].searchable_text for document_id in chunk_ids)
            embeddings = _validate_embeddings(
                encode_documents(texts),
                label="encode_documents",
                expected_rows=len(chunk_ids),
                expected_dimension=embedding_dimension,
            )
            yield EmbeddingChunk(document_ids=chunk_ids, embeddings=embeddings)

    rankings = streaming_top_k(
        query_ids,
        query_embeddings,
        document_chunks(),
        top_k=top_k,
        query_chunk_size=query_chunk_size,
        exclude_identical_ids=True,
    )
    metrics, raw_per_query = retrieval_metrics(qrels, rankings, k_values=cutoffs, require_all_queries=True)
    per_query: tuple[dict[str, object], ...] = tuple(
        {"dataset": dataset.name, "split": dataset.split, **row} for row in raw_per_query
    )

    artifacts: dict[str, Path] = {}
    if output_dir is not None:
        destination = Path(output_dir)
        ranking_rows = [
            {
                "query_id": query_id,
                "documents": [
                    {"document_id": document_id, "rank": rank, "score": score}
                    for rank, (document_id, score) in enumerate(rankings[query_id].items(), start=1)
                ],
            }
            for query_id in query_ids
        ]
        payloads: dict[str, object] = {
            "rankings.json": {
                "schema_version": 1,
                "dataset": dataset.name,
                "split": dataset.split,
                "top_k": top_k,
                "rankings": ranking_rows,
            },
            "metrics.json": {
                "schema_version": 1,
                "dataset": dataset.name,
                "split": dataset.split,
                "metrics": metrics,
            },
            "per_query.json": {
                "schema_version": 1,
                "dataset": dataset.name,
                "split": dataset.split,
                "rows": per_query,
            },
        }
        for name, payload in payloads.items():
            target = destination / name
            _atomic_write(target, payload)
            artifacts[name] = target

        run_record = {
            "schema_version": 1,
            "kind": "evaluation",
            "dataset": {
                "name": dataset.name,
                "split": dataset.split,
                "document_count": len(document_ids),
                "evaluated_query_count": len(query_ids),
                "qrel_count": len(dataset.qrels),
                "missing_qrel_document_count": len(dataset.missing_qrel_document_ids),
                "missing_qrel_document_ids": list(dataset.missing_qrel_document_ids),
                "query_document_id_overlap_count": len(set(query_ids).intersection(document_ids)),
                "snapshot_sha256": snapshot_sha256,
            },
            "model": public_model_identity,
            "config": dict(run_config) if run_config is not None else None,
            "evaluation": {
                "similarity": "normalized_cosine",
                "top_k": top_k,
                "metric_k_values": cutoffs,
                "embedding_dimension": embedding_dimension,
                "document_chunk_size": document_chunk_size,
                "query_chunk_size": query_chunk_size,
                "exclude_identical_ids": True,
            },
            "runtime": dict(runtime or {}),
        }
        run_path = destination / "run.json"
        _atomic_write(run_path, run_record)
        artifacts["run.json"] = run_path

    return RetrievalEvaluation(rankings=rankings, metrics=metrics, per_query=per_query, artifacts=artifacts)
