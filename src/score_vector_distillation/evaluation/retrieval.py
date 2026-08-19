"""Portable, memory-bounded full-corpus embedding retrieval.

The scorer normalizes query embeddings once, streams document embedding
chunks, and retains only the best ``k`` candidates for each query.  Rankings
are deterministic: scores are ordered descending and exact ties are resolved
by ascending document ID.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy import typing as npt

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]

__all__ = [
    "EmbeddingChunk",
    "load_embedding_chunk",
    "load_npy_embeddings",
    "read_ids",
    "select_top_k",
    "streaming_top_k",
]


@dataclass(frozen=True, slots=True)
class EmbeddingChunk:
    """One document-ID-aligned embedding matrix.

    ``embeddings`` may be a memory-mapped ``.npy`` array.  Validation is
    intentionally performed by :func:`streaming_top_k` immediately before a
    chunk is scored, so invalid values cannot silently enter a ranking.
    """

    document_ids: Sequence[str]
    embeddings: npt.NDArray[Any]


def _validated_ids(ids: Sequence[str], *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    values = tuple(ids)
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    invalid = [index for index, value in enumerate(values) if not isinstance(value, str) or not value.strip()]
    if invalid:
        raise ValueError(f"{name} must contain non-empty strings; invalid indices={invalid[:5]}.")
    if len(set(values)) != len(values):
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        raise ValueError(f"{name} must be unique; duplicate IDs={duplicates[:5]}.")
    return values


def _validated_embeddings(
    embeddings: npt.NDArray[Any],
    *,
    name: str,
    expected_rows: int,
    expected_dimension: int | None = None,
) -> FloatArray:
    values = np.asanyarray(embeddings)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix; got shape={values.shape}.")
    if values.shape[0] != expected_rows:
        raise ValueError(f"{name} row count must match its IDs; expected {expected_rows}, got {values.shape[0]}.")
    if values.shape[1] == 0:
        raise ValueError(f"{name} must have a non-empty embedding dimension.")
    if expected_dimension is not None and values.shape[1] != expected_dimension:
        raise ValueError(f"{name} dimension mismatch; expected {expected_dimension}, got {values.shape[1]}.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point values; got dtype={values.dtype}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return cast(FloatArray, values)


def _normalized_rows(embeddings: npt.NDArray[Any], *, name: str, eps: float) -> FloatArray:
    # Float32 avoids half-precision accumulation while keeping the large
    # query-by-document score blocks compact and portable across NumPy builds.
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= eps))
    if invalid.size:
        raise ValueError(f"{name} contains zero-norm or invalid rows at indices={invalid[:5].tolist()}.")
    return cast(FloatArray, values / norms[:, None])


def read_ids(path: str | Path) -> tuple[str, ...]:
    """Read and strictly validate one UTF-8 ID per line."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        values = tuple(line.rstrip("\r\n") for line in handle)
    return _validated_ids(values, name=f"IDs in {source}")


def load_npy_embeddings(path: str | Path, *, memory_map: bool = True) -> FloatArray:
    """Load a numeric 2D ``.npy`` matrix without permitting pickled objects.

    With ``memory_map=True`` (the default), document matrices remain on disk
    and are paged in only when their enclosing :class:`EmbeddingChunk` is
    scored.
    """

    source = Path(path)
    loaded = np.load(source, allow_pickle=False, mmap_mode="r" if memory_map else None)
    if not isinstance(loaded, np.ndarray):  # Defensive guard against archive-like inputs.
        raise ValueError(f"Expected a .npy array at {source}.")
    if loaded.ndim != 2:
        raise ValueError(f"Embedding array at {source} must be two-dimensional; got shape={loaded.shape}.")
    if not np.issubdtype(loaded.dtype, np.floating):
        raise TypeError(f"Embedding array at {source} must be floating-point; got dtype={loaded.dtype}.")
    return cast(FloatArray, loaded)


def load_embedding_chunk(
    ids_path: str | Path,
    embeddings_path: str | Path,
    *,
    memory_map: bool = True,
) -> EmbeddingChunk:
    """Load an ID file and its aligned ``.npy`` embedding matrix."""

    document_ids = read_ids(ids_path)
    embeddings = load_npy_embeddings(embeddings_path, memory_map=memory_map)
    if embeddings.shape[0] != len(document_ids):
        raise ValueError(
            f"Document ID and embedding row counts differ; ids={len(document_ids)}, embeddings={embeddings.shape[0]}."
        )
    return EmbeddingChunk(document_ids=document_ids, embeddings=embeddings)


def select_top_k(
    document_ids: Sequence[str],
    scores: Sequence[float] | npt.NDArray[Any],
    *,
    top_k: int,
) -> tuple[tuple[str, ...], FloatArray]:
    """Select an exact deterministic top-k from aligned IDs and scores.

    A score partition avoids sorting every candidate.  All candidates at the
    partition boundary are considered before the document-ID tie-break is
    applied, so the result does not depend on NumPy's partition ordering.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}.")
    ids = _validated_ids(document_ids, name="document_ids", allow_empty=True)
    values = np.asarray(scores)
    if values.ndim != 1 or values.shape[0] != len(ids):
        raise ValueError(f"scores must be a vector aligned with document_ids; got shape={values.shape}.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"scores must be numeric; got dtype={values.dtype}.")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values.")
    if not ids:
        return (), np.empty(0, dtype=np.float32)

    keep = min(top_k, len(ids))
    id_array = np.asarray(ids, dtype=np.str_)
    if keep == len(ids):
        selected: npt.NDArray[np.int64] = np.arange(len(ids), dtype=np.int64)
    else:
        boundary = np.partition(values, len(values) - keep)[len(values) - keep]
        above = np.flatnonzero(values > boundary)
        tied = np.flatnonzero(values == boundary)
        remaining = keep - len(above)
        tied_order = np.argsort(id_array[tied], kind="stable")
        selected = cast(npt.NDArray[np.int64], np.concatenate((above, tied[tied_order[:remaining]])))

    order = np.lexsort((id_array[selected], -values[selected]))
    ranked = selected[order]
    ranked_ids = tuple(str(value) for value in id_array[ranked].tolist())
    return ranked_ids, np.asarray(values[ranked], dtype=np.float32)


def streaming_top_k(
    query_ids: Sequence[str],
    query_embeddings: npt.NDArray[Any],
    document_chunks: Iterable[EmbeddingChunk],
    *,
    top_k: int,
    query_chunk_size: int = 128,
    normalization_eps: float = 1e-12,
    exclude_identical_ids: bool = True,
) -> dict[str, dict[str, float]]:
    """Return normalized-cosine top-k rankings over streamed documents.

    Memory is bounded by one document embedding chunk, one
    ``query_chunk_size`` by document-chunk score block, and ``O(B * top_k)``
    retained candidates.  Document IDs must be globally unique across chunks.
    By default, a document whose ID equals the query ID is excluded before
    top-k selection, matching the canonical BEIR dense-retrieval contract.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}.")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive; got {query_chunk_size}.")
    if not np.isfinite(normalization_eps) or normalization_eps <= 0.0:
        raise ValueError(f"normalization_eps must be finite and positive; got {normalization_eps}.")

    checked_query_ids = _validated_ids(query_ids, name="query_ids")
    checked_queries = _validated_embeddings(
        query_embeddings,
        name="query_embeddings",
        expected_rows=len(checked_query_ids),
    )
    normalized_queries = _normalized_rows(checked_queries, name="query_embeddings", eps=normalization_eps)
    embedding_dimension = normalized_queries.shape[1]

    best_ids: list[tuple[str, ...]] = [() for _ in checked_query_ids]
    best_scores: list[FloatArray] = [np.empty(0, dtype=np.float32) for _ in checked_query_ids]
    seen_document_ids: set[str] = set()
    document_count = 0

    for chunk_index, chunk in enumerate(document_chunks):
        chunk_ids = _validated_ids(chunk.document_ids, name=f"document_chunks[{chunk_index}].document_ids")
        duplicates = sorted(seen_document_ids.intersection(chunk_ids))
        if duplicates:
            raise ValueError(f"Document IDs must be unique across chunks; duplicate IDs={duplicates[:5]}.")
        seen_document_ids.update(chunk_ids)

        checked_documents = _validated_embeddings(
            chunk.embeddings,
            name=f"document_chunks[{chunk_index}].embeddings",
            expected_rows=len(chunk_ids),
            expected_dimension=embedding_dimension,
        )
        normalized_documents = _normalized_rows(
            checked_documents,
            name=f"document_chunks[{chunk_index}].embeddings",
            eps=normalization_eps,
        )
        document_count += len(chunk_ids)
        chunk_positions = {document_id: index for index, document_id in enumerate(chunk_ids)}

        for query_start in range(0, len(checked_query_ids), query_chunk_size):
            query_end = min(query_start + query_chunk_size, len(checked_query_ids))
            score_block = normalized_queries[query_start:query_end] @ normalized_documents.T
            if not np.isfinite(score_block).all():  # Defensive check after the matrix product.
                raise ValueError(f"Cosine scores for document chunk {chunk_index} are not finite.")
            for offset, chunk_scores in enumerate(score_block):
                query_index = query_start + offset
                excluded_index = chunk_positions.get(checked_query_ids[query_index]) if exclude_identical_ids else None
                if excluded_index is None:
                    new_ids = chunk_ids
                    new_scores = chunk_scores
                else:
                    new_ids = (*chunk_ids[:excluded_index], *chunk_ids[excluded_index + 1 :])
                    new_scores = np.concatenate((chunk_scores[:excluded_index], chunk_scores[excluded_index + 1 :]))
                candidate_ids = (*best_ids[query_index], *new_ids)
                candidate_scores = np.concatenate((best_scores[query_index], new_scores.astype(np.float32)))
                ranked_ids, ranked_scores = select_top_k(candidate_ids, candidate_scores, top_k=top_k)
                best_ids[query_index] = ranked_ids
                best_scores[query_index] = ranked_scores

    if document_count == 0:
        raise ValueError("document_chunks must contain at least one document.")

    return {
        query_id: {document_id: float(score) for document_id, score in zip(ids, scores, strict=True)}
        for query_id, ids, scores in zip(checked_query_ids, best_ids, best_scores, strict=True)
    }
