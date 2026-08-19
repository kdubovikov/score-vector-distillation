"""Direct loaders for corpus, query, and qrel snapshot files."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from score_vector_distillation.data.schema import Document, Qrel, Query, RetrievalDataset

_DecodedRow = TypeVar("_DecodedRow")


class SnapshotFormatError(ValueError):
    """Raised when snapshot contents violate the expected retrieval schema."""


def _read_jsonl_index(
    path: Path,
    decoder: Callable[[dict[str, Any], str], _DecodedRow],
) -> dict[str, _DecodedRow]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, _DecodedRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise SnapshotFormatError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise SnapshotFormatError(f"expected a JSON object at {path}:{line_number}")
            row_id = str(payload.get("_id") or payload.get("id") or "").strip()
            if not row_id:
                raise SnapshotFormatError(f"missing _id/id at {path}:{line_number}")
            if row_id in result:
                raise SnapshotFormatError(f"duplicate ID {row_id!r} at {path}:{line_number}")
            result[row_id] = decoder(payload, row_id)
    return result


def _decode_document(payload: dict[str, Any], document_id: str) -> Document:
    return Document(
        document_id=document_id,
        title=str(payload.get("title") or "").strip(),
        text=str(payload.get("text") or "").strip(),
    )


def _decode_query(payload: dict[str, Any], query_id: str) -> Query:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise SnapshotFormatError(f"query {query_id!r} has empty text")
    return Query(query_id=query_id, text=text)


def _field(row: dict[str, str | None], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _read_qrels(path: Path) -> tuple[Qrel, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    qrels: list[Qrel] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SnapshotFormatError(f"qrels file has no header: {path}")
        for line_number, row in enumerate(reader, start=2):
            query_id = _field(row, ("query-id", "query_id", "qid"))
            document_id = _field(row, ("corpus-id", "corpus_id", "docid"))
            relevance_text = _field(row, ("score", "relevance"))
            if not query_id or not document_id or not relevance_text:
                raise SnapshotFormatError(f"incomplete qrel at {path}:{line_number}")
            try:
                relevance = float(relevance_text)
            except ValueError as error:
                raise SnapshotFormatError(f"invalid relevance at {path}:{line_number}: {relevance_text!r}") from error
            key = (query_id, document_id)
            if key in seen:
                raise SnapshotFormatError(f"duplicate qrel {key!r} at {path}:{line_number}")
            seen.add(key)
            qrels.append(Qrel(query_id=query_id, document_id=document_id, relevance=relevance))
    return tuple(qrels)


def load_retrieval_dataset(
    dataset_root: str | Path,
    *,
    name: str,
    split: str,
    validate_references: bool = True,
    expected_missing_qrel_documents: int = 0,
) -> RetrievalDataset:
    """Load one normalized public snapshot split directly from disk.

    Expected files are ``corpus.jsonl``, ``queries.jsonl``, and
    ``qrels/<split>.tsv``. No benchmark Python package is imported.
    """

    if (
        not isinstance(expected_missing_qrel_documents, int)
        or isinstance(expected_missing_qrel_documents, bool)
        or expected_missing_qrel_documents < 0
    ):
        raise ValueError("expected_missing_qrel_documents must be a non-negative integer")
    root = Path(dataset_root)
    documents = _read_jsonl_index(root / "corpus.jsonl", _decode_document)
    queries = _read_jsonl_index(root / "queries.jsonl", _decode_query)
    qrels = _read_qrels(root / "qrels" / f"{split}.tsv")
    missing_document_ids: tuple[str, ...] = ()
    if validate_references:
        unknown_queries = sorted({qrel.query_id for qrel in qrels if qrel.query_id not in queries})
        unknown_documents = sorted({qrel.document_id for qrel in qrels if qrel.document_id not in documents})
        if unknown_queries or len(unknown_documents) != expected_missing_qrel_documents:
            query_preview = ", ".join(unknown_queries[:3]) or "none"
            document_preview = ", ".join(unknown_documents[:3]) or "none"
            raise SnapshotFormatError(
                f"qrels reference missing snapshot records: queries=[{query_preview}], "
                f"documents=[{document_preview}], observed_missing_documents={len(unknown_documents)}, "
                f"expected_missing_documents={expected_missing_qrel_documents}"
            )
        missing_document_ids = tuple(unknown_documents)
    return RetrievalDataset(
        name=name,
        split=split,
        documents=documents,
        queries=queries,
        qrels=qrels,
        missing_qrel_document_ids=missing_document_ids,
    )
