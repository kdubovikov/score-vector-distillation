"""Black-box embedding-teacher score materialization."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from score_vector_distillation.models.sentence_transformer import SentenceTransformerBackbone


@dataclass(frozen=True)
class ScoringRow:
    row_id: str
    query_text: str
    candidate_texts: tuple[str, ...]


@dataclass(frozen=True)
class TeacherScores:
    row_id: str
    scores: tuple[float, ...]


def score_candidate_rows(
    model: SentenceTransformerBackbone,
    rows: Sequence[ScoringRow],
    *,
    query_batch_size: int,
    document_batch_size: int,
    device: torch.device | str,
    query_instruction: str | None = None,
) -> list[TeacherScores]:
    """Encode unique texts once and compute aligned normalized-cosine rows."""

    if not rows:
        return []
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("row_id values must be unique.")
    widths = {len(row.candidate_texts) for row in rows}
    if 0 in widths or len(widths) != 1:
        raise ValueError(f"All rows must have one positive candidate width, got {sorted(widths)}")

    queries = list(dict.fromkeys(row.query_text for row in rows))
    documents = list(dict.fromkeys(text for row in rows for text in row.candidate_texts))
    query_embeddings = model.encode_texts(
        queries,
        kind="query",
        batch_size=query_batch_size,
        device=device,
        instruction=query_instruction,
    )
    document_embeddings = model.encode_texts(
        documents,
        kind="document",
        batch_size=document_batch_size,
        device=device,
    )
    query_index = {text: index for index, text in enumerate(queries)}
    document_index = {text: index for index, text in enumerate(documents)}

    output: list[TeacherScores] = []
    for row in rows:
        query = query_embeddings[query_index[row.query_text]]
        candidates = torch.stack([document_embeddings[document_index[text]] for text in row.candidate_texts])
        scores = torch.mv(candidates, query)
        output.append(TeacherScores(row_id=row.row_id, scores=tuple(float(value) for value in scores.tolist())))
    return output


def write_score_table(
    rows: Sequence[TeacherScores],
    path: str | Path,
    *,
    model_label: str,
) -> None:
    """Write the minimal row-aligned teacher-score table used by fusion."""

    if not model_label:
        raise ValueError("model_label must not be empty")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "row_id": row.row_id,
                "model_label": model_label,
                "scores": list(row.scores),
            }
            for row in rows
        ]
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp.parquet",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
