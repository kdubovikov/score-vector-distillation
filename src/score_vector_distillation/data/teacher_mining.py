"""Teacher-mined hard negatives with positive-aware filtering.

This implements the "TopK-PercPos" rule of NV-Retriever (Moreira et al., 2024)
for one existing positive-first candidate row.  The teacher scores every
document of the query's corpus (normalized cosine).  Every judged-relevant
document is removed, every remaining document whose teacher score exceeds
``max_relative_score`` times the labelled positive's teacher score is removed
as a likely false negative, and the ``negative_count`` highest-scoring
remaining documents become the row's negatives.  Exact score ties are broken
by ascending document ID, the convention of the evaluator's rankings, so the
selection does not depend on corpus order or on NumPy's partition order.

The threshold is written as ``s_pos - (1 - p) * |s_pos|``, which equals
``p * s_pos`` for the usual positive cosine and stays below the positive's
score if a positive's cosine is ever negative.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import typing as npt

__all__ = [
    "MinedNegatives",
    "MiningError",
    "document_id_ranks",
    "mine_negatives",
    "positive_aware_threshold",
]


class MiningError(ValueError):
    """Raised when a row cannot be mined under the requested rule."""


@dataclass(frozen=True, slots=True)
class MinedNegatives:
    """The negatives selected for one query and the bookkeeping of the rule.

    ``dropped_above_threshold`` counts every non-judged document whose teacher
    score exceeds the threshold; by construction all of them outrank every kept
    negative, so this is exactly the number of candidates the percentage rule
    removed from the top of the ranking.  ``judged_above_last`` counts the
    judged-relevant documents the teacher ranks above the last kept negative.
    """

    document_ids: tuple[str, ...]
    scores: tuple[float, ...]
    positive_score: float
    threshold: float
    dropped_above_threshold: int
    judged_above_last: int


def positive_aware_threshold(positive_score: float, *, max_relative_score: float = 0.95) -> float:
    """Return the largest teacher score a negative may have."""

    if not np.isfinite(positive_score):
        raise MiningError(f"positive_score must be finite; got {positive_score}.")
    if not 0.0 < max_relative_score <= 1.0:
        raise MiningError(f"max_relative_score must be in (0, 1]; got {max_relative_score}.")
    return float(positive_score - (1.0 - max_relative_score) * abs(positive_score))


def document_id_ranks(document_ids: Sequence[str]) -> npt.NDArray[np.int64]:
    """Rank of every document ID in ascending string order (the tie-break key)."""

    ids = np.asarray(list(document_ids), dtype=np.str_)
    if ids.ndim != 1 or ids.size == 0:
        raise MiningError("document_ids must be a non-empty sequence.")
    if len(set(ids.tolist())) != ids.size:
        raise MiningError("document_ids must be unique.")
    ranks = np.empty(ids.size, dtype=np.int64)
    ranks[np.argsort(ids, kind="stable")] = np.arange(ids.size, dtype=np.int64)
    return ranks


def mine_negatives(
    scores: npt.NDArray[Any],
    document_ids: Sequence[str],
    *,
    positive_id: str,
    judged_relevant_ids: Collection[str],
    negative_count: int = 31,
    max_relative_score: float = 0.95,
    id_ranks: npt.NDArray[np.int64] | None = None,
    index: Mapping[str, int] | None = None,
) -> MinedNegatives:
    """Select ``negative_count`` positive-aware hard negatives for one query.

    ``scores`` holds the teacher's score of every corpus document, aligned with
    ``document_ids``.  ``positive_id`` is the row's labelled positive (its score
    sets the threshold) and must be one of ``judged_relevant_ids``, all of which
    are excluded from the negatives.  ``id_ranks`` and ``index`` may be passed
    to avoid recomputing them for every query of one corpus.
    """

    raw = np.asarray(scores)
    if raw.ndim != 1 or raw.shape[0] != len(document_ids):
        raise MiningError(f"scores must be a vector aligned with document_ids; got shape {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
        raise MiningError("scores must be finite floating-point values.")
    # Compare in float64: with NumPy's weak scalar promotion a float32 array would otherwise be compared
    # with the threshold rounded to float32, which can keep a document scoring just above it.
    values = raw.astype(np.float64)
    if negative_count <= 0:
        raise MiningError("negative_count must be positive.")
    if positive_id not in judged_relevant_ids:
        raise MiningError(f"positive {positive_id!r} must be one of the judged-relevant documents.")
    positions = index if index is not None else {document_id: i for i, document_id in enumerate(document_ids)}
    ranks = id_ranks if id_ranks is not None else document_id_ranks(document_ids)
    if ranks.shape != values.shape:
        raise MiningError("id_ranks must be aligned with document_ids.")
    if positive_id not in positions:
        raise MiningError(f"positive {positive_id!r} is not in the corpus.")

    positive_score = float(values[positions[positive_id]])
    threshold = positive_aware_threshold(positive_score, max_relative_score=max_relative_score)
    judged = np.zeros(values.shape[0], dtype=bool)
    for document_id in judged_relevant_ids:
        position = positions.get(document_id)
        if position is not None:
            judged[position] = True
    above = (values > threshold) & ~judged
    eligible = np.flatnonzero(~judged & ~above)
    if eligible.size < negative_count:
        raise MiningError(f"only {eligible.size} eligible documents for {negative_count} negatives.")

    eligible_scores = values[eligible]
    if eligible.size > negative_count:
        boundary = np.partition(eligible_scores, eligible.size - negative_count)[eligible.size - negative_count]
        keep = eligible_scores >= boundary  # every document tied at the boundary competes on its ID
        eligible, eligible_scores = eligible[keep], eligible_scores[keep]
    order = np.lexsort((ranks[eligible], -eligible_scores))[:negative_count]
    selected = eligible[order]
    last_score = float(values[selected[-1]])
    last_rank = int(ranks[selected[-1]])
    judged_scores, judged_ranks = values[judged], ranks[judged]
    outranks_last = (judged_scores > last_score) | ((judged_scores == last_score) & (judged_ranks < last_rank))
    judged_above_last = int(np.sum(outranks_last))
    return MinedNegatives(
        document_ids=tuple(str(document_ids[i]) for i in selected),
        scores=tuple(float(values[i]) for i in selected),
        positive_score=positive_score,
        threshold=threshold,
        dropped_above_threshold=int(above.sum()),
        judged_above_last=judged_above_last,
    )
