"""Teacher-spread negatives: negatives sampled across the teacher's whole ranking.

Hard-negative mining (:mod:`score_vector_distillation.data.teacher_mining`) keeps
the documents the teacher ranks just below the labelled positive, so a row's
teacher scores are nearly flat.  This module builds rows whose negatives span the
teacher's ranking of the query's corpus instead, so that the teacher's score
vector varies widely within a row.

For one positive-first row the teacher scores every corpus document.  Every
document judged relevant to the query is removed, and every remaining document
whose teacher score exceeds ``max_relative_score`` times the labelled positive's
score is removed as a likely false negative (the positive-aware threshold of
:func:`~score_vector_distillation.data.teacher_mining.positive_aware_threshold`).
The remaining *eligible* documents are ranked by teacher score (descending, exact
ties by ascending document ID, as in the mining rule), and the negatives are drawn
uniformly without replacement from fixed strata of that ranking, by default 8 from
eligible ranks 1-30, 8 from 31-100, 8 from 101-1000 and 7 from 1001 to the end.

Short rankings are handled explicitly.  Quotas are filled from the deepest
stratum to the shallowest: a stratum that holds fewer eligible documents than its
quota plus the shortfall carried from deeper strata contributes all of its
documents, and the rest of the shortfall moves to the next shallower stratum.
Because strata are contiguous rank ranges, a stratum can only be short when every
deeper stratum is empty, so the rule moves draws towards the deepest ranks that
exist.  When fewer eligible documents exist than negatives are needed, the row
uses the minimal relaxation of the mining rule
(``mine_negatives(..., relax_threshold_if_short=True)``): every eligible document
plus the non-judged documents of lowest teacher score above the threshold.

Sampling is deterministic: the generator is seeded from the SHA-256 digest of a
caller-supplied key (for example the query ID), and draws depend only on the
eligible ranking, not on corpus order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import typing as npt

from score_vector_distillation.data.teacher_mining import (
    MiningError,
    document_id_ranks,
    mine_negatives,
    positive_aware_threshold,
)

__all__ = [
    "DEFAULT_SPREAD_STRATA",
    "SpreadNegatives",
    "SpreadStratum",
    "allocate_spread_quotas",
    "spread_negatives",
    "spread_seed",
]


@dataclass(frozen=True, slots=True)
class SpreadStratum:
    """A contiguous range of eligible ranks (1-based, inclusive) and its draw quota.

    ``last_rank=None`` means "to the end of the ranking" and is allowed only for
    the deepest stratum.
    """

    first_rank: int
    last_rank: int | None
    quota: int


DEFAULT_SPREAD_STRATA: tuple[SpreadStratum, ...] = (
    SpreadStratum(1, 30, 8),
    SpreadStratum(31, 100, 8),
    SpreadStratum(101, 1000, 8),
    SpreadStratum(1001, None, 7),
)


@dataclass(frozen=True, slots=True)
class SpreadNegatives:
    """The negatives selected for one query and the bookkeeping of the rule.

    ``eligible_ranks`` gives each negative's 1-based rank in the eligible ranking
    (0 for a negative added by the relaxation, which lies above the threshold) and
    ``strata`` the index of the stratum it was drawn from (-1 for every negative of
    a relaxed row, ``relaxed_fill > 0``, which takes all eligible documents and is
    not sampled).  ``taken_per_stratum`` is the number of draws per stratum after
    the shortfall carry (all zeros for a relaxed row).
    """

    document_ids: tuple[str, ...]
    scores: tuple[float, ...]
    eligible_ranks: tuple[int, ...]
    strata: tuple[int, ...]
    positive_score: float
    threshold: float
    eligible_count: int
    dropped_above_threshold: int
    taken_per_stratum: tuple[int, ...]
    relaxed_fill: int = 0


def spread_seed(seed_key: str) -> int:
    """Return the 64-bit generator seed of a key (first 8 bytes of its SHA-256, little endian)."""

    if not seed_key:
        raise MiningError("seed_key must be a non-empty string.")
    return int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little")


def _validate_strata(strata: Sequence[SpreadStratum]) -> None:
    if not strata:
        raise MiningError("strata must not be empty.")
    expected_first = 1
    for position, stratum in enumerate(strata):
        if stratum.quota <= 0:
            raise MiningError(f"stratum {position} must have a positive quota; got {stratum.quota}.")
        if stratum.first_rank != expected_first:
            raise MiningError(
                f"strata must be contiguous from rank 1; stratum {position} starts at {stratum.first_rank}, "
                f"expected {expected_first}."
            )
        if stratum.last_rank is None:
            if position != len(strata) - 1:
                raise MiningError("only the deepest stratum may be open-ended.")
            continue
        if stratum.last_rank < stratum.first_rank:
            raise MiningError(f"stratum {position} is empty: ranks {stratum.first_rank}-{stratum.last_rank}.")
        expected_first = stratum.last_rank + 1


def allocate_spread_quotas(
    eligible_count: int, strata: Sequence[SpreadStratum] = DEFAULT_SPREAD_STRATA
) -> tuple[int, ...]:
    """Return the number of draws per stratum for a ranking of ``eligible_count`` documents.

    Quotas are filled from the deepest stratum to the shallowest; a stratum's
    shortfall is carried to the next shallower stratum.  Raises
    :class:`MiningError` if the ranking is shorter than the total quota.
    """

    _validate_strata(strata)
    total = sum(stratum.quota for stratum in strata)
    if eligible_count < total:
        raise MiningError(f"only {eligible_count} eligible documents for {total} negatives.")
    sizes = []
    for stratum in strata:
        last = eligible_count if stratum.last_rank is None else min(stratum.last_rank, eligible_count)
        sizes.append(max(0, last - stratum.first_rank + 1))
    taken = [0] * len(strata)
    carry = 0
    for position in reversed(range(len(strata))):
        wanted = strata[position].quota + carry
        taken[position] = min(wanted, sizes[position])
        carry = wanted - taken[position]
    if carry:  # unreachable when eligible_count >= total; kept as an invariant check
        raise MiningError(f"could not place {carry} draws in a ranking of {eligible_count} documents.")
    return tuple(taken)


def spread_negatives(
    scores: npt.NDArray[Any],
    document_ids: Sequence[str],
    *,
    positive_id: str,
    judged_relevant_ids: Collection[str],
    seed_key: str,
    strata: Sequence[SpreadStratum] = DEFAULT_SPREAD_STRATA,
    max_relative_score: float = 0.95,
    id_ranks: npt.NDArray[np.int64] | None = None,
    index: Mapping[str, int] | None = None,
) -> SpreadNegatives:
    """Draw stratified negatives across the teacher's ranking for one query.

    ``scores`` holds the teacher's score of every corpus document, aligned with
    ``document_ids``; ``positive_id`` is the row's labelled positive and must be
    one of ``judged_relevant_ids``.  The number of negatives is the sum of the
    strata quotas.  Negatives are returned in ranking order (descending score,
    ascending document ID).  ``id_ranks`` and ``index`` may be passed to avoid
    recomputing them for every query of one corpus.
    """

    _validate_strata(strata)
    negative_count = sum(stratum.quota for stratum in strata)
    raw = np.asarray(scores)
    if raw.ndim != 1 or raw.shape[0] != len(document_ids):
        raise MiningError(f"scores must be a vector aligned with document_ids; got shape {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
        raise MiningError("scores must be finite floating-point values.")
    if positive_id not in judged_relevant_ids:
        raise MiningError(f"positive {positive_id!r} must be one of the judged-relevant documents.")
    positions = index if index is not None else {document_id: i for i, document_id in enumerate(document_ids)}
    ranks = id_ranks if id_ranks is not None else document_id_ranks(document_ids)
    # Compare in float64, as the mining rule does (see teacher_mining.mine_negatives).
    values = raw.astype(np.float64)
    if ranks.shape != values.shape:
        raise MiningError("id_ranks must be aligned with document_ids.")
    if positive_id not in positions:
        raise MiningError(f"positive {positive_id!r} is not in the corpus.")
    seed = spread_seed(seed_key)

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
        # Too few documents below the threshold: the mining rule's minimal relaxation decides the row.
        mined = mine_negatives(
            raw,
            document_ids,
            positive_id=positive_id,
            judged_relevant_ids=judged_relevant_ids,
            negative_count=negative_count,
            max_relative_score=max_relative_score,
            id_ranks=ranks,
            index=positions,
            relax_threshold_if_short=True,
        )
        # mine_negatives returns every negative in ranking order; the relaxed ones lie strictly above the
        # threshold, i.e. above every eligible document, and get rank 0.
        relaxed_ranks: list[int] = []
        next_rank = 1
        for score in mined.scores:
            if score > mined.threshold:
                relaxed_ranks.append(0)
            else:
                relaxed_ranks.append(next_rank)
                next_rank += 1
        return SpreadNegatives(
            document_ids=mined.document_ids,
            scores=mined.scores,
            eligible_ranks=tuple(relaxed_ranks),
            strata=(-1,) * negative_count,
            positive_score=mined.positive_score,
            threshold=mined.threshold,
            eligible_count=int(eligible.size),
            dropped_above_threshold=mined.dropped_above_threshold,
            taken_per_stratum=(0,) * len(strata),
            relaxed_fill=mined.relaxed_fill,
        )

    ranking = eligible[np.lexsort((ranks[eligible], -values[eligible]))]  # eligible rank r is ranking[r - 1]
    taken = allocate_spread_quotas(int(ranking.size), strata)
    generator = np.random.default_rng(seed)
    chosen_ranks: list[int] = []
    chosen_strata: list[int] = []
    for position, (stratum, count) in enumerate(zip(strata, taken, strict=True)):
        if count == 0:
            continue
        last = ranking.size if stratum.last_rank is None else min(stratum.last_rank, int(ranking.size))
        size = last - stratum.first_rank + 1
        offsets = np.sort(generator.choice(size, size=count, replace=False))
        chosen_ranks.extend(int(stratum.first_rank + offset) for offset in offsets)
        chosen_strata.extend([position] * count)
    selected = ranking[np.asarray(chosen_ranks, dtype=np.int64) - 1]
    return SpreadNegatives(
        document_ids=tuple(str(document_ids[i]) for i in selected),
        scores=tuple(float(values[i]) for i in selected),
        eligible_ranks=tuple(chosen_ranks),
        strata=tuple(chosen_strata),
        positive_score=positive_score,
        threshold=threshold,
        eligible_count=int(ranking.size),
        dropped_above_threshold=int(above.sum()),
        taken_per_stratum=taken,
        relaxed_fill=0,
    )
