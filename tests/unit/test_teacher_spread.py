import numpy as np
import pytest

from score_vector_distillation.data.teacher_mining import MiningError, document_id_ranks, mine_negatives
from score_vector_distillation.data.teacher_spread import (
    DEFAULT_SPREAD_STRATA,
    SpreadStratum,
    allocate_spread_quotas,
    spread_negatives,
    spread_seed,
)


def _corpus(n: int, seed: int, *, decimals: int | None = None):
    rng = np.random.default_rng(seed)
    ids = [f"d{int(v)}" for v in rng.permutation(10 * n)[:n]]
    scores = rng.uniform(-0.1, 0.9, size=n)
    if decimals is not None:
        scores = np.round(scores, decimals)  # creates exact ties
    return ids, scores.astype(np.float32)


def _eligible_ranking(scores, ids, positive_id, judged, p=0.95):
    """Reference: eligible documents sorted by (-score, id) in float64."""
    values = np.asarray(scores, dtype=np.float64)
    s_pos = values[ids.index(positive_id)]
    threshold = s_pos - (1 - p) * abs(s_pos)
    eligible = [i for i in range(len(ids)) if ids[i] not in judged and values[i] <= threshold]
    return [ids[i] for i in sorted(eligible, key=lambda i: (-values[i], ids[i]))], threshold


def test_default_strata_draw_8_8_8_7() -> None:
    assert [(s.first_rank, s.last_rank, s.quota) for s in DEFAULT_SPREAD_STRATA] == [
        (1, 30, 8),
        (31, 100, 8),
        (101, 1000, 8),
        (1001, None, 7),
    ]
    assert allocate_spread_quotas(57_000) == (8, 8, 8, 7)
    assert allocate_spread_quotas(1_007) == (8, 8, 8, 7)


def test_short_rankings_carry_the_shortfall_to_the_next_shallower_stratum() -> None:
    assert allocate_spread_quotas(1_003) == (8, 8, 12, 3)  # 3 documents in 1001-1003
    assert allocate_spread_quotas(1_001) == (8, 8, 14, 1)
    assert allocate_spread_quotas(1_000) == (8, 8, 15, 0)
    assert allocate_spread_quotas(500) == (8, 8, 15, 0)
    assert allocate_spread_quotas(60) == (8, 23, 0, 0)
    assert allocate_spread_quotas(40) == (21, 10, 0, 0)
    assert allocate_spread_quotas(31) == (30, 1, 0, 0)
    for n in range(31, 1_200):
        taken = allocate_spread_quotas(n)
        assert sum(taken) == 31
        assert taken[0] <= 30 and taken[1] <= 70 and taken[2] <= 900
    with pytest.raises(MiningError):
        allocate_spread_quotas(30)


def test_draws_respect_strata_threshold_and_judged_documents() -> None:
    ids, scores = _corpus(3_000, 0)
    positive = ids[int(np.argmax(scores))]
    judged = {positive, ids[5], ids[17]}
    spread = spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids=judged, seed_key="fiqa:1")
    ranking, threshold = _eligible_ranking(scores, ids, positive, judged)
    assert spread.eligible_count == len(ranking)
    assert spread.taken_per_stratum == (8, 8, 8, 7)
    assert spread.relaxed_fill == 0
    assert len(spread.document_ids) == len(set(spread.document_ids)) == 31
    assert not set(spread.document_ids) & judged
    assert all(score <= threshold for score in spread.scores)
    # every negative sits at its reported eligible rank, inside its stratum, in ranking order
    assert [ranking[r - 1] for r in spread.eligible_ranks] == list(spread.document_ids)
    assert list(spread.eligible_ranks) == sorted(spread.eligible_ranks)
    for rank, stratum in zip(spread.eligible_ranks, spread.strata, strict=True):
        bounds = DEFAULT_SPREAD_STRATA[stratum]
        assert bounds.first_rank <= rank <= (bounds.last_rank or spread.eligible_count)
    assert [spread.strata.count(k) for k in range(4)] == [8, 8, 8, 7]
    assert spread.scores == tuple(float(scores[ids.index(d)]) for d in spread.document_ids)


def test_draws_are_deterministic_per_key_and_independent_of_corpus_order() -> None:
    ids, scores = _corpus(2_500, 1, decimals=3)  # ties are broken by document ID
    positive = ids[int(np.argmax(scores))]
    first = spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="q:7")
    again = spread_negatives(
        scores,
        ids,
        positive_id=positive,
        judged_relevant_ids={positive},
        seed_key="q:7",
        id_ranks=document_id_ranks(ids),
        index={d: i for i, d in enumerate(ids)},
    )
    assert first == again
    permutation = np.random.default_rng(2).permutation(len(ids))
    shuffled = spread_negatives(
        scores[permutation],
        [ids[i] for i in permutation],
        positive_id=positive,
        judged_relevant_ids={positive},
        seed_key="q:7",
    )
    assert shuffled == first
    other = spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="q:8")
    assert other.document_ids != first.document_ids


def test_seed_is_the_sha256_prefix_of_the_key() -> None:
    import hashlib

    assert spread_seed("fiqa:1003") == int.from_bytes(hashlib.sha256(b"fiqa:1003").digest()[:8], "little")
    with pytest.raises(MiningError):
        spread_seed("")


def test_deep_stratum_draws_are_roughly_uniform_over_its_ranks() -> None:
    ids, scores = _corpus(5_000, 3)
    positive = ids[int(np.argmax(scores))]
    lower_half, total = 0, 0
    for key in range(300):
        spread = spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key=f"k{key}")
        deep = [r for r, s in zip(spread.eligible_ranks, spread.strata, strict=True) if s == 3]
        midpoint = (1_001 + spread.eligible_count) / 2
        lower_half += sum(r < midpoint for r in deep)
        total += len(deep)
    assert total == 300 * 7
    assert 0.45 < lower_half / total < 0.55


def test_short_corpus_uses_the_carry_rule() -> None:
    ids, scores = _corpus(70, 4)
    scores[0] = 1.0  # threshold 0.95 lies above every other score (all <= 0.9): 69 eligible documents
    positive = ids[0]
    spread = spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="short")
    ranking, _ = _eligible_ranking(scores, ids, positive, {positive})
    assert spread.eligible_count == len(ranking) == 69
    assert spread.taken_per_stratum == allocate_spread_quotas(69) == (8, 23, 0, 0)
    assert [ranking[r - 1] for r in spread.eligible_ranks] == list(spread.document_ids)


def test_rows_with_too_few_eligible_documents_use_the_mining_relaxation() -> None:
    ids, scores = _corpus(60, 5)
    order = np.argsort(scores)
    positive = ids[int(order[20])]  # 20 documents score below the positive; about 18 lie under 0.95 x s_pos
    judged = {positive, ids[int(order[0])]}
    spread = spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids=judged, seed_key="relaxed")
    mined = mine_negatives(
        scores, ids, positive_id=positive, judged_relevant_ids=judged, negative_count=31, relax_threshold_if_short=True
    )
    assert spread.relaxed_fill == mined.relaxed_fill > 0
    assert spread.document_ids == mined.document_ids and spread.scores == mined.scores
    assert spread.eligible_count == 31 - mined.relaxed_fill
    assert spread.taken_per_stratum == (0, 0, 0, 0) and set(spread.strata) == {-1}
    assert spread.eligible_ranks.count(0) == mined.relaxed_fill
    ranking, threshold = _eligible_ranking(scores, ids, positive, judged)
    kept = [d for d, r in zip(spread.document_ids, spread.eligible_ranks, strict=True) if r > 0]
    assert kept == ranking
    assert [r for r in spread.eligible_ranks if r > 0] == list(range(1, len(ranking) + 1))
    assert sum(score > threshold for score in spread.scores) == mined.relaxed_fill


def test_custom_strata_and_invalid_inputs() -> None:
    ids, scores = _corpus(200, 6)
    positive = ids[int(np.argmax(scores))]
    strata = (SpreadStratum(1, 10, 2), SpreadStratum(11, None, 3))
    spread = spread_negatives(
        scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="c", strata=strata
    )
    assert len(spread.document_ids) == 5 and spread.taken_per_stratum == (2, 3)
    for bad in (
        (SpreadStratum(2, 10, 2),),  # does not start at rank 1
        (SpreadStratum(1, 10, 2), SpreadStratum(12, None, 3)),  # gap
        (SpreadStratum(1, None, 2), SpreadStratum(11, 20, 3)),  # open stratum not last
        (SpreadStratum(1, 10, 0),),  # zero quota
        (SpreadStratum(1, 0, 1),),  # empty range
        (),
    ):
        with pytest.raises(MiningError):
            spread_negatives(
                scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="c", strata=bad
            )
    with pytest.raises(MiningError):
        spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids={ids[0]}, seed_key="c")  # not judged
    with pytest.raises(MiningError):
        spread_negatives(scores, ids, positive_id="x", judged_relevant_ids={"x"}, seed_key="c")
    with pytest.raises(MiningError):
        spread_negatives(scores[:10], ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="c")
    with pytest.raises(MiningError):
        spread_negatives(scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="")
    nan_scores = scores.copy()
    nan_scores[3] = np.nan
    with pytest.raises(MiningError):
        spread_negatives(nan_scores, ids, positive_id=positive, judged_relevant_ids={positive}, seed_key="c")
    with pytest.raises(MiningError):  # fewer non-judged documents than negatives, even with relaxation
        spread_negatives(scores[:20], ids[:20], positive_id=ids[0], judged_relevant_ids={ids[0]}, seed_key="c")
