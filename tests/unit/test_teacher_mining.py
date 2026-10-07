import numpy as np
import pytest

from score_vector_distillation.data.teacher_mining import (
    MiningError,
    document_id_ranks,
    mine_negatives,
    positive_aware_threshold,
)


def _brute_force(scores, ids, positive_id, judged, count, p):
    """Reference implementation: sort the whole corpus by (-score, id) and filter in order."""
    s_pos = scores[ids.index(positive_id)]
    threshold = s_pos - (1 - p) * abs(s_pos)
    ranked = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    kept, dropped = [], 0
    for i in ranked:
        if ids[i] in judged:
            continue
        if scores[i] > threshold:
            dropped += 1
            continue
        if len(kept) < count:
            kept.append(ids[i])
    return kept, dropped


def test_threshold_is_percentage_of_positive_and_stays_below_a_negative_positive() -> None:
    assert positive_aware_threshold(0.8) == pytest.approx(0.76)
    assert positive_aware_threshold(0.5, max_relative_score=0.9) == pytest.approx(0.45)
    assert positive_aware_threshold(-0.2) == pytest.approx(-0.21)
    assert positive_aware_threshold(0.0) == 0.0
    with pytest.raises(MiningError):
        positive_aware_threshold(float("nan"))
    with pytest.raises(MiningError):
        positive_aware_threshold(0.5, max_relative_score=0.0)


def test_drops_judged_and_near_positive_documents_then_takes_top_k() -> None:
    ids = ["p", "q", "a", "b", "c", "d", "e"]
    scores = np.array([0.80, 0.90, 0.79, 0.77, 0.75, 0.60, 0.10], dtype=np.float32)
    mined = mine_negatives(scores, ids, positive_id="p", judged_relevant_ids={"p", "q"}, negative_count=3)
    # threshold 0.76: q is judged (excluded, not counted as dropped); a (0.79) and b (0.77) exceed it
    assert mined.document_ids == ("c", "d", "e")
    assert mined.dropped_above_threshold == 2
    assert mined.judged_above_last == 2  # p and q both outrank e
    assert mined.positive_score == pytest.approx(0.80)
    assert mined.threshold == pytest.approx(0.76)
    assert mined.scores == pytest.approx((0.75, 0.60, 0.10))
    assert all(s <= mined.threshold for s in mined.scores)


def test_score_exactly_at_threshold_is_kept() -> None:
    ids = ["p", "x", "y"]
    scores = np.array([1.0, 0.95, 0.5], dtype=np.float64)
    mined = mine_negatives(scores, ids, positive_id="p", judged_relevant_ids={"p"}, negative_count=2)
    assert mined.document_ids == ("x", "y")
    assert mined.dropped_above_threshold == 0


def test_ties_are_broken_by_document_id_independent_of_corpus_order() -> None:
    ids = ["p", "z", "m", "a", "k"]
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.4], dtype=np.float32)
    first = mine_negatives(scores, ids, positive_id="p", judged_relevant_ids={"p"}, negative_count=2)
    assert first.document_ids == ("a", "m")
    permutation = [4, 2, 0, 3, 1]
    second = mine_negatives(
        scores[permutation], [ids[i] for i in permutation], positive_id="p", judged_relevant_ids={"p"}, negative_count=2
    )
    assert second.document_ids == first.document_ids


def test_matches_brute_force_on_random_corpora_with_ties() -> None:
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(40, 120))
        ids = [f"d{int(v)}" for v in rng.permutation(10 * n)[:n]]
        scores = np.round(rng.uniform(-0.2, 1.0, size=n), 2).astype(np.float32)  # rounding creates ties
        positive = ids[int(rng.integers(n))]
        judged = {positive, *(ids[int(i)] for i in rng.integers(0, n, size=int(rng.integers(0, 6))))}
        p = float(rng.choice([0.9, 0.95, 1.0]))
        count = int(rng.integers(1, 20))
        expected, dropped = _brute_force(scores.tolist(), ids, positive, judged, count, p)
        if len(expected) < count:
            with pytest.raises(MiningError):
                mine_negatives(
                    scores,
                    ids,
                    positive_id=positive,
                    judged_relevant_ids=judged,
                    negative_count=count,
                    max_relative_score=p,
                )
            continue
        mined = mine_negatives(
            scores,
            ids,
            positive_id=positive,
            judged_relevant_ids=judged,
            negative_count=count,
            max_relative_score=p,
            id_ranks=document_id_ranks(ids),
            index={d: i for i, d in enumerate(ids)},
        )
        assert list(mined.document_ids) == expected, trial
        assert mined.dropped_above_threshold == dropped, trial
        assert not set(mined.document_ids) & judged


def test_rejects_invalid_inputs() -> None:
    ids = ["p", "a", "b"]
    scores = np.array([0.9, 0.1, 0.2], dtype=np.float32)
    with pytest.raises(MiningError):
        mine_negatives(scores, ids, positive_id="p", judged_relevant_ids={"a"}, negative_count=1)  # positive not judged
    with pytest.raises(MiningError):
        mine_negatives(scores, ids, positive_id="x", judged_relevant_ids={"x"}, negative_count=1)  # not in corpus
    with pytest.raises(MiningError):
        mine_negatives(scores[:2], ids, positive_id="p", judged_relevant_ids={"p"}, negative_count=1)
    with pytest.raises(MiningError):
        mine_negatives(np.array([0.9, np.nan, 0.2]), ids, positive_id="p", judged_relevant_ids={"p"}, negative_count=1)
    with pytest.raises(MiningError):
        mine_negatives(scores, ids, positive_id="p", judged_relevant_ids={"p"}, negative_count=3)  # too few eligible
    with pytest.raises(MiningError):
        document_id_ranks(["a", "a"])


def test_document_id_ranks_follow_string_order() -> None:
    assert document_id_ranks(["b", "a", "c", "10", "9"]).tolist() == [3, 2, 4, 0, 1]


def test_float32_scores_are_compared_with_the_float64_threshold() -> None:
    # 0.9 * float32(0.7) = 0.62999998... in float64; float32(0.63) = 0.63000000 lies above it, but rounding the
    # threshold to float32 would make the two equal and keep the document.
    ids = ["p", "x", "y", "z"]
    scores = np.array([0.7, 0.63, 0.5, 0.4], dtype=np.float32)
    mined = mine_negatives(
        scores, ids, positive_id="p", judged_relevant_ids={"p"}, negative_count=2, max_relative_score=0.9
    )
    assert mined.document_ids == ("y", "z")
    assert mined.dropped_above_threshold == 1
