import pytest

from score_vector_distillation.evaluation.bootstrap import paired_stratified_bootstrap


def test_bootstrap_is_paired_stratified_and_deterministic() -> None:
    left = {"a": {"1": 0.3, "2": 0.4}, "b": {"1": 0.7}}
    right = {"a": {"1": 0.2, "2": 0.3}, "b": {"1": 0.5}}
    first = paired_stratified_bootstrap(left, right, datasets=["a", "b"], samples=1000, seed=7)
    second = paired_stratified_bootstrap(left, right, datasets=["a", "b"], samples=1000, seed=7)
    assert first == second
    assert first.mean_delta == pytest.approx(0.15)
    assert first.query_count == 3


def test_bootstrap_rejects_silent_query_intersection() -> None:
    with pytest.raises(ValueError, match="Query-ID mismatch"):
        paired_stratified_bootstrap(
            {"a": {"1": 1.0}},
            {"a": {"2": 1.0}},
            datasets=["a"],
            samples=10,
        )
