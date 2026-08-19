import pytest

from score_vector_distillation.evaluation.metrics import candidate_metrics, retrieval_metrics


def test_candidate_metrics_treat_first_candidate_as_positive() -> None:
    metrics = candidate_metrics([[3.0, 2.0, 1.0], [1.5, 2.0, 1.0]], k=2)
    assert metrics["MRR@2"] == pytest.approx(0.75)
    assert metrics["Recall@2"] == pytest.approx(1.0)


def test_retrieval_metrics_and_query_validation() -> None:
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 2.0, "x": 1.0}, "q2": {"x": 2.0, "d2": 1.0}}
    metrics, rows = retrieval_metrics(qrels, results, k_values=[1, 2])
    assert metrics["NDCG@1"] == pytest.approx(0.5)
    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert len(rows) == 2

    with pytest.raises(ValueError, match="Query-ID mismatch"):
        retrieval_metrics(qrels, {"q1": results["q1"]})
