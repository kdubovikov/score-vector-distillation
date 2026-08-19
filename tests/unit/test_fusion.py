import pytest

from score_vector_distillation.teachers.fusion import fuse_score_rows


def test_fusion_normalizes_weights_and_preserves_coordinates() -> None:
    scores = {
        "a": {"row-1": [1.0, 0.0], "row-2": [0.0, 1.0]},
        "b": {"row-1": [0.0, 2.0], "row-2": [2.0, 0.0]},
    }
    assert fuse_score_rows(scores, {"a": 1.0, "b": 3.0}) == {
        "row-1": pytest.approx([0.25, 1.5]),
        "row-2": pytest.approx([1.5, 0.25]),
    }


def test_fusion_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="Row-ID mismatch"):
        fuse_score_rows({"a": {"x": [1.0]}, "b": {"y": [1.0]}}, {"a": 1.0, "b": 1.0})
