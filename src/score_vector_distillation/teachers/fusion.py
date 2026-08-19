"""Pure coordinate-wise fusion with strict row alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate non-negative weights and normalize them to one."""

    if not weights:
        raise ValueError("At least one teacher weight is required.")
    materialized = {str(label): float(value) for label, value in weights.items()}
    negative = sorted(label for label, value in materialized.items() if value < 0.0)
    if negative:
        raise ValueError(f"Teacher weights must be non-negative: {negative}")
    total = sum(materialized.values())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Teacher weights must have a finite positive sum.")
    return {label: value / total for label, value in materialized.items()}


def fuse_score_rows(
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    weights: Mapping[str, float],
) -> dict[str, list[float]]:
    """Fuse aligned score vectors keyed by teacher then row ID."""

    normalized = normalize_weights(weights)
    missing_teachers = sorted(set(normalized).difference(scores))
    if missing_teachers:
        raise ValueError(f"Missing score tables for teachers: {missing_teachers}")

    labels = list(normalized)
    expected_rows = set(scores[labels[0]])
    for label in labels[1:]:
        actual = set(scores[label])
        if actual != expected_rows:
            missing = sorted(expected_rows.difference(actual))[:5]
            extra = sorted(actual.difference(expected_rows))[:5]
            raise ValueError(f"Row-ID mismatch for {label}: missing={missing}, extra={extra}")

    fused: dict[str, list[float]] = {}
    for row_id in sorted(expected_rows):
        value: NDArray[np.float64] | None = None
        shape: tuple[int, ...] | None = None
        for label in labels:
            current = np.asarray(scores[label][row_id], dtype=np.float64)
            if current.ndim != 1 or not np.isfinite(current).all():
                raise ValueError(f"Teacher {label!r}, row {row_id!r} has invalid scores.")
            if shape is None:
                shape = current.shape
            elif current.shape != shape:
                raise ValueError(f"Score-length mismatch for row {row_id!r}: expected {shape}, got {current.shape}")
            weighted = normalized[label] * current
            value = weighted if value is None else value + weighted
        assert value is not None
        fused[row_id] = value.astype(np.float32).tolist()
    return fused
