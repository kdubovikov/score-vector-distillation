"""Global scale matching for teachers whose score units differ from the target's."""

from __future__ import annotations

import math

import numpy as np
from numpy import typing as npt


def _as_matrix(rows: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"Expected a non-empty (rows, candidates>=2) score matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Score matrix contains non-finite values.")
    return matrix


def centered_rms(rows: npt.ArrayLike) -> float:
    """Root-mean-square of row-centered scores over all rows and candidates.

    ``RMS_c(X) = sqrt(mean_ij (x_ij - mean_j x_ij)^2)``.  Per-row shifts do not
    change it, matching the shift invariance of the centered distillation loss.
    """

    matrix = _as_matrix(rows)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return float(math.sqrt(float(np.mean(centered * centered))))


def rms_matched_scale(reference: npt.ArrayLike, target: npt.ArrayLike) -> float:
    """Scalar ``a`` such that ``centered_rms(a * target) == centered_rms(reference)``."""

    target_rms = centered_rms(target)
    if target_rms <= 0.0:
        raise ValueError("Target scores have zero centered RMS; a scale cannot be matched.")
    return centered_rms(reference) / target_rms
