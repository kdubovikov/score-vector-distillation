"""Losses for distilling a teacher score vector into a student retriever.

The primary objective operates directly on a ``[batch, candidates]`` residual
matrix.  It is therefore linear in the number of candidate scores and never
materializes candidate pairs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional

__all__ = [
    "centered_mse_loss",
    "forward_kl_loss",
    "margin_only_loss",
    "positive_negative_margin_mse_loss",
    "rank_weighted_pair_mse_loss",
    "unit_centered_mse_loss",
]


def _validate_score_vectors(student_scores: Tensor, teacher_scores: Tensor) -> None:
    """Validate the common score-vector contract."""
    if student_scores.ndim != 2 or teacher_scores.ndim != 2:
        raise ValueError(
            "student_scores and teacher_scores must both have shape "
            f"[batch, candidates]; got {tuple(student_scores.shape)} and "
            f"{tuple(teacher_scores.shape)}."
        )
    if student_scores.shape != teacher_scores.shape:
        raise ValueError(
            "student_scores and teacher_scores must have identical shapes; "
            f"got {tuple(student_scores.shape)} and {tuple(teacher_scores.shape)}."
        )
    if student_scores.shape[0] == 0 or student_scores.shape[1] == 0:
        raise ValueError("score vectors must contain at least one row and one candidate.")
    if not student_scores.is_floating_point() or not teacher_scores.is_floating_point():
        raise TypeError("student_scores and teacher_scores must be floating-point tensors.")
    if student_scores.device != teacher_scores.device:
        raise ValueError(
            "student_scores and teacher_scores must be on the same device; "
            f"got {student_scores.device} and {teacher_scores.device}."
        )


def centered_mse_loss(student_scores: Tensor, teacher_scores: Tensor) -> Tensor:
    """Return mean squared error after removing each row's additive offset.

    Let ``r = student_scores - teacher_scores``.  This computes

    ``mean((r - mean(r, dim=1)) ** 2)``.

    The implementation uses :math:`O(Bk)` time and memory.  Under the paper's
    ordered-pair normalization, the explicit all-pairs loss is exactly twice
    this value; pair tensors are intentionally absent from the production
    implementation.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    residuals = student_scores - teacher_scores
    centered_residuals = residuals - residuals.mean(dim=1, keepdim=True)
    return centered_residuals.square().mean()


def margin_only_loss(student_scores: Tensor, teacher_scores: Tensor) -> Tensor:
    """Return the mean squared residual margin ``m**2``, with no negative-order term.

    Candidate zero is the positive.  With residual ``r = student - teacher`` and
    ``m = r[0] - mean(r[1:])``, this is ``mean(m ** 2)`` over rows.  Row-centered
    MSE decomposes as ``((k - 1) / k) * (V + m**2 / k)``, with ``V`` the population
    variance of the negative residuals; this loss keeps only the margin part (up to
    a constant factor) and drops all negative-vs-negative structure.  Every negative
    receives the same gradient, ``-2 m / (k - 1)`` per row, so the student is never
    pushed to reorder its negatives.  A one-candidate row has no negatives and
    returns a differentiable zero.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    if student_scores.shape[1] == 1:
        return student_scores.sum() * 0.0

    residuals = student_scores - teacher_scores
    margins = residuals[:, 0] - residuals[:, 1:].mean(dim=1)
    return margins.square().mean()


def forward_kl_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Return temperature-squared forward KL, summed over candidates, averaged over rows.

    Both distributions use the same temperature. Teacher targets are detached.
    The fixed ``forward_kl_tau_0p1`` training recipe uses the historical 0.1
    temperature and multiplier tau**2 (not 2*tau**2). In the high-temperature
    limit this approaches half the centered-MSE loss and gradient.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    student_log_probs = functional.log_softmax(student_scores / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_scores.detach() / temperature, dim=-1)
    return temperature**2 * functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def positive_negative_margin_mse_loss(student_scores: Tensor, teacher_scores: Tensor) -> Tensor:
    """Return MarginMSE between candidate zero and every remaining candidate.

    Candidate zero is the positive.  A one-candidate row has no comparison and
    returns a differentiable zero.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    if student_scores.shape[1] == 1:
        return student_scores.sum() * 0.0

    student_margins = student_scores[:, :1] - student_scores[:, 1:]
    teacher_margins = teacher_scores[:, :1] - teacher_scores[:, 1:]
    return functional.mse_loss(student_margins, teacher_margins)


def unit_centered_mse_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return MSE between row-centered, unit-norm score vectors."""
    _validate_score_vectors(student_scores, teacher_scores)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")

    centered_student = student_scores - student_scores.mean(dim=1, keepdim=True)
    centered_teacher = teacher_scores - teacher_scores.mean(dim=1, keepdim=True)
    unit_student = functional.normalize(centered_student, p=2.0, dim=1, eps=eps)
    unit_teacher = functional.normalize(centered_teacher, p=2.0, dim=1, eps=eps)
    return functional.mse_loss(unit_student, unit_teacher)


def rank_weighted_pair_mse_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return the teacher-rank-weighted pairwise control objective.

    This supplementary control deliberately materializes the upper-triangular
    candidate pairs and therefore uses :math:`O(Bk^2)` memory.  It is separate
    from :func:`centered_mse_loss`, the linear-memory primary objective.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    candidate_count = student_scores.shape[1]
    if candidate_count == 1:
        return student_scores.sum() * 0.0

    pair_i, pair_j = torch.triu_indices(
        candidate_count,
        candidate_count,
        offset=1,
        device=student_scores.device,
    )
    student_pair_margins = student_scores[:, pair_i] - student_scores[:, pair_j]
    teacher_pair_margins = teacher_scores[:, pair_i] - teacher_scores[:, pair_j]

    teacher_order = torch.argsort(teacher_scores, dim=1, descending=True, stable=True)
    teacher_ranks = torch.empty_like(teacher_order)
    rank_values = torch.arange(candidate_count, device=teacher_scores.device).expand_as(teacher_order)
    teacher_ranks.scatter_(1, teacher_order, rank_values)

    best_pair_rank = torch.minimum(teacher_ranks[:, pair_i], teacher_ranks[:, pair_j]).to(dtype=student_scores.dtype)
    pair_weights = torch.log2(best_pair_rank + 2.0).reciprocal()
    pair_weights = pair_weights / pair_weights.mean(dim=1, keepdim=True).clamp_min(eps)
    return ((student_pair_margins - teacher_pair_margins).square() * pair_weights).mean()
