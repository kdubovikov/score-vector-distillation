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
    "forward_kl_teacher_temperature_loss",
    "margin_only_loss",
    "margin_only_scale_matched_loss",
    "pearson_row_weights",
    "positive_negative_margin_mse_loss",
    "rank_weighted_pair_mse_loss",
    "scale_matched_centered_mse_loss",
    "unit_centered_mse_loss",
    "weighted_unit_centered_mse_loss",
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


def margin_only_scale_matched_loss(student_scores: Tensor, teacher_scores: Tensor) -> Tensor:
    """Return the margin term of the row-centered MSE, ``((k - 1) / k**2) * m**2``.

    With ``k`` candidates per row, ``L_center = ((k - 1) / k) * (V + m**2 / k)``
    (see :func:`margin_only_loss`).  This loss is exactly ``L_center`` with the
    negatives' variance term ``((k - 1) / k) * V`` removed, so its margin gradient
    has the same scale as in the centered loss; it equals
    ``((k - 1) / k**2) * margin_only_loss``.
    """
    candidate_count = student_scores.shape[1]
    scale = (candidate_count - 1) / candidate_count**2
    return margin_only_loss(student_scores, teacher_scores) * scale


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


def forward_kl_teacher_temperature_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    student_temperature: float,
    teacher_temperature: float,
) -> Tensor:
    """Return forward KL with separate student and teacher temperatures.

    ``student_temperature**2 * KL(softmax(t / teacher_temperature) || softmax(s / student_temperature))``,
    summed over candidates and averaged over rows; teacher targets are detached.  The loss is
    zero exactly when ``s = (student_temperature / teacher_temperature) * t + c`` for some row
    offset ``c``, so the teacher temperature sets the score scale the student must reach.  With
    equal temperatures the computation is the same as :func:`forward_kl_loss`.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    for name, value in (("student_temperature", student_temperature), ("teacher_temperature", teacher_temperature)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    student_log_probs = functional.log_softmax(student_scores / student_temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_scores.detach() / teacher_temperature, dim=-1)
    return student_temperature**2 * functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def scale_matched_centered_mse_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    log_scale: Tensor,
) -> Tensor:
    """Return row-centered MSE after one learned global positive scale on the student.

    ``centered_mse_loss(exp(log_scale) * student_scores, teacher_scores)``: ``log_scale`` is a
    single scalar shared by every row (a trainable parameter of the student wrapper), so the
    student can match the teacher's score spread without shrinking its own similarity scale.
    At ``log_scale == 0`` the scale is exactly one and the loss, and its gradient with respect
    to the student scores, equal :func:`centered_mse_loss`.  The scale is a training device
    only: a global positive scale does not change any ranking, and it is not applied at
    inference.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    if log_scale.ndim != 0 or not log_scale.is_floating_point():
        raise ValueError(f"log_scale must be a floating-point scalar tensor; got shape {tuple(log_scale.shape)}.")
    if log_scale.device != student_scores.device:
        raise ValueError(f"log_scale must be on {student_scores.device}; got {log_scale.device}.")
    return centered_mse_loss(student_scores * log_scale.exp().to(dtype=student_scores.dtype), teacher_scores)


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


def _validate_row_weighting(
    *,
    student_exponent: float,
    teacher_exponent: float,
    student_scale: float,
    teacher_scale: float,
) -> None:
    for name, value in (("student_exponent", student_exponent), ("teacher_exponent", teacher_exponent)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative; got {value}.")
    for name, value in (("student_scale", student_scale), ("teacher_scale", teacher_scale)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive; got {value}.")


def pearson_row_weights(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    student_exponent: float,
    teacher_exponent: float,
    student_scale: float,
    teacher_scale: float,
) -> Tensor:
    """Return detached per-row weights ``(||P s|| / c_s)**a * (||P t|| / c_t)**b``.

    ``P`` removes each row's mean, so ``||P s||`` and ``||P t||`` are the rows' centered
    score norms (``sqrt(k)`` times the population standard deviation).  The scales
    ``c_s`` and ``c_t`` are fixed constants, never batch statistics.  The result is a
    ``[batch]`` tensor without autograd history: no gradient flows through the weights.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    _validate_row_weighting(
        student_exponent=student_exponent,
        teacher_exponent=teacher_exponent,
        student_scale=student_scale,
        teacher_scale=teacher_scale,
    )
    with torch.no_grad():
        student = student_scores.detach()
        teacher = teacher_scores.detach().to(dtype=student.dtype)
        student_norms = torch.linalg.vector_norm(student - student.mean(dim=1, keepdim=True), dim=1)
        teacher_norms = torch.linalg.vector_norm(teacher - teacher.mean(dim=1, keepdim=True), dim=1)
        weights: Tensor = (student_norms / student_scale).pow(student_exponent) * (teacher_norms / teacher_scale).pow(
            teacher_exponent
        )
    return weights


def weighted_unit_centered_mse_loss(
    student_scores: Tensor,
    teacher_scores: Tensor,
    *,
    student_exponent: float,
    teacher_exponent: float,
    student_scale: float,
    teacher_scale: float,
    eps: float = 1e-12,
) -> Tensor:
    """Return the per-row Pearson loss with fixed, detached row weights.

    Per row ``r`` the loss is ``w_r * (1/k) * ||u(s_r) - u(t_r)||**2 = w_r * (2/k) * (1 - rho_r)``,
    with ``u`` the row-centered unit vector, ``rho_r`` the within-row Pearson correlation and
    ``w_r`` from :func:`pearson_row_weights`; the result is the mean over rows.  With both
    exponents zero every weight is one and the loss is :func:`unit_centered_mse_loss`
    (up to floating-point summation order).  Because ``w_r`` is detached, the gradient of a
    row is ``w_r`` times the Pearson gradient of that row: its direction is invariant to a
    positive rescaling of the row's student scores.  With ``student_exponent = teacher_exponent = 1``
    the per-row term is ``(2/k) ||P s|| ||P t|| (1 - rho) / (c_s c_t)``, the shape term of the
    row-centered MSE decomposition, and its gradient is the row-centered MSE gradient with the
    component along ``P s`` (the part that only changes the row's spread) removed, divided by
    ``c_s c_t``.
    """
    _validate_score_vectors(student_scores, teacher_scores)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}.")
    weights = pearson_row_weights(
        student_scores,
        teacher_scores,
        student_exponent=student_exponent,
        teacher_exponent=teacher_exponent,
        student_scale=student_scale,
        teacher_scale=teacher_scale,
    )
    centered_student = student_scores - student_scores.mean(dim=1, keepdim=True)
    centered_teacher = teacher_scores - teacher_scores.mean(dim=1, keepdim=True)
    unit_student = functional.normalize(centered_student, p=2.0, dim=1, eps=eps)
    unit_teacher = functional.normalize(centered_teacher, p=2.0, dim=1, eps=eps)
    per_row = (unit_student - unit_teacher).square().mean(dim=1)
    return (weights.to(dtype=per_row.dtype) * per_row).mean()


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
