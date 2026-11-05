"""Tests for score-vector losses."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from score_vector_distillation.losses import (
    centered_mse_loss,
    margin_only_loss,
    margin_only_scale_matched_loss,
    positive_negative_margin_mse_loss,
    rank_weighted_pair_mse_loss,
    unit_centered_mse_loss,
)


def _ordered_pair_mse_reference(student_scores: Tensor, teacher_scores: Tensor) -> Tensor:
    """Explicit paper equation used only as a small-test oracle."""
    residuals = student_scores - teacher_scores
    pair_residuals = residuals[:, :, None] - residuals[:, None, :]
    return pair_residuals.square().mean()


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("candidate_count", [1, 2, 6, 32])
def test_ordered_pair_equation_is_twice_centered_mse(
    batch_size: int,
    candidate_count: int,
) -> None:
    generator = torch.Generator().manual_seed(1103 + candidate_count)
    student = torch.randn(batch_size, candidate_count, generator=generator, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(batch_size, candidate_count, generator=generator, dtype=torch.float32)

    centered = centered_mse_loss(student, teacher)
    ordered = _ordered_pair_mse_reference(student, teacher)
    centered_gradient = torch.autograd.grad(centered, student, retain_graph=True)[0]
    ordered_gradient = torch.autograd.grad(ordered, student)[0]

    torch.testing.assert_close(ordered, 2.0 * centered, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(ordered_gradient, 2.0 * centered_gradient, rtol=2e-6, atol=2e-7)


def test_centered_mse_is_invariant_to_independent_row_shifts() -> None:
    student = torch.tensor([[3.0, 1.5, -0.5], [0.5, 0.0, -1.0]], dtype=torch.float32)
    teacher = torch.tensor([[2.5, 1.0, 0.0], [1.0, -0.5, -0.25]], dtype=torch.float32)

    baseline = centered_mse_loss(student, teacher)
    shifted = centered_mse_loss(
        student + torch.tensor([[17.0], [-9.0]]),
        teacher + torch.tensor([[-4.0], [6.5]]),
    )

    torch.testing.assert_close(shifted, baseline)


def test_one_candidate_centered_loss_and_gradient_are_zero() -> None:
    student = torch.tensor([[3.0], [-1.0]], requires_grad=True)
    teacher = torch.tensor([[7.0], [2.0]])

    loss = centered_mse_loss(student, teacher)
    loss.backward()  # type: ignore[no-untyped-call]

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_positive_negative_margin_mse_matches_manual_formula() -> None:
    student = torch.tensor([[3.0, 1.5, -0.5], [0.5, 0.0, -1.0]])
    teacher = torch.tensor([[2.5, 1.0, 0.0], [1.0, -0.5, -0.25]])

    expected = ((student[:, :1] - student[:, 1:]) - (teacher[:, :1] - teacher[:, 1:])).square().mean()

    torch.testing.assert_close(positive_negative_margin_mse_loss(student, teacher), expected)


def test_unit_centered_loss_removes_positive_scale_and_offset() -> None:
    teacher = torch.tensor([[2.0, 0.0, -1.0, 0.5], [-0.5, 1.5, 0.0, -1.0]])
    student = 3.0 * teacher + torch.tensor([[5.0], [-7.0]])

    torch.testing.assert_close(
        unit_centered_mse_loss(student, teacher),
        torch.tensor(0.0),
        atol=1e-12,
        rtol=0.0,
    )


def test_rank_weighted_pair_control_matches_manual_teacher_rank_weights() -> None:
    student = torch.tensor([[1.7, 0.9, -0.4, 0.1], [0.2, -0.1, 1.1, -0.7]])
    teacher = torch.tensor([[1.3, 0.8, -0.2, 0.4], [-0.5, 0.0, 1.0, -0.1]])

    manual_rows = []
    for student_row, teacher_row in zip(student, teacher, strict=True):
        order = torch.argsort(teacher_row, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(teacher_row.numel())
        errors = []
        weights = []
        for i in range(teacher_row.numel()):
            for j in range(i + 1, teacher_row.numel()):
                errors.append(((student_row[i] - student_row[j]) - (teacher_row[i] - teacher_row[j])).square())
                weights.append(torch.log2(torch.minimum(ranks[i], ranks[j]).float() + 2.0).reciprocal())
        row_weights = torch.stack(weights)
        manual_rows.append((torch.stack(errors) * row_weights / row_weights.mean()).mean())
    expected = torch.stack(manual_rows).mean()

    torch.testing.assert_close(rank_weighted_pair_mse_loss(student, teacher), expected)


def test_losses_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        centered_mse_loss(torch.zeros(2, 3), torch.zeros(2, 2))


def _margin_reference(student: Tensor, teacher: Tensor) -> Tensor:
    """Per-row m = r_0 - mean(r_1..r_{k-1}) with r = student - teacher."""
    residuals = student - teacher
    return residuals[:, 0] - residuals[:, 1:].mean(dim=1)


def _scores(batch: int, k: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(batch, k, generator=generator, dtype=torch.float64)
    teacher = torch.randn(batch, k, generator=generator, dtype=torch.float64)
    return student, teacher


def test_margin_only_ignores_permutation_of_negatives() -> None:
    student, teacher = _scores(5, 32, 7)
    expected = margin_only_loss(student, teacher)
    generator = torch.Generator().manual_seed(11)
    for _ in range(3):
        student_perm = student.clone()
        teacher_perm = teacher.clone()
        for row in range(student.shape[0]):
            student_perm[row, 1:] = student[row, 1:][torch.randperm(31, generator=generator)]
            teacher_perm[row, 1:] = teacher[row, 1:][torch.randperm(31, generator=generator)]
        # permuting student and teacher negatives independently changes r_i, not mean(r_1..)
        torch.testing.assert_close(margin_only_loss(student_perm, teacher), expected, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(margin_only_loss(student, teacher_perm), expected, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(margin_only_loss(student_perm, teacher_perm), expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("candidate_count", [2, 4, 32])
def test_margin_only_equals_squared_margin_and_scaled_centered_margin_part(candidate_count: int) -> None:
    batch = 6
    k = candidate_count
    student, teacher = _scores(batch, k, 100 + k)
    margins = _margin_reference(student, teacher)
    loss = margin_only_loss(student, teacher)
    torch.testing.assert_close(loss, margins.square().mean(), atol=1e-14, rtol=0.0)

    residuals = student - teacher
    negative_variance = residuals[:, 1:].var(dim=1, unbiased=False)
    centered = centered_mse_loss(student, teacher)
    # L_center = ((k-1)/k) * (V + m^2/k), averaged over rows
    decomposed = ((k - 1) / k * (negative_variance + margins.square() / k)).mean()
    torch.testing.assert_close(centered, decomposed, atol=1e-13, rtol=0.0)
    margin_part = ((k - 1) / k * margins.square() / k).mean()
    torch.testing.assert_close(loss, k**2 / (k - 1) * margin_part, atol=1e-13, rtol=0.0)


def test_margin_only_gradient_is_identical_for_all_negatives() -> None:
    k = 32
    student, teacher = _scores(3, k, 5)
    student.requires_grad_(True)
    batch = student.shape[0]
    margins = _margin_reference(student.detach(), teacher)
    margin_only_loss(student, teacher).backward()
    grad = student.grad
    assert grad is not None
    for row in range(batch):
        negatives = grad[row, 1:]
        torch.testing.assert_close(negatives, negatives[:1].expand_as(negatives), atol=1e-15, rtol=0.0)
        torch.testing.assert_close(
            negatives[0],
            (-2.0 * margins[row] / (k - 1) / batch).to(negatives.dtype),
            atol=1e-14,
            rtol=0.0,
        )
        torch.testing.assert_close(grad[row, 0], 2.0 * margins[row] / batch, atol=1e-14, rtol=0.0)
        assert abs(float(grad[row].sum())) < 1e-13


def test_margin_only_handles_batches_as_row_mean() -> None:
    student, teacher = _scores(4, 8, 3)
    per_row = torch.stack([margin_only_loss(student[i : i + 1], teacher[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(margin_only_loss(student, teacher), per_row.mean(), atol=1e-14, rtol=0.0)
    # a single-row batch is allowed and equals that row's squared margin
    torch.testing.assert_close(per_row[0], _margin_reference(student[:1], teacher[:1]).square()[0])
    # the loss is invariant to a per-row additive offset of the student scores
    shifted = student + torch.tensor([[1.0], [-2.0], [0.5], [3.0]], dtype=student.dtype)
    torch.testing.assert_close(
        margin_only_loss(shifted, teacher), margin_only_loss(student, teacher), atol=1e-12, rtol=0.0
    )


def test_margin_only_single_candidate_is_differentiable_zero() -> None:
    student = torch.ones(2, 1, requires_grad=True)
    loss = margin_only_loss(student, torch.zeros(2, 1))
    loss.backward()
    assert float(loss) == 0.0
    assert student.grad is not None


def test_margin_only_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        margin_only_loss(torch.zeros(2, 3), torch.zeros(2, 2))


@pytest.mark.parametrize("candidate_count", [2, 4, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_margin_only_scale_matched_is_centered_loss_without_negative_variance(candidate_count: int, seed: int) -> None:
    k = candidate_count
    student, teacher = _scores(8, k, 1000 * seed + k)
    residuals = student - teacher
    negative_variance = residuals[:, 1:].var(dim=1, unbiased=False)
    # L_center - ((k-1)/k) * V, averaged over rows, is exactly the margin term ((k-1)/k^2) * m^2
    expected = centered_mse_loss(student, teacher) - ((k - 1) / k * negative_variance).mean()
    loss = margin_only_scale_matched_loss(student, teacher)
    torch.testing.assert_close(loss, expected, atol=1e-13, rtol=0.0)
    margins = _margin_reference(student, teacher)
    torch.testing.assert_close(loss, ((k - 1) / k**2 * margins.square()).mean(), atol=1e-14, rtol=0.0)
    torch.testing.assert_close(loss, (k - 1) / k**2 * margin_only_loss(student, teacher), atol=1e-14, rtol=0.0)


def test_margin_only_scale_matched_gradient_matches_centered_margin_gradient() -> None:
    k = 32
    student, teacher = _scores(3, k, 9)
    scaled_student = student.clone().requires_grad_(True)
    margin_only_scale_matched_loss(scaled_student, teacher).backward()
    # the centered loss gradient minus the gradient of its ((k-1)/k) V term
    centered_student = student.clone().requires_grad_(True)
    residuals = centered_student - teacher
    variance_term = ((k - 1) / k * residuals[:, 1:].var(dim=1, unbiased=False)).mean()
    (centered_mse_loss(centered_student, teacher) - variance_term).backward()
    assert scaled_student.grad is not None and centered_student.grad is not None
    torch.testing.assert_close(scaled_student.grad, centered_student.grad, atol=1e-14, rtol=0.0)
    negatives = scaled_student.grad[:, 1:]
    torch.testing.assert_close(negatives, negatives[:, :1].expand_as(negatives), atol=1e-15, rtol=0.0)


def test_margin_only_scale_matched_single_candidate_and_shape_checks() -> None:
    student = torch.ones(2, 1, requires_grad=True)
    loss = margin_only_scale_matched_loss(student, torch.zeros(2, 1))
    loss.backward()
    assert float(loss) == 0.0
    with pytest.raises(ValueError, match="identical shapes"):
        margin_only_scale_matched_loss(torch.zeros(2, 3), torch.zeros(2, 2))
