"""Tests for score-vector losses."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from score_vector_distillation.losses import (
    centered_mse_loss,
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
