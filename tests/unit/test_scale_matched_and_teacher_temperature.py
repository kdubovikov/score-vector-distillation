"""E028 losses: learned-global-scale centered MSE and forward KL with a separate teacher temperature."""

from __future__ import annotations

import math

import pytest
import torch

from score_vector_distillation.losses import (
    centered_mse_loss,
    forward_kl_loss,
    forward_kl_teacher_temperature_loss,
    scale_matched_centered_mse_loss,
)


def _rows(seed: int = 0, *, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = 0.4 + 0.05 * torch.randn(6, 32, generator=generator, dtype=dtype)
    teacher = 0.5 + 0.02 * torch.randn(6, 32, generator=generator, dtype=dtype)
    return student, teacher


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_scale_matched_reduces_to_centered_mse_at_unit_scale_with_frozen_log_scale(dtype: torch.dtype) -> None:
    student, teacher = _rows(dtype=dtype)
    frozen = torch.zeros((), dtype=dtype, requires_grad=False)

    left = student.clone().requires_grad_(True)
    right = student.clone().requires_grad_(True)
    scaled = scale_matched_centered_mse_loss(left, teacher, log_scale=frozen)
    reference = centered_mse_loss(right, teacher)
    scaled.backward()
    reference.backward()

    assert torch.equal(scaled.detach(), reference.detach())
    assert left.grad is not None and right.grad is not None
    assert torch.equal(left.grad, right.grad)


def test_scale_matched_is_invariant_to_moving_scale_between_student_and_parameter() -> None:
    student, teacher = _rows(1)
    log_scale = torch.tensor(-0.7, dtype=torch.float64)
    factor = 2.5

    moved = scale_matched_centered_mse_loss(student * factor, teacher, log_scale=log_scale)
    absorbed = scale_matched_centered_mse_loss(student, teacher, log_scale=log_scale + math.log(factor))

    assert moved.item() == pytest.approx(absorbed.item(), rel=1e-12, abs=1e-18)


def test_scale_matched_log_scale_gradient_and_closed_form_optimum() -> None:
    student, teacher = _rows(2)
    centered_student = student - student.mean(dim=1, keepdim=True)
    centered_teacher = teacher - teacher.mean(dim=1, keepdim=True)
    candidates = student.shape[1]

    log_scale = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    loss = scale_matched_centered_mse_loss(student, teacher, log_scale=log_scale)
    loss.backward()
    scale = math.exp(0.3)
    residual = scale * centered_student - centered_teacher
    expected = scale * float((2.0 / candidates) * (residual * centered_student).sum(dim=1).mean())
    assert log_scale.grad is not None
    assert log_scale.grad.item() == pytest.approx(expected, rel=1e-12)

    optimum = float((centered_student * centered_teacher).sum() / centered_student.square().sum())
    assert optimum > 0
    at_optimum = torch.tensor(math.log(optimum), dtype=torch.float64, requires_grad=True)
    scale_matched_centered_mse_loss(student, teacher, log_scale=at_optimum).backward()
    assert at_optimum.grad is not None
    assert abs(at_optimum.grad.item()) < 1e-15


def test_scale_matched_rejects_non_scalar_or_integer_log_scale() -> None:
    student, teacher = _rows(3)
    with pytest.raises(ValueError, match="scalar"):
        scale_matched_centered_mse_loss(student, teacher, log_scale=torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="scalar"):
        scale_matched_centered_mse_loss(student, teacher, log_scale=torch.tensor(0))


@pytest.mark.parametrize("temperature", [0.0125, 0.05, 0.1])
def test_teacher_temperature_kl_with_equal_temperatures_is_the_shared_temperature_kl(temperature: float) -> None:
    student, teacher = _rows(4, dtype=torch.float32)
    left = student.clone().requires_grad_(True)
    right = student.clone().requires_grad_(True)
    separate = forward_kl_teacher_temperature_loss(
        left, teacher, student_temperature=temperature, teacher_temperature=temperature
    )
    shared = forward_kl_loss(right, teacher, temperature=temperature)
    separate.backward()
    shared.backward()

    assert torch.equal(separate.detach(), shared.detach())
    assert left.grad is not None and right.grad is not None
    assert torch.equal(left.grad, right.grad)


@pytest.mark.parametrize("teacher_temperature", [0.0125, 0.025, 0.05, 0.1])
def test_teacher_temperature_kl_is_zero_at_the_implied_student_scale(teacher_temperature: float) -> None:
    _, teacher = _rows(5)
    student_temperature = 0.05
    implied = (student_temperature / teacher_temperature) * teacher + 0.3
    student = implied.clone().requires_grad_(True)

    loss = forward_kl_teacher_temperature_loss(
        student, teacher, student_temperature=student_temperature, teacher_temperature=teacher_temperature
    )
    loss.backward()

    assert abs(loss.item()) < 1e-12
    assert student.grad is not None
    assert student.grad.abs().max().item() < 1e-12
    unscaled = forward_kl_teacher_temperature_loss(
        teacher, teacher, student_temperature=student_temperature, teacher_temperature=teacher_temperature
    )
    if teacher_temperature != student_temperature:
        assert unscaled.item() > 1e-6


def test_teacher_temperature_kl_detaches_teacher_and_rejects_bad_temperatures() -> None:
    student, teacher = _rows(6)
    teacher = teacher.clone().requires_grad_(True)
    student = student.clone().requires_grad_(True)
    forward_kl_teacher_temperature_loss(
        student, teacher, student_temperature=0.05, teacher_temperature=0.025
    ).backward()
    assert teacher.grad is None
    for bad in (0.0, -0.1, math.inf, math.nan):
        with pytest.raises(ValueError):
            forward_kl_teacher_temperature_loss(student, teacher, student_temperature=0.05, teacher_temperature=bad)
        with pytest.raises(ValueError):
            forward_kl_teacher_temperature_loss(student, teacher, student_temperature=bad, teacher_temperature=0.05)
