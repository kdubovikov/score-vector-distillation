"""E032: per-row Pearson loss with fixed, detached row weights (weighted_unit_centered_mse)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from test_e028_training_integration import _batch, _components, _fake_row_scores, _FakeBackbone, _training_mapping
from torch import Tensor

from score_vector_distillation.config import ConfigError, DistillationLoss, RowWeightConfig, TrainingConfig
from score_vector_distillation.losses import (
    centered_mse_loss,
    pearson_row_weights,
    unit_centered_mse_loss,
    weighted_unit_centered_mse_loss,
)
from score_vector_distillation.training import model as model_module
from score_vector_distillation.training import trainer
from score_vector_distillation.training.model import _DISTILLATION_LOSSES, ScoreVectorStudent


@pytest.fixture
def fake_row_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "checkpointed_row_scores", _fake_row_scores)


ARM_A = {"student_exponent": 1.0, "teacher_exponent": 1.0, "student_scale": 0.27, "teacher_scale": 0.11}
ARM_B = {"student_exponent": 0.0, "teacher_exponent": 1.0, "student_scale": 0.27, "teacher_scale": 0.11}
UNIT = {"student_exponent": 0.0, "teacher_exponent": 0.0, "student_scale": 1.0, "teacher_scale": 1.0}


def _rows(seed: int = 0, *, rows: int = 6, dtype: torch.dtype = torch.float64) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    # rows with very different spreads, as on teacher-mined rows
    spread = torch.linspace(0.01, 0.12, rows, dtype=dtype)[:, None]
    student = 0.4 + spread * torch.randn(rows, 32, generator=generator, dtype=dtype)
    teacher = 0.5 + 0.5 * spread.flip(0) * torch.randn(rows, 32, generator=generator, dtype=dtype)
    return student, teacher


def _grad(function: Any, student: Tensor, teacher: Tensor, **kwargs: float) -> tuple[Tensor, Tensor]:
    leaf = student.clone().requires_grad_(True)
    loss = function(leaf, teacher, **kwargs)
    loss.backward()
    assert leaf.grad is not None
    return loss.detach(), leaf.grad


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_unit_weights_reduce_to_the_pearson_loss(dtype: torch.dtype) -> None:
    student, teacher = _rows(dtype=dtype)
    weighted, weighted_grad = _grad(weighted_unit_centered_mse_loss, student, teacher, **UNIT)
    pearson, pearson_grad = _grad(unit_centered_mse_loss, student, teacher)
    tolerance = 1e-12 if dtype is torch.float64 else 1e-6
    assert weighted.item() == pytest.approx(pearson.item(), rel=tolerance)
    assert torch.allclose(weighted_grad, pearson_grad, rtol=tolerance, atol=tolerance * pearson_grad.abs().max().item())
    assert torch.equal(pearson_row_weights(student, teacher, **UNIT), torch.ones(student.shape[0], dtype=dtype))


@pytest.mark.parametrize("arm", [ARM_A, ARM_B])
def test_float64_probe_of_the_per_row_formula(arm: dict[str, float]) -> None:
    student, teacher = _rows(1)
    k = student.shape[1]
    s, t = student.numpy(), teacher.numpy()
    rho = np.array([np.corrcoef(s[r], t[r])[0, 1] for r in range(len(s))])
    norm_s = np.linalg.norm(s - s.mean(1, keepdims=True), axis=1)
    norm_t = np.linalg.norm(t - t.mean(1, keepdims=True), axis=1)
    weights = (norm_s / arm["student_scale"]) ** arm["student_exponent"] * (norm_t / arm["teacher_scale"]) ** arm[
        "teacher_exponent"
    ]
    expected = float(np.mean(weights * (2.0 / k) * (1.0 - rho)))
    assert weighted_unit_centered_mse_loss(student, teacher, **arm).item() == pytest.approx(expected, rel=1e-12)
    assert np.allclose(pearson_row_weights(student, teacher, **arm).numpy(), weights, rtol=1e-13, atol=0)
    assert np.ptp(weights) > 0.5 * weights.mean()  # the fixture's rows are weighted very differently
    if arm is ARM_A:
        shape_term = (2.0 / k) * norm_s * norm_t * (1.0 - rho) / (arm["student_scale"] * arm["teacher_scale"])
        assert expected == pytest.approx(float(shape_term.mean()), rel=1e-12)


@pytest.mark.parametrize("arm", [ARM_A, ARM_B])
def test_weights_are_detached_each_row_gradient_is_weight_times_its_pearson_gradient(arm: dict[str, float]) -> None:
    student, teacher = _rows(2)
    rows = student.shape[0]
    _, grad = _grad(weighted_unit_centered_mse_loss, student, teacher, **arm)
    weights = pearson_row_weights(student, teacher, **arm)
    assert not weights.requires_grad and weights.grad_fn is None
    for r in range(rows):
        _, row_grad = _grad(unit_centered_mse_loss, student[r : r + 1], teacher[r : r + 1])
        assert torch.allclose(grad[r], weights[r] / rows * row_grad[0], rtol=1e-11, atol=1e-16)
    # the teacher side is never differentiated through the weights either
    teacher_leaf = teacher.clone().requires_grad_(True)
    student_leaf = student.clone().requires_grad_(True)
    weighted_unit_centered_mse_loss(student_leaf, teacher_leaf, **{**arm, "student_exponent": 0.0}).backward()
    assert teacher_leaf.grad is not None
    teacher_only = teacher.clone().requires_grad_(True)
    unit_centered_mse_loss(student, teacher_only).backward()
    assert teacher_only.grad is not None


@pytest.mark.parametrize("arm", [ARM_A, ARM_B, UNIT])
@pytest.mark.parametrize("factor", [0.25, 3.0])
def test_row_gradient_direction_is_invariant_to_positive_rescaling_of_the_student(
    arm: dict[str, float], factor: float
) -> None:
    student, teacher = _rows(3)
    _, base = _grad(weighted_unit_centered_mse_loss, student, teacher, **arm)
    _, scaled = _grad(weighted_unit_centered_mse_loss, factor * student, teacher, **arm)
    # direction unchanged row by row; magnitude scales by factor ** (student_exponent - 1)
    cosine = torch.nn.functional.cosine_similarity(base, scaled, dim=1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-12)
    expected = factor ** (arm["student_exponent"] - 1.0)
    assert torch.allclose(scaled, expected * base, rtol=1e-10, atol=1e-18)
    # row offsets change nothing
    _, shifted = _grad(weighted_unit_centered_mse_loss, student + 0.3, teacher - 0.1, **arm)
    assert torch.allclose(shifted, base, rtol=1e-10, atol=1e-18)


def test_arm_a_gradient_is_the_pairmse_gradient_without_its_spread_component() -> None:
    student, teacher = _rows(4)
    rows, k = student.shape
    unit_scales = {**ARM_A, "student_scale": 1.0, "teacher_scale": 1.0}
    _, grad_a = _grad(weighted_unit_centered_mse_loss, student, teacher, **unit_scales)
    _, grad_pair = _grad(centered_mse_loss, student, teacher)
    centered = student - student.mean(dim=1, keepdim=True)
    direction = centered / centered.norm(dim=1, keepdim=True)
    tangential = grad_pair - (grad_pair * direction).sum(dim=1, keepdim=True) * direction
    assert torch.allclose(grad_a, tangential, rtol=1e-10, atol=1e-16)
    assert (grad_a * direction).sum(dim=1).abs().max().item() < 1e-15  # no spread (radial) component
    # the constants only divide the gradient
    _, grad_scaled = _grad(weighted_unit_centered_mse_loss, student, teacher, **ARM_A)
    assert torch.allclose(grad_scaled, grad_a / (ARM_A["student_scale"] * ARM_A["teacher_scale"]), rtol=1e-12)
    del rows, k


def test_invalid_row_weighting_is_rejected() -> None:
    student, teacher = _rows(5)
    for bad in (
        {**ARM_A, "student_exponent": -1.0},
        {**ARM_A, "teacher_exponent": math.nan},
        {**ARM_A, "student_scale": 0.0},
        {**ARM_A, "teacher_scale": -0.1},
        {**ARM_A, "teacher_scale": math.inf},
    ):
        with pytest.raises(ValueError):
            weighted_unit_centered_mse_loss(student, teacher, **bad)


def test_config_row_weighting_is_owned_by_the_weighted_loss() -> None:
    mapping = {"student_exponent": 1, "teacher_exponent": 1, "student_scale": 0.27, "teacher_scale": 0.11}
    weighted = TrainingConfig.from_mapping(_training_mapping(loss="weighted_unit_centered_mse", row_weighting=mapping))
    assert weighted.loss is DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE
    assert weighted.row_weighting == RowWeightConfig(1.0, 1.0, 0.27, 0.11)
    assert TrainingConfig.from_mapping(_training_mapping(loss="unit_centered_mse")).row_weighting is None
    with pytest.raises(ConfigError, match="row_weighting is required"):
        TrainingConfig.from_mapping(_training_mapping(loss="weighted_unit_centered_mse"))
    with pytest.raises(ConfigError, match="row_weighting is only supported"):
        TrainingConfig.from_mapping(_training_mapping(loss="unit_centered_mse", row_weighting=mapping))
    with pytest.raises(ConfigError, match="missing required fields"):
        TrainingConfig.from_mapping(
            _training_mapping(loss="weighted_unit_centered_mse", row_weighting={"student_exponent": 1})
        )
    with pytest.raises(ConfigError, match="unsupported fields"):
        TrainingConfig.from_mapping(
            _training_mapping(loss="weighted_unit_centered_mse", row_weighting={**mapping, "normalize": "batch"})
        )
    with pytest.raises(ConfigError, match="must be positive"):
        TrainingConfig.from_mapping(
            _training_mapping(loss="weighted_unit_centered_mse", row_weighting={**mapping, "teacher_scale": 0})
        )
    with pytest.raises(ConfigError, match=r"greater than or equal to 0\.0"):
        TrainingConfig.from_mapping(
            _training_mapping(loss="weighted_unit_centered_mse", row_weighting={**mapping, "student_exponent": -1})
        )


def test_dispatch_of_existing_keys_is_unchanged_and_the_new_key_is_handled_by_the_student(
    fake_row_scoring: None,
) -> None:
    from score_vector_distillation import losses

    assert _DISTILLATION_LOSSES[DistillationLoss.UNIT_CENTERED_MSE] is losses.unit_centered_mse_loss
    assert _DISTILLATION_LOSSES[DistillationLoss.CENTERED_MSE] is losses.centered_mse_loss
    assert DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE not in _DISTILLATION_LOSSES
    student = ScoreVectorStudent(_FakeBackbone())  # type: ignore[arg-type]
    assert [name for name, _ in student.named_parameters()] == ["backbone.weight", "backbone.frozen"]
    batch = _batch(0)
    inputs = ({"x": batch["query"]}, {"x": batch["documents"]}, batch["teacher_scores"])
    pearson, scores = student(*inputs, document_chunk_size=5, loss=DistillationLoss.UNIT_CENTERED_MSE)
    unit, _ = student(
        *inputs,
        document_chunk_size=5,
        loss=DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE,
        row_weighting=RowWeightConfig(**UNIT),
    )
    assert unit.item() == pytest.approx(pearson.item(), rel=1e-6)
    weighted, _ = student(
        *inputs,
        document_chunk_size=5,
        loss=DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE,
        row_weighting=RowWeightConfig(**ARM_A),
    )
    expected = weighted_unit_centered_mse_loss(scores.float(), batch["teacher_scores"].float(), **ARM_A)
    assert torch.equal(weighted, expected)
    with pytest.raises(ValueError, match="row_weighting is required"):
        student(*inputs, document_chunk_size=5, loss=DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE)


def test_weighted_training_logs_row_weights_and_validates_with_the_weighted_loss(
    fake_row_scoring: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = trainer._build_training_components

    def build(prepared: Any, accelerator: Any) -> Any:
        prepared.run.training.row_weighting = RowWeightConfig(**ARM_A)
        return original(prepared, accelerator)

    monkeypatch.setattr(trainer, "_build_training_components", build)
    prepared, components, accelerator = _components(
        DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE, monkeypatch, max_norm=1.0
    )
    assert len(components.optimizer.param_groups) == 1
    request = SimpleNamespace(resume_from=None)
    steps, _ = trainer._run_training_epochs(request, prepared, components, accelerator, output_dir=tmp_path)  # type: ignore[arg-type]
    assert steps == 4
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train = [record for record in records if record["event"] == "train"]
    assert len(train) == 4
    for record in train:
        assert len(record["row_weight"]) == 2 and all(value > 0 for value in record["row_weight"])
        for weight, norm_s, norm_t in zip(
            record["row_weight"], record["row_student_centered_norm"], record["row_teacher_centered_norm"], strict=True
        ):
            assert weight == pytest.approx(
                norm_s * norm_t / (ARM_A["student_scale"] * ARM_A["teacher_scale"]), rel=1e-5
            )
    validation = [record for record in records if record["event"] == "validation"]
    batch = prepared.validation_dataset[0]
    with torch.no_grad():
        expected, _ = components.student(
            {"x": batch["query"]},
            {"x": batch["documents"]},
            batch["teacher_scores"],
            document_chunk_size=5,
            loss=DistillationLoss.WEIGHTED_UNIT_CENTERED_MSE,
            row_weighting=RowWeightConfig(**ARM_A),
        )
    assert validation[0]["validation_loss"] == pytest.approx(expected.item())


def test_existing_losses_log_no_row_weights(
    fake_row_scoring: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared, components, accelerator = _components(DistillationLoss.UNIT_CENTERED_MSE, monkeypatch, max_norm=1.0)
    trainer._run_training_epochs(
        SimpleNamespace(resume_from=None), prepared, components, accelerator, output_dir=tmp_path
    )  # type: ignore[arg-type]
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all("row_weight" not in record for record in records)
