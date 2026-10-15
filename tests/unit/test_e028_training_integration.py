"""E028 wiring: config keys, the learned-scale student, optimizer groups, clipping and gradient-norm logging."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from score_vector_distillation.config import ConfigError, DistillationLoss, TrainingConfig
from score_vector_distillation.losses import centered_mse_loss, scale_matched_centered_mse_loss
from score_vector_distillation.training import model as model_module
from score_vector_distillation.training import trainer
from score_vector_distillation.training.model import _DISTILLATION_LOSSES, ScoreVectorStudent


def _training_mapping(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "loss": "centered_mse",
        "epochs": 1,
        "seeds": [42],
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "reference_world_size": 8,
        "effective_batch_size": 16,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "max_gradient_norm": 1.0,
        "parameter_dtype": "float32",
        "mixed_precision": "none",
        "backbone_gradient_checkpointing": False,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "candidate_checkpointing": {"document_chunk_size": 4, "encode_query_once": True, "preserve_rng_state": True},
    }
    values.update(overrides)
    return values


def test_config_keys_for_the_new_losses_parse_and_are_owned_by_their_loss() -> None:
    scale = TrainingConfig.from_mapping(
        _training_mapping(loss="scale_matched_centered_mse", score_scale_learning_rate=0.05)
    )
    assert scale.loss is DistillationLoss.SCALE_MATCHED_CENTERED_MSE and scale.score_scale_learning_rate == 0.05
    kl = TrainingConfig.from_mapping(
        _training_mapping(loss="forward_kl_teacher_temperature", loss_temperature=0.05, teacher_temperature=0.025)
    )
    assert (kl.loss_temperature, kl.teacher_temperature) == (0.05, 0.025)
    centered = TrainingConfig.from_mapping(_training_mapping())
    assert (centered.loss_temperature, centered.teacher_temperature, centered.score_scale_learning_rate) == (
        None,
        None,
        None,
    )

    with pytest.raises(ConfigError, match="score_scale_learning_rate is required"):
        TrainingConfig.from_mapping(_training_mapping(loss="scale_matched_centered_mse"))
    with pytest.raises(ConfigError, match="score_scale_learning_rate is only supported"):
        TrainingConfig.from_mapping(_training_mapping(score_scale_learning_rate=0.05))
    with pytest.raises(ConfigError, match="teacher_temperature is required"):
        TrainingConfig.from_mapping(_training_mapping(loss="forward_kl_teacher_temperature", loss_temperature=0.05))
    with pytest.raises(ConfigError, match="loss_temperature is required"):
        TrainingConfig.from_mapping(_training_mapping(loss="forward_kl_teacher_temperature", teacher_temperature=0.05))
    with pytest.raises(ConfigError, match="teacher_temperature is only supported"):
        TrainingConfig.from_mapping(_training_mapping(teacher_temperature=0.05))
    with pytest.raises(ConfigError, match="loss_temperature is only supported"):
        TrainingConfig.from_mapping(_training_mapping(loss_temperature=0.05))
    with pytest.raises(ConfigError, match="must be positive"):
        TrainingConfig.from_mapping(
            _training_mapping(loss="forward_kl_teacher_temperature", loss_temperature=0.05, teacher_temperature=0.0)
        )
    with pytest.raises(ConfigError, match="must be positive"):
        TrainingConfig.from_mapping(_training_mapping(loss="scale_matched_centered_mse", score_scale_learning_rate=0.0))
    inbatch = TrainingConfig.from_mapping(
        _training_mapping(loss="label_only_inbatch_cross_entropy", loss_temperature=0.05)
    )
    assert inbatch.loss_temperature == 0.05


def test_existing_loss_dispatch_is_unchanged() -> None:
    from score_vector_distillation import losses

    assert _DISTILLATION_LOSSES[DistillationLoss.CENTERED_MSE] is losses.centered_mse_loss
    assert _DISTILLATION_LOSSES[DistillationLoss.UNIT_CENTERED_MSE] is losses.unit_centered_mse_loss
    assert _DISTILLATION_LOSSES[DistillationLoss.FORWARD_KL_TAU_0P1] is losses.forward_kl_loss
    assert (
        _DISTILLATION_LOSSES[DistillationLoss.POSITIVE_NEGATIVE_MARGIN_MSE] is losses.positive_negative_margin_mse_loss
    )
    assert DistillationLoss.SCALE_MATCHED_CENTERED_MSE not in _DISTILLATION_LOSSES
    assert DistillationLoss.FORWARD_KL_TEACHER_TEMPERATURE not in _DISTILLATION_LOSSES


class _FakeBackbone(nn.Module):
    normalize_eps = 1e-12

    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.weight = nn.Parameter(torch.eye(dimension) + 0.2 * torch.randn(dimension, dimension, generator=generator))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward_features(self, features: Mapping[str, Tensor]) -> Tensor:
        return features["x"] @ self.weight


def _fake_row_scores(
    *,
    encode_queries: Any,
    encode_documents: Any,
    query_inputs: Mapping[str, Tensor],
    document_inputs: Mapping[str, Tensor],
    document_chunk_size: int,
    normalize_embeddings: bool,
    normalization_eps: float,
) -> Tensor:
    del document_chunk_size, normalization_eps
    queries = encode_queries(query_inputs)
    documents = encode_documents(document_inputs)
    if normalize_embeddings:
        queries = torch.nn.functional.normalize(queries, dim=-1)
        documents = torch.nn.functional.normalize(documents, dim=-1)
    return torch.einsum("bd,bkd->bk", queries, documents)


@pytest.fixture
def fake_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "checkpointed_row_scores", _fake_row_scores)


def _batch(seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "query": torch.randn(2, 4, generator=generator),
        "documents": torch.randn(2, 5, 4, generator=generator),
        "teacher_scores": 0.5 + 0.02 * torch.randn(2, 5, generator=generator),
    }


def test_learned_scale_student_adds_one_zero_parameter_and_dispatches(fake_scoring: None) -> None:
    plain = ScoreVectorStudent(_FakeBackbone())  # type: ignore[arg-type]
    scaled = ScoreVectorStudent(_FakeBackbone(), learned_score_scale=True)  # type: ignore[arg-type]
    assert plain.learned_score_scale is None
    assert [name for name, _ in plain.named_parameters()] == ["backbone.weight", "backbone.frozen"]
    assert [name for name, _ in scaled.named_parameters()] == ["score_log_scale", "backbone.weight", "backbone.frozen"]
    assert scaled.learned_score_scale is not None and scaled.learned_score_scale.item() == 0.0

    batch = _batch(0)
    inputs = ({"x": batch["query"]}, {"x": batch["documents"]}, batch["teacher_scores"])
    centered, scores = plain(*inputs, document_chunk_size=5, loss=DistillationLoss.CENTERED_MSE)
    at_unit, _ = scaled(*inputs, document_chunk_size=5, loss=DistillationLoss.SCALE_MATCHED_CENTERED_MSE)
    assert torch.equal(centered, at_unit)
    assert torch.equal(centered, centered_mse_loss(scores.float(), batch["teacher_scores"].float()))
    with pytest.raises(ValueError, match="learned_score_scale=True"):
        plain(*inputs, document_chunk_size=5, loss=DistillationLoss.SCALE_MATCHED_CENTERED_MSE)
    with pytest.raises(ValueError, match="teacher_temperature"):
        plain(
            *inputs, document_chunk_size=5, loss=DistillationLoss.FORWARD_KL_TEACHER_TEMPERATURE, loss_temperature=0.05
        )
    kl, _ = plain(
        *inputs,
        document_chunk_size=5,
        loss=DistillationLoss.FORWARD_KL_TEACHER_TEMPERATURE,
        loss_temperature=0.05,
        teacher_temperature=0.025,
    )
    assert torch.isfinite(kl) and kl.item() > 0


class _FakeAccelerator:
    """Single-process stand-in: every micro-batch is an optimizer step; clip calls are recorded."""

    is_main_process = True
    num_processes = 1
    process_index = 0
    device = torch.device("cpu")
    sync_gradients = True

    def __init__(self) -> None:
        self.clip_calls: list[list[int]] = []

    @contextmanager
    def accumulate(self, model: nn.Module) -> Iterator[None]:
        del model
        yield

    def backward(self, loss: Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> Tensor:
        parameters = list(parameters)
        self.clip_calls.append([id(parameter) for parameter in parameters])
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def wait_for_everyone(self) -> None:
        return None

    def save_state(self, state_dir: Path) -> None:
        Path(state_dir).mkdir(parents=True, exist_ok=True)

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        return objects


def _training(loss: DistillationLoss, *, max_norm: float) -> SimpleNamespace:
    return SimpleNamespace(
        loss=loss,
        epochs=1,
        learning_rate=1e-2,
        weight_decay=0.01,
        warmup_ratio=0.0,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        eval_batch_size=1,
        max_gradient_norm=max_norm,
        loss_temperature=None,
        teacher_temperature=None,
        score_scale_learning_rate=0.05 if loss is DistillationLoss.SCALE_MATCHED_CENTERED_MSE else None,
        lora=SimpleNamespace(rank=4, alpha=8, dropout=0.0, target_modules=("x",)),
        backbone_gradient_checkpointing=False,
    )


def _components(loss: DistillationLoss, monkeypatch: pytest.MonkeyPatch, *, max_norm: float) -> tuple[Any, Any, Any]:
    monkeypatch.setattr(trainer, "SentenceTransformerBackbone", lambda *args, **kwargs: _FakeBackbone())
    monkeypatch.setattr(trainer, "profile_from_mapping", lambda data: None)
    monkeypatch.setattr(
        trainer,
        "_tokenize_batch",
        lambda model, batch, device: ({"x": batch["query"]}, {"x": batch["documents"]}),
    )
    rows = [_batch(seed) for seed in range(4)]
    prepared = SimpleNamespace(
        run=SimpleNamespace(training=_training(loss, max_norm=max_norm), sha256="run"),
        student_config=SimpleNamespace(data={}, sha256="student"),
        train_dataset=rows,
        validation_dataset=rows[:1],
        document_chunk_size=5,
    )
    monkeypatch.setattr(trainer, "collate_distillation_rows", lambda items: items[0])
    accelerator = _FakeAccelerator()
    components = trainer._build_training_components(prepared, accelerator)  # type: ignore[arg-type]
    return prepared, components, accelerator


def test_optimizer_groups_are_unchanged_for_existing_losses_and_split_for_the_learned_scale(
    fake_scoring: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plain, _ = _components(DistillationLoss.CENTERED_MSE, monkeypatch, max_norm=1.0)
    assert len(plain.optimizer.param_groups) == 1
    assert plain.optimizer.param_groups[0]["params"] == [plain.student.backbone.weight]
    assert plain.student.learned_score_scale is None

    _, scaled, _ = _components(DistillationLoss.SCALE_MATCHED_CENTERED_MSE, monkeypatch, max_norm=1.0)
    groups = scaled.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["params"] == [scaled.student.backbone.weight]
    assert groups[0]["lr"] == pytest.approx(1e-2 * 1.0) and groups[0]["weight_decay"] == 0.01
    assert groups[1]["params"] == [scaled.student.learned_score_scale]
    assert groups[1]["weight_decay"] == 0.0
    assert groups[1]["initial_lr"] == 0.05


def _manual_reference(loss_kind: DistillationLoss, monkeypatch: pytest.MonkeyPatch, *, max_norm: float) -> Tensor:
    """The 8c1fe09 update rule, written out: forward, backward, clip all student parameters, step."""

    prepared, components, _ = _components(loss_kind, monkeypatch, max_norm=max_norm)
    for batch in components.train_loader:
        loss, _ = components.student(
            {"x": batch["query"]},
            {"x": batch["documents"]},
            batch["teacher_scores"],
            document_chunk_size=5,
            loss=loss_kind,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(components.student.parameters(), max_norm)
        components.optimizer.step()
        components.scheduler.step()
        components.optimizer.zero_grad(set_to_none=True)
    del prepared
    return components.student.backbone.weight.detach().clone()


@pytest.mark.parametrize("max_norm", [1.0, 1e-3])
@pytest.mark.parametrize(
    "loss_kind",
    [DistillationLoss.CENTERED_MSE, DistillationLoss.UNIT_CENTERED_MSE, DistillationLoss.POSITIVE_NEGATIVE_MARGIN_MSE],
)
def test_gradient_norm_logging_does_not_change_existing_training(
    loss_kind: DistillationLoss, max_norm: float, fake_scoring: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    reference = _manual_reference(loss_kind, monkeypatch, max_norm=max_norm)

    torch.manual_seed(0)
    prepared, components, accelerator = _components(loss_kind, monkeypatch, max_norm=max_norm)
    request = SimpleNamespace(resume_from=None)
    steps, _ = trainer._run_training_epochs(request, prepared, components, accelerator, output_dir=tmp_path)  # type: ignore[arg-type]

    assert steps == 4
    assert torch.equal(components.student.backbone.weight.detach(), reference)
    every_parameter = [id(parameter) for parameter in components.student.parameters()]
    assert accelerator.clip_calls == [every_parameter] * 4
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train = [record for record in records if record["event"] == "train"]
    assert len(train) == 4
    assert all(record["grad_norm_pre_clip"] > 0 for record in train)
    assert all(record["grad_clipped"] == (record["grad_norm_pre_clip"] > max_norm) for record in train)
    assert all("score_scale" not in record for record in train)
    if max_norm == 1e-3:
        assert all(record["grad_clipped"] for record in train)


def test_learned_scale_training_logs_scale_and_excludes_it_from_clipping(
    fake_scoring: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared, components, accelerator = _components(
        DistillationLoss.SCALE_MATCHED_CENTERED_MSE, monkeypatch, max_norm=1e-6
    )
    scale = components.student.learned_score_scale
    request = SimpleNamespace(resume_from=None)
    trainer._run_training_epochs(request, prepared, components, accelerator, output_dir=tmp_path)  # type: ignore[arg-type]

    backbone_only = [id(components.student.backbone.weight), id(components.student.backbone.frozen)]
    assert accelerator.clip_calls == [backbone_only] * 4
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train = [record for record in records if record["event"] == "train"]
    assert all(record["grad_clipped"] for record in train)
    # Adam's first step moves the unclipped scale by its own learning rate (warmup ratio 0 here).
    assert train[0]["score_log_scale"] == pytest.approx(
        -0.05 * (1 if train[0]["score_log_scale_grad"] > 0 else -1), rel=1e-4
    )
    assert train[0]["score_scale"] == pytest.approx(torch.tensor(train[0]["score_log_scale"]).exp().item())
    assert train[0]["score_scale_learning_rate"] > 0 and all(
        record["score_scale_learning_rate"] >= 0 for record in train
    )
    assert train[-1]["score_log_scale"] == pytest.approx(float(scale.detach()))
    summary = trainer._score_scale_summary(components.student)
    assert summary["final_score_log_scale"] == pytest.approx(float(scale.detach()))
    assert trainer._score_scale_summary(ScoreVectorStudent(_FakeBackbone())) == {}  # type: ignore[arg-type]
    validation = [record for record in records if record["event"] == "validation"]
    assert len(validation) == 1
    batch = prepared.validation_dataset[0]
    with torch.no_grad():
        expected, _ = components.student(
            {"x": batch["query"]},
            {"x": batch["documents"]},
            batch["teacher_scores"],
            document_chunk_size=5,
            loss=DistillationLoss.SCALE_MATCHED_CENTERED_MSE,
        )
    assert validation[0]["validation_loss"] == pytest.approx(expected.item())
    assert scale_matched_centered_mse_loss  # imported for the public API contract


def test_wrapped_parameter_names_are_found() -> None:
    wrapper = nn.Module()
    wrapper.module = ScoreVectorStudent(_FakeBackbone(), learned_score_scale=True)  # type: ignore[arg-type]
    found = trainer._learned_scale_parameter(wrapper)
    assert found is wrapper.module.learned_score_scale
    assert trainer._learned_scale_parameter(ScoreVectorStudent(_FakeBackbone())) is None  # type: ignore[arg-type]
