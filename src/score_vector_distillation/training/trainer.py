"""Accelerate-compatible LoRA training for score-vector distillation."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from score_vector_distillation.config import (
    CandidateCheckpointingConfig,
    DistillationLoss,
    ExecutableRunConfig,
    LoraConfig,
    MixedPrecision,
    ParameterDType,
    ResolvedConfig,
    SchedulerKind,
    TrainingConfig,
    load_config,
    load_executable_run_config,
)
from score_vector_distillation.models.profiles import profile_from_mapping
from score_vector_distillation.models.sentence_transformer import (
    LoraSettings,
    SentenceTransformerBackbone,
    move_features,
    reshape_candidate_features,
)
from score_vector_distillation.run_record import (
    checkpoint_record,
    git_commit,
    runtime_summary,
    sha256_file,
    write_run_record,
)
from score_vector_distillation.training.dataset import (
    DistillationDataset,
    collate_distillation_rows,
)
from score_vector_distillation.training.model import ScoreVectorStudent

if TYPE_CHECKING:
    from accelerate import Accelerator


# Candidate-row validation intentionally runs once, unsharded, on the main
# process. It can take substantially longer than Accelerate's 600-second NCCL
# default, so synchronization around it needs a timeout sized for the job rather
# than for an ordinary gradient collective. One hour leaves substantial margin
# for the observed validation overrun without hiding multi-hour hangs.
DISTRIBUTED_PROCESS_GROUP_TIMEOUT = timedelta(hours=1)


@dataclass(frozen=True)
class TrainingRequest:
    run_config: Path
    train_file: Path
    eval_file: Path
    output_dir: Path
    seed: int | None = None
    resume_from: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class PreparedTrainingRun:
    """Validated configs and datasets needed to train."""

    run: ExecutableRunConfig
    student_config: ResolvedConfig
    data_config: ResolvedConfig
    teacher_configs: tuple[ResolvedConfig, ...]
    teacher_labels: tuple[str, ...]
    seed: int
    train_dataset: DistillationDataset
    validation_dataset: DistillationDataset
    candidate_count: int
    document_chunk_size: int

    def summary(self) -> dict[str, Any]:
        """Return the useful run shape available before allocating a model."""

        return {
            "run": self.run.name,
            "seed": self.seed,
            "train_rows": len(self.train_dataset),
            "eval_rows": len(self.validation_dataset),
            "candidate_count": self.candidate_count,
            "document_chunk_size": self.document_chunk_size,
            "teachers": list(self.teacher_labels),
        }


@dataclass(frozen=True)
class TrainingComponents:
    """Model and optimizer objects constructed for one Accelerate runtime."""

    student: ScoreVectorStudent
    optimizer: AdamW
    train_loader: DataLoader[Any]
    validation_loader: DataLoader[Any]
    scheduler: LambdaLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configured_candidate_count(data_config: ResolvedConfig) -> int:
    row_construction = data_config.data.get("row_construction")
    if not isinstance(row_construction, Mapping):
        raise ValueError("Data config requires row_construction metadata.")
    candidates = row_construction.get("candidates")
    if not isinstance(candidates, Mapping) or "count" not in candidates:
        raise ValueError("Data config requires row_construction.candidates.count.")
    count = int(candidates["count"])
    if count < 2:
        raise ValueError("Configured candidate count must be at least two.")
    return count


def _teacher_labels(configs: tuple[ResolvedConfig, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for config in configs:
        normalized_label = profile_from_mapping(config.data).label
        if normalized_label in labels:
            raise ValueError(f"Duplicate teacher label in run config: {normalized_label}")
        labels.append(normalized_label)
    return tuple(sorted(labels))


def _seed_for_request(request: TrainingRequest, training: TrainingConfig) -> int:
    if request.seed is not None:
        return request.seed
    return training.seeds[0]


def _lora_settings(config: LoraConfig) -> LoraSettings:
    return LoraSettings(
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=config.target_modules,
    )


def validate_runtime_batch(training: TrainingConfig, *, num_processes: int) -> int:
    """Return the realized global batch size or fail on a paper-config mismatch."""

    if num_processes <= 0:
        raise ValueError("num_processes must be positive.")
    per_device = training.per_device_batch_size
    accumulation = training.gradient_accumulation_steps
    realized = per_device * accumulation * num_processes
    expected_world_size = training.reference_world_size
    expected_batch = training.effective_batch_size
    if num_processes != expected_world_size or realized != expected_batch:
        raise ValueError(
            "Runtime topology does not match the configured paper protocol: "
            f"world_size={num_processes} (expected {expected_world_size}), "
            f"effective_batch_size={realized} (expected {expected_batch})."
        )
    return realized


def configured_document_chunk_size(
    checkpointing: CandidateCheckpointingConfig,
    *,
    candidate_count: int,
) -> int:
    """Validate the fixed checkpoint protocol and return its realized chunk width."""

    return checkpointing.realized_document_chunk_size(candidate_count=candidate_count)


def validate_fixed_training_protocol(training: TrainingConfig) -> None:
    """Reject configuration fields whose alternatives are not implemented by this trainer."""

    if training.parameter_dtype is not ParameterDType.FLOAT32:
        raise ValueError("The public trainer currently implements parameter_dtype=float32 only.")
    if training.scheduler is not SchedulerKind.COSINE:
        raise ValueError("The public trainer currently implements scheduler=cosine only.")


def load_resume_progress(
    state_dir: str | Path,
    *,
    run_config_sha256: str,
    student_config_sha256: str,
) -> tuple[int, int]:
    """Validate an epoch checkpoint and return ``(next_epoch, global_step)``."""

    metadata_path = Path(state_dir) / "trainer_state.json"
    if not metadata_path.is_file():
        raise ValueError(f"Resume checkpoint has no trainer_state.json: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Resume metadata must be a JSON object: {metadata_path}")
    expected_hashes = {
        "run_config_sha256": run_config_sha256,
        "student_config_sha256": student_config_sha256,
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            raise ValueError(f"Resume checkpoint {key} does not match the current configuration.")
    completed_epoch = payload.get("completed_epoch")
    global_step = payload.get("global_step")
    if not isinstance(completed_epoch, int) or completed_epoch < 0:
        raise ValueError("Resume completed_epoch must be a non-negative integer.")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("Resume global_step must be a non-negative integer.")
    return completed_epoch + 1, global_step


def prepare_output_directory(output_dir: str | Path, *, resume_from: str | Path | None) -> Path:
    """Create a fresh run directory or validate an in-place resume destination."""

    destination = Path(output_dir).resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Training output path exists but is not a directory: {destination}")
    if resume_from is None:
        if destination.is_dir() and any(destination.iterdir()):
            raise ValueError(
                f"Refusing to mix a new run with existing output files: {destination}. "
                "Choose an empty directory or pass --resume-from."
            )
    else:
        state_dir = Path(resume_from).resolve()
        if not destination.is_dir() or not state_dir.is_relative_to(destination):
            raise ValueError("Resume state must belong to the existing output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tokenize_batch(
    model: ScoreVectorStudent,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    query_texts = [str(value) for value in batch["query_texts"]]
    candidate_rows = [[str(value) for value in row] for row in batch["candidate_texts"]]
    candidate_count = len(candidate_rows[0])
    if any(len(row) != candidate_count for row in candidate_rows):
        raise ValueError("All rows in a training batch must have the same candidate count.")
    query_features = model.backbone.tokenize(query_texts, kind="query")
    flat_documents = [text for row in candidate_rows for text in row]
    document_features = reshape_candidate_features(
        model.backbone.tokenize(flat_documents, kind="document"),
        batch_size=len(candidate_rows),
        candidate_count=candidate_count,
    )
    return move_features(query_features, device), move_features(document_features, device)


@torch.no_grad()
def _validate_candidate_rows(
    model: ScoreVectorStudent,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    document_chunk_size: int,
    loss_kind: DistillationLoss,
    loss_temperature: float | None = None,
) -> dict[str, float | int]:
    """Measure held-out distillation loss, not full-corpus retrieval quality."""

    model.eval()
    losses: list[float] = []
    rows = 0
    for batch in loader:
        query_features, document_features = _tokenize_batch(model, batch, device=device)
        teacher = batch["teacher_scores"].to(device)
        validation_loss, _ = model(
            query_features,
            document_features,
            teacher,
            document_chunk_size=document_chunk_size,
            loss=loss_kind,
            candidate_ids=batch.get("candidate_ids"),
            loss_temperature=loss_temperature,
            # This loop runs unsharded on the main process only (see the
            # module docstring above); calling a distributed collective here
            # would hang waiting on ranks that never reach it.
            gather_across_processes=False,
        )
        losses.append(float(validation_loss.cpu()))
        rows += int(teacher.shape[0])
    return {"validation_loss": float(np.mean(losses)) if losses else 0.0, "rows": rows}


def _prepare_training_run(request: TrainingRequest) -> PreparedTrainingRun:
    """Resolve typed config references and validate both input tables."""

    run = load_executable_run_config(request.run_config)
    student_config = load_config(run.student_config)
    data_config = load_config(run.data_config)
    teacher_configs = tuple(load_config(path) for path in run.teacher_configs)
    teacher_labels = _teacher_labels(teacher_configs)
    if set(run.teacher_weights) != set(teacher_labels):
        raise ValueError(
            f"teacher_weights labels {sorted(run.teacher_weights)} do not match teacher_configs {list(teacher_labels)}."
        )
    validate_fixed_training_protocol(run.training)
    seed = _seed_for_request(request, run.training)
    if seed < 0:
        raise ValueError("Training seed must be non-negative.")

    train_dataset = DistillationDataset.from_parquet(request.train_file)
    validation_dataset = DistillationDataset.from_parquet(request.eval_file)
    candidate_count = _configured_candidate_count(data_config)
    document_chunk_size = configured_document_chunk_size(
        run.training.candidate_checkpointing,
        candidate_count=candidate_count,
    )
    for label, dataset in (("training", train_dataset), ("validation", validation_dataset)):
        widths = {len(row.candidate_texts) for row in dataset.rows}
        if widths != {candidate_count}:
            raise ValueError(
                f"{label.capitalize()} data candidate width {sorted(widths)} "
                f"does not match configured C{candidate_count}."
            )

    return PreparedTrainingRun(
        run=run,
        student_config=student_config,
        data_config=data_config,
        teacher_configs=teacher_configs,
        teacher_labels=teacher_labels,
        seed=seed,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        candidate_count=candidate_count,
        document_chunk_size=document_chunk_size,
    )


def _create_accelerator(training: TrainingConfig) -> Accelerator:
    """Initialize Accelerate from the typed runtime settings."""

    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    mixed_precision = "no" if training.mixed_precision is MixedPrecision.NONE else training.mixed_precision.value
    return Accelerator(
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=DISTRIBUTED_PROCESS_GROUP_TIMEOUT)],
    )


def _cosine_schedule(*, total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """Return the paper's warmup-plus-cosine learning-rate multiplier."""

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return multiplier


def _build_training_components(
    prepared: PreparedTrainingRun,
    accelerator: Accelerator,
) -> TrainingComponents:
    """Construct and distribute the student, optimizer, loaders, and scheduler."""

    training = prepared.run.training
    profile = profile_from_mapping(prepared.student_config.data)
    backbone = SentenceTransformerBackbone(
        profile,
        device=str(accelerator.device),
        torch_dtype=torch.float32,
        lora=_lora_settings(training.lora),
        gradient_checkpointing=training.backbone_gradient_checkpointing,
    )
    student = ScoreVectorStudent(backbone)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    for parameter in trainable:
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    optimizer = AdamW(
        trainable,
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    train_loader: DataLoader[Any] = DataLoader(
        prepared.train_dataset,
        batch_size=training.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_distillation_rows,
    )
    validation_loader: DataLoader[Any] = DataLoader(
        prepared.validation_dataset,
        batch_size=training.eval_batch_size,
        shuffle=False,
        collate_fn=collate_distillation_rows,
    )
    updates_per_epoch = math.ceil(len(train_loader) / training.gradient_accumulation_steps)
    total_steps = max(1, training.epochs * updates_per_epoch)
    warmup_steps = int(total_steps * training.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        _cosine_schedule(total_steps=total_steps, warmup_steps=warmup_steps),
    )
    distributed = accelerator.prepare(student, optimizer, train_loader, scheduler)
    prepared_student, prepared_optimizer, prepared_loader, prepared_scheduler = cast(
        tuple[ScoreVectorStudent, AdamW, DataLoader[Any], LambdaLR],
        distributed,
    )
    return TrainingComponents(
        student=prepared_student,
        optimizer=prepared_optimizer,
        train_loader=prepared_loader,
        validation_loader=validation_loader,
        scheduler=prepared_scheduler,
    )


def _append_metric(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")


def _resume_position(
    request: TrainingRequest,
    prepared: PreparedTrainingRun,
    accelerator: Accelerator,
) -> tuple[int, int]:
    """Restore Accelerate state and return the next epoch and optimizer step."""

    if request.resume_from is None:
        return 1, 0
    start_epoch, global_step = load_resume_progress(
        request.resume_from,
        run_config_sha256=prepared.run.sha256,
        student_config_sha256=prepared.student_config.sha256,
    )
    accelerator.load_state(request.resume_from)
    return start_epoch, global_step


def _log_distributed_stage(
    accelerator: Accelerator,
    *,
    epoch: int,
    global_step: int,
    stage: str,
) -> None:
    """Emit a rank-attributed lifecycle marker around long synchronization points."""

    print(
        "[training-stage] "
        f"rank={accelerator.process_index}/{accelerator.num_processes} "
        f"epoch={epoch} step={global_step} stage={stage}",
        flush=True,
    )


def _validate_and_checkpoint_epoch(
    prepared: PreparedTrainingRun,
    components: TrainingComponents,
    accelerator: Accelerator,
    *,
    output_dir: Path,
    metrics_path: Path,
    epoch: int,
    global_step: int,
) -> None:
    """Run main-only validation, then let every rank enter checkpointing together."""

    _log_distributed_stage(
        accelerator,
        epoch=epoch,
        global_step=global_step,
        stage="training_complete_wait",
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _log_distributed_stage(
            accelerator,
            epoch=epoch,
            global_step=global_step,
            stage="validation_start",
        )
        validation_model = cast(ScoreVectorStudent, accelerator.unwrap_model(components.student))
        validation = _validate_candidate_rows(
            validation_model,
            components.validation_loader,
            device=accelerator.device,
            document_chunk_size=prepared.document_chunk_size,
            loss_kind=prepared.run.training.loss,
            loss_temperature=prepared.run.training.loss_temperature,
        )
        _append_metric(
            metrics_path,
            {"event": "validation", "epoch": epoch, "step": global_step, **validation},
        )
        _log_distributed_stage(
            accelerator,
            epoch=epoch,
            global_step=global_step,
            stage="validation_complete",
        )
    else:
        _log_distributed_stage(
            accelerator,
            epoch=epoch,
            global_step=global_step,
            stage="validation_wait",
        )

    # Do not use save_state itself as the validation rendezvous. All ranks must
    # leave this explicit barrier before any rank starts writing checkpoint
    # state; the process-group timeout above allows main-only validation to run
    # for longer than Accelerate's default 600 seconds.
    accelerator.wait_for_everyone()
    state_dir = output_dir / f"state_epoch_{epoch}"
    _log_distributed_stage(
        accelerator,
        epoch=epoch,
        global_step=global_step,
        stage="checkpoint_start",
    )
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _write_json_atomic(
            state_dir / "trainer_state.json",
            {
                "schema_version": 1,
                "completed_epoch": epoch,
                "global_step": global_step,
                "run_config_sha256": prepared.run.sha256,
                "student_config_sha256": prepared.student_config.sha256,
            },
        )
    accelerator.wait_for_everyone()
    _log_distributed_stage(
        accelerator,
        epoch=epoch,
        global_step=global_step,
        stage="checkpoint_complete",
    )


def _run_training_epochs(
    request: TrainingRequest,
    prepared: PreparedTrainingRun,
    components: TrainingComponents,
    accelerator: Accelerator,
    *,
    output_dir: Path,
) -> tuple[int, float]:
    """Optimize, validate candidate-row loss, and checkpoint each epoch."""

    training = prepared.run.training
    start_epoch, global_step = _resume_position(request, prepared, accelerator)
    metrics_path = output_dir / "metrics.jsonl"
    started = time.time()

    for epoch in range(start_epoch, training.epochs + 1):
        components.student.train()
        for batch in components.train_loader:
            with accelerator.accumulate(components.student):
                unwrapped = cast(ScoreVectorStudent, accelerator.unwrap_model(components.student))
                query_features, document_features = _tokenize_batch(
                    unwrapped,
                    batch,
                    device=accelerator.device,
                )
                teacher = batch["teacher_scores"].to(accelerator.device)
                loss, _ = components.student(
                    query_features,
                    document_features,
                    teacher,
                    document_chunk_size=prepared.document_chunk_size,
                    loss=training.loss,
                    candidate_ids=batch.get("candidate_ids"),
                    loss_temperature=training.loss_temperature,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(components.student.parameters(), training.max_gradient_norm)
                components.optimizer.step()
                components.scheduler.step()
                components.optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    _append_metric(
                        metrics_path,
                        {
                            "event": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(loss.detach().cpu()),
                            "learning_rate": float(components.scheduler.get_last_lr()[0]),
                        },
                    )
        _validate_and_checkpoint_epoch(
            prepared,
            components,
            accelerator,
            output_dir=output_dir,
            metrics_path=metrics_path,
            epoch=epoch,
            global_step=global_step,
        )
    return global_step, time.time() - started


def _write_final_artifacts(
    request: TrainingRequest,
    prepared: PreparedTrainingRun,
    components: TrainingComponents,
    accelerator: Accelerator,
    summary: dict[str, Any],
    *,
    output_dir: Path,
    global_step: int,
    elapsed_seconds: float,
) -> None:
    """Persist the adapter, a compact run record, and the final summary."""

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_model = cast(ScoreVectorStudent, accelerator.unwrap_model(components.student))
        model_dir = output_dir / "model_final"
        final_model.backbone.model.save_pretrained(model_dir)
        checkpoint = checkpoint_record(model_dir)
        write_run_record(
            model_dir / "run.json",
            {
                "schema_version": 1,
                "kind": "training",
                "run": prepared.run.name,
                "seed": prepared.seed,
                "git_commit": git_commit(Path(__file__).resolve().parents[3]),
                "config": prepared.run.resolved.data,
                "models": {
                    "student": prepared.student_config.data,
                    "teachers": [config.data for config in prepared.teacher_configs],
                },
                "dataset": prepared.data_config.data,
                "inputs": {
                    "train": {"file": request.train_file.name, "sha256": sha256_file(request.train_file)},
                    "eval": {"file": request.eval_file.name, "sha256": sha256_file(request.eval_file)},
                },
                "checkpoint": checkpoint,
                "runtime": runtime_summary(
                    attention_implementation=final_model.backbone.effective_attention_implementation()
                ),
                "result": {
                    "global_steps": global_step,
                    "elapsed_seconds": elapsed_seconds,
                    "world_size": accelerator.num_processes,
                },
            },
        )
        summary.update(
            {
                "completed": True,
                "global_steps": global_step,
                "elapsed_seconds": elapsed_seconds,
                "world_size": accelerator.num_processes,
                "document_chunk_size": prepared.document_chunk_size,
                "checkpoint": "model_final",
            }
        )
        _write_json_atomic(output_dir / "summary.json", summary)
    # Keep every rank alive until rank zero has durably published the final
    # model and metadata. No rank should report a clean training exit while the
    # main process is still writing release artifacts.
    accelerator.wait_for_everyone()


def train(request: TrainingRequest) -> dict[str, Any]:
    """Train one configured seed and write model, state, metrics, and a run record."""

    prepared = _prepare_training_run(request)
    summary = prepared.summary()
    if request.dry_run:
        return summary

    set_seed(prepared.seed)
    accelerator = _create_accelerator(prepared.run.training)
    if accelerator.is_main_process:
        prepare_output_directory(request.output_dir, resume_from=request.resume_from)
    accelerator.wait_for_everyone()
    output_dir = request.output_dir.resolve()
    summary["effective_batch_size"] = validate_runtime_batch(
        prepared.run.training,
        num_processes=accelerator.num_processes,
    )
    components = _build_training_components(prepared, accelerator)
    global_step, elapsed_seconds = _run_training_epochs(
        request,
        prepared,
        components,
        accelerator,
        output_dir=output_dir,
    )
    _write_final_artifacts(
        request,
        prepared,
        components,
        accelerator,
        summary,
        output_dir=output_dir,
        global_step=global_step,
        elapsed_seconds=elapsed_seconds,
    )
    return summary
