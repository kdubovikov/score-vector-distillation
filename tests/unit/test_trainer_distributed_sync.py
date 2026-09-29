from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import accelerate
import pytest
import torch

from score_vector_distillation.config import DistillationLoss, MixedPrecision
from score_vector_distillation.training import trainer


class _FakeAccelerator:
    def __init__(
        self,
        *,
        process_index: int,
        num_processes: int,
        barrier: threading.Barrier,
        events: list[tuple[str, int]],
        event_lock: threading.Lock,
    ) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.is_main_process = process_index == 0
        self.device = torch.device("cpu")
        self._barrier = barrier
        self._events = events
        self._event_lock = event_lock

    def _record(self, event: str) -> None:
        with self._event_lock:
            self._events.append((event, self.process_index))

    def wait_for_everyone(self) -> None:
        self._record("barrier_enter")
        self._barrier.wait(timeout=5)
        self._record("barrier_exit")

    def unwrap_model(self, model: object) -> object:
        return model

    def save_state(self, state_dir: Path) -> None:
        self._record("save_state")
        state_dir.mkdir(parents=True, exist_ok=True)


def _prepared() -> Any:
    return SimpleNamespace(
        document_chunk_size=4,
        run=SimpleNamespace(
            sha256="run-sha",
            training=SimpleNamespace(loss=DistillationLoss.CENTERED_MSE, loss_temperature=None),
        ),
        student_config=SimpleNamespace(sha256="student-sha"),
    )


def _components() -> Any:
    return SimpleNamespace(student=object(), validation_loader=object())


def test_accelerator_extends_process_group_timeout_beyond_long_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeInitProcessGroupKwargs:
        def __init__(self, *, timeout: object) -> None:
            captured["timeout"] = timeout

    class FakeAccelerator:
        def __init__(self, **kwargs: object) -> None:
            captured["accelerator_kwargs"] = kwargs

    monkeypatch.setattr(accelerate, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(accelerate.utils, "InitProcessGroupKwargs", FakeInitProcessGroupKwargs)
    training = SimpleNamespace(
        gradient_accumulation_steps=2,
        mixed_precision=MixedPrecision.NONE,
    )

    result = trainer._create_accelerator(training)

    assert isinstance(result, FakeAccelerator)
    assert captured["timeout"] == trainer.DISTRIBUTED_PROCESS_GROUP_TIMEOUT
    assert captured["timeout"].total_seconds() == 3600
    assert len(captured["accelerator_kwargs"]["kwargs_handlers"]) == 1


def test_non_main_rank_cannot_checkpoint_while_main_validation_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    event_lock = threading.Lock()
    events: list[tuple[str, int]] = []
    validation_started = threading.Event()
    release_validation = threading.Event()
    exceptions: list[BaseException] = []

    def validate(*args: object, **kwargs: object) -> dict[str, float | int]:
        with event_lock:
            events.append(("validation_start", 0))
        validation_started.set()
        assert release_validation.wait(timeout=5)
        with event_lock:
            events.append(("validation_end", 0))
        return {"validation_loss": 0.25, "rows": 32}

    def append_metric(path: Path, payload: object) -> None:
        with event_lock:
            events.append(("metric", 0))
        assert payload == {
            "event": "validation",
            "epoch": 1,
            "step": 557,
            "validation_loss": 0.25,
            "rows": 32,
        }

    def write_metadata(path: Path, payload: object) -> None:
        with event_lock:
            events.append(("metadata", 0))

    monkeypatch.setattr(trainer, "_validate_candidate_rows", validate)
    monkeypatch.setattr(trainer, "_append_metric", append_metric)
    monkeypatch.setattr(trainer, "_write_json_atomic", write_metadata)

    def run_rank(process_index: int) -> None:
        accelerator = _FakeAccelerator(
            process_index=process_index,
            num_processes=2,
            barrier=barrier,
            events=events,
            event_lock=event_lock,
        )
        try:
            trainer._validate_and_checkpoint_epoch(
                _prepared(),
                _components(),
                accelerator,
                output_dir=tmp_path,
                metrics_path=tmp_path / "metrics.jsonl",
                epoch=1,
                global_step=557,
            )
        except BaseException as error:  # pragma: no cover - assertion below reports thread failures
            exceptions.append(error)

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    assert validation_started.wait(timeout=5)
    with event_lock:
        assert not any(event == "save_state" for event, _ in events)
    release_validation.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not exceptions
    assert all(not thread.is_alive() for thread in threads)
    with event_lock:
        event_names = [event for event, _ in events]
        validation_end_index = event_names.index("validation_end")
        save_indices = [index for index, event in enumerate(event_names) if event == "save_state"]
        assert len(save_indices) == 2
        assert all(index > validation_end_index for index in save_indices)
        assert event_names.count("validation_start") == 1
        assert event_names.count("metric") == 1
        assert event_names.count("metadata") == 1


def test_single_process_validation_and_checkpoint_semantics_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_lock = threading.Lock()
    events: list[tuple[str, int]] = []
    metric_payloads: list[object] = []
    metadata_payloads: list[object] = []
    accelerator = _FakeAccelerator(
        process_index=0,
        num_processes=1,
        barrier=threading.Barrier(1),
        events=events,
        event_lock=event_lock,
    )
    monkeypatch.setattr(
        trainer,
        "_validate_candidate_rows",
        lambda *args, **kwargs: {"validation_loss": 0.5, "rows": 8},
    )
    monkeypatch.setattr(trainer, "_append_metric", lambda path, payload: metric_payloads.append(payload))
    monkeypatch.setattr(trainer, "_write_json_atomic", lambda path, payload: metadata_payloads.append(payload))

    trainer._validate_and_checkpoint_epoch(
        _prepared(),
        _components(),
        accelerator,
        output_dir=tmp_path,
        metrics_path=tmp_path / "metrics.jsonl",
        epoch=1,
        global_step=7,
    )

    assert metric_payloads == [
        {
            "event": "validation",
            "epoch": 1,
            "step": 7,
            "validation_loss": 0.5,
            "rows": 8,
        }
    ]
    assert metadata_payloads == [
        {
            "schema_version": 1,
            "completed_epoch": 1,
            "global_step": 7,
            "run_config_sha256": "run-sha",
            "student_config_sha256": "student-sha",
        }
    ]
    assert [event for event, _ in events].count("save_state") == 1


def test_non_main_rank_waits_for_main_final_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    event_lock = threading.Lock()
    events: list[tuple[str, int]] = []
    save_started = threading.Event()
    release_save = threading.Event()
    rank_completed = [threading.Event(), threading.Event()]
    exceptions: list[BaseException] = []

    class BlockingModelWriter:
        def save_pretrained(self, path: Path) -> None:
            save_started.set()
            assert release_save.wait(timeout=5)

    backbone = SimpleNamespace(
        model=BlockingModelWriter(),
        effective_attention_implementation=lambda: "sdpa",
    )
    components = SimpleNamespace(student=SimpleNamespace(backbone=backbone))
    prepared = SimpleNamespace(
        seed=0,
        document_chunk_size=4,
        run=SimpleNamespace(name="run", resolved=SimpleNamespace(data={})),
        student_config=SimpleNamespace(data={}),
        teacher_configs=(),
        data_config=SimpleNamespace(data={}),
    )
    request = SimpleNamespace(
        train_file=tmp_path / "train.jsonl",
        eval_file=tmp_path / "eval.jsonl",
    )
    monkeypatch.setattr(trainer, "checkpoint_record", lambda path: {})
    monkeypatch.setattr(trainer, "write_run_record", lambda path, payload: None)
    monkeypatch.setattr(trainer, "git_commit", lambda path: "commit")
    monkeypatch.setattr(trainer, "runtime_summary", lambda **kwargs: {})
    monkeypatch.setattr(trainer, "sha256_file", lambda path: "sha256")
    monkeypatch.setattr(trainer, "_write_json_atomic", lambda path, payload: None)

    def run_rank(process_index: int) -> None:
        accelerator = _FakeAccelerator(
            process_index=process_index,
            num_processes=2,
            barrier=barrier,
            events=events,
            event_lock=event_lock,
        )
        try:
            trainer._write_final_artifacts(
                request,
                prepared,
                components,
                accelerator,
                {},
                output_dir=tmp_path,
                global_step=557,
                elapsed_seconds=1.0,
            )
            rank_completed[process_index].set()
        except BaseException as error:  # pragma: no cover - assertion below reports thread failures
            exceptions.append(error)

    threads = [threading.Thread(target=run_rank, args=(rank,), daemon=True) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    assert save_started.wait(timeout=5)
    assert not rank_completed[1].is_set()
    release_save.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not exceptions
    assert all(not thread.is_alive() for thread in threads)
    assert all(completed.is_set() for completed in rank_completed)
