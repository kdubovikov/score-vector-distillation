from __future__ import annotations

import json
from pathlib import Path

import pytest

from score_vector_distillation.training.trainer import load_resume_progress, prepare_output_directory


def test_resume_progress_starts_after_completed_epoch(tmp_path: Path) -> None:
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "completed_epoch": 1,
                "global_step": 557,
                "run_config_sha256": "run-hash",
                "student_config_sha256": "student-hash",
            }
        ),
        encoding="utf-8",
    )

    assert load_resume_progress(
        tmp_path,
        run_config_sha256="run-hash",
        student_config_sha256="student-hash",
    ) == (2, 557)


def test_resume_progress_rejects_different_config(tmp_path: Path) -> None:
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "completed_epoch": 1,
                "global_step": 557,
                "run_config_sha256": "old-run",
                "student_config_sha256": "student-hash",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_config_sha256"):
        load_resume_progress(
            tmp_path,
            run_config_sha256="new-run",
            student_config_sha256="student-hash",
        )


def test_new_run_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "metrics.jsonl").write_text("old metrics\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to mix"):
        prepare_output_directory(output_dir, resume_from=None)


def test_resume_state_must_belong_to_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    outside_state = tmp_path / "other" / "state_epoch_1"
    outside_state.mkdir(parents=True)

    with pytest.raises(ValueError, match="must belong"):
        prepare_output_directory(output_dir, resume_from=outside_state)

    inside_state = output_dir / "state_epoch_1"
    inside_state.mkdir()
    assert prepare_output_directory(output_dir, resume_from=inside_state) == output_dir.resolve()
