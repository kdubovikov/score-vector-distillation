from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from score_vector_distillation.cli import (
    _handle_config,
    _resolve_evaluation_protocol,
    _select_stride_shard,
    build_parser,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_cli_import_does_not_require_training_extra() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import score_vector_distillation.cli; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cli_reports_expected_config_failure_without_traceback(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("config_kind: executable_run\nschema_version: 1\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "score_vector_distillation.cli", "config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("error: ")
    assert "missing required fields" in result.stderr
    assert "Traceback" not in result.stderr


def _paper_evaluate_args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "evaluate",
            "--run-config",
            str(REPOSITORY_ROOT / "configs/paper/runs/qwen_main.yaml"),
            "--cache-dir",
            "cache",
            "--dataset-name",
            "SciFact",
            "--model-config",
            "released-model.yaml",
            "--output-dir",
            "evaluation",
            *extra,
        ]
    )


def test_evaluate_uses_typed_protocol_from_run_config() -> None:
    protocol = _resolve_evaluation_protocol(_paper_evaluate_args())

    assert protocol.data_config == (REPOSITORY_ROOT / "configs/paper/data/eight_task_panel.yaml").resolve()
    assert protocol.primary_metric == "NDCG@10"
    assert protocol.metric_k_values == (10, 100)
    assert protocol.top_k == 100
    assert protocol.aggregation == "dataset_macro"
    assert protocol.run_config is not None


def test_config_command_validates_the_typed_executable_view() -> None:
    args = build_parser().parse_args(["config", str(REPOSITORY_ROOT / "configs/paper/runs/qwen_main.yaml")])

    result = _handle_config(args)

    assert result["config_kind"] == "executable_run"
    assert len(result["sha256"]) == 64


def test_evaluate_rejects_cli_override_that_conflicts_with_run_config() -> None:
    with pytest.raises(ValueError, match="--top-k conflicts"):
        _resolve_evaluation_protocol(_paper_evaluate_args("--top-k", "10"))


def test_evaluate_parses_required_resolved_attention() -> None:
    args = _paper_evaluate_args("--require-resolved-attention", "eager")

    assert args.require_resolved_attention == "eager"


def test_teacher_score_parses_stride_shard_and_attention_gate() -> None:
    args = build_parser().parse_args(
        [
            "teacher-score",
            "--rows",
            "rows.jsonl",
            "--model-config",
            "teacher.yaml",
            "--output",
            "scores.parquet",
            "--shard-index",
            "2",
            "--shard-count",
            "8",
            "--require-resolved-attention",
            "eager",
        ]
    )

    assert args.shard_index == 2
    assert args.shard_count == 8
    assert args.require_resolved_attention == "eager"


def test_stride_shards_are_disjoint_and_reconstruct_input() -> None:
    rows = list(range(19))
    shards = [_select_stride_shard(rows, shard_index=index, shard_count=8) for index in range(8)]

    assert sorted(item for shard in shards for item in shard) == rows
    assert sum(len(shard) for shard in shards) == len(rows)


@pytest.mark.parametrize(
    ("shard_index", "shard_count", "message"),
    [
        (0, None, "provided together"),
        (None, 8, "provided together"),
        (0, 0, "must be positive"),
        (8, 8, "must be in"),
        (1, 8, "contains no rows"),
    ],
)
def test_stride_shard_rejects_invalid_selection(
    shard_index: int | None,
    shard_count: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _select_stride_shard(["only-row"], shard_index=shard_index, shard_count=shard_count)
