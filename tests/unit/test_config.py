from pathlib import Path

import pytest

from score_vector_distillation.config import (
    ConfigError,
    DistillationLoss,
    EvaluationAggregation,
    ExecutableRunConfig,
    RetrievalSimilarity,
    content_hash,
    load_executable_run_config,
    resolve_run_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_content_hash_is_order_independent() -> None:
    assert content_hash({"a": 1, "b": [2, 3]}) == content_hash({"b": [2, 3], "a": 1})


def test_run_config_inheritance_deep_merges_and_hashes_executable_config(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("name: base\ntraining:\n  loss: centered_mse\n  epochs: 1\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("base_run: base.yaml\nname: child\ntraining:\n  loss: unit_centered_mse\n", encoding="utf-8")

    resolved = resolve_run_config(child)

    assert resolved.path == child.resolve()
    assert resolved.data == {
        "name": "child",
        "training": {"loss": "unit_centered_mse", "epochs": 1},
    }
    assert resolved.sha256 == content_hash(resolved.data)


def test_paper_run_loads_as_a_strict_typed_executable_config() -> None:
    path = REPOSITORY_ROOT / "configs/paper/runs/qwen_main.yaml"
    resolved = resolve_run_config(path)
    run = load_executable_run_config(path)

    assert run.resolved.data == resolved.data
    assert run.sha256 == resolved.sha256
    assert run.training.loss is DistillationLoss.CENTERED_MSE
    assert run.training.lora.rank == 256
    assert run.training.candidate_checkpointing.document_chunk_size == 4
    assert run.evaluation.primary_metric.label == "NDCG@10"
    assert run.evaluation.metric_k_values == (10, 100)
    assert run.evaluation.similarity is RetrievalSimilarity.NORMALIZED_COSINE
    assert run.evaluation.aggregation is EvaluationAggregation.DATASET_MACRO
    assert run.evaluation.top_k == 100


@pytest.mark.parametrize(
    "name",
    [
        "e5_main.yaml",
        "e5_nv_only.yaml",
        "e5_qwen8_only.yaml",
        "qwen_control_centered.yaml",
        "qwen_control_label_only.yaml",
        "qwen_control_margin.yaml",
        "qwen_control_margin_only.yaml",
        "qwen_control_rank_weighted.yaml",
        "qwen_control_unit_centered.yaml",
        "qwen_main.yaml",
        "qwen_nv_only.yaml",
        "qwen_qwen8_only.yaml",
    ],
)
def test_every_concrete_paper_run_parses_after_inheritance(name: str) -> None:
    run = load_executable_run_config(REPOSITORY_ROOT / "configs/paper/runs" / name)

    assert run.name
    assert run.training.seeds


def test_loss_controls_is_explicitly_a_non_executable_sweep_index() -> None:
    path = REPOSITORY_ROOT / "configs/paper/runs/loss_controls.yaml"

    with pytest.raises(ConfigError, match="sweep_index, not an executable run"):
        load_executable_run_config(path)


def test_executable_run_rejects_unknown_top_level_field() -> None:
    path = REPOSITORY_ROOT / "configs/paper/runs/qwen_main.yaml"
    resolved = resolve_run_config(path)
    resolved.data["silently_ignored"] = True

    with pytest.raises(ConfigError, match=r"unsupported fields.*silently_ignored"):
        ExecutableRunConfig.from_resolved(resolved)


def test_evaluation_contract_rejects_unreported_cutoff() -> None:
    path = REPOSITORY_ROOT / "configs/paper/runs/qwen_main.yaml"
    resolved = resolve_run_config(path)
    evaluation = resolved.data["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["primary_metric"] = "NDCG@20"

    with pytest.raises(ConfigError, match="cutoff must be included"):
        ExecutableRunConfig.from_resolved(resolved)
