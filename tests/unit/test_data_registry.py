from pathlib import Path

from score_vector_distillation.data.registry import load_registry

REPOSITORY_ROOT = Path(__file__).parents[2]
PAPER_CONFIG = REPOSITORY_ROOT / "configs" / "paper" / "data" / "eight_task_panel.yaml"


def test_paper_registry_declares_exact_panel_and_strict_row_splits() -> None:
    registry = load_registry(PAPER_CONFIG)

    assert [dataset.name for dataset in registry.panel.datasets] == [
        "SciFact",
        "NFCorpus",
        "FiQA",
        "ArguAna",
        "SciDocs",
        "TREC-COVID",
        "Webis-Touche2020",
        "Quora",
    ]
    assert [(item.dataset, item.split) for item in registry.panel.train] == [
        ("SciFact", "train"),
        ("NFCorpus", "train"),
        ("FiQA", "train"),
    ]
    assert [(item.dataset, item.split, item.max_rows) for item in registry.panel.row_eval] == [
        ("SciFact", "test", 200),
        ("NFCorpus", "dev", 324),
        ("FiQA", "dev", 500),
    ]
    assert sum(item.expected_rows or 0 for item in registry.panel.train) == 8_899
    assert sum(item.expected_rows or 0 for item in registry.panel.row_eval) == 1_024
    assert registry.panel.candidates.positive_index == 0
    assert registry.panel.candidates.require_disjoint_query_ids
    assert registry.panel.provider == "BEIR public dataset snapshots"
    assert registry.dataset("ArguAna").expected_missing_qrel_documents == 5
    assert all(
        dataset.expected_missing_qrel_documents == 0 for dataset in registry.panel.datasets if dataset.name != "ArguAna"
    )


def test_paper_snapshot_pins_are_resolved_and_include_upstream_md5() -> None:
    registry = load_registry(PAPER_CONFIG, require_resolved_pins=True)
    unresolved = [issue for issue in registry.validation_issues() if issue.code == "unresolved_pin"]

    assert not unresolved
    assert all(dataset.snapshot.sha256 is not None for dataset in registry.panel.datasets)
    assert all(dataset.snapshot.upstream_md5 is not None for dataset in registry.panel.datasets)
