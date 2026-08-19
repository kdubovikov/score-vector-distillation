import shutil
from pathlib import Path

import pytest

from score_vector_distillation.data.pipeline import PanelPreparationError, prepare_panel_rows
from score_vector_distillation.data.registry import DatasetRegistry
from score_vector_distillation.data.schema import (
    CandidateConstruction,
    DatasetSpec,
    DatasetSplit,
    PanelConfig,
    SnapshotSpec,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "retrieval" / "toy"


def _registry(*, expected_train_rows: int = 2) -> DatasetRegistry:
    return DatasetRegistry(
        PanelConfig(
            schema_version=1,
            provider="Public toy snapshots",
            citation_title="Toy retrieval fixture",
            citation_url="https://example.invalid/toy",
            datasets=(
                DatasetSpec(
                    name="Toy",
                    snapshot=SnapshotSpec(
                        url="https://example.invalid/toy.zip",
                        archive_name="toy.zip",
                        sha256="0" * 64,
                    ),
                ),
            ),
            train=(DatasetSplit(dataset="Toy", split="train", expected_rows=expected_train_rows),),
            row_eval=(DatasetSplit(dataset="Toy", split="test", expected_rows=2),),
            full_corpus_eval=(DatasetSplit(dataset="Toy", split="test"),),
            candidates=CandidateConstruction(
                candidate_count=4,
                positive_index=0,
                seed=42,
                tfidf_max_features=100,
                tfidf_top_pool=4,
                query_batch_size=1,
                require_disjoint_query_ids=True,
            ),
        )
    )


def _materialize_fixture(cache_dir: Path) -> None:
    shutil.copytree(FIXTURE_ROOT, cache_dir / "datasets" / "Toy")


def test_prepare_panel_rows_commits_exact_rows_without_sidecar(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    output = tmp_path / "prepared"
    _materialize_fixture(cache)

    result = prepare_panel_rows(_registry(), cache_dir=cache, output_dir=output)

    assert result.train_rows == 2
    assert result.eval_rows == 2
    assert result.candidate_count == 4
    assert result.train_path.is_file()
    assert result.eval_path.is_file()
    assert not (output / "manifest.json").exists()
    assert not list(tmp_path.glob(".prepared.*.staging"))


def test_prepare_panel_rows_fails_before_publication_on_count_mismatch(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    output = tmp_path / "prepared"
    _materialize_fixture(cache)

    with pytest.raises(PanelPreparationError, match="expected 3, built 2"):
        prepare_panel_rows(_registry(expected_train_rows=3), cache_dir=cache, output_dir=output)

    assert not output.exists()


def test_prepare_panel_rows_never_fetches_missing_materialization(tmp_path: Path) -> None:
    output = tmp_path / "prepared"

    with pytest.raises(PanelPreparationError, match="prepare the pinned snapshot explicitly"):
        prepare_panel_rows(_registry(), cache_dir=tmp_path / "empty-cache", output_dir=output)

    assert not output.exists()
