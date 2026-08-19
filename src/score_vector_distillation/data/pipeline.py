"""End-to-end preparation of deterministic candidate rows."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from score_vector_distillation.data.candidates import (
    assert_query_disjoint,
    build_candidate_rows,
    write_candidate_rows_jsonl,
)
from score_vector_distillation.data.loaders import load_retrieval_dataset
from score_vector_distillation.data.registry import DatasetRegistry
from score_vector_distillation.data.schema import CandidateRow, DatasetSplit


class PanelPreparationError(RuntimeError):
    """Raised when the configured panel rows cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class PreparedPanel:
    """Paths and counts for prepared candidate rows."""

    output_dir: Path
    train_path: Path
    eval_path: Path
    train_rows: int
    eval_rows: int
    candidate_count: int


def _require_snapshot_files(cache_dir: Path, splits: Sequence[DatasetSplit]) -> None:
    for split in splits:
        root = cache_dir / "datasets" / split.dataset
        required = {
            f"{split.dataset}/corpus.jsonl": root / "corpus.jsonl",
            f"{split.dataset}/queries.jsonl": root / "queries.jsonl",
            f"{split.dataset}/qrels/{split.split}.tsv": root / "qrels" / f"{split.split}.tsv",
        }
        for logical_path, physical_path in required.items():
            if not physical_path.is_file():
                raise PanelPreparationError(
                    f"required materialized snapshot file is missing: {logical_path}; "
                    "prepare the pinned snapshot explicitly before building rows"
                )


def _build_partition(
    registry: DatasetRegistry,
    cache_dir: Path,
    splits: Sequence[DatasetSplit],
    *,
    logical_split: str,
) -> tuple[CandidateRow, ...]:
    settings = registry.panel.candidates
    rows: list[CandidateRow] = []
    for split in splits:
        if split.expected_rows is None:
            raise PanelPreparationError(
                f"{logical_split} source {split.dataset}:{split.split} has no expected_rows assertion"
            )
        dataset = load_retrieval_dataset(
            cache_dir / "datasets" / split.dataset,
            name=split.dataset,
            split=split.split,
        )
        source_rows = build_candidate_rows(
            dataset,
            logical_split=logical_split,
            candidate_count=settings.candidate_count,
            seed=settings.seed,
            max_rows=split.max_rows,
            tfidf_max_features=settings.tfidf_max_features,
            tfidf_top_pool=settings.tfidf_top_pool,
            query_batch_size=settings.query_batch_size,
        )
        if len(source_rows) != split.expected_rows:
            raise PanelPreparationError(
                f"row-count mismatch for {split.dataset}:{split.split}: "
                f"expected {split.expected_rows}, built {len(source_rows)}"
            )
        rows.extend(source_rows)
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
        raise PanelPreparationError(f"duplicate {logical_split} row IDs: {duplicates[:5]}")
    return tuple(rows)


def _publish_staged_panel(staging: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    if not output_dir.is_dir():
        raise PanelPreparationError(f"output path exists but is not a directory: {output_dir}")
    for filename in ("train.candidates.jsonl", "eval.candidates.jsonl"):
        os.replace(staging / filename, output_dir / filename)
    (output_dir / "manifest.json").unlink(missing_ok=True)
    staging.rmdir()


def prepare_panel_rows(
    registry: DatasetRegistry,
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    allow_unresolved: bool = False,
) -> PreparedPanel:
    """Build train/eval candidate rows from already materialized snapshots."""

    registry.validate(require_resolved_pins=not allow_unresolved)
    settings = registry.panel.candidates
    if not settings.require_disjoint_query_ids:
        raise PanelPreparationError("candidate preparation requires require_disjoint_query_ids: true")

    cache = Path(cache_dir)
    destination = Path(output_dir)
    _require_snapshot_files(cache, (*registry.panel.train, *registry.panel.row_eval))
    train_rows = _build_partition(registry, cache, registry.panel.train, logical_split="train")
    eval_rows = _build_partition(registry, cache, registry.panel.row_eval, logical_split="eval")
    assert_query_disjoint(train_rows, eval_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent))
    try:
        write_candidate_rows_jsonl(train_rows, staging / "train.candidates.jsonl")
        write_candidate_rows_jsonl(eval_rows, staging / "eval.candidates.jsonl")
        _publish_staged_panel(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PreparedPanel(
        output_dir=destination,
        train_path=destination / "train.candidates.jsonl",
        eval_path=destination / "eval.candidates.jsonl",
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        candidate_count=settings.candidate_count,
    )
