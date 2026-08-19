"""Retrieval snapshot loading and candidate-row construction."""

from score_vector_distillation.data.candidates import (
    assert_query_disjoint,
    build_candidate_rows,
    read_candidate_rows_jsonl,
    write_candidate_rows_jsonl,
)
from score_vector_distillation.data.fetch import fetch_file, fetch_snapshot
from score_vector_distillation.data.loaders import load_retrieval_dataset
from score_vector_distillation.data.pipeline import PanelPreparationError, PreparedPanel, prepare_panel_rows
from score_vector_distillation.data.registry import DatasetRegistry, load_registry
from score_vector_distillation.data.schema import (
    CandidateConstruction,
    CandidateRow,
    DatasetSpec,
    DatasetSplit,
    PanelConfig,
    RetrievalDataset,
    SnapshotSpec,
)

__all__ = [
    "CandidateConstruction",
    "CandidateRow",
    "DatasetRegistry",
    "DatasetSpec",
    "DatasetSplit",
    "PanelConfig",
    "PanelPreparationError",
    "PreparedPanel",
    "RetrievalDataset",
    "SnapshotSpec",
    "assert_query_disjoint",
    "build_candidate_rows",
    "fetch_file",
    "fetch_snapshot",
    "load_registry",
    "load_retrieval_dataset",
    "prepare_panel_rows",
    "read_candidate_rows_jsonl",
    "write_candidate_rows_jsonl",
]
