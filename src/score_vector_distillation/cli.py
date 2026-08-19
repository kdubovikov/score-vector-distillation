"""Command-line interface for validation and reproduction workflows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy import typing as npt

from score_vector_distillation.config import (
    load_config,
    load_executable_run_config,
    resolve_run_config,
)


def _assignments(
    values: Sequence[str],
    *,
    value_parser: Callable[[str], Any] = str,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=VALUE, got {value!r}.")
        label, raw_item = value.split("=", 1)
        if not label.strip() or not raw_item.strip():
            raise ValueError(f"Expected non-empty LABEL=VALUE, got {value!r}.")
        if label in parsed:
            raise ValueError(f"Duplicate assignment for {label!r}.")
        parsed[label] = value_parser(raw_item)
    return parsed


def _select_stride_shard(
    rows: Sequence[Any],
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> list[Any]:
    """Select one deterministic stride shard while rejecting ambiguous inputs."""

    if (shard_index is None) != (shard_count is None):
        raise ValueError("--shard-index and --shard-count must be provided together.")
    if shard_count is None:
        selected = list(rows)
    else:
        if shard_count <= 0:
            raise ValueError("--shard-count must be positive.")
        if shard_index is None or shard_index < 0 or shard_index >= shard_count:
            raise ValueError("--shard-index must be in [0, --shard-count).")
        selected = list(rows[shard_index::shard_count])
    if not selected:
        raise ValueError("The selected teacher-score shard contains no rows.")
    return selected


def _add_config_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("config", help="Validate a YAML configuration.")
    parser.add_argument("path", type=Path)


def _add_smoke_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("smoke", help="Run a synthetic CPU training smoke test.")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke/tiny.yaml"))


def _add_train_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="Train one configured student seed.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")


def _add_data_fetch_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("data-fetch", help="Fetch checksum-pinned public snapshots.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--allow-unresolved", action="store_true", help="Exploratory only; never exact reproduction.")


def _add_data_prepare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("data-prepare", help="Build strict candidate rows from materialized snapshots.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-unresolved", action="store_true", help="Exploratory only; never exact reproduction.")


def _add_teacher_score_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("teacher-score", help="Materialize one teacher's candidate-row scores.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--document-batch-size", type=int)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Zero-based stride shard index. Must be used with --shard-count.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Number of deterministic stride shards. Must be used with --shard-index.",
    )
    parser.add_argument(
        "--require-resolved-attention",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Fail unless the loaded Transformers backbone resolves to this attention implementation.",
    )


def _add_teacher_fuse_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("teacher-fuse", help="Fuse teacher scores into a training Parquet.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--teacher", action="append", required=True, help="Repeat LABEL=score-table.parquet.")
    parser.add_argument("--output", type=Path, required=True)


def _add_evaluate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("evaluate", help="Run deterministic full-corpus retrieval evaluation.")
    parser.add_argument(
        "--run-config",
        type=Path,
        help="Use the typed retrieval protocol and dataset reference from an executable run config.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        help="Dataset config for an exploratory evaluation without --run-config.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional run-local PEFT adapter directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--query-instruction")
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--document-batch-size", type=int, default=128)
    parser.add_argument("--document-chunk-size", type=int, default=4_096)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--metric-k", action="append", type=int, dest="metric_k_values")
    parser.add_argument(
        "--require-resolved-attention",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Fail unless the loaded Transformers backbone resolves to this attention implementation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="svd", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_config_parser(subparsers)
    _add_smoke_parser(subparsers)
    _add_data_fetch_parser(subparsers)
    _add_data_prepare_parser(subparsers)
    _add_teacher_score_parser(subparsers)
    _add_teacher_fuse_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_train_parser(subparsers)
    return parser


def _handle_config(args: argparse.Namespace) -> dict[str, Any]:
    """Validate one configuration graph."""

    config = resolve_run_config(args.path)
    config_kind = config.data.get("config_kind")
    if config_kind == "executable_run":
        config = load_executable_run_config(args.path).resolved
    return {
        "path": str(config.path),
        "sha256": config.sha256,
        "config_kind": config_kind,
    }


def _handle_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the synthetic training smoke test."""

    from score_vector_distillation.smoke import run_smoke

    return run_smoke(args.config)


def _handle_data_fetch(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch and verify the selected public dataset snapshots."""

    from score_vector_distillation.data.fetch import fetch_snapshot
    from score_vector_distillation.data.registry import load_registry

    registry = load_registry(args.config, require_resolved_pins=not args.allow_unresolved)
    names = list(args.dataset) if args.dataset else [spec.name for spec in registry.panel.datasets]
    unknown = sorted(set(names).difference(spec.name for spec in registry.panel.datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    roots = {
        name: str(
            fetch_snapshot(
                registry.dataset(name),
                args.cache_dir,
                allow_unresolved=bool(args.allow_unresolved),
            )
        )
        for name in names
    }
    return {"datasets": roots, "verified": not bool(args.allow_unresolved)}


def _handle_data_prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Build deterministic candidate rows for the configured panel."""

    from score_vector_distillation.data.pipeline import prepare_panel_rows
    from score_vector_distillation.data.registry import load_registry

    registry = load_registry(args.config, require_resolved_pins=not args.allow_unresolved)
    prepared = prepare_panel_rows(
        registry,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        allow_unresolved=bool(args.allow_unresolved),
    )
    return {
        "train_path": str(prepared.train_path),
        "eval_path": str(prepared.eval_path),
        "train_rows": prepared.train_rows,
        "eval_rows": prepared.eval_rows,
        "candidate_count": prepared.candidate_count,
    }


def _handle_teacher_score(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize one teacher's row-aligned score vectors."""

    import torch

    from score_vector_distillation.data.candidates import read_candidate_rows_jsonl
    from score_vector_distillation.models.profiles import profile_from_mapping
    from score_vector_distillation.models.sentence_transformer import SentenceTransformerBackbone
    from score_vector_distillation.teachers.scoring import ScoringRow, score_candidate_rows, write_score_table

    model_config = load_config(args.model_config)
    profile = profile_from_mapping(model_config.data)
    raw_scoring = model_config.data.get("teacher_scoring", {})
    if not isinstance(raw_scoring, dict):
        raise ValueError("Model config teacher_scoring must be a mapping.")
    dtype_name = str(args.dtype or raw_scoring.get("dtype", "float32"))
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    query_batch_size = int(args.query_batch_size or raw_scoring.get("query_batch_size", 1))
    document_batch_size = int(args.document_batch_size or raw_scoring.get("document_batch_size", 1))
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformerBackbone(profile, device=device, torch_dtype=dtypes[dtype_name])
    source_rows = read_candidate_rows_jsonl(args.rows)
    rows = _select_stride_shard(
        source_rows,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    effective_attention = model.effective_attention_implementation()
    if args.require_resolved_attention is not None and effective_attention != args.require_resolved_attention:
        raise RuntimeError(
            "Loaded backbone resolved an unexpected attention implementation: "
            f"required {args.require_resolved_attention!r}, got {effective_attention!r}."
        )
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset].append(row)
    raw_instructions = model_config.data.get("dataset_instructions", {})
    if not isinstance(raw_instructions, dict):
        raise ValueError("dataset_instructions must be a mapping.")
    instructions = {str(key).lower(): str(value) for key, value in raw_instructions.items()}
    scores_by_row: dict[str, Any] = {}
    for dataset, dataset_rows in sorted(grouped.items()):
        instruction = instructions.get(dataset.lower())
        scored = score_candidate_rows(
            model,
            [
                ScoringRow(
                    row_id=row.row_id,
                    query_text=row.query_text,
                    candidate_texts=row.candidate_texts,
                )
                for row in dataset_rows
            ],
            query_batch_size=query_batch_size,
            document_batch_size=document_batch_size,
            device=device,
            query_instruction=instruction,
        )
        scores_by_row.update({row.row_id: row for row in scored})
    ordered_scores = [scores_by_row[row.row_id] for row in rows]
    write_score_table(
        ordered_scores,
        args.output,
        model_label=profile.label,
    )
    return {
        "path": str(args.output),
        "rows": len(ordered_scores),
        "source_rows": len(source_rows),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "model_label": profile.label,
        "model_id": profile.model_id,
        "revision": profile.revision,
        "dtype": dtype_name,
        "query_batch_size": query_batch_size,
        "document_batch_size": document_batch_size,
        "resolved_attention_implementation": effective_attention,
    }


def _handle_teacher_fuse(args: argparse.Namespace) -> dict[str, Any]:
    """Fuse aligned teacher tables into the training artifact."""

    from score_vector_distillation.teachers.io import fuse_teacher_tables_to_training_parquet

    run = load_executable_run_config(args.run_config)
    teacher_paths = _assignments(args.teacher, value_parser=Path)
    artifact = fuse_teacher_tables_to_training_parquet(
        args.rows,
        teacher_paths,
        run.teacher_weights,
        args.output,
    )
    return {
        "path": str(artifact.path),
        "row_count": artifact.row_count,
        "candidate_count": artifact.candidate_count,
        "teacher_labels": list(artifact.teacher_labels),
    }


@dataclass(frozen=True)
class _ResolvedEvaluationProtocol:
    """Executable retrieval settings selected for one CLI evaluation."""

    data_config: Path
    top_k: int
    metric_k_values: tuple[int, ...]
    run_config: dict[str, Any] | None = None
    primary_metric: str | None = None
    aggregation: str | None = None


def _resolve_evaluation_protocol(args: argparse.Namespace) -> _ResolvedEvaluationProtocol:
    """Resolve paper settings from a run config or explicit exploratory flags."""

    if args.run_config is None:
        if args.data_config is None:
            raise ValueError("evaluate requires --run-config or --data-config.")
        cutoffs = tuple(sorted(set(args.metric_k_values or (10, 100))))
        return _ResolvedEvaluationProtocol(
            data_config=args.data_config,
            top_k=100 if args.top_k is None else int(args.top_k),
            metric_k_values=cutoffs,
        )

    run = load_executable_run_config(args.run_config)
    evaluation = run.evaluation
    if args.data_config is not None:
        explicit_data = load_config(args.data_config)
        configured_data = load_config(run.data_config)
        if explicit_data.sha256 != configured_data.sha256:
            raise ValueError("--data-config does not match the dataset referenced by --run-config.")
    if args.top_k is not None and int(args.top_k) != evaluation.top_k:
        raise ValueError("--top-k conflicts with the executable run evaluation protocol.")
    if args.metric_k_values is not None:
        explicit_cutoffs = tuple(sorted(set(int(value) for value in args.metric_k_values)))
        if explicit_cutoffs != evaluation.metric_k_values:
            raise ValueError("--metric-k conflicts with the executable run evaluation protocol.")
    return _ResolvedEvaluationProtocol(
        data_config=run.data_config,
        top_k=evaluation.top_k,
        metric_k_values=evaluation.metric_k_values,
        run_config=run.resolved.data,
        primary_metric=evaluation.primary_metric.label,
        aggregation=evaluation.aggregation.value,
    )


def _handle_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run one configured full-corpus retrieval evaluation."""

    import torch

    from score_vector_distillation.data.fetch import fetch_snapshot
    from score_vector_distillation.data.loaders import load_retrieval_dataset
    from score_vector_distillation.data.registry import load_registry
    from score_vector_distillation.evaluation.pipeline import evaluate_retrieval_model
    from score_vector_distillation.models.profiles import profile_from_mapping
    from score_vector_distillation.models.sentence_transformer import SentenceTransformerBackbone
    from score_vector_distillation.run_record import checkpoint_record, runtime_summary

    protocol = _resolve_evaluation_protocol(args)
    model_config = load_config(args.model_config)
    registry = load_registry(protocol.data_config, require_resolved_pins=True)
    dataset_spec = registry.dataset(str(args.dataset_name))
    requested_split = (str(args.dataset_name), str(args.split))
    panel_splits = {(item.dataset, item.split) for item in registry.panel.full_corpus_eval}
    if requested_split not in panel_splits:
        raise ValueError(f"Requested dataset split {requested_split!r} is not in full_corpus_evaluation.")
    profile = profile_from_mapping(model_config.data)
    dataset_root = fetch_snapshot(dataset_spec, args.cache_dir)
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformerBackbone(profile, device=device, torch_dtype=dtypes[str(args.dtype)])
    model_identity: dict[str, str] = {
        "model_id": profile.model_id,
        "revision": profile.revision,
    }
    if args.checkpoint is not None:
        model.load_lora_adapter(args.checkpoint)
        checkpoint = checkpoint_record(args.checkpoint)
        model_identity.update(
            {
                "checkpoint_file": checkpoint["file"],
                "checkpoint_sha256": checkpoint["sha256"],
            }
        )
    effective_attention = model.effective_attention_implementation()
    if args.require_resolved_attention is not None and effective_attention != args.require_resolved_attention:
        raise RuntimeError(
            "Loaded backbone resolved an unexpected attention implementation: "
            f"required {args.require_resolved_attention!r}, got {effective_attention!r}."
        )
    retrieval_dataset = load_retrieval_dataset(
        dataset_root,
        name=str(args.dataset_name),
        split=str(args.split),
        expected_missing_qrel_documents=dataset_spec.expected_missing_qrel_documents,
    )

    def encode_queries(texts: Sequence[str]) -> npt.NDArray[Any]:
        return model.encode_texts(
            texts,
            kind="query",
            batch_size=int(args.query_batch_size),
            device=device,
            instruction=args.query_instruction,
        ).numpy()

    def encode_documents(texts: Sequence[str]) -> npt.NDArray[Any]:
        return model.encode_texts(
            texts,
            kind="document",
            batch_size=int(args.document_batch_size),
            device=device,
        ).numpy()

    if dataset_spec.snapshot.sha256 is None:  # guarded by registry validation; keeps the type explicit
        raise ValueError("Full-corpus evaluation requires a pinned snapshot checksum.")
    evaluation = evaluate_retrieval_model(
        retrieval_dataset,
        encode_queries=encode_queries,
        encode_documents=encode_documents,
        model_identity=model_identity,
        snapshot_sha256=dataset_spec.snapshot.sha256,
        run_config={
            "run": protocol.run_config,
            "model": model_config.data,
            "invocation": {
                "device": device,
                "dtype": str(args.dtype),
                "query_instruction": args.query_instruction,
                "query_batch_size": int(args.query_batch_size),
                "document_batch_size": int(args.document_batch_size),
            },
        },
        runtime=runtime_summary(attention_implementation=effective_attention),
        top_k=protocol.top_k,
        metric_k_values=protocol.metric_k_values,
        document_chunk_size=int(args.document_chunk_size),
        query_chunk_size=int(args.query_chunk_size),
        output_dir=args.output_dir,
    )
    return {
        "dataset": str(args.dataset_name),
        "split": str(args.split),
        "protocol": {
            "primary_metric": protocol.primary_metric,
            "metric_k_values": protocol.metric_k_values,
            "aggregation": protocol.aggregation,
            "top_k": protocol.top_k,
        },
        "metrics": evaluation.metrics,
        "artifacts": {name: str(path) for name, path in evaluation.artifacts.items()},
    }


def _handle_train(args: argparse.Namespace) -> dict[str, Any]:
    """Train one configured student seed."""

    from score_vector_distillation.training.trainer import TrainingRequest, train

    return train(
        TrainingRequest(
            run_config=args.config,
            train_file=args.train_file,
            eval_file=args.eval_file,
            output_dir=args.output_dir,
            seed=args.seed,
            resume_from=args.resume_from,
            dry_run=args.dry_run,
        )
    )


CommandHandler = Callable[[argparse.Namespace], dict[str, Any]]

_COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "config": _handle_config,
    "smoke": _handle_smoke,
    "data-fetch": _handle_data_fetch,
    "data-prepare": _handle_data_prepare,
    "teacher-score": _handle_teacher_score,
    "teacher-fuse": _handle_teacher_fuse,
    "evaluate": _handle_evaluate,
    "train": _handle_train,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Parse one command, delegate it to its handler, and print JSON."""

    args = build_parser().parse_args(argv)
    try:
        handler = _COMMAND_HANDLERS[str(args.command)]
    except KeyError as error:  # pragma: no cover - argparse enforces known subcommands
        raise AssertionError(f"Unhandled command: {args.command}") from error
    result = handler(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def entrypoint() -> None:
    """Run the CLI with concise messages for expected validation failures."""

    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    entrypoint()
