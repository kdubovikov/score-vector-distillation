"""Dataset-stratified paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt


@dataclass(frozen=True)
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    bootstrap_tail_probability: float
    query_count: int
    dataset_count: int


def paired_stratified_bootstrap(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
    *,
    datasets: Sequence[str],
    samples: int = 10_000,
    seed: int = 42,
    require_identical_queries: bool = True,
) -> BootstrapResult:
    """Bootstrap macro-averaged paired query deltas within datasets."""

    if samples <= 0:
        raise ValueError("samples must be positive.")
    deltas: list[npt.NDArray[np.float64]] = []
    for dataset in datasets:
        left_rows = left.get(dataset, {})
        right_rows = right.get(dataset, {})
        if require_identical_queries and set(left_rows) != set(right_rows):
            missing = sorted(set(left_rows).difference(right_rows))[:5]
            extra = sorted(set(right_rows).difference(left_rows))[:5]
            raise ValueError(f"Query-ID mismatch for {dataset}: missing_right={missing}, extra_right={extra}")
        query_ids = sorted(set(left_rows).intersection(right_rows))
        if not query_ids:
            raise ValueError(f"No paired queries for dataset {dataset!r}.")
        deltas.append(np.asarray([left_rows[qid] - right_rows[qid] for qid in query_ids], dtype=np.float64))

    observed = float(np.mean([values.mean() for values in deltas]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        means = [float(values[rng.integers(0, len(values), len(values))].mean()) for values in deltas]
        draws[sample_index] = float(np.mean(means))
    return BootstrapResult(
        mean_delta=observed,
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        bootstrap_tail_probability=float(np.mean(draws <= 0.0)),
        query_count=sum(len(values) for values in deltas),
        dataset_count=len(deltas),
    )
