#!/usr/bin/env python3
"""Render compact Markdown tables from the committed result fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/paper/results"))
    args = parser.parse_args()
    main_results = pd.read_csv(args.artifact_root / "main_results.csv")
    controls = pd.read_csv(args.artifact_root / "loss_controls.csv")
    fusion = pd.read_csv(args.artifact_root / "query_disjoint_fusion_summary.csv")
    mechanism = pd.read_csv(args.artifact_root / "query_disjoint_mechanism_summary.csv")
    calibrated = pd.read_csv(args.artifact_root / "query_disjoint_calibrated_fusion_summary.csv")
    print("# Main results\n")
    print(main_results.to_markdown(index=False))
    print("\n# Loss controls\n")
    print(controls.to_markdown(index=False))
    print("\n# Query-disjoint fusion diagnostic\n")
    print(
        fusion[
            [
                "dataset",
                "row_count",
                "nv_ndcg10",
                "equal_ndcg10",
                "qwen8_ndcg10",
                "best_weight",
                "best_ndcg10",
            ]
        ].to_markdown(index=False)
    )
    print("\n# Query-disjoint teacher agreement\n")
    print(
        mechanism[
            [
                "dataset",
                "row_count",
                "pearson_mean",
                "top1_agreement",
                "top5_jaccard_mean",
            ]
        ].to_markdown(index=False)
    )
    print("\n# Query-disjoint calibrated fusion diagnostic\n")
    print(calibrated.to_markdown(index=False))


if __name__ == "__main__":
    main()
