#!/usr/bin/env python3
"""Generate the corrected recovery and cost-quality paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

E5_COLOR = "#1F8A4C"
QWEN_COLOR = "#2458A6"
QWEN_C6_COLOR = "#6F91C8"
TEACHER_COLOR = "#7B55A3"
GRID_MAJOR = "#DDE3EA"
GRID_MINOR = "#EEF2F6"
TEXT_GRAY = "#5F6772"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        type=Path,
        default=Path("artifacts/paper/results/figure_systems.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper/figures"),
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def load_systems(path: Path) -> pd.DataFrame:
    systems = pd.read_csv(path)
    required = {
        "system_id",
        "role",
        "all8_ndcg10",
        "ci_low",
        "ci_high",
        "base_system_id",
        "parameters_b",
        "query_latency_ms",
        "document_latency_ms",
    }
    missing = required - set(systems.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if systems["system_id"].duplicated().any():
        raise ValueError(f"Duplicate system_id values in {path}")
    return systems.set_index("system_id", drop=False)


def save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / filename
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"wrote {output}")
    return output


def score(systems: pd.DataFrame, system_id: str) -> float:
    return float(systems.loc[system_id, "all8_ndcg10"])


def latency(systems: pd.DataFrame, system_id: str, column: str = "query_latency_ms") -> float:
    return float(systems.loc[system_id, column])


def add_interval(ax: plt.Axes, systems: pd.DataFrame, system_id: str, *, color: str) -> None:
    row = systems.loc[system_id]
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    value = float(row["all8_ndcg10"])
    ax.errorbar(
        [float(row["query_latency_ms"])],
        [value],
        yerr=[[value - low], [high - value]],
        fmt="none",
        ecolor=color,
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )


def generate_cost_quality(systems: pd.DataFrame, output_dir: Path) -> Path:
    points = [
        ("e5_large_base", "E5 base", E5_COLOR, "o", False, (5, -9), "left"),
        ("e5_large_equal", "E5 equal", E5_COLOR, "o", True, (5, 8), "left"),
        ("qwen_base", "Qwen base", QWEN_COLOR, "o", False, (-5, 8), "right"),
        ("qwen_equal", "Qwen equal", QWEN_COLOR, "o", True, (5, -9), "left"),
        ("qwen8_teacher", "Qwen 8B", TEACHER_COLOR, "D", True, (-5, -8), "right"),
    ]
    fig, ax = plt.subplots(figsize=(3.55, 2.05))

    for base_id, student_id, color in [
        ("e5_large_base", "e5_large_equal", E5_COLOR),
        ("qwen_base", "qwen_equal", QWEN_COLOR),
    ]:
        ax.plot(
            [latency(systems, base_id), latency(systems, student_id)],
            [score(systems, base_id), score(systems, student_id)],
            color=color,
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )

    add_interval(ax, systems, "e5_large_equal", color=E5_COLOR)
    add_interval(ax, systems, "qwen_equal", color=QWEN_COLOR)

    for system_id, label, color, marker, filled, offset, horizontal_alignment in points:
        x = latency(systems, system_id)
        y = score(systems, system_id)
        facecolor = color if filled else "none"
        size = 45 if system_id in {"qwen_equal", "qwen8_teacher"} else 36
        ax.scatter(
            x,
            y,
            s=size,
            facecolor=facecolor,
            edgecolor=color,
            marker=marker,
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha=horizontal_alignment,
            va="center",
            fontsize=7,
            fontweight="bold" if system_id == "qwen_equal" else "normal",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.35},
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=TEXT_GRAY,
            label="frozen base",
            markersize=4.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TEXT_GRAY,
            markeredgecolor=TEXT_GRAY,
            label="distilled",
            markersize=4.5,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=TEXT_GRAY,
            markeredgecolor=TEXT_GRAY,
            label="teacher",
            markersize=4.5,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        ncol=1,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.86, 11.5)
    ax.set_ylim(0.47, 0.632)
    ax.set_xlabel("Query latency (ms/text, batch size 8, log scale)")
    ax.set_ylabel("All-8 macro NDCG@10")
    ax.grid(True, which="major", color=GRID_MAJOR, linewidth=0.55)
    ax.grid(True, which="minor", color=GRID_MINOR, linewidth=0.35)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.20, top=0.98)
    return save(fig, output_dir, "cost_quality.pdf")


def recovery(systems: pd.DataFrame, system_id: str, teacher_id: str) -> tuple[float, float, float]:
    row = systems.loc[system_id]
    base_id = str(row["base_system_id"])
    base = score(systems, base_id)
    teacher = score(systems, teacher_id)
    denominator = teacher - base
    if denominator <= 0:
        raise ValueError(f"Teacher score must exceed base score for {system_id}")
    value = 100.0 * (float(row["all8_ndcg10"]) - base) / denominator
    low = 100.0 * (float(row["ci_low"]) - base) / denominator
    high = 100.0 * (float(row["ci_high"]) - base) / denominator
    return value, low, high


def generate_recovery_size(systems: pd.DataFrame, output_dir: Path) -> Path:
    teacher_id = "qwen8_teacher"
    students = [
        ("e5_large_equal", "E5 equal", E5_COLOR, (5, 10), "left"),
        ("qwen_equal", "Qwen equal", QWEN_COLOR, (5, 8), "left"),
        ("qwen_c6", "Qwen C6", QWEN_C6_COLOR, (5, 5), "left"),
    ]
    fig, ax = plt.subplots(figsize=(3.55, 2.05))

    for base_id, color in [("e5_large_base", E5_COLOR), ("qwen_base", QWEN_COLOR)]:
        x = float(systems.loc[base_id, "parameters_b"])
        ax.scatter(
            x,
            0.0,
            s=30,
            facecolor="none",
            edgecolor=color,
            marker="o",
            linewidth=0.95,
            zorder=3,
        )

    for system_id, label, color, offset, horizontal_alignment in students:
        x = float(systems.loc[system_id, "parameters_b"])
        value, low, high = recovery(systems, system_id, teacher_id)
        ax.errorbar(
            [x],
            [value],
            yerr=[[value - low], [high - value]],
            fmt="o",
            markersize=5.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=3,
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            label,
            xy=(x, value),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha=horizontal_alignment,
            va="center",
            fontsize=7,
            fontweight="bold" if system_id != "qwen_c6" else "normal",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.35},
        )

    teacher_x = float(systems.loc[teacher_id, "parameters_b"])
    ax.scatter(
        teacher_x,
        100.0,
        s=44,
        facecolor=TEACHER_COLOR,
        edgecolor="white",
        marker="D",
        linewidth=0.7,
        zorder=4,
    )
    ax.annotate(
        "Qwen 8B reference",
        xy=(teacher_x, 100.0),
        xytext=(-5, -8),
        textcoords="offset points",
        color=TEACHER_COLOR,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.35},
    )

    ax.axhline(0, color="#AEB7C2", linewidth=0.7)
    ax.axhline(100, color="#B8A6CA", linestyle=(0, (2, 2)), linewidth=0.65)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=TEXT_GRAY,
            label="frozen base",
            markersize=4.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TEXT_GRAY,
            markeredgecolor=TEXT_GRAY,
            label="three-seed mean",
            markersize=4.5,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=TEXT_GRAY,
            markeredgecolor=TEXT_GRAY,
            label="teacher",
            markersize=4.5,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        frameon=False,
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.25, 10.5)
    ax.set_ylim(-5, 107)
    ax.set_xlabel("Model parameters (billions, log scale)")
    ax.set_ylabel("Qwen 8B reference gap recovered (%)")
    ax.grid(True, which="major", color=GRID_MAJOR, linewidth=0.55)
    ax.grid(True, which="minor", color=GRID_MINOR, linewidth=0.35)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.20, top=0.84)
    return save(fig, output_dir, "recovery_size.pdf")


def report_summary(systems: pd.DataFrame) -> None:
    for system_id in ["e5_large_equal", "qwen_equal", "qwen_c6"]:
        value, low, high = recovery(systems, system_id, "qwen8_teacher")
        print(f"{system_id}: recovery={value:.2f}% [{low:.2f}, {high:.2f}]")
    online_query = latency(systems, "online_fusion")
    online_document = latency(systems, "online_fusion", "document_latency_ms")
    qwen_query = latency(systems, "qwen_equal")
    qwen_document = latency(systems, "qwen_equal", "document_latency_ms")
    print(
        "qwen_equal speedup over sequential online fusion: "
        f"query={online_query / qwen_query:.2f}x, "
        f"document={online_document / qwen_document:.2f}x"
    )


def main() -> None:
    args = parse_args()
    setup_style()
    systems = load_systems(args.systems)
    generate_cost_quality(systems, args.output_dir)
    generate_recovery_size(systems, args.output_dir)
    report_summary(systems)


if __name__ == "__main__":
    main()
