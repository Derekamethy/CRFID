from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def set_budget_ticks(ax: plt.Axes, budgets: np.ndarray) -> None:
    labels = ["20\n" if budget == 20 else "\n21" if budget == 21 else str(budget) for budget in budgets]
    ax.set_xticks(budgets, labels=labels)


def calibration_plot(rows: list[dict[str, str]], output: Path, metric: str) -> None:
    budgets = np.asarray([int(row["budget"]) for row in rows])
    values = np.asarray([float(row[metric]) for row in rows])
    lower = np.asarray([float(row[f"{metric}_ci_95_lower"]) for row in rows])
    upper = np.asarray([float(row[f"{metric}_ci_95_upper"]) for row in rows])
    title_metric = "Macro-F1" if metric == "macro_f1" else "Accuracy"
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.errorbar(
        budgets[1:],
        values[1:],
        yerr=np.vstack((values[1:] - lower[1:], upper[1:] - values[1:])),
        fmt="o-",
        color="#C84C4C",
        ecolor="#C84C4C",
        capsize=3,
        linewidth=1.8,
        markersize=5,
        label="Target-assisted labelled P4",
    )
    ax.errorbar(
        budgets[:1],
        values[:1],
        yerr=np.vstack((values[:1] - lower[:1], upper[:1] - values[:1])),
        fmt="s",
        color="#2468A2",
        ecolor="#2468A2",
        capsize=4,
        markersize=7,
        label="Strict source-only DG",
        zorder=4,
    )
    ax.axvline(42, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(42, ax.get_ylim()[0], " 42-block coverage ceiling", rotation=90, va="bottom", ha="right", color="#555555", fontsize=8)
    for budget, label in ((7, "1-shot"), (21, "3-shot"), (35, "5-shot")):
        index = int(np.flatnonzero(budgets == budget)[0])
        ax.annotate(label, (budget, values[index]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xscale("symlog", linthresh=7, linscale=1.0)
    set_budget_ticks(ax, budgets)
    ax.set_xlabel("Number of labelled P4 target measurements")
    ax.set_ylabel(f"P4 {title_metric}")
    ax.set_title(f"P4 labelled-target calibration curve: {title_metric}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.text(0.5, 0.005, "Points are equal-weight seed means; bars are 95% crossed hierarchical block/bootstrap intervals.", ha="center", fontsize=8, color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    stem = "p4_macro_f1_calibration_curve" if metric == "macro_f1" else "p4_accuracy_calibration_curve"
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def incremental_plot(rows: list[dict[str, str]], output: Path) -> None:
    budgets = np.asarray([int(row["to_budget"]) for row in rows])
    values = np.asarray([float(row["macro_f1_change"]) for row in rows])
    lower = np.asarray([float(row["macro_f1_change_ci_95_lower"]) for row in rows])
    upper = np.asarray([float(row["macro_f1_change_ci_95_upper"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.errorbar(
        budgets,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        fmt="o-",
        color="#7A4EAB",
        capsize=3,
        linewidth=1.6,
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(42, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xscale("symlog", linthresh=7, linscale=1.0)
    set_budget_ticks(ax, budgets)
    ax.set_xlabel("Ending labelled-P4 budget")
    ax.set_ylabel("Incremental Macro-F1 change")
    ax.set_title("Marginal gain from each additional label tranche")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "incremental_macro_f1_gain.png", bbox_inches="tight")
    plt.close(fig)


def coverage_plot(rows: list[dict[str, str]], output: Path) -> None:
    budgets = np.asarray([int(row["total_labelled_measurements"]) for row in rows])
    blocks = np.asarray([float(row["unique_condition_blocks_mean"]) for row in rows])
    signals = np.asarray([float(row["unique_exact_signals_mean"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(budgets, blocks, "o-", color="#2B8C6B", linewidth=1.8, label="Independent condition blocks")
    ax.axhline(42, color="#2B8C6B", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xscale("symlog", linthresh=7, linscale=1.0)
    set_budget_ticks(ax, budgets)
    ax.set_xlabel("Number of labelled P4 target measurements")
    ax.set_ylabel("Unique condition blocks", color="#2B8C6B")
    ax.tick_params(axis="y", labelcolor="#2B8C6B")
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(budgets, signals, "^-", color="#D8872A", linewidth=1.5, label="Unique exact signals")
    twin.set_ylabel("Unique exact signal digests", color="#D8872A")
    twin.tick_params(axis="y", labelcolor="#D8872A")
    ax.set_title("Raw-label budget versus independent and exact-signal coverage")
    ax.grid(axis="y", alpha=0.2)
    handles = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(handles, [item.get_label() for item in handles], loc="lower right")
    fig.savefig(output / "support_condition_coverage.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render publication figures from frozen calibration CSVs.")
    parser.add_argument(
        "--results-directory",
        default=str(ROOT / "results/canonical_metrics/p4_large_calibration_curve"),
    )
    arguments = parser.parse_args()
    results = Path(arguments.results_directory)
    configure()
    curve = read_rows(results / "15_CALIBRATION_CURVE.csv")
    incremental = read_rows(results / "16_INCREMENTAL_GAINS.csv")
    coverage = read_rows(results / "17_SUPPORT_COVERAGE_SUMMARY.csv")
    figures = results / "figures"
    calibration_plot(curve, figures, "macro_f1")
    calibration_plot(curve, figures, "accuracy")
    incremental_plot(incremental, figures)
    coverage_plot(coverage, figures)
    print(figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
