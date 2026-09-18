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


def set_budget_ticks(axis: plt.Axes, budgets: np.ndarray) -> None:
    labels = ["20\n" if value == 20 else "\n21" if value == 21 else str(value) for value in budgets]
    axis.set_xticks(budgets, labels=labels)


def calibration_plot(rows: list[dict[str, str]], output: Path) -> None:
    budgets = np.asarray([int(row["budget"]) for row in rows])
    prototype = np.asarray([float(row["prototype_macro_f1"]) for row in rows])
    prototype_low = np.asarray([float(row["prototype_macro_f1_ci_95_lower"]) for row in rows])
    prototype_high = np.asarray([float(row["prototype_macro_f1_ci_95_upper"]) for row in rows])
    linear = np.asarray([float(row["linear_macro_f1"]) for row in rows])
    linear_low = np.asarray([float(row["linear_macro_f1_ci_95_lower"]) for row in rows])
    linear_high = np.asarray([float(row["linear_macro_f1_ci_95_upper"]) for row in rows])
    fig, axis = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    axis.errorbar(
        budgets[1:], prototype[1:],
        yerr=np.vstack((prototype[1:] - prototype_low[1:], prototype_high[1:] - prototype[1:])),
        fmt="o-", color="#C84C4C", capsize=3, linewidth=1.6, markersize=4.5,
        label="Cosine prototype (target-assisted)",
    )
    axis.errorbar(
        budgets[1:], linear[1:],
        yerr=np.vstack((linear[1:] - linear_low[1:], linear_high[1:] - linear[1:])),
        fmt="D-", color="#2B8C6B", capsize=3, linewidth=1.8, markersize=4.5,
        label="Trainable linear readout (target-assisted)",
    )
    axis.errorbar(
        budgets[:1], linear[:1],
        yerr=np.vstack((linear[:1] - linear_low[:1], linear_high[:1] - linear[:1])),
        fmt="s", color="#2468A2", capsize=4, markersize=7,
        label="Strict source-only DG anchor", zorder=5,
    )
    axis.axvline(42, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    axis.text(42, axis.get_ylim()[0], " 42-block coverage ceiling", rotation=90, va="bottom", ha="right", fontsize=8, color="#555555")
    axis.set_xscale("symlog", linthresh=7, linscale=1.0)
    set_budget_ticks(axis, budgets)
    axis.set_xlabel("Number of labelled P4 target measurements")
    axis.set_ylabel("Held-condition P4 Macro-F1")
    axis.set_title("Frozen C1: cosine prototype versus trainable linear readout")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(output / "p4_linear_vs_prototype_calibration.png", bbox_inches="tight")
    fig.savefig(output / "p4_linear_vs_prototype_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def difference_plot(rows: list[dict[str, str]], output: Path) -> None:
    positive = [row for row in rows if int(row["budget"]) > 0]
    budgets = np.asarray([int(row["budget"]) for row in positive])
    values = np.asarray([float(row["macro_f1_linear_minus_prototype"]) for row in positive])
    lower = np.asarray([float(row["macro_f1_linear_minus_prototype_ci_95_lower"]) for row in positive])
    upper = np.asarray([float(row["macro_f1_linear_minus_prototype_ci_95_upper"]) for row in positive])
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.errorbar(
        budgets, values, yerr=np.vstack((values - lower, upper - values)),
        fmt="o-", color="#7A4EAB", capsize=3, linewidth=1.8,
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.axhline(0.05, color="#B07A1A", linestyle=":", linewidth=1.2, label="Practical threshold (+0.05)")
    axis.axvline(42, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    axis.set_xscale("symlog", linthresh=7, linscale=1.0)
    set_budget_ticks(axis, budgets)
    axis.set_xlabel("Number of labelled P4 target measurements")
    axis.set_ylabel("Macro-F1: linear minus prototype")
    axis.set_title("Paired readout-capacity contrast")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(output / "p4_linear_minus_prototype.png", bbox_inches="tight")
    fig.savefig(output / "p4_linear_minus_prototype.pdf", bbox_inches="tight")
    plt.close(fig)


def train_query_plot(rows: list[dict[str, str]], output: Path) -> None:
    budgets = np.asarray([int(row["budget"]) for row in rows])
    support = np.asarray([float(row["support_macro_f1_mean"]) for row in rows])
    query = np.asarray([float(row["held_query_macro_f1"]) for row in rows])
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.plot(budgets, support, "o-", color="#D8872A", linewidth=1.7, label="Support/train Macro-F1")
    axis.plot(budgets, query, "D-", color="#2B8C6B", linewidth=1.7, label="Held-condition query Macro-F1")
    axis.set_xscale("symlog", linthresh=7, linscale=1.0)
    set_budget_ticks(axis, budgets)
    axis.set_xlabel("Number of labelled P4 target measurements")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Linear-head support fit versus held-condition transfer")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(output / "p4_linear_train_vs_query.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-directory",
        default=str(ROOT / "results/canonical_metrics/p4_trainable_linear_readout"),
    )
    arguments = parser.parse_args()
    results = Path(arguments.results_directory)
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    configure()
    primary = read_rows(results / "13_PRIMARY_RESULTS.csv")
    train_query = read_rows(results / "19_TRAIN_VS_QUERY.csv")
    calibration_plot(primary, output)
    difference_plot(primary, output)
    train_query_plot(train_query, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
