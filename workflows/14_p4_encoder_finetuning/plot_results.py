from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"frozen_linear": "#4C78A8", "partial_ft": "#F58518", "full_ft": "#54A24B"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False})


def plot(output_directory: Path) -> None:
    _style()
    output = output_directory.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = _read_csv(output / "18_PRIMARY_RESULTS.csv")
    status = json.loads((output / "32_FINAL_STATUS.json").read_text(encoding="utf-8"))
    budgets = np.asarray([int(row["budget"]) for row in primary], dtype=np.int64)
    positions = np.arange(1, len(budgets) + 1, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for key, label in (("frozen_linear", "Frozen C1 + linear"), ("partial_ft", "Partial encoder FT + linear"), ("full_ft", "Full encoder FT + linear")):
        values = np.asarray([float(row[f"{key}_macro_f1"]) for row in primary])
        lower = np.asarray([float(row[f"{key}_macro_f1_ci_95_lower"]) for row in primary])
        upper = np.asarray([float(row[f"{key}_macro_f1_ci_95_upper"]) for row in primary])
        ax.errorbar(positions, values, yerr=np.vstack([values - lower, upper - values]), color=COLORS[key], marker="o", linewidth=2, capsize=3, label=label)
    zero = 0.11223088363457531
    zero_low, zero_high = 0.061166882460495876, 0.16200529817788056
    ax.errorbar([0], [zero], yerr=[[zero - zero_low], [zero_high - zero]], color="#777777", marker="D", linestyle="none", capsize=3, label="Strict source-only DG reference")
    ax.axhline(1 / 7, color="#999999", linestyle=":", linewidth=1, label="Chance accuracy reference (1/7)")
    ax.axvline(0.5, color="#BBBBBB", linewidth=1)
    ax.text(0, 0.025, "source-only", ha="center", va="bottom", color="#666666")
    ax.text(2.5, 0.025, "target-assisted adaptation", ha="center", va="bottom", color="#666666")
    ax.set_xticks(np.arange(0, len(budgets) + 1), ["0", *[str(value) for value in budgets]])
    ax.set_xlabel("P4 labelled support measurements")
    ax.set_ylabel("Held-condition P4 Macro-F1")
    ax.set_title("P4 encoder fine-tuning under block-disjoint held conditions")
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "p4_encoder_finetuning_main.png", bbox_inches="tight")
    fig.savefig(figures / "p4_encoder_finetuning_main.pdf", bbox_inches="tight")
    plt.close(fig)

    diagnostic = _read_csv(output / "20_SUPPORT_VS_QUERY.csv")
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for arm, color, label in (("PARTIAL_FT", COLORS["partial_ft"], "Partial"), ("FULL_FT", COLORS["full_ft"], "Full")):
        rows = sorted((row for row in diagnostic if row["arm"] == arm), key=lambda row: int(row["budget"]))
        x = np.arange(len(rows))
        support = [float(row["support_macro_f1_mean"]) for row in rows]
        query = [float(row["held_query_macro_f1"]) for row in rows]
        ax.plot(x, support, color=color, marker="o", linewidth=2, label=f"{label} support")
        ax.plot(x, query, color=color, marker="s", linestyle="--", linewidth=2, label=f"{label} held condition")
    ax.set_xticks(np.arange(len(budgets)), [str(value) for value in budgets])
    ax.set_xlabel("P4 labelled support measurements")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Support fitting versus held-condition transfer")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "p4_encoder_finetuning_support_query.png", bbox_inches="tight")
    fig.savefig(figures / "p4_encoder_finetuning_support_query.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    plot(Path(args.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

