"""Public report rendering for Pre-DG results and comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_results(
    path: str | Path,
    aggregate: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pre-DG results",
        "",
        "These are single-dataset, non-strict baseline results. They are not fair",
        "zero-shot domain-generalization evidence.",
        "",
        "| Dataset | Split | Accuracy mean | Macro-F1 mean |",
        "|---|---|---:|---:|",
    ]
    for row in aggregate["split_results"]:
        lines.append(
            f"| {row['dataset_id']} | {row['split_name']} | "
            f"{row['accuracy_mean']:.12f} | {row['macro_f1_mean']:.12f} |"
        )
    lines.extend(
        [
            "",
            f"Numerical tolerance comparison: "
            f"`{'PASS' if comparison['numerical_tolerance_passed'] else 'FAIL'}`.",
            "",
            "Historical full predictions and checkpoints were not retained, so exact",
            "cross-archive prediction/checkpoint identity cannot be tested.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_comparison(path: str | Path, comparison: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pre-DG reproduction comparison",
        "",
        "| Dataset | Split | Historical mean | v2 mean | Difference | Tolerance | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison["split_mean_comparisons"]:
        lines.append(
            f"| {row['dataset_id']} | {row['split_name']} | "
            f"{row['authoritative_accuracy_mean']:.12f} | "
            f"{row['reproduced_accuracy_mean']:.12f} | "
            f"{row['absolute_difference']:.12f} | {row['tolerance']:.3f} | "
            f"{'PASS' if row['within_tolerance'] else 'FAIL'} |"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
