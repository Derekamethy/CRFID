"""Concise public markdown reporting for the Data1 branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_results_report(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    methods = result["methods"]
    lines = [
        "# External Data1 pipeline-validation results",
        "",
        f"Primary verdict: `{result['primary_verdict']}`.",
        f"EV4 verdict: `{result['ev4_verdict']}`.",
        "",
        "This is a frozen within-Data1 pipeline-validation result. It is not",
        "cross-dataset model transfer or direct validation of a seven-class model.",
        "",
        "## Pooled results",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| {method['method']} | {method['accuracy']:.12f} | "
            f"{method['macro_f1']:.12f} |"
        )
    lines.extend(
        [
            "",
            "The raw and first-difference CNN rows report equal-weight five-seed means.",
            "The PCA-logistic score applies only to the frozen set-3 versus sets-4–9 split.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")

