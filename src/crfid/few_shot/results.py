"""Readers for the frozen Few-Shot result tables.

These helpers return the stored rows unchanged. They never recompute, rescale or
re-rank a frozen value; recomputation belongs in the verification entry point,
which compares against, rather than replaces, the stored numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .paths import FrozenBranch

CANONICAL_RESULTS = "06_final_comparative_audit_and_archive/FINAL_CANONICAL_RESULTS_TABLE.csv"
METHOD_COMPARISON = "06_final_comparative_audit_and_archive/FINAL_METHOD_COMPARISON.csv"
PAIRED_EPISODE_DELTAS = "06_final_comparative_audit_and_archive/FINAL_PAIRED_EPISODE_DELTAS.csv"
PER_CLASS_COMPARISON = "06_final_comparative_audit_and_archive/FINAL_PER_CLASS_COMPARISON.csv"
BOOTSTRAP_INTERVALS = "06_final_comparative_audit_and_archive/FINAL_BOOTSTRAP_INTERVALS.csv"
CALIBRATION_COST = "06_final_comparative_audit_and_archive/FINAL_CALIBRATION_COST_TABLE.csv"

UNIT_RESULTS = {
    "FS3": "04_prototype_adaptation/fs3_execution/FS3_UNIT_RESULTS.csv",
    "FS5": "05_head_finetuning/fs5_p4_execution/FS5_UNIT_RESULTS.csv",
}
PER_CLASS_RESULTS = {
    "FS3": "04_prototype_adaptation/fs3_execution/FS3_PER_CLASS_RESULTS.csv",
    "FS5": "05_head_finetuning/fs5_p4_execution/FS5_PER_CLASS_RESULTS.csv",
}


def read_csv_rows(root: str | Path | None, relative_path: str) -> list[dict[str, str]]:
    branch = FrozenBranch.resolve(root)
    with branch.path(relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_canonical_results(root: str | Path | None = None) -> list[dict[str, str]]:
    return read_csv_rows(root, CANONICAL_RESULTS)


def load_method_comparison(root: str | Path | None = None) -> list[dict[str, str]]:
    return read_csv_rows(root, METHOD_COMPARISON)


def load_paired_episode_deltas(root: str | Path | None = None) -> list[dict[str, str]]:
    return read_csv_rows(root, PAIRED_EPISODE_DELTAS)


def load_per_class_comparison(root: str | Path | None = None) -> list[dict[str, str]]:
    return read_csv_rows(root, PER_CLASS_COMPARISON)


def load_unit_results(stage: str, root: str | Path | None = None) -> list[dict[str, str]]:
    if stage not in UNIT_RESULTS:
        raise KeyError(f"Unknown unit-result stage: {stage}")
    return read_csv_rows(root, UNIT_RESULTS[stage])


def load_per_class_results(stage: str, root: str | Path | None = None) -> list[dict[str, str]]:
    if stage not in PER_CLASS_RESULTS:
        raise KeyError(f"Unknown per-class stage: {stage}")
    return read_csv_rows(root, PER_CLASS_RESULTS[stage])
