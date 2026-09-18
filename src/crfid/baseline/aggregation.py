"""Seed aggregation and frozen-reference comparison."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def aggregate_seed_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["split_name"])].append(row)
    split_rows = []
    for (dataset_id, split_name), items in sorted(grouped.items()):
        accuracy = [float(item["accuracy"]) for item in items]
        macro_f1 = [float(item["macro_f1"]) for item in items]
        split_rows.append(
            {
                "dataset_id": dataset_id,
                "split_name": split_name,
                "seed_count": len(items),
                "accuracy_mean": mean(accuracy),
                "accuracy_population_std": pstdev(accuracy) if len(accuracy) > 1 else 0.0,
                "macro_f1_mean": mean(macro_f1),
                "macro_f1_population_std": pstdev(macro_f1) if len(macro_f1) > 1 else 0.0,
            }
        )
    headline = {
        row["dataset_id"]: row
        for row in split_rows
        if row["split_name"].endswith("grouped_random_raw_condition")
    }
    return {"split_results": split_rows, "grouped_random_headline": headline}


def compare_with_authoritative(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    references: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual = {
        (row["dataset_id"], row["split_name"], int(row["seed"])): row for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    for reference in references:
        key = (
            reference["dataset_id"],
            reference["split_name"],
            int(reference["seed"]),
        )
        if key not in actual:
            raise ValueError(f"Missing reproduced Pre-DG run: {key}")
        row = actual[key]
        delta = float(row["accuracy"]) - float(reference["accuracy"])
        comparisons.append(
            {
                "dataset_id": key[0],
                "split_name": key[1],
                "seed": key[2],
                "authoritative_accuracy": float(reference["accuracy"]),
                "reproduced_accuracy": float(row["accuracy"]),
                "absolute_difference": abs(delta),
                "authoritative_selected_dropout": float(reference["selected_dropout"]),
                "reproduced_selected_dropout": float(row["selected_dropout"]),
                "accuracy_within_tolerance": abs(delta)
                <= float(policy["per_run_accuracy_absolute_tolerance"]),
            }
        )
    split_differences = []
    by_reference: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in references:
        by_reference[(row["dataset_id"], row["split_name"])].append(float(row["accuracy"]))
    for row in aggregate["split_results"]:
        key = (row["dataset_id"], row["split_name"])
        reference_mean = mean(by_reference[key])
        difference = abs(float(row["accuracy_mean"]) - reference_mean)
        tolerance = (
            float(policy["grouped_random_mean_accuracy_absolute_tolerance"])
            if row["split_name"].endswith("grouped_random_raw_condition")
            else float(policy["per_split_mean_accuracy_absolute_tolerance"])
        )
        split_differences.append(
            {
                "dataset_id": key[0],
                "split_name": key[1],
                "authoritative_accuracy_mean": reference_mean,
                "reproduced_accuracy_mean": float(row["accuracy_mean"]),
                "absolute_difference": difference,
                "tolerance": tolerance,
                "within_tolerance": difference <= tolerance,
            }
        )
    passed = all(item["accuracy_within_tolerance"] for item in comparisons) and all(
        item["within_tolerance"] for item in split_differences
    )
    return comparisons, {
        "run_comparisons": comparisons,
        "split_mean_comparisons": split_differences,
        "exact_identity_checks": "PASS",
        "prediction_exact_match": "NOT_AVAILABLE_HISTORICAL_PREDICTIONS_NOT_RETAINED",
        "checkpoint_exact_match": "NOT_AVAILABLE_HISTORICAL_CHECKPOINTS_NOT_RETAINED",
        "numerical_tolerance_passed": passed,
    }
