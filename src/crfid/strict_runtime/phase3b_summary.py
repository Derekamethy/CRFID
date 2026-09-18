"""Deterministic Phase 3B result recomputation, aggregation, and ranking."""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .hashing import canonical_json_sha256, sha256_file
from .neutral_metrics import evaluate_all_levels
from .phase3b_execution import C0, C1, C2, C3, CANDIDATE_IDS, FOLDS, SEEDS


CLASS_COUNT = 7


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _precision(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    return [
        float(value)
        for value in np.divide(
            diagonal,
            predicted,
            out=np.zeros_like(diagonal),
            where=predicted > 0,
        )
    ]


def _entropy(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(
        np.mean(
            -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-300, None)), axis=1
            )
        )
    )


def recompute_prediction_metrics(path: Path) -> dict:
    """Recompute every reported held metric directly from a prediction bundle."""
    with np.load(path, allow_pickle=False) as bundle:
        labels = np.asarray(bundle["true_labels"], dtype=np.int64)
        logits = np.asarray(bundle["logits"], dtype=np.float32)
        conditions = np.asarray(bundle["condition_ids"], dtype=str)
        weights = np.asarray(bundle["unique_signal_weights"], dtype=np.float64)
        saved_predictions = np.asarray(bundle["predictions"], dtype=np.int64)
    result = evaluate_all_levels(labels, logits, conditions, weights)
    predictions = np.asarray(result["predictions"], dtype=np.int64)
    if not np.array_equal(predictions, saved_predictions):
        raise RuntimeError(f"Saved predictions do not match logits: {path}")
    sample = result["sample"]
    condition = result["condition"]["metrics"]
    unique = result["unique_signal_weighted"]
    for metrics in (sample, condition, unique):
        metrics["per_class_precision"] = _precision(metrics["confusion_matrix"])
    counts = np.bincount(predictions, minlength=CLASS_COUNT)
    return {
        "sample": sample,
        "condition": condition,
        "unique_signal_weighted": unique,
        "dominant_predicted_class_fraction": float(counts.max() / len(predictions)),
        "mean_prediction_entropy": _entropy(logits),
    }


def _float_equal(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _float_equal(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _float_equal(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def collect_runs(execution_root: Path) -> list[dict]:
    """Load exactly 60 complete canonical units and independently recompute metrics."""
    records: list[dict] = []
    for candidate_id in CANDIDATE_IDS:
        for fold_id in FOLDS:
            for seed in SEEDS:
                run_dir = execution_root / "runs" / candidate_id / fold_id / f"seed_{seed}"
                manifest_path = run_dir / "RUN_MANIFEST.json"
                manifest = read_json(manifest_path)
                identity = (manifest["candidate_id"], manifest["fold_id"], manifest["seed"])
                if identity != (candidate_id, fold_id, seed):
                    raise RuntimeError(f"Run identity mismatch: {manifest_path}")
                if not manifest["run_complete"] or manifest["seed_selected_or_ranked"]:
                    raise RuntimeError(f"Incomplete or seed-selected run: {manifest_path}")
                if manifest["p4_numerical_content_accessed"]:
                    raise RuntimeError(f"P4-contaminated run: {manifest_path}")
                held_path = run_dir / "outer_held_predictions.npz"
                declared = manifest["predictions"]["outer_held"]
                if (
                    held_path.stat().st_size != declared["size_bytes"]
                    or sha256_file(held_path) != declared["sha256"]
                ):
                    raise RuntimeError(f"Prediction custody mismatch: {held_path}")
                recomputed = recompute_prediction_metrics(held_path)
                saved = manifest["metrics"]["values"]["outer_held"]
                if not _float_equal(recomputed, saved):
                    raise RuntimeError(f"Metric recomputation mismatch: {manifest_path}")
                development = manifest["metrics"]["values"]["outer_development"]
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "fold_id": fold_id,
                        "seed": seed,
                        "execution_status": manifest["execution_status"],
                        "selected_inner_epoch": manifest["selected_inner_epoch"],
                        "inner_selected_sample_macro_f1": manifest[
                            "inner_selected_sample_macro_f1"
                        ],
                        "sample_accuracy": recomputed["sample"]["accuracy"],
                        "sample_macro_f1": recomputed["sample"]["macro_f1"],
                        "sample_worst_class_recall": recomputed["sample"][
                            "worst_class_recall"
                        ],
                        "sample_zero_recall_class_count": recomputed["sample"][
                            "zero_recall_class_count"
                        ],
                        "condition_accuracy": recomputed["condition"]["accuracy"],
                        "condition_macro_f1": recomputed["condition"]["macro_f1"],
                        "unique_signal_weighted_accuracy": recomputed[
                            "unique_signal_weighted"
                        ]["accuracy"],
                        "unique_signal_weighted_macro_f1": recomputed[
                            "unique_signal_weighted"
                        ]["macro_f1"],
                        "dominant_predicted_class_fraction": recomputed[
                            "dominant_predicted_class_fraction"
                        ],
                        "mean_prediction_entropy": recomputed[
                            "mean_prediction_entropy"
                        ],
                        "outer_development_sample_accuracy": development["sample"][
                            "accuracy"
                        ],
                        "outer_development_sample_macro_f1": development["sample"][
                            "macro_f1"
                        ],
                        "development_to_held_sample_accuracy_gap": manifest["metrics"][
                            "values"
                        ]["development_to_held_sample_accuracy_gap"],
                        "development_to_held_sample_macro_f1_gap": manifest["metrics"][
                            "values"
                        ]["development_to_held_sample_macro_f1_gap"],
                        "metrics": recomputed,
                        "manifest": manifest,
                        "manifest_path": manifest_path,
                    }
                )
    if len(records) != 60:
        raise RuntimeError(f"Expected 60 runs, found {len(records)}")
    return records


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _sd(values: list[float]) -> float:
    return float(statistics.pstdev(values))


def fold_summaries(records: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    metrics = (
        "sample_accuracy",
        "sample_macro_f1",
        "sample_worst_class_recall",
        "sample_zero_recall_class_count",
        "condition_accuracy",
        "condition_macro_f1",
        "unique_signal_weighted_accuracy",
        "unique_signal_weighted_macro_f1",
        "dominant_predicted_class_fraction",
        "mean_prediction_entropy",
        "selected_inner_epoch",
        "outer_development_sample_accuracy",
        "outer_development_sample_macro_f1",
        "development_to_held_sample_accuracy_gap",
        "development_to_held_sample_macro_f1_gap",
    )
    for candidate_id in CANDIDATE_IDS:
        for fold_id in FOLDS:
            selected = [
                row
                for row in records
                if row["candidate_id"] == candidate_id and row["fold_id"] == fold_id
            ]
            if [row["seed"] for row in selected] != list(SEEDS):
                raise RuntimeError(f"Seed coverage mismatch: {candidate_id}/{fold_id}")
            summary = {
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                "seed_count": len(selected),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in selected]
                summary[f"{metric}_mean"] = _mean(values)
                summary[f"{metric}_population_sd"] = _sd(values)
            summaries.append(summary)
    return summaries


def candidate_comparison(records: list[dict], folds: list[dict]) -> list[dict]:
    comparisons: list[dict] = []
    for candidate_id in CANDIDATE_IDS:
        selected = [row for row in records if row["candidate_id"] == candidate_id]
        selected_folds = [row for row in folds if row["candidate_id"] == candidate_id]
        per_class_mean_recall = [
            _mean([row["metrics"]["sample"]["per_class_recall"][index] for row in selected])
            for index in range(CLASS_COUNT)
        ]
        seed_means = [
            _mean([row["sample_macro_f1"] for row in selected if row["seed"] == seed])
            for seed in SEEDS
        ]
        comparisons.append(
            {
                "candidate_id": candidate_id,
                "canonical_unit_count": len(selected),
                "worst_outer_held_position_macro_f1": min(
                    row["sample_macro_f1_mean"] for row in selected_folds
                ),
                "mean_outer_held_position_macro_f1": _mean(
                    [row["sample_macro_f1"] for row in selected]
                ),
                "outer_held_position_macro_f1_population_sd_all_units": _sd(
                    [row["sample_macro_f1"] for row in selected]
                ),
                "mean_outer_held_position_accuracy": _mean(
                    [row["sample_accuracy"] for row in selected]
                ),
                "worst_class_recall": min(per_class_mean_recall),
                "worst_class_index": int(np.argmin(per_class_mean_recall)),
                "per_class_mean_recall_json": json.dumps(per_class_mean_recall),
                "zero_recall_class_frequency": sum(
                    row["sample_zero_recall_class_count"] for row in selected
                )
                / (len(selected) * CLASS_COUNT),
                "condition_block_macro_f1": _mean(
                    [row["condition_macro_f1"] for row in selected]
                ),
                "unique_signal_weighted_macro_f1": _mean(
                    [row["unique_signal_weighted_macro_f1"] for row in selected]
                ),
                "across_seed_standard_deviation": _sd(seed_means),
                "per_seed_cross_fold_macro_f1_json": json.dumps(
                    dict(zip((str(seed) for seed in SEEDS), seed_means)), sort_keys=True
                ),
                "mean_development_to_held_macro_f1_gap": _mean(
                    [row["development_to_held_sample_macro_f1_gap"] for row in selected]
                ),
            }
        )
    control = next(row for row in comparisons if row["candidate_id"] == C0)
    for row in comparisons:
        row["worst_fold_macro_f1_delta_vs_c0"] = (
            row["worst_outer_held_position_macro_f1"]
            - control["worst_outer_held_position_macro_f1"]
        )
        row["mean_macro_f1_delta_vs_c0"] = (
            row["mean_outer_held_position_macro_f1"]
            - control["mean_outer_held_position_macro_f1"]
        )
        row["zero_recall_frequency_delta_vs_c0"] = (
            row["zero_recall_class_frequency"] - control["zero_recall_class_frequency"]
        )
    return comparisons


def deterministic_ranking(comparisons: list[dict], selection_rule: dict) -> dict:
    expected = [
        ("worst_outer_held_position_macro_f1", "maximize"),
        ("mean_outer_held_position_macro_f1", "maximize"),
        ("worst_class_recall", "maximize"),
        ("zero_recall_class_frequency", "minimize"),
        ("condition_block_macro_f1", "maximize"),
        ("unique_signal_weighted_macro_f1", "maximize"),
        ("across_seed_standard_deviation", "minimize"),
        ("candidate_id", "ascending_deterministic"),
    ]
    observed = [
        (row["metric"], row["operation"])
        for row in selection_rule["ordered_selector"]
    ]
    if observed != expected:
        raise RuntimeError(f"Frozen selector differs: {observed}")

    def key(row: dict) -> tuple:
        return (
            -row["worst_outer_held_position_macro_f1"],
            -row["mean_outer_held_position_macro_f1"],
            -row["worst_class_recall"],
            row["zero_recall_class_frequency"],
            -row["condition_block_macro_f1"],
            -row["unique_signal_weighted_macro_f1"],
            row["across_seed_standard_deviation"],
            row["candidate_id"],
        )

    ordered = sorted(comparisons, key=key)
    return {
        "schema_version": 1,
        "phase": "3B",
        "selection_rule_semantic_sha256": selection_rule["selection_rule_sha256"],
        "aggregation_definitions": {
            "worst_outer_held_position_macro_f1": "minimum_across_three_fold_means_each_mean_over_five_equal_weight_seeds",
            "mean_outer_held_position_macro_f1": "arithmetic_mean_over_all_15_equal_weight_fold_seed_units",
            "worst_class_recall": "minimum_across_seven_classes_of_each_class_mean_recall_over_all_15_units",
            "zero_recall_class_frequency": "zero_recall_class_occurrences_divided_by_15_units_times_7_classes",
            "condition_block_macro_f1": "arithmetic_mean_over_all_15_equal_weight_units",
            "unique_signal_weighted_macro_f1": "arithmetic_mean_over_all_15_equal_weight_units",
            "across_seed_standard_deviation": "population_sd_of_five_seed_means_each_seed_mean_over_three_folds",
            "candidate_id": "ascending_lexicographic_complete_tie_break",
        },
        "ordered_selector": selection_rule["ordered_selector"],
        "ranking": [
            {
                "rank": index,
                **{key: value for key, value in row.items() if key != "per_class_mean_recall_json"},
            }
            for index, row in enumerate(ordered, start=1)
        ],
        "provisional_source_only_winner": ordered[0]["candidate_id"],
        "all_seeds_equal_weight": True,
        "best_seed_selection": False,
        "p4_used": False,
    }


def per_class_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        metrics = record["metrics"]
        for class_index in range(CLASS_COUNT):
            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "fold_id": record["fold_id"],
                    "seed": record["seed"],
                    "class_index": class_index,
                    "sample_precision": metrics["sample"]["per_class_precision"][class_index],
                    "sample_recall": metrics["sample"]["per_class_recall"][class_index],
                    "sample_f1": metrics["sample"]["per_class_f1"][class_index],
                    "condition_precision": metrics["condition"]["per_class_precision"][class_index],
                    "condition_recall": metrics["condition"]["per_class_recall"][class_index],
                    "condition_f1": metrics["condition"]["per_class_f1"][class_index],
                    "unique_signal_weighted_precision": metrics["unique_signal_weighted"]["per_class_precision"][class_index],
                    "unique_signal_weighted_recall": metrics["unique_signal_weighted"]["per_class_recall"][class_index],
                    "unique_signal_weighted_f1": metrics["unique_signal_weighted"]["per_class_f1"][class_index],
                }
            )
    return rows


def _pairwise_seed_agreement(execution_root: Path, candidate_id: str, fold_id: str) -> float:
    predictions = {}
    for seed in SEEDS:
        path = execution_root / "runs" / candidate_id / fold_id / f"seed_{seed}" / "outer_held_predictions.npz"
        with np.load(path, allow_pickle=False) as bundle:
            predictions[seed] = np.asarray(bundle["predictions"], dtype=np.int64)
    agreements = [
        float(np.mean(predictions[left] == predictions[right]))
        for left, right in itertools.combinations(SEEDS, 2)
    ]
    return _mean(agreements)


def mechanism_rows(execution_root: Path, records: list[dict]) -> list[dict]:
    control = {
        (row["fold_id"], row["seed"]): row for row in records if row["candidate_id"] == C0
    }
    seed_agreement = {
        (candidate_id, fold_id): _pairwise_seed_agreement(
            execution_root, candidate_id, fold_id
        )
        for candidate_id in (C0, C2)
        for fold_id in FOLDS
    }
    rows = []
    for record in records:
        path = record["manifest_path"].parent / "MECHANISM.json"
        mechanism = read_json(path)
        semantic = dict(mechanism)
        declared_hash = semantic.pop("mechanism_sha256")
        if canonical_json_sha256(semantic) != declared_hash:
            raise RuntimeError(f"Mechanism hash mismatch: {path}")
        baseline = control[(record["fold_id"], record["seed"])]
        input_geometry = mechanism.get("input_geometry", {})
        embedding = mechanism.get("embedding_geometry", {})
        ncm = mechanism.get("ncm_readout", {})
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "fold_id": record["fold_id"],
                "seed": record["seed"],
                "input_tag_silhouette": input_geometry.get("tag_silhouette_condition_blocks", ""),
                "input_position_silhouette": input_geometry.get("position_silhouette_condition_blocks", ""),
                "input_position_probe_accuracy": input_geometry.get("position_probe_validation_accuracy", ""),
                "embedding_tag_silhouette": embedding.get("tag_silhouette_condition_blocks", ""),
                "embedding_position_silhouette": embedding.get("position_silhouette_condition_blocks", ""),
                "embedding_position_probe_accuracy": embedding.get("position_probe_validation_accuracy", ""),
                "held_sample_macro_f1": record["sample_macro_f1"],
                "held_macro_f1_delta_vs_matched_c0": record["sample_macro_f1"] - baseline["sample_macro_f1"],
                "held_worst_class_recall": record["sample_worst_class_recall"],
                "held_zero_recall_class_count": record["sample_zero_recall_class_count"],
                "development_to_held_macro_f1_gap": record["development_to_held_sample_macro_f1_gap"],
                "ncm_nearest_margin": ncm.get("mean_nearest_to_second_nearest_squared_distance_margin", ""),
                "ncm_minimum_prototype_distance": ncm.get("minimum_class_prototype_distance", ""),
                "mean_pairwise_seed_prediction_agreement": seed_agreement[
                    (record["candidate_id"], record["fold_id"])
                ]
                if record["candidate_id"] in {C0, C2}
                else "",
                "mechanism_file_sha256": sha256_file(path),
                "mechanism_recomputed": True,
                "p4_used": False,
            }
        )
    return rows


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in names})


def public_seed_rows(records: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key not in {"metrics", "manifest", "manifest_path"}}
        for row in records
    ]


def condition_rows(records: list[dict]) -> list[dict]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "fold_id": row["fold_id"],
            "seed": row["seed"],
            "accuracy": row["condition_accuracy"],
            "macro_f1": row["condition_macro_f1"],
            "per_class_recall_json": json.dumps(row["metrics"]["condition"]["per_class_recall"]),
            "confusion_matrix_json": json.dumps(row["metrics"]["condition"]["confusion_matrix"]),
        }
        for row in records
    ]


def unique_signal_rows(records: list[dict]) -> list[dict]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "fold_id": row["fold_id"],
            "seed": row["seed"],
            "accuracy": row["unique_signal_weighted_accuracy"],
            "macro_f1": row["unique_signal_weighted_macro_f1"],
            "per_class_recall_json": json.dumps(row["metrics"]["unique_signal_weighted"]["per_class_recall"]),
            "confusion_matrix_json": json.dumps(row["metrics"]["unique_signal_weighted"]["confusion_matrix"]),
        }
        for row in records
    ]
