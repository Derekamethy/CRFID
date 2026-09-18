from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "canonical_metrics" / "fixed_position_grouped_validity"
SPLITS = ROOT / "manifests" / "fixed_position_grouped_validity" / "exact_group_split_manifest.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(matrix: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(values)
    true_support = values.sum(axis=1)
    predicted_support = values.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(7),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive, true_support, out=np.zeros(7), where=true_support > 0
    )
    denominator = 2 * true_positive + predicted_support - true_positive + true_support - true_positive
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(7), where=denominator > 0)
    return float(true_positive.sum() / values.sum()), float(f1.mean()), precision, recall, f1


def test_executed_results_recompute_from_compact_artifacts() -> None:
    runs = _rows(RESULTS / "05_RUN_REGISTER.csv")
    per_run = _rows(RESULTS / "06_PER_RUN_METRICS.csv")
    aggregates = _rows(RESULTS / "07_POSITION_AGGREGATES.csv")
    per_class = _rows(RESULTS / "08_PER_CLASS_RESULTS.csv")
    histograms = _rows(RESULTS / "09_PREDICTED_CLASS_HISTOGRAMS.csv")
    learning = _rows(RESULTS / "11_LEARNING_VALIDITY_RESULTS.csv")
    confusions = _rows(RESULTS / "confusion_matrices.csv")
    splits = _rows(SPLITS)
    curves = _rows(RESULTS / "learning_curves.csv")

    assert len(runs) == 60
    assert len(per_run) == 60
    assert len(aggregates) == 56
    assert len(per_class) == 840
    assert len(histograms) == 420
    assert len(confusions) == 840
    assert len(splits) == 3780
    assert curves
    assert len({row["run_id"] for row in runs}) == 60
    assert all(row["status"] == "PASS" for row in runs)
    assert len(learning) == 4 and all(row["status"] == "PASS" for row in learning)

    metric_by_run = {row["run_id"]: row for row in per_run}
    confusion_by_run: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in confusions:
        confusion_by_run[
            (row["position"], row["fold"], row["seed"], row["evaluation_level"])
        ].append(row)
    histogram_by_run: dict[tuple[str, str, str], dict[int, int]] = defaultdict(dict)
    for row in histograms:
        histogram_by_run[(row["position"], row["fold"], row["seed"])][
            int(row["predicted_class_index"])
        ] = int(row["row_count"])
    class_by_key = {
        (
            row["position"],
            row["fold"],
            row["seed"],
            row["evaluation_level"],
            int(row["class_index"]),
        ): row
        for row in per_class
    }

    for values in metric_by_run.values():
        key = (values["position"], values["fold"], values["seed"])
        for level, total, accuracy_column, macro_column in (
            ("row", 1050, "held_row_accuracy", "held_row_macro_f1"),
            ("condition_block", 21, "held_block_accuracy", "held_block_macro_f1"),
        ):
            rows = sorted(
                confusion_by_run[key + (level,)],
                key=lambda row: int(row["true_class_index"]),
            )
            matrix = np.asarray(
                [[int(row[f"predicted_{column}"]) for column in range(7)] for row in rows],
                dtype=np.int64,
            )
            assert int(matrix.sum()) == total
            accuracy, macro_f1, precision, recall, f1 = _metrics(matrix)
            assert math.isclose(accuracy, float(values[accuracy_column]), abs_tol=1e-15)
            assert math.isclose(macro_f1, float(values[macro_column]), abs_tol=1e-15)
            for class_index in range(7):
                reported = class_by_key[key + (level, class_index)]
                assert math.isclose(precision[class_index], float(reported["precision"]), abs_tol=1e-15)
                assert math.isclose(recall[class_index], float(reported["recall"]), abs_tol=1e-15)
                assert math.isclose(f1[class_index], float(reported["f1"]), abs_tol=1e-15)
            if level == "row":
                assert list(matrix.sum(axis=0).astype(int)) == [
                    histogram_by_run[key][class_index] for class_index in range(7)
                ]

    for aggregate in aggregates:
        values = [
            float(row[aggregate["metric"]])
            for row in per_run
            if row["position"] == aggregate["position"]
        ]
        assert len(values) == 15
        assert math.isclose(statistics.fmean(values), float(aggregate["mean"]), abs_tol=1e-15)
        assert math.isclose(
            statistics.pstdev(values),
            float(aggregate["population_standard_deviation"]),
            abs_tol=1e-15,
        )
        assert math.isclose(statistics.median(values), float(aggregate["median"]), abs_tol=1e-15)
        assert math.isclose(min(values), float(aggregate["minimum"]), abs_tol=1e-15)
        assert math.isclose(max(values), float(aggregate["maximum"]), abs_tol=1e-15)
        if aggregate["confidence_interval_95_low"]:
            point = float(aggregate["confidence_interval_point_estimate"])
            low = float(aggregate["confidence_interval_95_low"])
            high = float(aggregate["confidence_interval_95_high"])
            assert 0 <= low <= point <= high <= 1
            assert aggregate["confidence_interval_estimand"].startswith(
                "mean across five seed metrics"
            )

    assert all(
        math.isclose(float(row["uniform_seven_class_chance"]), 1 / 7, abs_tol=1e-15)
        for row in per_run
    )
    for position in ("P1", "P2", "P3", "P4"):
        for seed in (42, 43, 44, 45, 46):
            held_counts = Counter(
                row["condition_block_identifier"]
                for row in splits
                if row["position"] == position
                and int(row["seed"]) == seed
                and row["split_assignment"] == "test"
            )
            assert len(held_counts) == 63 and set(held_counts.values()) == {1}
    assert all(1 <= int(row["selected_epoch"]) <= 50 for row in runs)
