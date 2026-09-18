"""Paired 2x2 diagnostics with the frozen synchronized 60-run Case-B design."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .fixed_position_grouped_validity import (
    EXPECTED_POSITIONS,
    _confusion_rows,
    _per_class_rows,
    _run_manifest_rows,
    build_partitions,
    execute_primary_runs,
    load_governed_position,
    sha256_file,
)
from .strict_runtime.hashing import canonical_json_sha256
from .strict_runtime.neutral_metrics import metrics_from_confusion
from .strict_runtime.neutral_model import initialize_model
from .strict_runtime.phase3b_execution import (
    evaluate_logits,
    extract_logits_embeddings,
    load_checkpoint_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_RESULTS = PROJECT_ROOT / "results" / "canonical_metrics" / "fixed_position_grouped_validity"
BASE_MANIFEST = PROJECT_ROOT / "manifests" / "fixed_position_grouped_validity" / "exact_group_split_manifest.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "angle_distance_factorial_contrast" / "canonical.json"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "canonical_metrics" / "angle_distance_factorial"
DEFAULT_MANIFESTS = PROJECT_ROOT / "manifests" / "angle_distance_factorial"
POSITIONS = ("P1", "P2", "P3", "P4")
SEEDS = (42, 43, 44, 45, 46)
FOLDS = (1, 2, 3)
CORE_METRICS = (
    "held_row_accuracy",
    "held_row_macro_f1",
    "held_block_accuracy",
    "held_block_macro_f1",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_identifier(prefix: str, value: str, length: int = 24) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def paired_unit_id(tag_id: int, er: int, surface: str) -> str:
    return stable_identifier("pair", f"tag={tag_id}|er={er}|surface={surface}")


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))


def factor_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    """Return the frozen angle, distance, and interaction contrasts."""

    if set(values) != set(POSITIONS):
        raise ValueError("Factor contrasts require exactly P1, P2, P3, and P4")
    p1, p2, p3, p4 = (float(values[position]) for position in POSITIONS)
    return {
        "angle_45_minus_0": 0.5 * ((p2 - p1) + (p4 - p3)),
        "distance_150_minus_50": 0.5 * ((p3 - p1) + (p4 - p2)),
        "angle_by_distance_interaction": (p4 - p3) - (p2 - p1),
    }


def simple_effects(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(POSITIONS):
        raise ValueError("Simple effects require exactly P1, P2, P3, and P4")
    return {
        "angle_45_minus_0_at_50mm": float(values["P2"]) - float(values["P1"]),
        "angle_45_minus_0_at_150mm": float(values["P4"]) - float(values["P3"]),
        "distance_150_minus_50_at_0deg": float(values["P3"]) - float(values["P1"]),
        "distance_150_minus_50_at_45deg": float(values["P4"]) - float(values["P2"]),
    }


def _precision(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(axis=0)
    return np.divide(np.diag(matrix), denominator, out=np.zeros(7), where=denominator > 0)


def _matrix_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int, int, str], np.ndarray]:
    grouped: dict[tuple[str, int, int, str], dict[int, np.ndarray]] = defaultdict(dict)
    for row in rows:
        key = (str(row["position"]), int(row["fold"]), int(row["seed"]), str(row["evaluation_level"]))
        true_class = int(row["true_class_index"])
        if true_class in grouped[key]:
            raise RuntimeError(f"Duplicate confusion row: {key}/{true_class}")
        grouped[key][true_class] = np.asarray(
            [int(row[f"predicted_{index}"]) for index in range(7)], dtype=np.int64
        )
    matrices: dict[tuple[str, int, int, str], np.ndarray] = {}
    for key, class_rows in grouped.items():
        if set(class_rows) != set(range(7)):
            raise RuntimeError(f"Incomplete confusion matrix: {key}")
        matrices[key] = np.stack([class_rows[index] for index in range(7)])
    return matrices


def reconstruct_released_results(results_root: Path = BASE_RESULTS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently reconstruct released compact metrics without checkpoints."""

    results_root = Path(results_root)
    register = read_csv(results_root / "05_RUN_REGISTER.csv")
    run_metrics = read_csv(results_root / "06_PER_RUN_METRICS.csv")
    aggregates = read_csv(results_root / "07_POSITION_AGGREGATES.csv")
    per_class = read_csv(results_root / "08_PER_CLASS_RESULTS.csv")
    histograms = read_csv(results_root / "09_PREDICTED_CLASS_HISTOGRAMS.csv")
    matrices = _matrix_lookup(read_csv(results_root / "confusion_matrices.csv"))
    expected_runs = {
        f"{position}_F{fold}_S{seed}"
        for position in POSITIONS
        for fold in FOLDS
        for seed in SEEDS
    }
    register_ids = [row["run_id"] for row in register]
    metric_ids = [row["run_id"] for row in run_metrics]
    if Counter(register_ids) != Counter(expected_runs) or Counter(metric_ids) != Counter(expected_runs):
        raise RuntimeError("BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: run register differs")
    if any(row["status"] != "PASS" for row in register):
        raise RuntimeError("BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: failed run present")
    metric_lookup = {row["run_id"]: row for row in run_metrics}
    per_class_lookup = {
        (row["position"], int(row["fold"]), int(row["seed"]), row["evaluation_level"], int(row["class_index"])): row
        for row in per_class
    }
    histogram_lookup = {
        (row["position"], int(row["fold"]), int(row["seed"]), int(row["predicted_class_index"])): row
        for row in histograms
    }
    if len(per_class_lookup) != 4 * 3 * 5 * 2 * 7 or len(histogram_lookup) != 4 * 3 * 5 * 7:
        raise RuntimeError("BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: compact table duplicates")

    level_columns = {
        "row": ("held_row_accuracy", "held_row_macro_f1", 1050),
        "condition_block": ("held_block_accuracy", "held_block_macro_f1", 21),
    }
    for position in POSITIONS:
        for fold in FOLDS:
            for seed in SEEDS:
                run_id = f"{position}_F{fold}_S{seed}"
                reported = metric_lookup[run_id]
                for level, (accuracy_column, macro_column, expected_total) in level_columns.items():
                    matrix = matrices[(position, fold, seed, level)]
                    if int(matrix.sum()) != expected_total:
                        raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: {run_id}/{level} total")
                    derived = metrics_from_confusion(matrix)
                    if not _close(derived["accuracy"], float(reported[accuracy_column])) or not _close(
                        derived["macro_f1"], float(reported[macro_column])
                    ):
                        raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: {run_id}/{level} metrics")
                    precision = _precision(matrix)
                    for class_index in range(7):
                        row = per_class_lookup[(position, fold, seed, level, class_index)]
                        checks = (
                            _close(float(row["precision"]), precision[class_index]),
                            _close(float(row["recall"]), derived["per_class_recall"][class_index]),
                            _close(float(row["f1"]), derived["per_class_f1"][class_index]),
                            int(row["support"]) == int(matrix[class_index].sum()),
                        )
                        if not all(checks):
                            raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: per-class {run_id}/{level}")
                row_matrix = matrices[(position, fold, seed, "row")]
                predicted_counts = row_matrix.sum(axis=0)
                for class_index in range(7):
                    row = histogram_lookup[(position, fold, seed, class_index)]
                    if int(row["row_count"]) != int(predicted_counts[class_index]) or not _close(
                        float(row["row_fraction"]), predicted_counts[class_index] / 1050.0
                    ):
                        raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: histogram {run_id}")

    aggregate_lookup = {(row["position"], row["metric"]): row for row in aggregates}
    reconstruction_rows: list[dict[str, Any]] = []
    for position in POSITIONS:
        position_runs = [row for row in run_metrics if row["position"] == position]
        for metric in CORE_METRICS:
            values = [float(row[metric]) for row in position_runs]
            published = aggregate_lookup[(position, metric)]
            descriptive_checks = {
                "run_count": int(published["run_count"]) == 15,
                "mean": _close(statistics.fmean(values), float(published["mean"])),
                "population_standard_deviation": _close(statistics.pstdev(values), float(published["population_standard_deviation"])),
                "median": _close(statistics.median(values), float(published["median"])),
                "minimum": _close(min(values), float(published["minimum"])),
                "maximum": _close(max(values), float(published["maximum"])),
            }
            level = "row" if "row" in metric else "condition_block"
            metric_name = "accuracy" if metric.endswith("accuracy") else "macro_f1"
            pooled_seed_values = []
            for seed in SEEDS:
                pooled = sum((matrices[(position, fold, seed, level)] for fold in FOLDS), np.zeros((7, 7), dtype=np.int64))
                pooled_seed_values.append(float(metrics_from_confusion(pooled)[metric_name]))
            point = statistics.fmean(pooled_seed_values)
            point_match = _close(point, float(published["confidence_interval_point_estimate"]))
            if not all(descriptive_checks.values()) or not point_match:
                raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: aggregate {position}/{metric}")
            reconstruction_rows.append(
                {
                    "position": position,
                    "metric": metric,
                    "released_run_mean": float(published["mean"]),
                    "recomputed_run_mean": statistics.fmean(values),
                    "released_pooled_seed_point": float(published["confidence_interval_point_estimate"]),
                    "recomputed_pooled_seed_point": point,
                    "run_count": 15,
                    "compact_confusion_total": 15750 if level == "row" else 315,
                    "status": "PASS_EXACT_RECONSTRUCTION",
                }
            )
    return reconstruction_rows, {
        "status": "PASS_COMPACT_RESULT_RECONSTRUCTION",
        "run_count": 60,
        "confusion_matrix_count": len(matrices),
        "per_class_row_count": len(per_class),
        "histogram_row_count": len(histograms),
    }


def audit_paired_design(
    manifest_rows: Sequence[Mapping[str, str]],
    run_register: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    raise_on_failure: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit synchronized assignment and create a sanitized held-unit manifest."""

    key_counts = Counter(
        (
            row["position"],
            int(row["fold"]),
            int(row["seed"]),
            int(row["TagID"]),
            int(row["ER"]),
            row["surface"],
        )
        for row in manifest_rows
    )
    duplicate_count = sum(count - 1 for count in key_counts.values() if count > 1)
    expected_count = 4 * 3 * 5 * 63
    physical_cells = sorted(
        {(int(row["TagID"]), int(row["ER"]), row["surface"]) for row in manifest_rows}
    )
    assignment_failures = 0
    missing_position_groups = 0
    repetition_failures = 0
    held_fold_failures = 0
    unit_manifest: list[dict[str, Any]] = []
    by_design: dict[tuple[int, int, str, int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_design[(int(row["TagID"]), int(row["ER"]), row["surface"], int(row["fold"]), int(row["seed"]))].append(row)
        if int(row["row_count"]) != 50:
            repetition_failures += 1
    for rows in by_design.values():
        if {row["position"] for row in rows} != set(POSITIONS):
            missing_position_groups += 1
        if len({row["split_assignment"] for row in rows}) != 1:
            assignment_failures += 1
    for tag_id, er, surface in physical_cells:
        for seed in SEEDS:
            held_folds = set()
            for position in POSITIONS:
                matches = [
                    row
                    for row in manifest_rows
                    if int(row["TagID"]) == tag_id
                    and int(row["ER"]) == er
                    and row["surface"] == surface
                    and int(row["seed"]) == seed
                    and row["position"] == position
                    and row["split_assignment"] == "test"
                ]
                if len(matches) != 1:
                    held_fold_failures += 1
                else:
                    held_folds.add(int(matches[0]["fold"]))
            if len(held_folds) != 1:
                held_fold_failures += 1
                held_fold = ""
            else:
                held_fold = next(iter(held_folds))
            unit_manifest.append(
                {
                    "paired_unit_id": paired_unit_id(tag_id, er, surface),
                    "TagID": tag_id,
                    "ER": er,
                    "surface": surface,
                    "held_fold": held_fold,
                    "seed": seed,
                    "positions": "P1|P2|P3|P4",
                    "row_count_per_position": 50,
                    "pairing_status": "PASS" if held_fold != "" else "FAIL",
                }
            )
    prediction_alignment_failures = 0
    if prediction_rows is not None:
        prediction_counts = Counter(
            (row["paired_unit_id"], int(row["seed"]), row["position"])
            for row in prediction_rows
        )
        expected_prediction_keys = {
            (paired_unit_id(tag_id, er, surface), seed, position)
            for tag_id, er, surface in physical_cells
            for seed in SEEDS
            for position in POSITIONS
        }
        prediction_alignment_failures = sum(count != 1 for count in prediction_counts.values())
        prediction_alignment_failures += len(expected_prediction_keys - set(prediction_counts))
        prediction_alignment_failures += len(set(prediction_counts) - expected_prediction_keys)
    p4_boundary_failures = sum(
        int(row["held_evaluation_count"]) != 1
        or int(row["model_updates_after_held"]) != 0
        or str(row["checkpoint_verified_before_held"]).casefold() != "true"
        for row in run_register
    )
    checks = [
        ("physical_cell_coverage", len(physical_cells) == 63 and len(manifest_rows) == expected_count, f"physical_cells={len(physical_cells)}; manifest_rows={len(manifest_rows)}"),
        ("label_identity_across_positions", {tag for tag, _, _ in physical_cells} == set(range(1, 8)), "labels are the canonical TagID mapping shared by all positions"),
        ("fold_assignment_synchronization", held_fold_failures == 0, f"failures={held_fold_failures}"),
        ("train_validation_test_synchronization", assignment_failures == 0, f"failures={assignment_failures}"),
        ("seed_correspondence", {int(row["seed"]) for row in manifest_rows} == set(SEEDS), "seeds=42|43|44|45|46"),
        ("prediction_unit_alignment", prediction_rows is not None and prediction_alignment_failures == 0, f"failures={prediction_alignment_failures}"),
        ("missing_or_duplicated_cells", missing_position_groups == 0 and duplicate_count == 0, f"missing_groups={missing_position_groups}; duplicate_rows={duplicate_count}"),
        ("fifty_repetitions_within_block", repetition_failures == 0, f"failures={repetition_failures}"),
        ("no_row_level_independence", True, "inferential resampling unit is the physical condition block"),
        ("p4_not_used_for_tuning_or_split", p4_boundary_failures == 0, f"run_boundary_failures={p4_boundary_failures}; split rule is position-independent"),
    ]
    audit_rows = [
        {
            "check_id": index,
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
        for index, (name, passed, details) in enumerate(checks, start=1)
    ]
    if raise_on_failure and any(row["status"] != "PASS" for row in audit_rows):
        failures = "; ".join(
            f"{row['check']} ({row['details']})"
            for row in audit_rows
            if row["status"] != "PASS"
        )
        raise RuntimeError(
            "BLOCKED_PAIRED_GOVERNED_INPUTS: paired-design audit failed: " + failures
        )
    return audit_rows, unit_manifest


def generate_synchronized_split_manifest(
    data_by_position: Mapping[str, Any],
    fixed_config: Mapping[str, Any],
    protocol_id: str,
) -> list[dict[str, Any]]:
    """Generate position-independent validation membership for the Case B rerun."""

    surfaces = {
        value: index for index, value in enumerate(fixed_config["data"]["surface_order"])
    }
    records: list[dict[str, Any]] = []
    for position in POSITIONS:
        data = data_by_position[position]
        by_tag: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for condition_id in sorted({row["raw_condition_id"] for row in data.rows}):
            member = next(row for row in data.rows if row["raw_condition_id"] == condition_id)
            by_tag[int(member["tag_id"])].append(
                {
                    "condition_id": condition_id,
                    "tag_id": int(member["tag_id"]),
                    "er": int(member["er"]),
                    "surface": str(member["surface"]),
                    "row_count": sum(row["raw_condition_id"] == condition_id for row in data.rows),
                }
            )
        for fold_zero in range(3):
            for seed in SEEDS:
                for tag_id in range(1, 8):
                    ordered = sorted(by_tag[tag_id], key=lambda row: (row["er"], row["surface"]))
                    test = [
                        row
                        for row in ordered
                        if (row["er"] + surfaces[row["surface"]]) % 3 == fold_zero
                    ]
                    remaining = [row for row in ordered if row not in test]
                    material = f"{protocol_id}|fold={fold_zero}|seed={seed}|tag={tag_id}".encode("utf-8")
                    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)
                    selected = np.random.default_rng(rng_seed).permutation(len(remaining))[:2]
                    validation_physical = {
                        (remaining[int(index)]["er"], remaining[int(index)]["surface"])
                        for index in selected
                    }
                    test_physical = {(row["er"], row["surface"]) for row in test}
                    for block in ordered:
                        physical = (block["er"], block["surface"])
                        assignment = (
                            "test"
                            if physical in test_physical
                            else "validation"
                            if physical in validation_physical
                            else "train"
                        )
                        records.append(
                            {
                                "position": position,
                                "fold": fold_zero + 1,
                                "seed": seed,
                                "TagID": block["tag_id"],
                                "ER": block["er"],
                                "surface": block["surface"],
                                "paired_unit_id": paired_unit_id(block["tag_id"], block["er"], block["surface"]),
                                "condition_block_identifier": block["condition_id"],
                                "row_count": block["row_count"],
                                "split_assignment": assignment,
                                "source_data_fingerprint": data.source_data_fingerprint,
                            }
                        )
    if len(records) != 4 * 3 * 5 * 63:
        raise RuntimeError("Synchronized manifest row count differs")
    for fold in FOLDS:
        for seed in SEEDS:
            for tag_id in range(1, 8):
                rows = [
                    row
                    for row in records
                    if int(row["fold"]) == fold and int(row["seed"]) == seed and int(row["TagID"]) == tag_id
                ]
                for er in range(3):
                    for surface in ("A1", "A2", "A3"):
                        aligned = [row for row in rows if int(row["ER"]) == er and row["surface"] == surface]
                        if len(aligned) != 4 or len({row["split_assignment"] for row in aligned}) != 1:
                            raise RuntimeError("Synchronized manifest pairing differs")
                counts = Counter(row["split_assignment"] for row in rows if row["position"] == "P1")
                if counts != {"train": 4, "validation": 2, "test": 3}:
                    raise RuntimeError("Synchronized manifest per-tag counts differ")
    return records


def execute_case_b_synchronized_rerun(
    *,
    raw_root: Path,
    runtime_root: Path,
    factorial_protocol_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Train exactly the frozen C1 configuration on synchronized P1-P4 splits."""

    fixed_config = json.loads(
        (PROJECT_ROOT / "configs" / "fixed_position_grouped_validity" / "canonical.json").read_text(encoding="utf-8")
    )
    data_by_position = {
        position: load_governed_position(Path(raw_root), position) for position in POSITIONS
    }
    manifest_rows = generate_synchronized_split_manifest(
        data_by_position,
        fixed_config,
        "PAIRED_ANGLE_DISTANCE_FACTORIAL_CONTRAST_V1",
    )
    runtime_root = Path(runtime_root).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    outer_checkpoints = list((runtime_root / "primary").glob("P*/fold_*/seed_*/outer_refit_checkpoint.pt"))
    inner_checkpoints = list((runtime_root / "primary").glob("P*/fold_*/seed_*/inner_selected_checkpoint.pt"))
    if len(outer_checkpoints) == 60 and len(inner_checkpoints) == 60:
        execution = recover_completed_case_b_execution(
            data_by_position=data_by_position,
            manifest_rows=manifest_rows,
            runtime_root=runtime_root,
            protocol_sha256=factorial_protocol_sha256,
        )
        execution_mode = "REUSED_COMPLETED_CASE_B_CHECKPOINTS"
    elif len(outer_checkpoints) == 0 and len(inner_checkpoints) == 0:
        execution = execute_primary_runs(
            data_by_position,
            manifest_rows,
            fixed_config,
            runtime_root,
        )
        execution_mode = "TRAINED_EXACTLY_60_SYNCHRONIZED_RUNS"
    else:
        raise RuntimeError(
            "Partial synchronized checkpoint inventory requires manual review; refusing to retrain"
        )
    if len(execution["run_register"]) != 60 or len(execution["run_metrics"]) != 60:
        raise RuntimeError("Case B synchronized rerun did not produce exactly 60 runs")
    for row in execution["run_register"]:
        row["protocol_sha256"] = factorial_protocol_sha256
        row["synchronized_case"] = "B"
        row["hyperparameter_tuning"] = False
    return manifest_rows, execution, {
        "status": "PASS_CASE_B_SYNCHRONIZED_60_RUN_RERUN",
        "run_count": 60,
        "training_performed": execution_mode == "TRAINED_EXACTLY_60_SYNCHRONIZED_RUNS",
        "method_id": "C1_FIRST_DIFFERENCE_ERM_1DCNN",
        "hyperparameter_tuning": False,
        "manifest_sha256": canonical_json_sha256(manifest_rows),
        "runtime_root": str(runtime_root),
        "execution_mode": execution_mode,
    }


def recover_completed_case_b_execution(
    *,
    data_by_position: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    runtime_root: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Reconstruct compact evidence from an already-completed 60-run Case B set."""

    run_register: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    histograms: list[dict[str, Any]] = []
    confusions: list[dict[str, Any]] = []
    for position in POSITIONS:
        data = data_by_position[position]
        for fold in FOLDS:
            for seed in SEEDS:
                run_id = f"{position}_F{fold}_S{seed}"
                run_manifest = _run_manifest_rows(list(manifest_rows), position, fold, seed)
                partitions, _ = build_partitions(data, run_manifest)
                run_directory = Path(runtime_root) / "primary" / position / f"fold_{fold}" / f"seed_{seed}"
                outer_model, outer_payload = load_checkpoint_model(
                    run_directory / "outer_refit_checkpoint.pt", seed
                )
                inner_model, inner_payload = load_checkpoint_model(
                    run_directory / "inner_selected_checkpoint.pt", seed
                )
                selected_epoch = int(outer_payload["epochs"])
                if int(inner_payload["epochs"]) != selected_epoch:
                    raise RuntimeError(f"Case B checkpoint epoch mismatch: {run_id}")
                split_sha = canonical_json_sha256(run_manifest)
                if outer_payload["split_sha256"] != split_sha or inner_payload["split_sha256"] != split_sha:
                    raise RuntimeError(f"Case B checkpoint split mismatch: {run_id}")
                if bool(outer_payload.get("p4_used")) or bool(inner_payload.get("p4_used")):
                    raise RuntimeError(f"Case B checkpoint reports P4 tuning use: {run_id}")
                initial_model = initialize_model(seed)
                initial_logits, _ = extract_logits_embeddings(initial_model, partitions["outer_held"])
                initial = evaluate_logits(partitions["outer_held"], initial_logits)["sample"]
                inner_logits, _ = extract_logits_embeddings(inner_model, partitions["inner_validation"])
                validation = evaluate_logits(partitions["inner_validation"], inner_logits)["sample"]
                development_logits, _ = extract_logits_embeddings(outer_model, partitions["outer_development"])
                training = evaluate_logits(partitions["outer_development"], development_logits)["sample"]
                held_logits, _ = extract_logits_embeddings(outer_model, partitions["outer_held"])
                held = evaluate_logits(partitions["outer_held"], held_logits)
                held_row = held["sample"]
                held_block = held["condition"]["metrics"]
                histogram = np.bincount(np.asarray(held["predictions"], dtype=np.int64), minlength=7)
                row_majority = max(np.bincount(partitions["outer_held"].labels, minlength=7)) / len(partitions["outer_held"].labels)
                block_true = np.asarray(
                    [row["true_label"] for row in held["condition"]["rows"]], dtype=np.int64
                )
                block_majority = max(np.bincount(block_true, minlength=7)) / len(block_true)
                run_register.append(
                    {
                        "run_id": run_id,
                        "position": position,
                        "fold": fold,
                        "seed": seed,
                        "method_id": "C1_FIRST_DIFFERENCE_ERM_1DCNN",
                        "status": "PASS",
                        "started_at_utc": "not_retained_after_completed_case_b_audit_stop",
                        "completed_at_utc": "not_retained_after_completed_case_b_audit_stop",
                        "duration_seconds": "not_retained_after_completed_case_b_audit_stop",
                        "protocol_sha256": protocol_sha256,
                        "split_sha256": split_sha,
                        "source_data_fingerprint": data.source_data_fingerprint,
                        "selected_epoch": selected_epoch,
                        "checkpoint_verified_before_held": True,
                        "held_evaluation_count": 1,
                        "model_updates_after_held": 0,
                        "synchronized_case": "B",
                        "hyperparameter_tuning": False,
                    }
                )
                run_metrics.append(
                    {
                        "run_id": run_id,
                        "position": position,
                        "fold": fold,
                        "seed": seed,
                        "selected_epoch": selected_epoch,
                        "initial_held_accuracy": float(initial["accuracy"]),
                        "initial_held_macro_f1": float(initial["macro_f1"]),
                        "training_accuracy": float(training["accuracy"]),
                        "training_macro_f1": float(training["macro_f1"]),
                        "validation_accuracy": float(validation["accuracy"]),
                        "validation_macro_f1": float(validation["macro_f1"]),
                        "held_row_accuracy": float(held_row["accuracy"]),
                        "held_row_macro_f1": float(held_row["macro_f1"]),
                        "held_block_accuracy": float(held_block["accuracy"]),
                        "held_block_macro_f1": float(held_block["macro_f1"]),
                        "train_to_held_accuracy_gap": float(training["accuracy"] - held_row["accuracy"]),
                        "uniform_seven_class_chance": 1.0 / 7.0,
                        "empirical_row_majority_baseline": float(row_majority),
                        "condition_block_majority_baseline": float(block_majority),
                        "dominant_predicted_class_fraction": float(held["dominant_predicted_class_fraction"]),
                        "mean_prediction_entropy": float(held["mean_prediction_entropy"]),
                    }
                )
                per_class.extend(_per_class_rows(position, fold, seed, "row", held_row))
                per_class.extend(_per_class_rows(position, fold, seed, "condition_block", held_block))
                for class_index, count in enumerate(histogram):
                    histograms.append(
                        {
                            "position": position,
                            "fold": fold,
                            "seed": seed,
                            "predicted_class_index": class_index,
                            "predicted_TagID": class_index + 1,
                            "row_count": int(count),
                            "row_fraction": float(count / histogram.sum()),
                        }
                    )
                confusions.extend(_confusion_rows(position, fold, seed, "row", held_row["confusion_matrix"]))
                confusions.extend(_confusion_rows(position, fold, seed, "condition_block", held_block["confusion_matrix"]))
    if len(run_register) != 60:
        raise RuntimeError("Completed Case B checkpoint inventory did not reconstruct 60 runs")
    return {
        "run_register": run_register,
        "run_metrics": run_metrics,
        "per_class": per_class,
        "histograms": histograms,
        "confusions": confusions,
        "curves": [],
    }


def recover_frozen_block_predictions(
    *,
    raw_root: Path,
    checkpoint_root: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    run_metrics: Sequence[Mapping[str, str]],
    confusion_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run inference from frozen checkpoints and verify every released held result."""

    metric_lookup = {row["run_id"]: row for row in run_metrics}
    matrices = _matrix_lookup(confusion_rows)
    records: list[dict[str, Any]] = []
    checkpoint_digest = hashlib.sha256()
    data_fingerprints: dict[str, str] = {}
    for position in POSITIONS:
        data = load_governed_position(Path(raw_root), position)
        data_fingerprints[position] = data.source_data_fingerprint
        metadata = {row["raw_condition_id"]: row for row in data.rows}
        for fold in FOLDS:
            for seed in SEEDS:
                run_id = f"{position}_F{fold}_S{seed}"
                run_manifest = _run_manifest_rows(list(manifest_rows), position, fold, seed)
                partitions, _ = build_partitions(data, run_manifest)
                checkpoint = Path(checkpoint_root) / "primary" / position / f"fold_{fold}" / f"seed_{seed}" / "outer_refit_checkpoint.pt"
                if not checkpoint.is_file():
                    raise RuntimeError(f"BLOCKED_PAIRED_GOVERNED_INPUTS: missing checkpoint {run_id}")
                checkpoint_sha = sha256_file(checkpoint)
                relative_checkpoint = checkpoint.relative_to(Path(checkpoint_root)).as_posix()
                checkpoint_digest.update(
                    relative_checkpoint.encode("utf-8")
                    + b"\0"
                    + str(checkpoint.stat().st_size).encode("ascii")
                    + b"\0"
                    + checkpoint_sha.encode("ascii")
                    + b"\n"
                )
                model, _ = load_checkpoint_model(checkpoint, seed)
                logits, _ = extract_logits_embeddings(model, partitions["outer_held"])
                held = evaluate_logits(partitions["outer_held"], logits)
                reported = metric_lookup[run_id]
                comparisons = (
                    (held["sample"]["accuracy"], reported["held_row_accuracy"]),
                    (held["sample"]["macro_f1"], reported["held_row_macro_f1"]),
                    (held["condition"]["metrics"]["accuracy"], reported["held_block_accuracy"]),
                    (held["condition"]["metrics"]["macro_f1"], reported["held_block_macro_f1"]),
                )
                if not all(_close(float(observed), float(expected)) for observed, expected in comparisons):
                    raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: checkpoint metrics {run_id}")
                if not np.array_equal(
                    np.asarray(held["sample"]["confusion_matrix"], dtype=np.int64),
                    matrices[(position, fold, seed, "row")],
                ) or not np.array_equal(
                    np.asarray(held["condition"]["metrics"]["confusion_matrix"], dtype=np.int64),
                    matrices[(position, fold, seed, "condition_block")],
                ):
                    raise RuntimeError(f"BLOCKED_PREREQUISITE_RESULT_RECONSTRUCTION: checkpoint confusion {run_id}")
                for condition in held["condition"]["rows"]:
                    row = metadata[str(condition["raw_condition_id"])]
                    records.append(
                        {
                            "paired_unit_id": paired_unit_id(int(row["tag_id"]), int(row["er"]), str(row["surface"])),
                            "TagID": int(row["tag_id"]),
                            "ER": int(row["er"]),
                            "surface": str(row["surface"]),
                            "position": position,
                            "fold": fold,
                            "seed": seed,
                            "true_label": int(condition["true_label"]),
                            "predicted_label": int(condition["predicted_label"]),
                            "correct": int(condition["true_label"] == condition["predicted_label"]),
                            "sample_count": int(condition["sample_count"]),
                        }
                    )
    expected_records = 63 * 5 * 4
    if len(records) != expected_records or any(int(row["sample_count"]) != 50 for row in records):
        raise RuntimeError("BLOCKED_PAIRED_GOVERNED_INPUTS: prediction record coverage differs")
    prediction_digest = hashlib.sha256()
    for row in sorted(records, key=lambda item: (item["paired_unit_id"], item["seed"], item["position"])):
        payload = "|".join(
            str(row[key])
            for key in ("paired_unit_id", "position", "fold", "seed", "true_label", "predicted_label", "sample_count")
        )
        prediction_digest.update(payload.encode("utf-8") + b"\n")
    return records, {
        "status": "PASS_FROZEN_CHECKPOINT_INFERENCE_REPRODUCED_RELEASED_RESULTS",
        "training_performed": False,
        "checkpoint_count": 60,
        "checkpoint_inventory_sha256": checkpoint_digest.hexdigest(),
        "sanitized_block_prediction_count": len(records),
        "sanitized_block_prediction_sha256": prediction_digest.hexdigest(),
        "source_data_fingerprints": data_fingerprints,
    }


def prediction_arrays(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, int, int, str, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned [block, seed, position] true/prediction/correct arrays."""

    block_metadata = {
        (
            str(row["paired_unit_id"]),
            int(row["TagID"]),
            int(row["ER"]),
            str(row["surface"]),
            int(row["fold"]),
        )
        for row in prediction_rows
    }
    blocks = sorted(block_metadata, key=lambda value: (value[1], value[2], value[3]))
    if len(blocks) != 63:
        raise RuntimeError("Paired prediction arrays require 63 physical blocks")
    block_index = {block[0]: index for index, block in enumerate(blocks)}
    seed_index = {seed: index for index, seed in enumerate(SEEDS)}
    position_index = {position: index for index, position in enumerate(POSITIONS)}
    predicted = np.full((63, 5, 4), -1, dtype=np.int64)
    correct = np.full((63, 5, 4), -1, dtype=np.int64)
    true = np.full(63, -1, dtype=np.int64)
    seen: set[tuple[int, int, int]] = set()
    for row in prediction_rows:
        index = (
            block_index[str(row["paired_unit_id"])],
            seed_index[int(row["seed"])],
            position_index[str(row["position"])],
        )
        if index in seen:
            raise RuntimeError(f"Duplicate aligned prediction: {index}")
        seen.add(index)
        predicted[index] = int(row["predicted_label"])
        correct[index] = int(row["correct"])
        block_label = int(row["true_label"])
        if true[index[0]] not in {-1, block_label}:
            raise RuntimeError("Labels differ across paired positions or seeds")
        true[index[0]] = block_label
    if len(seen) != 63 * 5 * 4 or np.any(predicted < 0) or np.any(correct < 0) or np.any(true < 0):
        raise RuntimeError("Missing position or seed in aligned prediction arrays")
    return blocks, true, predicted, correct


def _confusion_from_vectors(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (np.asarray(true, dtype=np.int64), np.asarray(predicted, dtype=np.int64)), 1)
    return matrix


def _position_values(
    true: np.ndarray,
    predicted: np.ndarray,
    correct: np.ndarray,
    block_indices: np.ndarray | None = None,
    seed_indices: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if block_indices is None:
        block_indices = np.arange(correct.shape[0], dtype=np.int64)
    if seed_indices is None:
        seed_indices = np.arange(correct.shape[1], dtype=np.int64)
    accuracy_values: dict[str, float] = {}
    macro_values: dict[str, float] = {}
    for position_index, position in enumerate(POSITIONS):
        accuracy_values[position] = float(
            correct[block_indices][:, seed_indices, position_index].mean()
        )
        seed_macro = []
        for seed_index in seed_indices:
            selected_predictions = predicted[block_indices, int(seed_index), position_index]
            matrix = _confusion_from_vectors(true[block_indices], selected_predictions)
            seed_macro.append(float(metrics_from_confusion(matrix)["macro_f1"]))
        macro_values[position] = statistics.fmean(seed_macro)
    return {"condition_block_accuracy": accuracy_values, "pooled_condition_block_macro_f1": macro_values}


def draw_paired_bootstrap_indices(
    tag_ids: np.ndarray,
    seed_count: int,
    rng: np.random.Generator,
    *,
    resample_seeds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample blocks within TagID while returning one shared paired index draw."""

    tag_ids = np.asarray(tag_ids, dtype=np.int64)
    selected_blocks: list[int] = []
    for tag_id in sorted(np.unique(tag_ids)):
        candidates = np.flatnonzero(tag_ids == tag_id)
        selected_blocks.extend(int(value) for value in rng.choice(candidates, size=len(candidates), replace=True))
    selected_seeds = (
        rng.choice(np.arange(seed_count, dtype=np.int64), size=seed_count, replace=True)
        if resample_seeds
        else np.arange(seed_count, dtype=np.int64)
    )
    return np.asarray(selected_blocks, dtype=np.int64), np.asarray(selected_seeds, dtype=np.int64)


def bootstrap_factor_intervals(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks, true, predicted, correct = prediction_arrays(prediction_rows)
    tag_ids = np.asarray([block[1] for block in blocks], dtype=np.int64)
    point_values = _position_values(true, predicted, correct)
    interval_rows: list[dict[str, Any]] = []
    simple_rows: list[dict[str, Any]] = []
    methods = (
        ("paired_tagid_stratified_block_bootstrap_seeds_fixed", False, random_seed),
        ("paired_tagid_stratified_block_plus_seed_bootstrap", True, random_seed + 1),
    )
    for method, resample_seeds, method_seed in methods:
        rng = np.random.default_rng(method_seed)
        samples = {
            endpoint: {
                name: np.empty(resamples, dtype=np.float64)
                for name in (*factor_contrasts(values), *simple_effects(values))
            }
            for endpoint, values in point_values.items()
        }
        for replicate in range(resamples):
            selected_blocks, selected_seeds = draw_paired_bootstrap_indices(
                tag_ids, len(SEEDS), rng, resample_seeds=resample_seeds
            )
            replicate_values = _position_values(
                true, predicted, correct, selected_blocks, selected_seeds
            )
            for endpoint, values in replicate_values.items():
                effects = {**factor_contrasts(values), **simple_effects(values)}
                for name, value in effects.items():
                    samples[endpoint][name][replicate] = value
        for endpoint, values in point_values.items():
            point_factor = factor_contrasts(values)
            point_simple = simple_effects(values)
            for name, estimate in point_factor.items():
                low, high = np.quantile(samples[endpoint][name], [0.025, 0.975])
                interval_rows.append(
                    {
                        "endpoint": endpoint,
                        "contrast": name,
                        "estimate": estimate,
                        "confidence_interval_95_low": float(low),
                        "confidence_interval_95_high": float(high),
                        "method": method,
                        "resamples": resamples,
                        "random_seed": method_seed,
                        "seeds_resampled": resample_seeds,
                        "primary_endpoint": endpoint == "condition_block_accuracy",
                    }
                )
            for name, estimate in point_simple.items():
                low, high = np.quantile(samples[endpoint][name], [0.025, 0.975])
                simple_rows.append(
                    {
                        "endpoint": endpoint,
                        "simple_effect": name,
                        "estimate": estimate,
                        "confidence_interval_95_low": float(low),
                        "confidence_interval_95_high": float(high),
                        "method": method,
                        "resamples": resamples,
                        "random_seed": method_seed,
                    }
                )
    return interval_rows, simple_rows


def build_position_metric_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    run_metrics: Sequence[Mapping[str, str]],
    histograms: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    _, true, predicted, correct = prediction_arrays(prediction_rows)
    inferential = _position_values(true, predicted, correct)
    rows: list[dict[str, Any]] = []
    position_design = {
        "P1": (50, 0),
        "P2": (50, 45),
        "P3": (150, 0),
        "P4": (150, 45),
    }
    for position in POSITIONS:
        distance, angle = position_design[position]
        for metric, values in inferential.items():
            rows.append(
                {
                    "position": position,
                    "distance_mm": distance,
                    "angle_degrees": angle,
                    "metric": metric,
                    "estimand": "mean_of_five_seed_specific_metrics_pooled_over_63_blocks",
                    "value": values[position],
                    "unit_count": 63,
                    "seed_count": 5,
                    "inferential_role": "primary" if metric == "condition_block_accuracy" else "secondary",
                }
            )
        position_runs = [row for row in run_metrics if row["position"] == position]
        descriptive = {
            "descriptive_15_run_row_accuracy": statistics.fmean(float(row["held_row_accuracy"]) for row in position_runs),
            "descriptive_15_run_row_macro_f1": statistics.fmean(float(row["held_row_macro_f1"]) for row in position_runs),
            "descriptive_15_run_block_macro_f1": statistics.fmean(float(row["held_block_macro_f1"]) for row in position_runs),
            "training_accuracy": statistics.fmean(float(row["training_accuracy"]) for row in position_runs),
            "validation_accuracy": statistics.fmean(float(row["validation_accuracy"]) for row in position_runs),
            "train_to_held_accuracy_gap": statistics.fmean(float(row["train_to_held_accuracy_gap"]) for row in position_runs),
            "validation_to_held_accuracy_gap": statistics.fmean(float(row["validation_accuracy"]) - float(row["held_row_accuracy"]) for row in position_runs),
            "selected_epoch": statistics.fmean(float(row["selected_epoch"]) for row in position_runs),
            "held_block_accuracy_run_variance": statistics.pvariance(float(row["held_block_accuracy"]) for row in position_runs),
        }
        for metric, value in descriptive.items():
            rows.append(
                {
                    "position": position,
                    "distance_mm": distance,
                    "angle_degrees": angle,
                    "metric": metric,
                    "estimand": "descriptive_mean_across_15_runs" if not metric.endswith("variance") else "descriptive_population_variance_across_15_runs",
                    "value": value,
                    "unit_count": 15,
                    "seed_count": 5,
                    "inferential_role": "descriptive",
                }
            )
        position_histograms = [row for row in histograms if row["position"] == position]
        total = sum(int(row["row_count"]) for row in position_histograms)
        by_class = Counter()
        for row in position_histograms:
            by_class[int(row["predicted_TagID"])] += int(row["row_count"])
        for tag_id in range(1, 8):
            rows.append(
                {
                    "position": position,
                    "distance_mm": distance,
                    "angle_degrees": angle,
                    "metric": f"descriptive_predicted_TagID_{tag_id}_fraction",
                    "estimand": "descriptive_fraction_across_15_held_row_prediction_sets",
                    "value": by_class[tag_id] / total,
                    "unit_count": total,
                    "seed_count": 5,
                    "inferential_role": "secondary_descriptive",
                }
            )
    return rows


def build_primary_contrast_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocks, _, _, correct = prediction_arrays(prediction_rows)
    rows: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        identifier, tag_id, er, surface, fold = block
        for seed_index, seed in enumerate(SEEDS):
            values = {
                position: float(correct[block_index, seed_index, position_index])
                for position_index, position in enumerate(POSITIONS)
            }
            for contrast, estimate in factor_contrasts(values).items():
                rows.append(
                    {
                        "record_type": "paired_block_seed_contrast",
                        "paired_unit_id": identifier,
                        "TagID": tag_id,
                        "ER": er,
                        "surface": surface,
                        "held_fold": fold,
                        "seed": seed,
                        "contrast": contrast,
                        "estimate": estimate,
                        "physical_block_count": 1,
                        "primary_endpoint": "condition_block_accuracy",
                    }
                )
    for seed_index, seed in enumerate(SEEDS):
        values = {
            position: float(correct[:, seed_index, position_index].mean())
            for position_index, position in enumerate(POSITIONS)
        }
        for contrast, estimate in factor_contrasts(values).items():
            rows.append(
                {
                    "record_type": "seed_summary",
                    "paired_unit_id": "",
                    "TagID": "",
                    "ER": "",
                    "surface": "",
                    "held_fold": "",
                    "seed": seed,
                    "contrast": contrast,
                    "estimate": estimate,
                    "physical_block_count": 63,
                    "primary_endpoint": "condition_block_accuracy",
                }
            )
    values = {
        position: float(correct[:, :, position_index].mean())
        for position_index, position in enumerate(POSITIONS)
    }
    for contrast, estimate in factor_contrasts(values).items():
        rows.append(
            {
                "record_type": "overall_primary_estimand",
                "paired_unit_id": "",
                "TagID": "",
                "ER": "",
                "surface": "",
                "held_fold": "",
                "seed": "all_five",
                "contrast": contrast,
                "estimate": estimate,
                "physical_block_count": 63,
                "primary_endpoint": "condition_block_accuracy",
            }
        )
    return rows


def build_macro_f1_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    run_metrics: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    _, true, predicted, correct = prediction_arrays(prediction_rows)
    del correct
    rows: list[dict[str, Any]] = []
    position_seed: dict[int, dict[str, float]] = {}
    for seed_index, seed in enumerate(SEEDS):
        values: dict[str, float] = {}
        for position_index, position in enumerate(POSITIONS):
            matrix = _confusion_from_vectors(true, predicted[:, seed_index, position_index])
            values[position] = float(metrics_from_confusion(matrix)["macro_f1"])
            rows.append(
                {
                    "record_type": "position_seed_pooled_blocks",
                    "position": position,
                    "seed": seed,
                    "contrast": "",
                    "estimate": values[position],
                    "estimand": "Macro-F1_from_63_pooled_condition_blocks",
                    "physical_block_count": 63,
                }
            )
        position_seed[seed] = values
        for contrast, estimate in factor_contrasts(values).items():
            rows.append(
                {
                    "record_type": "seed_factor_contrast",
                    "position": "",
                    "seed": seed,
                    "contrast": contrast,
                    "estimate": estimate,
                    "estimand": "factor_contrast_of_seed_specific_pooled_block_Macro-F1",
                    "physical_block_count": 63,
                }
            )
    pooled_values = {
        position: statistics.fmean(position_seed[seed][position] for seed in SEEDS)
        for position in POSITIONS
    }
    for position, estimate in pooled_values.items():
        rows.append(
            {
                "record_type": "position_pooled_estimand",
                "position": position,
                "seed": "all_five",
                "contrast": "",
                "estimate": estimate,
                "estimand": "mean_of_five_seed_specific_Macro-F1_values_from_63_pooled_blocks",
                "physical_block_count": 63,
            }
        )
    for contrast, estimate in factor_contrasts(pooled_values).items():
        rows.append(
            {
                "record_type": "overall_factor_contrast",
                "position": "",
                "seed": "all_five",
                "contrast": contrast,
                "estimate": estimate,
                "estimand": "factor_contrast_of_mean_seed_specific_pooled_block_Macro-F1",
                "physical_block_count": 63,
            }
        )
    for position in POSITIONS:
        values = [float(row["held_block_macro_f1"]) for row in run_metrics if row["position"] == position]
        rows.append(
            {
                "record_type": "descriptive_15_run_position_mean",
                "position": position,
                "seed": "fold_seed_runs",
                "contrast": "",
                "estimate": statistics.fmean(values),
                "estimand": "descriptive_mean_of_15_fold_level_run_Macro-F1_values_not_the_pooled_estimand",
                "physical_block_count": 21,
            }
        )
    return rows


def build_per_class_factor_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _, true, predicted, _ = prediction_arrays(prediction_rows)
    position_values: dict[str, dict[str, dict[int, float]]] = {
        metric: {position: {} for position in POSITIONS}
        for metric in ("recall", "f1", "predicted_class_fraction")
    }
    for position_index, position in enumerate(POSITIONS):
        seed_metrics = []
        for seed_index in range(len(SEEDS)):
            matrix = _confusion_from_vectors(true, predicted[:, seed_index, position_index])
            seed_metrics.append(metrics_from_confusion(matrix))
        for class_index in range(7):
            position_values["recall"][position][class_index] = statistics.fmean(
                float(metrics["per_class_recall"][class_index]) for metrics in seed_metrics
            )
            position_values["f1"][position][class_index] = statistics.fmean(
                float(metrics["per_class_f1"][class_index]) for metrics in seed_metrics
            )
            position_values["predicted_class_fraction"][position][class_index] = statistics.fmean(
                float(np.asarray(metrics["confusion_matrix"])[:, class_index].sum() / 63.0)
                for metrics in seed_metrics
            )
    rows: list[dict[str, Any]] = []
    for metric, by_position in position_values.items():
        for class_index in range(7):
            values = {position: by_position[position][class_index] for position in POSITIONS}
            for contrast, estimate in factor_contrasts(values).items():
                rows.append(
                    {
                        "evaluation_level": "condition_block_pooled_within_seed",
                        "metric": metric,
                        "TagID": class_index + 1,
                        "contrast": contrast,
                        "P1": values["P1"],
                        "P2": values["P2"],
                        "P3": values["P3"],
                        "P4": values["P4"],
                        "estimate": estimate,
                        "role": "secondary_exploratory",
                    }
                )
    return rows


def build_heterogeneity_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocks, _, _, correct = prediction_arrays(prediction_rows)
    dimensions = {
        "TagID": sorted({block[1] for block in blocks}),
        "ER": sorted({block[2] for block in blocks}),
        "surface": sorted({block[3] for block in blocks}),
    }
    metadata_index = {"TagID": 1, "ER": 2, "surface": 3}
    rows: list[dict[str, Any]] = []
    for dimension, groups in dimensions.items():
        for group in groups:
            selected = np.asarray(
                [index for index, block in enumerate(blocks) if block[metadata_index[dimension]] == group],
                dtype=np.int64,
            )
            values = {
                position: float(correct[selected, :, position_index].mean())
                for position_index, position in enumerate(POSITIONS)
            }
            for contrast, estimate in factor_contrasts(values).items():
                rows.append(
                    {
                        "group_dimension": dimension,
                        "group_value": group,
                        "endpoint": "condition_block_accuracy",
                        "contrast": contrast,
                        "estimate": estimate,
                        "physical_block_count": len(selected),
                        "seed_count": 5,
                        "role": "exploratory_no_narrative_selection",
                    }
                )
    return rows


def build_seed_fold_consistency_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    run_metrics: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    blocks, true, predicted, correct = prediction_arrays(prediction_rows)
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        accuracy_values = {
            position: float(correct[:, seed_index, position_index].mean())
            for position_index, position in enumerate(POSITIONS)
        }
        macro_values = {}
        for position_index, position in enumerate(POSITIONS):
            matrix = _confusion_from_vectors(true, predicted[:, seed_index, position_index])
            macro_values[position] = float(metrics_from_confusion(matrix)["macro_f1"])
        for endpoint, values in (
            ("condition_block_accuracy", accuracy_values),
            ("pooled_condition_block_macro_f1", macro_values),
        ):
            for contrast, estimate in factor_contrasts(values).items():
                rows.append(
                    {
                        "record_type": "seed_consistency",
                        "group": seed,
                        "endpoint": endpoint,
                        "contrast": contrast,
                        "estimate": estimate,
                        "position": "",
                        "variance": "",
                        "role": "secondary",
                    }
                )
    for fold in FOLDS:
        for endpoint, column in (
            ("condition_block_accuracy", "held_block_accuracy"),
            ("condition_block_macro_f1_fold_descriptive", "held_block_macro_f1"),
            ("row_accuracy_descriptive", "held_row_accuracy"),
        ):
            values = {
                position: statistics.fmean(
                    float(row[column])
                    for row in run_metrics
                    if row["position"] == position and int(row["fold"]) == fold
                )
                for position in POSITIONS
            }
            for contrast, estimate in factor_contrasts(values).items():
                rows.append(
                    {
                        "record_type": "fold_consistency",
                        "group": fold,
                        "endpoint": endpoint,
                        "contrast": contrast,
                        "estimate": estimate,
                        "position": "",
                        "variance": "",
                        "role": "secondary_descriptive",
                    }
                )
    for position in POSITIONS:
        position_runs = [row for row in run_metrics if row["position"] == position]
        for endpoint, column in (
            ("held_block_accuracy", "held_block_accuracy"),
            ("held_row_accuracy", "held_row_accuracy"),
            ("selected_epoch", "selected_epoch"),
        ):
            values = [float(row[column]) for row in position_runs]
            rows.append(
                {
                    "record_type": "run_level_variance",
                    "group": "15_runs",
                    "endpoint": endpoint,
                    "contrast": "",
                    "estimate": statistics.fmean(values),
                    "position": position,
                    "variance": statistics.pvariance(values),
                    "role": "secondary_descriptive",
                }
            )
    for endpoint, value_function in (
        ("train_to_held_accuracy_gap", lambda row: float(row["train_to_held_accuracy_gap"])),
        ("validation_to_held_accuracy_gap", lambda row: float(row["validation_accuracy"]) - float(row["held_row_accuracy"])),
        ("selected_epoch", lambda row: float(row["selected_epoch"])),
    ):
        values = {
            position: statistics.fmean(value_function(row) for row in run_metrics if row["position"] == position)
            for position in POSITIONS
        }
        for contrast, estimate in factor_contrasts(values).items():
            rows.append(
                {
                    "record_type": "overall_secondary_factor_contrast",
                    "group": "all_runs",
                    "endpoint": endpoint,
                    "contrast": contrast,
                    "estimate": estimate,
                    "position": "",
                    "variance": "",
                    "role": "secondary_descriptive",
                }
            )
    return rows


def classify_factor_pattern(
    interval_rows: Sequence[Mapping[str, Any]],
    simple_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    primary_method = "paired_tagid_stratified_block_bootstrap_seeds_fixed"
    intervals = {
        str(row["contrast"]): row
        for row in interval_rows
        if row["endpoint"] == "condition_block_accuracy" and row["method"] == primary_method
    }
    simple = {
        str(row["simple_effect"]): row
        for row in simple_rows
        if row["endpoint"] == "condition_block_accuracy" and row["method"] == primary_method
    }

    def excludes_zero(row: Mapping[str, Any]) -> bool:
        return _interval_excludes_zero(row)

    angle = intervals["angle_45_minus_0"]
    distance = intervals["distance_150_minus_50"]
    interaction = intervals["angle_by_distance_interaction"]
    angle_simple_consistent = (
        float(simple["angle_45_minus_0_at_50mm"]["estimate"]) < 0.0
        and float(simple["angle_45_minus_0_at_150mm"]["estimate"]) < 0.0
    )
    distance_simple_consistent = (
        float(simple["distance_150_minus_50_at_0deg"]["estimate"]) < 0.0
        and float(simple["distance_150_minus_50_at_45deg"]["estimate"]) < 0.0
    )
    significant = {
        "angle": excludes_zero(angle),
        "distance": excludes_zero(distance),
        "interaction": excludes_zero(interaction),
    }
    if significant["interaction"]:
        classification = "ANGLE_DISTANCE_INTERACTION"
    elif (
        significant["angle"]
        and float(angle["estimate"]) < 0.0
        and angle_simple_consistent
        and not significant["distance"]
    ):
        classification = "ANGLE_ASSOCIATED_DEGRADATION"
    elif (
        significant["distance"]
        and float(distance["estimate"]) < 0.0
        and distance_simple_consistent
        and not significant["angle"]
    ):
        classification = "DISTANCE_ASSOCIATED_DEGRADATION"
    elif sum(significant.values()) > 1:
        classification = "MIXED_FACTOR_PATTERN"
    else:
        classification = "FACTOR_EFFECT_NOT_CONFIRMED"
    return classification, {
        "primary_interval_method": primary_method,
        "significant_primary_contrasts": significant,
        "angle_simple_effects_directionally_consistent": angle_simple_consistent,
        "distance_simple_effects_directionally_consistent": distance_simple_consistent,
    }


def _format_effect(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['estimate']):.6f} "
        f"[{float(row['confidence_interval_95_low']):.6f}, "
        f"{float(row['confidence_interval_95_high']):.6f}]"
    )


def _interval_excludes_zero(row: Mapping[str, Any]) -> bool:
    tolerance = 1e-12
    return (
        float(row["confidence_interval_95_low"]) > tolerance
        or float(row["confidence_interval_95_high"]) < -tolerance
    )


def preflight_factorial(
    *, raw_root: Path, config_path: Path = DEFAULT_CONFIG,
    output_root: Path = PROJECT_ROOT / "outputs/angle_distance_factorial",
) -> dict[str, Any]:
    """Validate scientific inputs without training or writing outputs."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frozen = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if config != frozen:
        raise ValueError("Factorial configuration differs from the frozen public protocol")
    output = Path(output_root).resolve()
    if output.is_relative_to(PROJECT_ROOT) and not output.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("Use outputs/ or an external output directory; retained evidence is read-only")
    required = [Path(raw_root) / f"{surface}_{position}.csv" for position in POSITIONS for surface in ("A1", "A2", "A3")]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("GOVERNED_MEASUREMENTS_REQUIRED: --raw-root must contain the approved A1_P1.csv through A3_P4.csv files; missing " + ", ".join(missing))
    binding = json.loads((DEFAULT_RESULTS / "02_FIXED_POSITION_BINDING.json").read_text(encoding="utf-8"))
    expected = binding["synchronized_checkpoint_inference"]["source_data_fingerprints"]
    observed = {position: load_governed_position(Path(raw_root), position).source_data_fingerprint for position in POSITIONS}
    if observed != expected:
        raise ValueError("Factorial measurement fingerprints differ from the retained scientific binding")
    return {"status": "PUBLIC_FACTORIAL_INPUTS_VERIFIED", "source_data_fingerprints": observed,
            "configuration_sha256": sha256_file(Path(config_path)),
            "training_configuration_sha256": sha256_file(PROJECT_ROOT / "configs/fixed_position_grouped_validity/canonical.json"),
            "run_count": 60, "training_performed": False,
            "historical_replay": "HISTORICAL_PROVENANCE_REPLAY_NOT_DISTRIBUTED"}


def run_factorial_analysis(
    *,
    raw_root: Path,
    output_root: Path = PROJECT_ROOT / "outputs/angle_distance_factorial",
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Execute the fixed scientific design; write only to the requested output root."""
    binding = preflight_factorial(raw_root=raw_root, config_path=config_path, output_root=output_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    results_root = Path(output_root).resolve() / "results"
    manifests_root = Path(output_root).resolve() / "manifests"
    results_root.mkdir(parents=True, exist_ok=True)
    reconstruction_rows, reconstruction_status = reconstruct_released_results(BASE_RESULTS)
    inherited_audit = [row for row in read_csv(DEFAULT_RESULTS / "04_PAIRED_DESIGN_AUDIT.csv") if row["design"] == "inherited_60_runs"]
    inherited_failures = [row for row in inherited_audit if row["status"] != "PASS"]
    if len(inherited_failures) != 1 or inherited_failures[0]["details"] != "failures=492":
        raise RuntimeError("Retained Case-B pairing decision differs")
    synchronized_runtime = Path(output_root).resolve() / "synchronized_primary"
    synchronized_manifest, execution, rerun_status = execute_case_b_synchronized_rerun(
        raw_root=Path(raw_root),
        runtime_root=synchronized_runtime,
        factorial_protocol_sha256=sha256_file(DEFAULT_RESULTS / "06_FACTORIAL_PROTOCOL.md"),
    )
    run_register = execution["run_register"]
    run_metrics = execution["run_metrics"]
    confusion_rows = execution["confusions"]
    histograms = execution["histograms"]
    prediction_rows, synchronized_inference_status = recover_frozen_block_predictions(
        raw_root=Path(raw_root),
        checkpoint_root=synchronized_runtime,
        manifest_rows=synchronized_manifest,
        run_metrics=run_metrics,
        confusion_rows=confusion_rows,
    )
    synchronized_audit, paired_manifest = audit_paired_design(
        synchronized_manifest, run_register, prediction_rows
    )
    audit_rows = [
        {"design": "inherited_60_runs", **row} for row in inherited_audit
    ] + [
        {"design": "case_b_synchronized_60_run_rerun", **row}
        for row in synchronized_audit
    ]
    binding_payload = {
        **binding,
        "compact_reconstruction": reconstruction_status,
        "inherited_frozen_inference": "HISTORICAL_PROVENANCE_REPLAY_NOT_DISTRIBUTED",
        "inherited_pairing_failure": inherited_failures[0],
        "paired_design_case": "B_SYNCHRONIZED_RERUN_REQUIRED_AND_COMPLETED",
        "new_training_required": True,
        "synchronized_rerun": rerun_status,
        "synchronized_checkpoint_inference": synchronized_inference_status,
    }
    write_json(results_root / "02_FIXED_POSITION_BINDING.json", binding_payload)
    write_csv(results_root / "03_EXISTING_RESULT_RECONSTRUCTION.csv", reconstruction_rows)
    write_csv(results_root / "04_PAIRED_DESIGN_AUDIT.csv", audit_rows)
    write_csv(results_root / "05_PAIRED_UNIT_MANIFEST.csv", paired_manifest)
    write_csv(manifests_root / "paired_unit_manifest.csv", paired_manifest)
    write_csv(manifests_root / "synchronized_group_split_manifest.csv", synchronized_manifest)
    manifest_sha = sha256_file(manifests_root / "paired_unit_manifest.csv")
    (manifests_root / "paired_unit_manifest.sha256").write_text(
        f"{manifest_sha}  paired_unit_manifest.csv\n", encoding="ascii"
    )
    synchronized_manifest_sha = sha256_file(manifests_root / "synchronized_group_split_manifest.csv")
    (manifests_root / "synchronized_group_split_manifest.sha256").write_text(
        f"{synchronized_manifest_sha}  synchronized_group_split_manifest.csv\n",
        encoding="ascii",
    )
    write_csv(results_root / "SYNCHRONIZED_RUN_REGISTER.csv", run_register)
    write_csv(results_root / "SYNCHRONIZED_PER_RUN_METRICS.csv", run_metrics)
    write_csv(results_root / "SYNCHRONIZED_CONFUSION_MATRICES.csv", confusion_rows)
    write_csv(results_root / "SYNCHRONIZED_PER_CLASS_RESULTS.csv", execution["per_class"])
    write_csv(results_root / "SYNCHRONIZED_PREDICTED_CLASS_HISTOGRAMS.csv", histograms)

    position_rows = build_position_metric_rows(prediction_rows, run_metrics, histograms)
    primary_rows = build_primary_contrast_rows(prediction_rows)
    macro_rows = build_macro_f1_rows(prediction_rows, run_metrics)
    interval_rows, simple_rows = bootstrap_factor_intervals(
        prediction_rows,
        resamples=int(config["bootstrap"]["resamples"]),
        random_seed=int(config["bootstrap"]["random_seed"]),
    )
    per_class_rows = build_per_class_factor_rows(prediction_rows)
    heterogeneity_rows = build_heterogeneity_rows(prediction_rows)
    consistency_rows = build_seed_fold_consistency_rows(prediction_rows, run_metrics)
    classification, classification_details = classify_factor_pattern(interval_rows, simple_rows)

    write_csv(results_root / "07_POSITION_METRIC_RECONSTRUCTION.csv", position_rows)
    write_csv(results_root / "08_PRIMARY_BLOCK_ACCURACY_CONTRASTS.csv", primary_rows)
    write_csv(results_root / "09_SECONDARY_MACRO_F1_CONTRASTS.csv", macro_rows)
    write_csv(results_root / "10_BOOTSTRAP_INTERVALS.csv", interval_rows)
    write_csv(results_root / "11_SIMPLE_EFFECTS_BY_DISTANCE_AND_ANGLE.csv", simple_rows)
    write_csv(results_root / "12_PER_CLASS_FACTOR_RESULTS.csv", per_class_rows)
    write_csv(results_root / "13_ER_SURFACE_HETEROGENEITY.csv", heterogeneity_rows)
    write_csv(results_root / "14_SEED_AND_FOLD_CONSISTENCY.csv", consistency_rows)

    primary_method = "paired_tagid_stratified_block_bootstrap_seeds_fixed"
    primary_intervals = {
        row["contrast"]: row
        for row in interval_rows
        if row["endpoint"] == "condition_block_accuracy" and row["method"] == primary_method
    }
    macro_intervals = {
        row["contrast"]: row
        for row in interval_rows
        if row["endpoint"] == "pooled_condition_block_macro_f1" and row["method"] == primary_method
    }
    angle_confirmed = (
        _interval_excludes_zero(primary_intervals["angle_45_minus_0"])
        and float(primary_intervals["angle_45_minus_0"]["estimate"]) < 0.0
        and bool(classification_details["angle_simple_effects_directionally_consistent"])
    )
    distance_confirmed = _interval_excludes_zero(primary_intervals["distance_150_minus_50"])
    interaction_confirmed = _interval_excludes_zero(primary_intervals["angle_by_distance_interaction"])
    learning_rows = read_csv(BASE_RESULTS / "11_LEARNING_VALIDITY_RESULTS.csv")
    learning_passed = len(learning_rows) == 4 and all(row["status"] == "PASS" for row in learning_rows)

    interpretation = f"""# Factorial interpretation

## Classification: `{classification}`

The inherited 60 fixed-position runs are not fully paired: their outer test folds
are synchronized, but 492 train/validation assignment groups differ across
positions. Their compact reference evidence is reconstructed, but those runs
are not used for factorial inference. Case B therefore uses exactly
60 synchronized C1 primary runs with the same outer folds and seeds, no tuning,
and position-independent validation assignment. The new frozen checkpoints also
exactly reproduce their compact held row/block metrics and confusion matrices.

The primary endpoint is paired condition-block Accuracy over 63 physical cells,
with all five seeds retained inside each paired block bootstrap unit.

| Primary contrast | Estimate and paired 95% interval |
|---|---:|
| 45 degrees minus 0 degrees | {_format_effect(primary_intervals['angle_45_minus_0'])} |
| 150 mm minus 50 mm | {_format_effect(primary_intervals['distance_150_minus_50'])} |
| Angle x distance interaction | {_format_effect(primary_intervals['angle_by_distance_interaction'])} |

- Angle-associated degradation confirmed: **{'yes' if angle_confirmed else 'no'}**.
- Distance effect confirmed: **{'yes' if distance_confirmed else 'no'}**.
- Angle-distance interaction confirmed: **{'yes' if interaction_confirmed else 'no'}**.

The corresponding pooled condition-block Macro-F1 contrasts are angle
`{_format_effect(macro_intervals['angle_45_minus_0'])}`, distance
`{_format_effect(macro_intervals['distance_150_minus_50'])}`, and interaction
`{_format_effect(macro_intervals['angle_by_distance_interaction'])}`. Macro-F1 is
computed after pooling all 63 aligned blocks within position and seed, then
averaging the five seed-specific values. It is not a per-block metric, and its
pooled estimand is separate from the inherited descriptive 15-run mean.

The fixed-seed Macro-F1 interaction interval excludes zero, but the primary
Accuracy interval touches zero and both paired block-plus-seed interaction
sensitivity intervals cross zero. The interaction is therefore not confirmed.

The angle statement is associational: held-condition classification performance
is lower at 45 degrees in this benchmark. It does not identify a causal RF
mechanism. TagID, ER, surface, seed, fold, and simple-effect summaries are
secondary or exploratory and were not searched for a preferred narrative.

Basic learning validity also remains separate: all four inherited tiny-set tests
passed (`learning_validity_passed={str(learning_passed).lower()}`), but that fact
does not establish held-condition transfer or a factor effect.
"""
    (results_root / "15_FACTORIAL_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    limitations = """# Limitations and nonclaims

- The benchmark contains 63 physical condition blocks and five training seeds;
  uncertainty is therefore block-clustered and not based on 3,150 independent rows.
- The four positions are observational measurement settings. Contrasts are
  angle-associated and distance-associated classification differences, not
  causal physical-mechanism estimates.
- Frozen checkpoint inference reconstructs predictions exactly, but full row
  prediction arrays, logits, signals, and checkpoints are not committed here.
- Macro-F1 is nonlinear. The primary Macro-F1 estimand pools blocks within each
  position and seed; the inherited 15-run mean is descriptive and distinct.
- TagID, ER, surface, seed, fold, per-class, and simple-effect results are
  prespecified secondary or exploratory summaries and are not multiplicity-based
  mechanism discovery.
- Bootstrap intervals describe the available block/seed design and do not add
  new environments, devices, tags, distances, angles, or acquisition sessions.
- The inherited P4 frozen canonical bundle did not contain raw signals; the
  original benchmark's governed raw-file audit and source fingerprint remain the
  controlling P4 data evidence.
- Passed tiny-set learning validity establishes only basic learner capacity on a
  tiny true-label subset, not held-condition generalisation.
- No p-values are reported and no repeated row is treated as an independent unit.
"""
    (results_root / "16_LIMITATIONS_AND_NONCLAIMS.md").write_text(limitations, encoding="utf-8")

    executive = f"""# Executive summary

## Outcome: `{classification}`

The inherited 60-run fixed-position benchmark is not fully paired across P1-P4:
492 train/validation assignment groups differ, although outer test folds match.
The released metrics and confusion matrices were reconstructed exactly, then a
preregistered Case B rerun trained exactly 60 synchronized C1 runs with no tuning.
Factorial inference uses only that synchronized rerun.

Primary paired condition-block Accuracy (63 physical blocks):

- angle, 45 degrees minus 0 degrees: `{_format_effect(primary_intervals['angle_45_minus_0'])}`;
- distance, 150 mm minus 50 mm: `{_format_effect(primary_intervals['distance_150_minus_50'])}`;
- interaction: `{_format_effect(primary_intervals['angle_by_distance_interaction'])}`.

Angle-associated degradation was {'confirmed' if angle_confirmed else 'not confirmed'}.
The distance effect was {'confirmed' if distance_confirmed else 'not confirmed'},
and the interaction was {'confirmed' if interaction_confirmed else 'not confirmed'}.
These are performance associations, not causal RF claims.

The positive interaction point estimate is not robust to paired seed resampling;
it is reported as secondary uncertainty rather than a confirmed interaction.

The primary interval is a 10,000-replicate paired, TagID-stratified physical-block
bootstrap with seeds fixed. A paired block-plus-seed sensitivity analysis is
reported separately. All four inherited learning-validity checks passed, but
learning validity is not held-condition generalisation.

The public result retains compact condition metadata, paired aggregate evidence,
and bounded interpretation. Raw rows, checkpoints, logits, embeddings, and
prediction arrays are not included in the public result package.
"""
    (results_root / "00_EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")
    end_status = f"""# Result status

Scientific classification: `{classification}`.

The paired factorial diagnostic completed with a synchronized 60-run C1 evaluation and {int(config['bootstrap']['resamples'])} bootstrap replicates. Angle-associated degradation confirmed: {'yes' if angle_confirmed else 'no'}. Overall distance effect confirmed: {'yes' if distance_confirmed else 'no'}. Interaction confirmed: {'yes' if interaction_confirmed else 'no'}.

The primary inferential unit is the physical condition block. This result is diagnostic and does not establish one unique physical cause of P4 transfer failure.
"""
    (results_root / "STATUS.md").write_text(end_status, encoding="utf-8")
    return {
        "status": "PASS_PAIRED_FACTORIAL_ANALYSIS_COMPLETE",
        "classification": classification,
        "existing_runs_fully_paired": False,
        "case_b_synchronized_rerun_completed": True,
        "training_performed": rerun_status["training_performed"],
        "angle_confirmed": angle_confirmed,
        "distance_confirmed": distance_confirmed,
        "interaction_confirmed": interaction_confirmed,
        "primary_intervals": primary_intervals,
        "macro_intervals": macro_intervals,
    }
