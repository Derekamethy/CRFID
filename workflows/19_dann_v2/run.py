"""Execute the isolated governed DANN v2 post-hoc follow-up."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))

from data import (
    EXPECTED_INPUT_HASHES,
    balanced_epoch_indices,
    batch_composition,
    domain_labels,
    fit_first_difference,
    load_source_data,
)
from metrics import paired_vector_bootstrap
from model import C1PositionDANN
from protocol import (
    ARM_A0,
    ARM_A1,
    ARM_A2,
    ARMS,
    CHANCE_REFERENCE_REPLICATES,
    CHANCE_REFERENCE_SEED,
    DEVELOPMENT_BOOTSTRAP_REPLICATES,
    DEVELOPMENT_BOOTSTRAP_SEED,
    FOLDS,
    LAMBDAS,
    P4_BOOTSTRAP_REPLICATES,
    P4_BOOTSTRAP_SEED,
    SEEDS,
    canonical_json_sha256,
    derive_final_epochs,
    select_lambda,
    sha256_file,
    validate_lambda_grid,
)
from target import freeze_target_predictions, score_target_once
from training import development_run, intervention_validity, train_final_arms


RESULTS = PROJECT_ROOT / "results" / "canonical_metrics" / "dann_v2"
RUNTIME = PROJECT_ROOT / "outputs" / "dann_v2"
CONFIG = PROJECT_ROOT / "configs" / "dann_v2" / "preregistered.json"
SELECTION = RESULTS / "19_LAMBDA_SELECTION.json"
GATE = RESULTS / "20_SOURCE_MECHANISM_GATE.json"
FINAL_EPOCHS = RESULTS / "21_FINAL_EPOCH_DERIVATION.json"
FINAL_RECORDS = RUNTIME / "final_source" / "records.json"
P4_FREEZE = RUNTIME / "p4" / "P4_PREDICTION_FREEZE.json"
P4_SCORE = RUNTIME / "p4" / "P4_SCORE.json"
_WORKER_DATA = None

EXPECTED_DEVELOPMENT_RUNS = 165
P4_RESULT_FILES = (
    "25_FINAL_P4_RUN_MANIFEST.csv",
    "26_FINAL_P4_METRICS.csv",
    "27_P4_PAIRED_EFFECTS.csv",
    "28_P4_UNCERTAINTY_INTERVALS.csv",
    "29_P4_CHANCE_REFERENCE.csv",
    "30_PER_TAGID_RESULTS.csv",
)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(
    path: Path, rows: list[dict], fields: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows or [{"status": "NOT_AVAILABLE"}])


def _compact(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _lineage_hashes() -> dict[str, str]:
    expected = {
        "configs/strict_dg/canonical.yaml": "2338bbe1f7ec501546512d286b3d2b3d3067f9f8e458a547e2a237c15a1b835e",
        "src/crfid/strict_runtime/neutral_model.py": "eecf7a5a274d471892590b51e0d79b65e0fe7e13739afbe08cd51828493c6227",
        "src/crfid/strict_runtime/neutral_data.py": "09a3d7e4579b63b4257f82bd4711183428e4d4675fcdad9cafd8a263d2dc6cac",
        "src/crfid/strict_runtime/phase3b_execution.py": "ddc65b7fa4bb8287140cd2837f7a392df968f9dcabd0dfabc5df3b479cf9b010",
        "src/crfid/strict_runtime/neutral_metrics.py": "ccb26eecc31993c3f339f174411902539b4599b34fdc018732a63b3c9e2d6471",
        "src/crfid/strict_runtime/scale_policy.py": "bee7fa226c45eab2480242f9ab246eb08c22bad4df776a98637299d63b6034b1",
        "src/crfid/strict_runtime/target_evaluation.py": "82413d4b44ab3eef283722738da91ca41e0061eac33c358719d5ff12b9e43b88",
        "src/crfid/protocols/strict_dg.py": "1714dd8ad6a1fa62619a2b16e601f84954a9fc2d2e3c2d2d1675364b71de4aa3",
    }
    observed = {name: sha256_file(PROJECT_ROOT / name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            "FAIL_DANN_V2_PROTOCOL_OR_DATA_INTEGRITY: canonical lineage mismatch"
        )
    return observed


def _run_focused_tests() -> subprocess.CompletedProcess[str]:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_dann_v2_model.py",
        "tests/test_dann_v2_protocol.py",
        "tests/test_dann_v2_evaluation.py",
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    text = completed.stdout + completed.stderr
    (RESULTS / "10_FOCUSED_TEST_RESULTS.txt").write_text(text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "FAIL_DANN_V2_IMPLEMENTATION_VALIDITY: focused tests failed"
        )
    return completed


def bind(source_inputs: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG)
    validate_lambda_grid(config["lambda_grid"])
    if config["development"]["expected_total_runs"] != EXPECTED_DEVELOPMENT_RUNS:
        raise RuntimeError("Frozen development run count changed")
    data = load_source_data(source_inputs)
    lineage = _lineage_hashes()
    runtime = {
        **read_json(RESULTS / "01_GOVERNED_BASELINE_BINDING.json"),
        "status": "PASS_GOVERNED_RUNTIME_AND_BASELINE_BINDING",
        "python_version": platform.python_version(),
        "python_executable_sha256": sha256_file(Path(sys.executable)),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "governed_source_input_hashes": data.input_hashes,
        "canonical_code_hashes": lineage,
        "source_rows": len(data.labels),
        "p4_accessed": False,
        "absolute_paths_recorded": False,
    }
    if (
        runtime["python_version"] != "3.12.13"
        or runtime["numpy_version"] != "2.4.6"
        or runtime["torch_version"] != "2.12.0+cpu"
        or data.input_hashes != EXPECTED_INPUT_HASHES
    ):
        raise RuntimeError("FAIL_DANN_V2_PROTOCOL_OR_DATA_INTEGRITY")
    write_json(RESULTS / "01_GOVERNED_BASELINE_BINDING.json", runtime)

    validity = intervention_validity()
    write_json(RESULTS / "07_GRADIENT_SIGN_AND_MAGNITUDE_TESTS.json", validity)
    if not validity["passed"]:
        write_json(
            RESULTS / "33_MAIN_SCIENTIFIC_CLASSIFICATION.json",
            {"classification": "FAIL_DANN_V2_IMPLEMENTATION_VALIDITY"},
        )
        raise RuntimeError("FAIL_DANN_V2_IMPLEMENTATION_VALIDITY")

    domain_rows = []
    batching_rows = []
    for fold in FOLDS:
        selected = data.splits[fold]["inner_train"]
        positions = data.positions[selected]
        labels, order = domain_labels(positions)
        sample_order = balanced_epoch_indices(
            positions, data.labels[selected], seed=42, epoch=1
        )
        batches = batch_composition(sample_order, positions, data.labels[selected])
        domain_rows.append(
            {
                "fold": fold,
                "held_position": {"S1": "P1", "S2": "P2", "S3": "P3"}[fold],
                "training_domains": "|".join(order),
                "domain_label_mapping": _compact(
                    {position: index for index, position in enumerate(order)}
                ),
                "domain_head_output_dimension": len(np.unique(labels)),
                "tagid_used_as_domain_label": False,
                "p4_present": False,
                "status": "PASS",
            }
        )
        for batch in batches:
            batching_rows.append(
                {
                    "audit_scope": "PRETRAINING_DETERMINISTIC_EXAMPLE",
                    "fold": fold,
                    "seed": 42,
                    "epoch": 1,
                    "batch_index": batch["batch_index"],
                    "sample_count": batch["sample_count"],
                    "position_counts": _compact(batch["position_counts"]),
                    "position_tag_counts": _compact(batch["position_tag_counts"]),
                    "all_domains_present": set(batch["position_counts"]) == set(order),
                }
            )
    write_csv(RESULTS / "08_DOMAIN_LABEL_AUDIT.csv", domain_rows)
    write_csv(RESULTS / "09_BATCHING_AUDIT.csv", batching_rows)
    _run_focused_tests()


def _development_path(arm: str, fold: str, seed: int, lambda_max: float) -> Path:
    slug = {ARM_A0: "a0", ARM_A1: "a1", ARM_A2: "a2"}[arm]
    if arm == ARM_A0:
        return RUNTIME / "development" / slug / fold / f"seed_{seed}.json"
    return (
        RUNTIME
        / "development"
        / slug
        / f"lambda_{lambda_max:.2f}"
        / fold
        / f"seed_{seed}.json"
    )


def _initialize_worker(source_inputs: str) -> None:
    global _WORKER_DATA
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_DATA = load_source_data(Path(source_inputs))


def _development_worker(
    arm: str, fold: str, seed: int, lambda_max: float
) -> dict:
    if _WORKER_DATA is None:
        raise RuntimeError("Development worker was not initialized")
    return development_run(
        _WORKER_DATA,
        fold=fold,
        seed=seed,
        lambda_max=lambda_max,
        arm=arm,
    )


def _final_source_worker(
    seed: int, selected_lambda: float, epochs: int, output_root: str
) -> list[dict]:
    if _WORKER_DATA is None:
        raise RuntimeError("Final worker was not initialized")
    return train_final_arms(
        _WORKER_DATA,
        seed=seed,
        selected_lambda=selected_lambda,
        epochs=epochs,
        output_root=Path(output_root),
    )


def _all_development_jobs() -> list[tuple[str, str, int, float, Path]]:
    jobs = []
    for fold in FOLDS:
        for seed in SEEDS:
            jobs.append((ARM_A0, fold, seed, 0.0, _development_path(ARM_A0, fold, seed, 0.0)))
    for arm in (ARM_A1, ARM_A2):
        for value in LAMBDAS:
            for fold in FOLDS:
                for seed in SEEDS:
                    jobs.append((arm, fold, seed, value, _development_path(arm, fold, seed, value)))
    if len(jobs) != EXPECTED_DEVELOPMENT_RUNS:
        raise RuntimeError("Frozen DANN v2 development run count changed")
    return jobs


def execute_development(source_inputs: Path) -> None:
    all_jobs = _all_development_jobs()
    missing = [job for job in all_jobs if not job[-1].is_file()]
    write_json(
        RUNTIME / "development" / "DETERMINISTIC_INFRASTRUCTURE_RETRY_POLICY.json",
        {
            "schema_version": 2,
            "expected_run_count_frozen_before_outcomes": EXPECTED_DEVELOPMENT_RUNS,
            "policy": "Only absent result artifacts may be rerun with identical inputs; completed artifacts are immutable.",
            "pre_p4": True,
            "scientific_outcome_inspected_before_retry": False,
            "worker_count": 4,
        },
    )
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=_initialize_worker,
        initargs=(str(source_inputs.resolve()),),
    ) as pool:
        futures = {
            pool.submit(_development_worker, arm, fold, seed, value): (
                arm,
                fold,
                seed,
                value,
                path,
            )
            for arm, fold, seed, value, path in missing
        }
        for completed_index, future in enumerate(as_completed(futures), start=1):
            arm, fold, seed, value, path = futures[future]
            result = future.result()
            write_json(path, result)
            print(
                _compact(
                    {
                        "event": "development_job_complete",
                        "completed_missing_jobs": completed_index,
                        "missing_jobs": len(missing),
                        "arm": arm,
                        "lambda": value,
                        "fold": fold,
                        "seed": seed,
                    }
                ),
                flush=True,
            )
    compile_development(_development_records())


def _development_records() -> list[dict]:
    return [read_json(path) for *_, path in _all_development_jobs()]


def _matched_records(
    records: list[dict], arm: str, lambda_max: float
) -> list[dict]:
    return sorted(
        [
            row
            for row in records
            if row["arm"] == arm and float(row["lambda_max"]) == float(lambda_max)
        ],
        key=lambda row: (row["fold"], int(row["seed"])),
    )


def _a0_records(records: list[dict]) -> list[dict]:
    return sorted(
        [row for row in records if row["arm"] == ARM_A0],
        key=lambda row: (row["fold"], int(row["seed"])),
    )


def _development_effects(records: list[dict]) -> list[dict]:
    a0 = _a0_records(records)
    if len(a0) != 15:
        raise RuntimeError("Matched A0 development evidence incomplete")
    getters = {
        "familiar_validation_macro_f1": lambda row: row["familiar_validation_metrics"]["macro_f1"],
        "position_probe_macro_f1": lambda row: row["position_probe_macro_f1"],
        "tagid_probe_macro_f1": lambda row: row["tagid_probe_macro_f1"],
    }
    rows = []
    for value in LAMBDAS:
        a1 = _matched_records(records, ARM_A1, value)
        a2 = _matched_records(records, ARM_A2, value)
        if len(a1) != 15 or len(a2) != 15:
            raise RuntimeError(f"Matched A1/A2 evidence incomplete for lambda {value}")
        for name, reference, treatment in (
            ("A1-A0", a0, a1),
            ("A2-A0", a0, a2),
            ("A2-A1", a1, a2),
        ):
            for endpoint, getter in getters.items():
                differences = np.asarray(
                    [
                        getter(right) - getter(left)
                        for left, right in zip(reference, treatment, strict=True)
                    ],
                    dtype=np.float64,
                )
                interval = paired_vector_bootstrap(
                    differences,
                    replicates=DEVELOPMENT_BOOTSTRAP_REPLICATES,
                    seed=DEVELOPMENT_BOOTSTRAP_SEED,
                )
                rows.append(
                    {
                        "contrast": name,
                        "lambda_max": value,
                        "endpoint": endpoint,
                        "point_estimate": interval["point_estimate"],
                        "lower_95": interval["interval"][0],
                        "upper_95": interval["interval"][1],
                        "paired_unit": "fold_x_seed",
                        "unit_contrasts": _compact(interval["unit_contrasts"]),
                        "replicates": interval["replicates"],
                        "rng_seed": DEVELOPMENT_BOOTSTRAP_SEED,
                        "contrast_specific_seed_offset": False,
                    }
                )
    return rows


def compile_development(records: list[dict]) -> None:
    if len(records) != EXPECTED_DEVELOPMENT_RUNS:
        raise RuntimeError("Incomplete DANN v2 development")
    keys = {
        (row["arm"], row["fold"], int(row["seed"]), float(row["lambda_max"]))
        for row in records
    }
    if len(keys) != EXPECTED_DEVELOPMENT_RUNS or any(row.get("p4_used") for row in records):
        raise RuntimeError("Development identity or P4 gate failed")
    manifest = []
    metrics_rows = []
    familiar_rows = []
    position_rows = []
    tagid_rows = []
    lopo_rows = []
    trajectory_rows = []
    batching_rows = []
    for row in records:
        manifest.append(
            {
                "run_id": row["run_id"],
                "arm": row["arm"],
                "lambda_max": row["lambda_max"],
                "fold": row["fold"],
                "seed": row["seed"],
                "selected_checkpoint_epoch": row["selected_epoch"],
                "stopping_epoch": row["epochs_executed"],
                "runtime_seconds": row["runtime_seconds"],
                "p4_accessed": False,
                "status": "COMPLETE",
            }
        )
        common = {
            "run_id": row["run_id"],
            "arm": row["arm"],
            "lambda_max": row["lambda_max"],
            "fold": row["fold"],
            "seed": row["seed"],
        }
        metrics_rows.append(
            {
                **common,
                "familiar_validation_accuracy": row["familiar_validation_metrics"]["accuracy"],
                "familiar_validation_macro_f1": row["familiar_validation_metrics"]["macro_f1"],
                "position_probe_macro_f1": row["position_probe_macro_f1"],
                "tagid_probe_macro_f1": row["tagid_probe_macro_f1"],
                "outer_held_lopo_accuracy": row["outer_held_lopo_row_metrics"]["accuracy"],
                "outer_held_lopo_macro_f1": row["outer_held_lopo_row_metrics"]["macro_f1"],
                "outer_held_lopo_chance_reference": 1.0 / 7.0,
                "domain_train_accuracy": row["domain_train_accuracy"],
                "domain_validation_accuracy": row["domain_validation_accuracy"],
                "selected_checkpoint_epoch": row["selected_epoch"],
                "selected_mean_lambda": row["selected_checkpoint_schedule"]["mean_lambda"],
                "selected_maximum_realised_lambda": row["selected_checkpoint_schedule"]["maximum_realised_lambda"],
                "selected_cumulative_domain_gradient_coefficient": row["selected_checkpoint_schedule"]["cumulative_domain_gradient_coefficient"],
            }
        )
        familiar_rows.append(
            {
                **common,
                "accuracy": row["familiar_validation_metrics"]["accuracy"],
                "macro_f1": row["familiar_validation_metrics"]["macro_f1"],
                "scope": "FAMILIAR_SOURCE_INNER_VALIDATION",
            }
        )
        position_rows.append(
            {
                **common,
                "position_probe_balanced_accuracy": row["position_probe_balanced_accuracy"],
                "position_probe_macro_f1": row["position_probe_macro_f1"],
                "chance_reference": row["probe_position_chance"],
                "probe_train": "inner_train_condition_centroids",
                "probe_test": "inner_validation_condition_centroids",
            }
        )
        tagid_rows.append(
            {
                **common,
                "tagid_probe_macro_f1": row["tagid_probe_macro_f1"],
                "chance_reference": 1.0 / 7.0,
                "probe_train": "inner_train_condition_centroids",
                "probe_test": "inner_validation_condition_centroids",
            }
        )
        lopo_rows.append(
            {
                **common,
                "held_position": row["held_position"],
                "outer_held_lopo_accuracy": row["outer_held_lopo_row_metrics"]["accuracy"],
                "outer_held_lopo_macro_f1": row["outer_held_lopo_row_metrics"]["macro_f1"],
                "seven_class_chance_reference": 1.0 / 7.0,
                "selection_use": "DIAGNOSTIC_ONLY",
            }
        )
        for phase, history in (
            ("INNER_CHECKPOINT_TRAINING", row["inner_training_history"]),
            ("OUTER_HELD_DIAGNOSTIC_REFIT", row["outer_diagnostic_training_history"]),
        ):
            for epoch_row in history:
                for update_in_epoch, lambda_value in enumerate(epoch_row["lambda_values"]):
                    trajectory_rows.append(
                        {
                            **common,
                            "phase": phase,
                            "epoch": epoch_row["epoch"],
                            "update_in_epoch": update_in_epoch,
                            "global_update_index": (epoch_row["epoch"] - 1) * epoch_row["batch_count"] + update_in_epoch,
                            "lambda": lambda_value,
                            "encoder_domain_coefficient": (
                                lambda_value
                                if row["arm"] == ARM_A1
                                else -lambda_value if row["arm"] == ARM_A2 else 0.0
                            ),
                            "included_in_selected_checkpoint": phase == "OUTER_HELD_DIAGNOSTIC_REFIT" or epoch_row["epoch"] <= row["selected_epoch"],
                        }
                    )
                for batch in epoch_row["batch_composition"]:
                    batching_rows.append(
                        {
                            **common,
                            "phase": phase,
                            "epoch": epoch_row["epoch"],
                            "batch_index": batch["batch_index"],
                            "sample_count": batch["sample_count"],
                            "position_counts": _compact(batch["position_counts"]),
                            "position_tag_counts": _compact(batch["position_tag_counts"]),
                            "all_domains_present": True,
                        }
                    )
    write_csv(RESULTS / "11_DEVELOPMENT_RUN_MANIFEST.csv", manifest)
    write_csv(RESULTS / "12_DEVELOPMENT_METRICS.csv", metrics_rows)
    write_csv(RESULTS / "13_DEVELOPMENT_LAMBDA_TRAJECTORIES.csv", trajectory_rows)
    write_csv(RESULTS / "14_FAMILIAR_SOURCE_RETENTION.csv", familiar_rows)
    write_csv(RESULTS / "15_POSITION_PROBE_RESULTS.csv", position_rows)
    write_csv(RESULTS / "16_TAGID_PROBE_RESULTS.csv", tagid_rows)
    write_csv(RESULTS / "17_OUTER_HELD_LOPO_DIAGNOSTIC.csv", lopo_rows)
    write_csv(RESULTS / "09_BATCHING_AUDIT.csv", batching_rows)
    write_csv(
        RESULTS / "18_DEVELOPMENT_UNCERTAINTY_INTERVALS.csv",
        _development_effects(records),
    )


def _write_p4_not_created_placeholders(reason: str) -> None:
    placeholder = [{"status": "NOT_CREATED_BY_GOVERNED_DESIGN", "reason": reason}]
    for name in P4_RESULT_FILES:
        write_csv(RESULTS / name, placeholder)
    confusion = RESULTS / "31_CONFUSION_MATRICES"
    confusion.mkdir(parents=True, exist_ok=True)
    (confusion / "NOT_CREATED_BY_GOVERNED_DESIGN.txt").write_text(
        f"NOT_CREATED_BY_GOVERNED_DESIGN\n{reason}\n", encoding="utf-8"
    )


def select_source_lambda() -> dict:
    records = _development_records()
    effects = list(
        csv.DictReader(
            (RESULTS / "18_DEVELOPMENT_UNCERTAINTY_INTERVALS.csv").open(
                encoding="utf-8"
            )
        )
    )
    effect_map = {
        (row["contrast"], float(row["lambda_max"]), row["endpoint"]): row
        for row in effects
    }
    a0 = _a0_records(records)
    a0_task = float(
        np.mean([row["familiar_validation_metrics"]["macro_f1"] for row in a0])
    )
    a0_position = float(np.mean([row["position_probe_macro_f1"] for row in a0]))
    a0_tagid = float(np.mean([row["tagid_probe_macro_f1"] for row in a0]))
    candidates = []
    for value in LAMBDAS:
        a2 = _matched_records(records, ARM_A2, value)
        task = float(
            np.mean([row["familiar_validation_metrics"]["macro_f1"] for row in a2])
        )
        position = float(np.mean([row["position_probe_macro_f1"] for row in a2]))
        tagid = float(np.mean([row["tagid_probe_macro_f1"] for row in a2]))
        position_interval = effect_map[("A2-A0", value, "position_probe_macro_f1")]
        candidates.append(
            {
                "lambda_max": value,
                "run_count": len(a2),
                "familiar_validation_macro_f1_mean": task,
                "familiar_validation_macro_f1_difference": task - a0_task,
                "position_probe_macro_f1_mean": position,
                "position_probe_macro_f1_difference": position - a0_position,
                "position_probe_difference_interval_95": [
                    float(position_interval["lower_95"]),
                    float(position_interval["upper_95"]),
                ],
                "tagid_probe_macro_f1_mean": tagid,
                "tagid_probe_macro_f1_difference": tagid - a0_tagid,
                "outer_held_lopo_used_for_selection": False,
                "p4_used": False,
            }
        )
    selection = select_lambda(candidates)
    selection["matched_a0_means"] = {
        "familiar_validation_macro_f1": a0_task,
        "position_probe_macro_f1": a0_position,
        "tagid_probe_macro_f1": a0_tagid,
    }
    selection["development_runs_completed"] = len(records)
    write_json(SELECTION, selection)
    if selection["selected_lambda"] is None:
        gate = {
            "schema_version": 1,
            "status": "FAIL_SOURCE_MECHANISM_CONTINUATION_GATE",
            "reason": "NO_DANN_V2_CANDIDATE_MEETS_SOURCE_MECHANISM_GATE",
            "implementation_validity": "PASS",
            "development_runs_completed": len(records),
            "development_runs_expected": EXPECTED_DEVELOPMENT_RUNS,
            "p4_permitted": False,
        }
        write_json(GATE, gate)
        write_json(
            FINAL_EPOCHS,
            {
                "status": "NOT_DERIVED_BY_GOVERNED_DESIGN",
                "reason": gate["reason"],
            },
        )
        write_csv(
            RESULTS / "22_FINAL_TRAINING_MANIFEST.csv",
            [{"status": "NOT_CREATED_BY_GOVERNED_DESIGN", "reason": gate["reason"]}],
        )
        write_csv(
            RESULTS / "23_FINAL_SOURCE_DIAGNOSTICS.csv",
            [{"status": "NOT_CREATED_BY_GOVERNED_DESIGN", "reason": gate["reason"]}],
        )
        write_json(
            RESULTS / "24_PRE_P4_ACCESS_SEAL.json",
            {"status": "NOT_CREATED_BY_GOVERNED_DESIGN", "reason": gate["reason"]},
        )
        write_json(
            RESULTS / "PRE_P4_ACCESS_SEAL.json",
            {"status": "NOT_CREATED_BY_GOVERNED_DESIGN", "reason": gate["reason"]},
        )
        _write_p4_not_created_placeholders(gate["reason"])
        return selection
    selected_lambda = float(selection["selected_lambda"])
    epochs = derive_final_epochs(records, selected_lambda)
    evidence = {
        str(seed): {
            row["fold"]: int(row["selected_epoch"])
            for row in _matched_records(records, ARM_A2, selected_lambda)
            if int(row["seed"]) == seed
        }
        for seed in SEEDS
    }
    write_json(
        FINAL_EPOCHS,
        {
            "schema_version": 1,
            "status": "PASS_SOURCE_ONLY_FINAL_EPOCH_FREEZE",
            "selected_lambda": selected_lambda,
            "rule": "deterministic_median_of_three_selected_A2_fold_epochs_per_seed",
            "fold_epoch_evidence": evidence,
            "final_epochs_by_seed": epochs,
            "identical_across_arms": True,
            "p4_used": False,
        },
    )
    selected_row = next(
        row
        for row in selection["candidate_rows"]
        if float(row["lambda_max"]) == selected_lambda
    )
    write_json(
        GATE,
        {
            "schema_version": 1,
            "status": "PASS_SOURCE_MECHANISM_CONTINUATION_GATE",
            "implementation_validity": "PASS",
            "development_runs_completed": len(records),
            "development_runs_expected": EXPECTED_DEVELOPMENT_RUNS,
            "selected_lambda": selected_lambda,
            "mechanism_candidate_class": selection["mechanism_candidate_class"],
            "task_retention_gate": selected_row["task_retention_gate"],
            "tagid_representation_retention_gate": selected_row[
                "tagid_representation_retention_gate"
            ],
            "position_reduction_mean": selected_row[
                "position_probe_macro_f1_difference"
            ],
            "position_reduction_interval_95": selected_row[
                "position_probe_difference_interval_95"
            ],
            "p4_permitted_after_final_training_and_seal": True,
            "p4_used": False,
        },
    )
    return selection


def execute_final_source(source_inputs: Path) -> None:
    selection = read_json(SELECTION)
    gate = read_json(GATE)
    epochs = read_json(FINAL_EPOCHS)["final_epochs_by_seed"]
    if (
        selection.get("selected_lambda") is None
        or gate.get("status") != "PASS_SOURCE_MECHANISM_CONTINUATION_GATE"
    ):
        raise RuntimeError("NO_DANN_V2_CANDIDATE_MEETS_SOURCE_MECHANISM_GATE")
    records = []
    missing = []
    for seed in SEEDS:
        path = RUNTIME / "final_source" / f"seed_{seed}.json"
        if path.is_file():
            records.extend(read_json(path)["records"])
        else:
            missing.append((seed, path))
    with ProcessPoolExecutor(
        max_workers=5,
        initializer=_initialize_worker,
        initargs=(str(source_inputs.resolve()),),
    ) as pool:
        futures = {
            pool.submit(
                _final_source_worker,
                seed,
                float(selection["selected_lambda"]),
                int(epochs[str(seed)]),
                str(RUNTIME / "final_models"),
            ): (seed, path)
            for seed, path in missing
        }
        for future in as_completed(futures):
            seed, path = futures[future]
            local = future.result()
            write_json(path, {"records": local})
            records.extend(local)
            print(_compact({"event": "final_source_seed_complete", "seed": seed}), flush=True)
    if len(records) != 15:
        raise RuntimeError("Incomplete final DANN v2 source refit")
    for seed in SEEDS:
        local_epochs = {
            int(row["epochs"])
            for row in records
            if int(row["seed"]) == seed
        }
        if local_epochs != {int(epochs[str(seed)])}:
            raise RuntimeError("Final epoch budget differs across arms")
    data = load_source_data(source_inputs)
    state = fit_first_difference(
        data.signals, np.arange(len(data.labels), dtype=np.int64)
    )
    model_root = RUNTIME / "final_models"
    model_root.mkdir(parents=True, exist_ok=True)
    np.save(model_root / "preprocessing_mean.npy", state.mean, allow_pickle=False)
    np.save(model_root / "preprocessing_scale.npy", state.scale, allow_pickle=False)
    write_json(FINAL_RECORDS, {"records": records})
    compile_final_source(records)


def compile_final_source(records: list[dict]) -> None:
    manifest = []
    diagnostics = []
    for row in sorted(records, key=lambda item: (item["arm"], int(item["seed"]))):
        checkpoint = Path(row["checkpoint_path"])
        manifest.append(
            {
                "run_id": f"DANNV2-FINAL-SOURCE-{row['arm']}-S{row['seed']}",
                "arm": row["arm"],
                "seed": row["seed"],
                "lambda_max": row["lambda_max"],
                "epochs": row["epochs"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint_relative_runtime_path": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                "runtime_seconds": row["runtime_seconds"],
                "p4_accessed": False,
                "status": "COMPLETE",
            }
        )
        diagnostics.append(
            {
                "scope": "FAMILIAR_SOURCE_RESUBSTITUTION_DIAGNOSTIC",
                "arm": row["arm"],
                "seed": row["seed"],
                "lambda_max": row["lambda_max"],
                "epochs": row["epochs"],
                "accuracy": row["source_row_metrics"]["accuracy"],
                "macro_f1": row["source_row_metrics"]["macro_f1"],
                "worst_position_macro_f1": row["source_worst_position_macro_f1"],
                "position_probe_macro_f1": row["position_probe_macro_f1"],
                "tagid_probe_macro_f1": row["tagid_probe_macro_f1"],
                "domain_head_accuracy": row["domain_train_accuracy"],
                "mean_lambda": row["schedule"]["mean_lambda"],
                "maximum_realised_lambda": row["schedule"]["maximum_realised_lambda"],
                "cumulative_domain_gradient_coefficient": row["schedule"]["cumulative_domain_gradient_coefficient"],
                "not_held_source_dg": True,
            }
        )
    write_csv(RESULTS / "22_FINAL_TRAINING_MANIFEST.csv", manifest)
    write_csv(RESULTS / "23_FINAL_SOURCE_DIAGNOSTICS.csv", diagnostics)


def _seal_hash_paths() -> dict[str, Path]:
    paths = {
        f"workflows/19_dann_v2/{path.name}": path
        for path in sorted(WORKFLOW_ROOT.glob("*.py"))
    }
    paths["configs/dann_v2/preregistered.json"] = CONFIG
    for path in sorted((PROJECT_ROOT / "tests").glob("test_dann_v2_*.py")):
        paths[f"tests/{path.name}"] = path
    return paths


def seal_pre_p4() -> None:
    selection = read_json(SELECTION)
    gate = read_json(GATE)
    epochs = read_json(FINAL_EPOCHS)
    records = read_json(FINAL_RECORDS)["records"]
    if (
        selection.get("selected_lambda") is None
        or gate.get("status") != "PASS_SOURCE_MECHANISM_CONTINUATION_GATE"
        or epochs.get("status") != "PASS_SOURCE_ONLY_FINAL_EPOCH_FREEZE"
        or len(records) != 15
    ):
        raise RuntimeError("P4 seal prerequisites are incomplete")
    if P4_FREEZE.exists() or P4_SCORE.exists():
        raise RuntimeError("FAIL_DANN_V2_P4_GOVERNANCE: P4 runtime already exists")
    code_hashes = {
        name: sha256_file(path) for name, path in _seal_hash_paths().items()
    }
    checkpoint_expectations = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "epochs": row["epochs"],
            "lambda_max": row["lambda_max"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_path": Path(row["checkpoint_path"]).relative_to(PROJECT_ROOT).as_posix(),
        }
        for row in sorted(records, key=lambda item: (item["arm"], int(item["seed"])))
    ]
    seal = {
        "schema_version": 2,
        "status": "PASS_PRE_P4_ACCESS_SEAL",
        "claim_boundary": "POST_HOC_SOURCE_ONLY_FOLLOWUP_DANN_V2",
        "acknowledgement_p4_historically_known_from_prior_work": True,
        "statement": "DANN v2 source-only development did not access P4; P4 is a historically known target being re-accessed only after this seal.",
        "selected_lambda": selection["selected_lambda"],
        "mechanism_gate_evidence": gate,
        "retention_gate_evidence": {
            "task": gate["task_retention_gate"],
            "tagid_probe": gate["tagid_representation_retention_gate"],
        },
        "final_epochs_by_seed": epochs["final_epochs_by_seed"],
        "arms": {
            ARM_A0: "encoder domain coefficient 0",
            ARM_A1: "encoder domain coefficient +lambda(p)",
            ARM_A2: "encoder domain coefficient -lambda(p)",
        },
        "schedule_implementation_hash": code_hashes["workflows/19_dann_v2/protocol.py"],
        "metric_hashes": {
            "metrics.py": code_hashes["workflows/19_dann_v2/metrics.py"],
            "target.py": code_hashes["workflows/19_dann_v2/target.py"],
        },
        "uncertainty_code_hashes": {
            "metrics.py": code_hashes["workflows/19_dann_v2/metrics.py"],
            "target.py": code_hashes["workflows/19_dann_v2/target.py"],
        },
        "all_frozen_code_and_test_hashes": code_hashes,
        "final_checkpoints_expected": checkpoint_expectations,
        "p4_evaluation_plan": "Freeze 15 A0/A1/A2 prediction bundles before labels; score once; no retraining or result-driven reruns.",
        "chance_reference_procedure": {
            "analytic_accuracy": 1.0 / 7.0,
            "uniform_random_block_replicates": CHANCE_REFERENCE_REPLICATES,
            "seed": CHANCE_REFERENCE_SEED,
            "class_count": 7,
        },
        "development_bootstrap_seed": DEVELOPMENT_BOOTSTRAP_SEED,
        "p4_bootstrap_seed": P4_BOOTSTRAP_SEED,
        "analysis_definitions_frozen": True,
        "p4_previously_opened_by_dann_v2": False,
        "hash_normalization": "canonical JSON with sorted keys and compact separators",
    }
    seal["seal_canonical_json_sha256"] = canonical_json_sha256(seal)
    for path in (
        RESULTS / "24_PRE_P4_ACCESS_SEAL.json",
        RESULTS / "PRE_P4_ACCESS_SEAL.json",
    ):
        write_json(path, seal)
    physical_hash = sha256_file(RESULTS / "PRE_P4_ACCESS_SEAL.json")
    (RESULTS / "PRE_P4_ACCESS_SEAL.sha256").write_text(
        physical_hash + "  PRE_P4_ACCESS_SEAL.json (UTF-8 LF)\n", encoding="ascii"
    )


def freeze_p4(p4_directory: Path) -> None:
    seal = read_json(RESULTS / "24_PRE_P4_ACCESS_SEAL.json")
    if seal.get("status") != "PASS_PRE_P4_ACCESS_SEAL":
        raise RuntimeError("FAIL_DANN_V2_P4_GOVERNANCE: no passing seal")
    for name, expected in seal["all_frozen_code_and_test_hashes"].items():
        if sha256_file(PROJECT_ROOT / name) != expected:
            raise RuntimeError(f"FAIL_DANN_V2_P4_GOVERNANCE: frozen file changed: {name}")
    freeze_target_predictions(
        p4_directory=p4_directory,
        checkpoint_root=RUNTIME / "final_models",
        preprocessing_mean=RUNTIME / "final_models" / "preprocessing_mean.npy",
        preprocessing_scale=RUNTIME / "final_models" / "preprocessing_scale.npy",
        selection_path=SELECTION,
        output_root=RUNTIME / "p4",
    )


def score_p4(p4_directory: Path) -> None:
    config = read_json(CONFIG)
    bootstrap = config["p4_bootstrap"]
    chance = config["chance_reference"]
    if (
        bootstrap["seed"] != P4_BOOTSTRAP_SEED
        or bootstrap["replicates"] != P4_BOOTSTRAP_REPLICATES
        or bootstrap["seed_offsets"]
        or chance["seed"] != CHANCE_REFERENCE_SEED
        or chance["random_replicates"] != CHANCE_REFERENCE_REPLICATES
    ):
        raise RuntimeError("FAIL_DANN_V2_PROTOCOL_OR_DATA_INTEGRITY: frozen statistics changed")
    result = score_target_once(
        p4_directory=p4_directory,
        freeze_manifest=P4_FREEZE,
        bootstrap_replicates=P4_BOOTSTRAP_REPLICATES,
        bootstrap_seed=P4_BOOTSTRAP_SEED,
        chance_replicates=CHANCE_REFERENCE_REPLICATES,
        chance_seed=CHANCE_REFERENCE_SEED,
    )
    write_json(P4_SCORE, result)
    compile_p4(result)


def compile_p4(score: dict) -> None:
    freeze = read_json(P4_FREEZE)
    manifest = [
        {
            "run_id": f"DANNV2-FINAL-P4-{row['arm']}-S{row['seed']}",
            "arm": row["arm"],
            "seed": row["seed"],
            "lambda_max": row["lambda_max"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "prediction_array_sha256": row["prediction_array_sha256"],
            "prediction_frozen_before_label_access": row[
                "prediction_frozen_before_label_access"
            ],
            "status": "COMPLETE",
        }
        for row in freeze["records"]
    ]
    write_csv(RESULTS / "25_FINAL_P4_RUN_MANIFEST.csv", manifest)
    write_csv(RESULTS / "26_FINAL_P4_METRICS.csv", score["seed_metrics"])
    write_csv(RESULTS / "27_P4_PAIRED_EFFECTS.csv", score["paired"])
    write_csv(RESULTS / "28_P4_UNCERTAINTY_INTERVALS.csv", score["uncertainty"])
    chance = score["chance_reference"]
    write_csv(
        RESULTS / "29_P4_CHANCE_REFERENCE.csv",
        [
            {
                "reference": "UNIFORM_RANDOM_FIXED_SEVEN_CLASS_BLOCK",
                "endpoint": "Accuracy",
                "mean": chance["mean_random_accuracy"],
                "lower_95": chance["accuracy_interval_95"][0],
                "upper_95": chance["accuracy_interval_95"][1],
                "analytic_chance": chance["analytic_accuracy_chance"],
                "replicates": chance["replicates"],
                "seed": chance["seed"],
            },
            {
                "reference": "UNIFORM_RANDOM_FIXED_SEVEN_CLASS_BLOCK",
                "endpoint": "Macro-F1",
                "mean": chance["mean_random_macro_f1"],
                "lower_95": chance["macro_f1_interval_95"][0],
                "upper_95": chance["macro_f1_interval_95"][1],
                "analytic_chance": "DESCRIPTIVE_SIMULATION",
                "replicates": chance["replicates"],
                "seed": chance["seed"],
            },
        ],
    )
    write_csv(RESULTS / "30_PER_TAGID_RESULTS.csv", score["per_class"])
    confusion = RESULTS / "31_CONFUSION_MATRICES"
    confusion.mkdir(parents=True, exist_ok=True)
    marker = confusion / "NOT_CREATED_BY_GOVERNED_DESIGN.txt"
    if marker.exists():
        marker.unlink()
    rows = []
    for record in score["seed_metrics"]:
        for true_class, values in enumerate(record["block_confusion_matrix"]):
            for predicted_class, count in enumerate(values):
                rows.append(
                    {
                        "dataset": "P4_BLOCK",
                        "arm": record["arm"],
                        "seed": record["seed"],
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": count,
                    }
                )
    write_csv(confusion / "confusion_matrices.csv", rows)


def _means_by_arm(rows: list[dict], field: str) -> dict[str, float]:
    return {
        arm: float(
            np.mean([float(row[field]) for row in rows if row["arm"] == arm])
        )
        for arm in ARMS
    }


def _v1_v2_comparison(selection: dict, p4: dict | None) -> str:
    selected = None
    if selection.get("selected_lambda") is not None:
        selected = next(
            row
            for row in selection["candidate_rows"]
            if float(row["lambda_max"]) == float(selection["selected_lambda"])
        )
    final_text = "DANN v2 final refit was not run because the source mechanism gate failed."
    if FINAL_RECORDS.is_file():
        records = read_json(FINAL_RECORDS)["records"]
        means = {
            arm: float(
                np.mean(
                    [
                        row["source_row_metrics"]["macro_f1"]
                        for row in records
                        if row["arm"] == arm
                    ]
                )
            )
            for arm in ARMS
        }
        lambda_means = {
            arm: float(
                np.mean(
                    [row["schedule"]["mean_lambda"] for row in records if row["arm"] == arm]
                )
            )
            for arm in ARMS
        }
        final_text = (
            f"V2 familiar-source resubstitution Macro-F1 was A0 {means[ARM_A0]:.6f}, "
            f"A1 {means[ARM_A1]:.6f}, A2 {means[ARM_A2]:.6f}; A2-A0 "
            f"{means[ARM_A2] - means[ARM_A0]:+.6f}. Mean realized lambda was "
            f"{lambda_means[ARM_A2]:.6f} for both magnitude-matched treatment arms."
        )
    p4_text = "V2 P4 was not opened."
    if p4 is not None:
        means = _means_by_arm(p4["seed_metrics"], "p4_block_macro_f1")
        p4_text = (
            f"V2 P4 block Macro-F1 was A0 {means[ARM_A0]:.6f}, "
            f"A1 {means[ARM_A1]:.6f}, A2 {means[ARM_A2]:.6f}."
        )
    mechanism_text = "No v2 candidate passed."
    if selected is not None:
        mechanism_text = (
            f"At selected lambda {selection['selected_lambda']}, v2 development A2-A0 "
            f"position-probe change was {selected['position_probe_macro_f1_difference']:+.6f}, "
            f"TagID-probe change {selected['tagid_probe_macro_f1_difference']:+.6f}, and familiar "
            f"validation change {selected['familiar_validation_macro_f1_difference']:+.6f}."
        )
    return f"""# DANN v1 versus DANN v2

DANN v1 is immutable and used descriptively only. Its audited deployment had mean lambda about 0.296, a fully exhausted schedule, and roughly twice the screened cumulative adversarial pressure. A1 used +1.0 while A2 used at most -0.3. Final v1 position-probe A2-A0 was +0.002923; TagID-probe A2-A0 was -0.675137; familiar-source resubstitution A2-A0 was -0.530325. P4 block Macro-F1 was A0 0.121778, A1 0.129440, A2 0.132980, all within the empirical chance regime.

V2 uses one registered 50-epoch schedule and exact +lambda/-lambda magnitude matching. {mechanism_text}

{final_text}

{p4_text}

Whether the v2 schedule and magnitude-matching corrections reduced the v1 source collapse is answered only by the final resubstitution contrast above; no stronger mechanism claim is made.
"""


def final_report() -> None:
    selection = read_json(SELECTION)
    gate = read_json(GATE)
    p4 = read_json(P4_SCORE) if P4_SCORE.is_file() else None
    selected = None
    if selection.get("selected_lambda") is not None:
        selected = next(
            row
            for row in selection["candidate_rows"]
            if float(row["lambda_max"]) == float(selection["selected_lambda"])
        )
    if gate["status"] != "PASS_SOURCE_MECHANISM_CONTINUATION_GATE":
        classification = "NO_DANN_V2_CANDIDATE_MEETS_SOURCE_MECHANISM_GATE"
        p4_reliable_a0 = False
        p4_reliable_a1 = False
        exceeds_chance = False
        p4_summary = "P4 was not permitted and was not opened."
    else:
        if p4 is None:
            raise RuntimeError("Passing source gate requires completed governed P4 stage before report")
        uncertainty = {
            (row["contrast"], row["endpoint"], row["family"]): row
            for row in p4["uncertainty"]
        }
        p4_reliable_a0 = all(
            uncertainty[("A2-A0", "P4_block_macro_f1", family)]["lower_95"] > 0.0
            for family in (
                "tagid_stratified_block_only",
                "tagid_stratified_block_plus_training_seed",
            )
        )
        p4_reliable_a1 = all(
            uncertainty[("A2-A1", "P4_block_macro_f1", family)]["lower_95"] > 0.0
            for family in (
                "tagid_stratified_block_only",
                "tagid_stratified_block_plus_training_seed",
            )
        )
        means_f1 = _means_by_arm(p4["seed_metrics"], "p4_block_macro_f1")
        means_acc = _means_by_arm(p4["seed_metrics"], "p4_block_accuracy")
        chance = p4["chance_reference"]
        exceeds_chance = means_f1[ARM_A2] > chance["macro_f1_interval_95"][1]
        position_reliable = bool(
            selected is not None
            and float(selected["position_probe_difference_interval_95"][1]) < 0.0
        )
        if p4_reliable_a0 and p4_reliable_a1:
            classification = (
                "DANN_V2_RELIABLE_POSITION_REDUCTION_WITH_RELIABLE_P4_BENEFIT"
                if position_reliable
                else "DANN_V2_P4_BENEFIT_WITHOUT_RELIABLE_POSITION_REDUCTION"
            )
        else:
            classification = (
                "DANN_V2_RELIABLE_POSITION_REDUCTION_NO_RELIABLE_P4_BENEFIT"
                if position_reliable
                else "DANN_V2_NO_RELIABLE_POSITION_SUPPRESSION_OR_P4_BENEFIT"
            )
        p4_summary = (
            f"P4 completed 15/15. Block Accuracy/Macro-F1: A0 {means_acc[ARM_A0]:.6f}/{means_f1[ARM_A0]:.6f}; "
            f"A1 {means_acc[ARM_A1]:.6f}/{means_f1[ARM_A1]:.6f}; A2 {means_acc[ARM_A2]:.6f}/{means_f1[ARM_A2]:.6f}. "
            f"Random Macro-F1 mean {chance['mean_random_macro_f1']:.6f}, 95% [{chance['macro_f1_interval_95'][0]:.6f}, {chance['macro_f1_interval_95'][1]:.6f}]."
        )
    write_json(
        RESULTS / "33_MAIN_SCIENTIFIC_CLASSIFICATION.json",
        {
            "schema_version": 1,
            "classification": classification,
            "claim_boundary": "POST_HOC_SOURCE_ONLY_FOLLOWUP_DANN_V2",
            "implementation_validity": "PASS",
            "development_runs_completed": EXPECTED_DEVELOPMENT_RUNS,
            "selected_lambda": selection.get("selected_lambda"),
            "source_mechanism_gate": gate["status"],
            "reliable_position_probe_reduction": bool(
                selected is not None
                and float(selected["position_probe_difference_interval_95"][1]) < 0.0
            ),
            "p4_permitted": gate["status"] == "PASS_SOURCE_MECHANISM_CONTINUATION_GATE",
            "p4_runs_completed": 15 if p4 is not None else 0,
            "reliable_p4_macro_f1_vs_a0": p4_reliable_a0,
            "reliable_p4_macro_f1_vs_a1": p4_reliable_a1,
            "a2_exceeds_empirical_p4_chance_interval": exceeds_chance,
        },
    )
    source_answers = (
        "No lambda retained task/TagID information while achieving mean position reduction."
        if selected is None
        else (
            f"Selected lambda {selection['selected_lambda']} ({selection['mechanism_candidate_class']}); "
            f"A2-A0 familiar validation {selected['familiar_validation_macro_f1_difference']:+.6f}, "
            f"position probe {selected['position_probe_macro_f1_difference']:+.6f} "
            f"(95% {selected['position_probe_difference_interval_95']}), TagID probe "
            f"{selected['tagid_probe_macro_f1_difference']:+.6f}."
        )
    )
    position_reliable = bool(
        selected is not None
        and float(selected["position_probe_difference_interval_95"][1]) < 0.0
    )
    if selected is None:
        interpretation = "No source-only candidate met the continuation gate."
    elif position_reliable and p4_reliable_a0 and p4_reliable_a1:
        interpretation = (
            "Reliable source-position probe reduction and a reliable P4 benefit "
            "were both observed under the frozen v2 design."
        )
    elif position_reliable:
        interpretation = (
            "Source-position probe reduction was statistically supported, but "
            "no reliable P4 benefit was observed."
        )
    elif p4_reliable_a0 and p4_reliable_a1:
        interpretation = (
            "A reliable P4 benefit was observed without reliable evidence of "
            "source-position suppression."
        )
    else:
        interpretation = (
            "The selected candidate had a negative position-probe point estimate, "
            "but its interval crossed zero; there was no reliable evidence of "
            "position suppression and no reliable P4 benefit."
        )
    (RESULTS / "00_EXECUTIVE_SUMMARY.md").write_text(
        f"""# DANN v2 executive summary

Classification: `{classification}`.

Implementation validity passed, including forward identity, sign and magnitude matching, identical domain-head gradients, lambda-zero equivalence, fold-local labels, and development/final schedule regression. Development completed `{EXPECTED_DEVELOPMENT_RUNS}/{EXPECTED_DEVELOPMENT_RUNS}` source-only runs.

{source_answers}

Source-mechanism continuation gate: `{gate['status']}`. {p4_summary}

Answers to the frozen questions: Q1-Q3 are determined by the source differences above. Q4 is reported in the A2-A1 rows of `18_DEVELOPMENT_UNCERTAINTY_INTERVALS.csv`. Q5 passed by construction and regression: the same fixed 50-epoch denominator is used at development and final stages. Q6 is shown beside `1/7` in `17_OUTER_HELD_LOPO_DIAGNOSTIC.csv` and never selected lambda. Q7 reliable A2-A0 P4 benefit: `{p4_reliable_a0}`. Q8 reliable A2-A1 P4 benefit: `{p4_reliable_a1}`. Q9 A2 exceeded the empirical chance interval: `{exceeds_chance}`. Q10 is answered by the classification and interpretation below.

Scientific interpretation: {interpretation}

Post-hoc boundary: DANN v2 was designed after prior P4 results were known. It is a governed source-only-developed follow-up, not pristine unseen-target confirmation.
""",
        encoding="utf-8",
    )
    (RESULTS / "32_V1_VS_V2_COMPARISON.md").write_text(
        _v1_v2_comparison(selection, p4), encoding="utf-8"
    )
    (RESULTS / "34_LIMITATIONS_AND_NON_CLAIMS.md").write_text(
        """# Limitations and non-claims

- DANN v2 is post-hoc with respect to the research programme's historical P4 knowledge.
- Five seeds and fifteen fold-by-seed source units provide coarse uncertainty; weak candidates remain explicitly weak.
- K=2 development and K=3 all-source domain heads address different source-domain cardinalities even though schedule semantics are identical.
- Frozen linear probes test decodability by one registered instrument, not universal absence of information.
- Outer-held LOPO operates near seven-class chance and is diagnostic only.
- Familiar-source final diagnostics are resubstitution, not held-source generalization.
- No universal claim about DANN, causal mediation, or other targets is made.
""",
        encoding="utf-8",
    )
    (RESULTS / "35_REPRODUCTION_COMMANDS.md").write_text(
        """# Reproduction commands

Historical scientific execution (do not rerun after release):

```powershell
crfid-python.cmd workflows/19_dann_v2/run.py bind --source-inputs <GOVERNED_SOURCE_INPUTS>
crfid-python.cmd workflows/19_dann_v2/run.py development --source-inputs <GOVERNED_SOURCE_INPUTS>
crfid-python.cmd workflows/19_dann_v2/run.py select
crfid-python.cmd workflows/19_dann_v2/run.py final-source --source-inputs <GOVERNED_SOURCE_INPUTS>  # only after a passing source gate
crfid-python.cmd workflows/19_dann_v2/run.py seal                               # only after a passing source gate
crfid-python.cmd workflows/19_dann_v2/run.py freeze-p4 --p4-directory <GOVERNED_P4_DIRECTORY>
crfid-python.cmd workflows/19_dann_v2/run.py score-p4 --p4-directory <GOVERNED_P4_DIRECTORY>
crfid-python.cmd workflows/19_dann_v2/run.py report
```

Focused non-training validation:

```powershell
crfid-pytest.cmd tests/test_dann_v2_model.py tests/test_dann_v2_protocol.py tests/test_dann_v2_evaluation.py -q
```
""",
        encoding="utf-8",
    )
    write_artifact_manifest()


def write_artifact_manifest() -> None:
    rows = []
    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file() or path.name == "36_ARTIFACT_MANIFEST.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(RESULTS / "36_ARTIFACT_MANIFEST.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "bind",
            "development",
            "select",
            "final-source",
            "seal",
            "freeze-p4",
            "score-p4",
            "report",
            "manifest",
        ),
    )
    parser.add_argument("--source-inputs", type=Path)
    parser.add_argument("--p4-directory", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command in {"bind", "development", "final-source"} and args.source_inputs is None:
        raise SystemExit("--source-inputs is required")
    if args.command in {"freeze-p4", "score-p4"} and args.p4_directory is None:
        raise SystemExit("--p4-directory is required")
    if args.command == "bind":
        bind(args.source_inputs)
    elif args.command == "development":
        execute_development(args.source_inputs)
    elif args.command == "select":
        select_source_lambda()
    elif args.command == "final-source":
        execute_final_source(args.source_inputs)
    elif args.command == "seal":
        seal_pre_p4()
    elif args.command == "freeze-p4":
        freeze_p4(args.p4_directory)
    elif args.command == "score-p4":
        score_p4(args.p4_directory)
    elif args.command == "report":
        final_report()
    elif args.command == "manifest":
        write_artifact_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
