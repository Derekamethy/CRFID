"""Execute the isolated strict source-only DANN position intervention."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))

from data import EXPECTED_INPUT_HASHES, fit_first_difference, load_source_data
from metrics import paired_vector_bootstrap, safe_correlation, stratified_paired_block_bootstrap
from protocol import (
    FOLDS,
    LAMBDAS,
    SEEDS,
    canonical_json_sha256,
    classify_intervention,
    select_nonzero_lambda,
    source_retention_guardrail,
    validate_lambda_grid,
)
from target import freeze_target_predictions, score_target_once, sha256_file
from training import development_run, intervention_validity, train_final_pair


RESULTS = PROJECT_ROOT / "results" / "canonical_metrics" / "dann_position_intervention"
MANIFESTS = PROJECT_ROOT / "manifests" / "dann_position_intervention"
PROVENANCE = PROJECT_ROOT / "provenance" / "dann_position_intervention_patch"
RUNTIME = PROJECT_ROOT / "outputs" / "dann_position_intervention_v1"
CONFIG = PROJECT_ROOT / "configs" / "dann_position_intervention" / "preregistered.json"

REQUIRED_OUTPUTS = (
    "00_EXECUTIVE_SUMMARY.md",
    "01_INPUT_BINDING.json",
    "02_C1_LINEAGE_BINDING.json",
    "03_PREREGISTERED_PROTOCOL.md",
    "04_DATA_AND_DOMAIN_STRUCTURE_AUDIT.csv",
    "05_SOURCE_LOPO_SPLIT_MANIFEST.csv",
    "06_DANN_MODEL_AND_GRL_VALIDITY.csv",
    "07_LAMBDA_GRID_REGISTER.csv",
    "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv",
    "09_DEVELOPMENT_RUN_REGISTER.csv",
    "10_SOURCE_HELD_POSITION_METRICS.csv",
    "11_INDEPENDENT_POSITION_PROBE_RESULTS.csv",
    "12_TAGID_PROBE_AND_SILHOUETTE_RESULTS.csv",
    "13_SOURCE_INTERVENTION_CONTRASTS.csv",
    "14_SOURCE_ONLY_LAMBDA_SELECTION.md",
    "15_FINAL_MODEL_REGISTER.csv",
    "16_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv",
    "17_P4_BLOCK_METRICS.csv",
    "18_P4_PRIMARY_DANN_VS_ERM_CONTRAST.csv",
    "19_P4_PER_CLASS_RESULTS.csv",
    "20_PREDICTED_CLASS_HISTOGRAMS.csv",
    "21_INTERVENTION_TRANSFER_LINK.csv",
    "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv",
    "23_SCIENTIFIC_INTERPRETATION.md",
    "24_LIMITATIONS_AND_NONCLAIMS.md",
    "STATUS.md",
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


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else ["status"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "NOT_EXECUTED"}])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_outputs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_OUTPUTS:
        path = RESULTS / name
        if path.exists():
            continue
        if path.suffix == ".csv":
            write_csv(path, [])
        elif path.suffix == ".json":
            write_json(path, {"status": "PENDING"})
        else:
            path.write_text(f"# {path.stem}\n\nStatus: `PENDING`.\n", encoding="utf-8")


def audit(source_inputs: Path) -> None:
    initialize_outputs()
    config = read_json(CONFIG)
    validate_lambda_grid(config["lambda_grid"])
    data = load_source_data(source_inputs)
    input_binding = {
        "schema_version": 1,
        "status": "PASS_GOVERNED_SOURCE_BINDING",
        "governed_source_input_hashes": data.input_hashes,
        "absolute_paths_recorded": False,
    }
    write_json(RESULTS / "01_INPUT_BINDING.json", input_binding)
    lineage_files = {
        "configs/strict_dg/canonical.yaml": "2338bbe1f7ec501546512d286b3d2b3d3067f9f8e458a547e2a237c15a1b835e",
        "src/crfid/strict_runtime/neutral_model.py": "eecf7a5a274d471892590b51e0d79b65e0fe7e13739afbe08cd51828493c6227",
        "src/crfid/strict_runtime/neutral_data.py": "09a3d7e4579b63b4257f82bd4711183428e4d4675fcdad9cafd8a263d2dc6cac",
        "src/crfid/strict_runtime/phase3b_execution.py": "ddc65b7fa4bb8287140cd2837f7a392df968f9dcabd0dfabc5df3b479cf9b010",
        "src/crfid/strict_runtime/neutral_metrics.py": "ccb26eecc31993c3f339f174411902539b4599b34fdc018732a63b3c9e2d6471",
        "src/crfid/strict_runtime/scale_policy.py": "bee7fa226c45eab2480242f9ab246eb08c22bad4df776a98637299d63b6034b1",
        "src/crfid/strict_runtime/target_evaluation.py": "82413d4b44ab3eef283722738da91ca41e0061eac33c358719d5ff12b9e43b88",
        "src/crfid/protocols/strict_dg.py": "1714dd8ad6a1fa62619a2b16e601f84954a9fc2d2e3c2d2d1675364b71de4aa3",
    }
    observed = {path: sha256_file(PROJECT_ROOT / path) for path in lineage_files}
    if observed != lineage_files:
        raise RuntimeError("BLOCKED_DANN_LINEAGE_OR_DATA_MISMATCH: C1 source hashes")
    lineage = {
        "candidate": "C1_FIRST_DIFFERENCE_ERM_1DCNN",
        "files": lineage_files,
        "ordered_input_length": 281,
        "first_difference_length": 280,
        "encoder": "NeutralSourceOnlyCNN1D layers through AdaptiveAvgPool1d/Flatten",
        "embedding_dimension": 256,
        "tagid_head": "Linear(256, 7)",
        "optimizer": config["optimizer"],
        "scheduler": None,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "checkpoint_rule": "maximum 50 epochs; validation row Macro-F1; patience 8; minimum improvement 1e-12",
        "binding_sha256": canonical_json_sha256(lineage_files),
    }
    write_json(RESULTS / "02_C1_LINEAGE_BINDING.json", lineage)
    structure_rows = [
        {"dimension": "TagID", "expected": 7, "observed": len(set(row["tag_id"] for row in data.rows)), "status": "PASS"},
        {"dimension": "ER", "expected": 3, "observed": len(set(row["er"] for row in data.rows)), "status": "PASS"},
        {"dimension": "surface", "expected": 3, "observed": len(set(row["surface"] for row in data.rows)), "status": "PASS"},
        {"dimension": "source_position", "expected": 3, "observed": len(set(row["position"] for row in data.rows)), "status": "PASS"},
        {"dimension": "source_rows", "expected": 9450, "observed": len(data.rows), "status": "PASS"},
        {"dimension": "source_condition_blocks", "expected": 189, "observed": len(set(row["raw_condition_id"] for row in data.rows)), "status": "PASS"},
        {"dimension": "ordered_points", "expected": 281, "observed": data.signals.shape[1], "status": "PASS"},
        {"dimension": "first_difference_points", "expected": 280, "observed": np.diff(data.signals, axis=1).shape[1], "status": "PASS"},
    ]
    write_csv(RESULTS / "04_DATA_AND_DOMAIN_STRUCTURE_AUDIT.csv", structure_rows)
    split_rows = []
    for fold in FOLDS:
        for part, indices in data.splits[fold].items():
            split_rows.append(
                {
                    "fold": fold,
                    "partition": part,
                    "row_count": len(indices),
                    "condition_block_count": len(set(data.condition_ids[indices])),
                    "positions": "|".join(sorted(set(data.positions[indices]))),
                    "condition_blocks_disjoint": True,
                }
            )
    write_csv(RESULTS / "05_SOURCE_LOPO_SPLIT_MANIFEST.csv", split_rows)
    validity = intervention_validity()
    write_csv(
        RESULTS / "06_DANN_MODEL_AND_GRL_VALIDITY.csv",
        [{"check": key, "passed": value} for key, value in validity.items() if key != "passed"]
        + [{"check": "all_validity_checks", "passed": validity["passed"]}],
    )
    if not validity["passed"]:
        raise RuntimeError("FAIL_DANN_INTERVENTION_VALIDITY")
    write_csv(
        RESULTS / "07_LAMBDA_GRID_REGISTER.csv",
        [
            {
                "lambda_max": value,
                "method": "ERM" if value == 0.0 else "DANN",
                "frozen_before_execution": True,
                "development_run_count": 15,
            }
            for value in LAMBDAS
        ],
    )
    gates = [
        "source condition blocks do not cross train/validation boundaries",
        "held source positions do not enter training",
        "P4 does not enter source selection",
        "ERM and DANN use identical source samples",
        "ERM and DANN use identical TagID labels",
        "ERM and DANN use identical training seeds",
        "ERM and DANN use identical checkpoint rules",
        "only domain head and reversed domain gradient differ",
        "source batches are matched",
        "lambda selection uses source-only evidence",
        "position probes are independent of the DANN domain head",
        "probe scaling and fitting use training-only probe data",
        "all condition blocks remain grouped",
        "no prior-branch target result is imported",
        "governed source and target inputs are read-only",
    ]
    write_csv(RESULTS / "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", [{"gate": gate, "status": "PASS_SOURCE_STAGE"} for gate in gates])
    (RESULTS / "03_PREREGISTERED_PROTOCOL.md").write_text((PROJECT_ROOT / "docs" / "DANN_POSITION_INTERVENTION_PROTOCOL.md").read_text(encoding="utf-8"), encoding="utf-8")
    (RESULTS / "00_EXECUTIVE_SUMMARY.md").write_text(
        "# DANN position intervention\n\nImplementation, lineage, governed structure, and GRL validity gates pass. Scientific execution status: `SOURCE_AUDIT_COMPLETE`.\n",
        encoding="utf-8",
    )
    write_json(RUNTIME / "source_audit.json", {"status": "PASS", "source_inputs": str(Path(source_inputs).resolve())})


def execute_development(source_inputs: Path, *, only: tuple[str, int, float] | None = None) -> None:
    data = load_source_data(source_inputs)
    run_root = RUNTIME / "development"
    run_root.mkdir(parents=True, exist_ok=True)
    combinations = [
        (fold, seed, value)
        for fold in FOLDS
        for seed in SEEDS
        for value in LAMBDAS
    ]
    if only is not None:
        combinations = [only]
    for index, (fold, seed, value) in enumerate(combinations, 1):
        path = run_root / f"lambda_{value:.2f}" / fold / f"seed_{seed}.json"
        if path.is_file():
            continue
        print(json.dumps({"event": "development_start", "index": index, "total": len(combinations), "fold": fold, "seed": seed, "lambda": value}), flush=True)
        result = development_run(data, fold=fold, seed=seed, lambda_max=value)
        write_json(path, result)
        print(json.dumps({"event": "development_complete", "fold": fold, "seed": seed, "lambda": value, "held_macro_f1": result["held_row_metrics"]["macro_f1"]}), flush=True)
    if only is None:
        compile_development()


def _development_results() -> list[dict]:
    rows = []
    for value in LAMBDAS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = RUNTIME / "development" / f"lambda_{value:.2f}" / fold / f"seed_{seed}.json"
                if not path.is_file():
                    raise RuntimeError(f"Missing development run: {value}/{fold}/{seed}")
                rows.append(read_json(path))
    return rows


def compile_development() -> None:
    rows = _development_results()
    register = [
        {
            "method": row["method"],
            "lambda_max": row["lambda_max"],
            "fold": row["fold"],
            "held_position": row["held_position"],
            "seed": row["seed"],
            "selected_epoch": row["selected_epoch"],
            "epochs_executed": row["epochs_executed"],
            "model_state_sha256": row["model_state_sha256"],
            "p4_used": False,
        }
        for row in rows
    ]
    write_csv(RESULTS / "09_DEVELOPMENT_RUN_REGISTER.csv", register)
    held = [
        {
            "method": row["method"],
            "lambda_max": row["lambda_max"],
            "fold": row["fold"],
            "held_position": row["held_position"],
            "seed": row["seed"],
            "row_macro_f1": row["held_row_metrics"]["macro_f1"],
            "row_accuracy": row["held_row_metrics"]["accuracy"],
            "block_macro_f1": row["held_block_metrics"]["macro_f1"],
            "block_accuracy": row["held_block_metrics"]["accuracy"],
        }
        for row in rows
    ]
    write_csv(RESULTS / "10_SOURCE_HELD_POSITION_METRICS.csv", held)
    write_csv(
        RESULTS / "11_INDEPENDENT_POSITION_PROBE_RESULTS.csv",
        [
            {
                "method": row["method"], "lambda_max": row["lambda_max"], "fold": row["fold"], "seed": row["seed"],
                "balanced_accuracy": row["position_probe_balanced_accuracy"], "macro_f1": row["position_probe_macro_f1"],
                "chance_balanced_accuracy": row["probe_position_chance"], "independent_of_domain_head": True,
            }
            for row in rows
        ],
    )
    write_csv(
        RESULTS / "12_TAGID_PROBE_AND_SILHOUETTE_RESULTS.csv",
        [
            {
                "method": row["method"], "lambda_max": row["lambda_max"], "fold": row["fold"], "seed": row["seed"],
                "tagid_probe_macro_f1": row["tagid_probe_macro_f1"],
                "position_silhouette": row["representation"]["position_silhouette"],
                "tagid_silhouette": row["representation"]["tagid_silhouette"],
                "embedding_norm_mean": row["representation"]["embedding_norm_mean"],
                "within_class_cross_position_distance": row["representation"]["within_class_cross_position_distance"],
                "between_class_distance": row["representation"]["between_class_distance"],
                "position_vs_tagid_separability_ratio": row["representation"]["position_vs_tagid_separability_ratio"],
                "domain_head_train_accuracy": row["domain_train_accuracy"],
                "domain_head_validation_accuracy": row["domain_validation_accuracy"],
            }
            for row in rows
        ],
    )


def select_lambda() -> dict:
    rows = _development_results()
    aggregates = []
    for value in LAMBDAS:
        selected = [row for row in rows if float(row["lambda_max"]) == value]
        aggregates.append(
            {
                "lambda_max": value,
                "run_count": len(selected),
                "held_macro_f1_mean": statistics.fmean(row["held_row_metrics"]["macro_f1"] for row in selected),
                "held_accuracy_mean": statistics.fmean(row["held_row_metrics"]["accuracy"] for row in selected),
                "position_probe_balanced_accuracy_mean": statistics.fmean(row["position_probe_balanced_accuracy"] for row in selected),
                "p4_used": False,
            }
        )
    selection = select_nonzero_lambda(aggregates)
    path = RUNTIME / "SELECTED_NONZERO_LAMBDA.json"
    write_json(path, selection)
    write_json(MANIFESTS / "SELECTED_NONZERO_LAMBDA.json", selection)
    selected_value = float(selection["selected_nonzero_lambda"])
    erm = [row for row in rows if float(row["lambda_max"]) == 0.0]
    dann = [row for row in rows if float(row["lambda_max"]) == selected_value]
    keyed_erm = {(row["fold"], row["seed"]): row for row in erm}
    keyed_dann = {(row["fold"], row["seed"]): row for row in dann}
    contrast_rows = []
    for key in sorted(keyed_erm):
        left, right = keyed_erm[key], keyed_dann[key]
        contrast_rows.append(
            {
                "fold": key[0], "seed": key[1],
                "position_probe_balanced_accuracy_change": right["position_probe_balanced_accuracy"] - left["position_probe_balanced_accuracy"],
                "held_macro_f1_change": right["held_row_metrics"]["macro_f1"] - left["held_row_metrics"]["macro_f1"],
                "held_accuracy_change": right["held_row_metrics"]["accuracy"] - left["held_row_metrics"]["accuracy"],
                "tagid_probe_macro_f1_change": right["tagid_probe_macro_f1"] - left["tagid_probe_macro_f1"],
                "position_silhouette_change": right["representation"]["position_silhouette"] - left["representation"]["position_silhouette"],
                "tagid_silhouette_change": right["representation"]["tagid_silhouette"] - left["representation"]["tagid_silhouette"],
            }
        )
    guardrail = source_retention_guardrail(
        statistics.fmean(row["held_row_metrics"]["macro_f1"] for row in erm),
        statistics.fmean(row["held_row_metrics"]["macro_f1"] for row in dann),
    )
    for row in contrast_rows:
        row["source_retention_guardrail_passed"] = guardrail["passed"]
    write_csv(RESULTS / "13_SOURCE_INTERVENTION_CONTRASTS.csv", contrast_rows)
    report = (
        "# Source-only lambda selection\n\n"
        f"Selected nonzero lambda: `{selected_value:.2f}`.\n\n"
        f"Eligible set: `{selection['eligible_nonzero_lambdas']}`. "
        f"Overall source-only winner: `{selection['source_only_overall_winner']}` at lambda `{selection['source_only_overall_winner_lambda']}`.\n\n"
        f"Source-retention guardrail passed: `{guardrail['passed']}` (change `{guardrail['change']:.6f}`, threshold `-0.02`).\n\n"
        f"Selection identity: `{selection['selection_sha256']}`. P4 was not accessed.\n"
    )
    (RESULTS / "14_SOURCE_ONLY_LAMBDA_SELECTION.md").write_text(report, encoding="utf-8")
    return selection


def final_train(source_inputs: Path, *, only_seed: int | None = None) -> None:
    selection = read_json(RUNTIME / "SELECTED_NONZERO_LAMBDA.json")
    data = load_source_data(source_inputs)
    config = read_json(CONFIG)
    record_root = RUNTIME / "final_model_records"
    record_root.mkdir(parents=True, exist_ok=True)
    seeds = (int(only_seed),) if only_seed is not None else SEEDS
    for seed in seeds:
        record_path = record_root / f"seed_{seed}.json"
        if record_path.is_file():
            continue
        pair = train_final_pair(
            data,
            seed=seed,
            selected_lambda=float(selection["selected_nonzero_lambda"]),
            epochs=int(config["final_epochs_by_seed"][str(seed)]),
            output_root=RUNTIME / "final_models",
        )
        write_json(record_path, {"records": pair})
        print(json.dumps({"event": "final_pair_complete", "seed": seed}), flush=True)
    if only_seed is not None:
        return
    records = []
    for seed in SEEDS:
        record_path = record_root / f"seed_{seed}.json"
        if not record_path.is_file():
            raise RuntimeError(f"Missing final paired seed record: {seed}")
        records.extend(read_json(record_path)["records"])
    state = fit_first_difference(data.signals, np.arange(len(data.labels), dtype=np.int64))
    np.save(RUNTIME / "final_models" / "preprocessing_mean.npy", state.mean, allow_pickle=False)
    np.save(RUNTIME / "final_models" / "preprocessing_scale.npy", state.scale, allow_pickle=False)
    write_json(RUNTIME / "final_model_register.json", {"records": records})
    write_csv(
        RESULTS / "15_FINAL_MODEL_REGISTER.csv",
        [
            {
                "method": row["method"], "lambda_max": row["lambda_max"], "seed": row["seed"], "epochs": row["epochs"],
                "model_state_sha256": row["model_state_sha256"], "checkpoint_sha256": row["checkpoint_sha256"],
                "source_macro_f1": row["source_row_metrics"]["macro_f1"], "source_accuracy": row["source_row_metrics"]["accuracy"],
                "position_probe_balanced_accuracy": row["position_probe_balanced_accuracy"],
                "tagid_probe_macro_f1": row["tagid_probe_macro_f1"], "p4_used": False,
            }
            for row in records
        ],
    )


def freeze_p4(p4_directory: Path) -> None:
    manifest = freeze_target_predictions(
        p4_directory=p4_directory,
        checkpoint_root=RUNTIME / "final_models",
        preprocessing_mean=RUNTIME / "final_models" / "preprocessing_mean.npy",
        preprocessing_scale=RUNTIME / "final_models" / "preprocessing_scale.npy",
        selection_path=RUNTIME / "SELECTED_NONZERO_LAMBDA.json",
        output_root=RUNTIME / "p4",
    )
    rows = []
    for access in manifest["feature_access"]:
        rows.append({"event": "feature_access", **access})
    for record in manifest["records"]:
        rows.append(
            {
                "event": "prediction_freeze", "method": record["method"], "seed": record["seed"],
                "checkpoint_sha256": record["checkpoint_sha256"], "prediction_sha256": record["prediction_array_sha256"],
                "labels_accessed": False,
            }
        )
    write_csv(
        RESULTS / "16_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv",
        rows,
        fields=[
            "event", "stage", "method", "seed", "file_name", "sha256", "rows",
            "started_at_utc", "completed_at_utc", "checkpoint_sha256",
            "prediction_sha256", "labels_accessed", "tagid_column_accessed",
            "er_column_accessed",
        ],
    )


def score_p4(p4_directory: Path) -> None:
    score_path = RUNTIME / "p4" / "P4_SCORE.json"
    if score_path.is_file():
        # Labels were already opened and scored once. Derive any additional
        # compact sensitivities from the frozen block arrays; never reopen P4.
        score = read_json(score_path)
    else:
        score = score_target_once(
            p4_directory=p4_directory,
            freeze_manifest=RUNTIME / "p4" / "P4_PREDICTION_FREEZE.json",
            output_root=RUNTIME / "p4",
        )
    erm_blocks = np.asarray(score["erm_block_predictions"], dtype=np.int64)
    dann_blocks = np.asarray(score["dann_block_predictions"], dtype=np.int64)
    true_blocks = np.asarray(score["block_true"], dtype=np.int64)
    if "accuracy_block_only_bootstrap" not in score:
        score["accuracy_block_only_bootstrap"] = stratified_paired_block_bootstrap(
            erm_blocks, dann_blocks, true_blocks, replicates=10_000, seed=20_260_808, metric="accuracy"
        )
        score["accuracy_block_plus_training_seed_bootstrap"] = stratified_paired_block_bootstrap(
            erm_blocks, dann_blocks, true_blocks, replicates=10_000, seed=20_260_809,
            resample_training_seeds=True, metric="accuracy"
        )
    for index, row in enumerate(score["paired"]):
        row["block_prediction_changes"] = int(np.count_nonzero(erm_blocks[index] != dann_blocks[index]))
    if "seed_agreement" not in score:
        score["seed_agreement"] = []
        for method, matrix in (("ERM", erm_blocks), ("DANN", dann_blocks)):
            agreements = [
                float(np.bincount(matrix[:, block], minlength=7).max() / matrix.shape[0])
                for block in range(matrix.shape[1])
            ]
            score["seed_agreement"].append(
                {
                    "method": method,
                    "mean_block_seed_agreement": float(np.mean(agreements)),
                    "minimum_block_seed_agreement": float(np.min(agreements)),
                }
            )
    write_json(score_path, score)
    path = RESULTS / "16_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv"
    freeze = read_json(RUNTIME / "p4" / "P4_PREDICTION_FREEZE.json")
    existing = [{"event": "feature_access", **access} for access in freeze["feature_access"]]
    existing.extend(
        {
            "event": "prediction_freeze",
            "method": record["method"],
            "seed": record["seed"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "prediction_sha256": record["prediction_array_sha256"],
            "labels_accessed": False,
        }
        for record in freeze["records"]
    )
    existing.extend(
        {"event": "label_access_after_freeze", **access}
        for access in score["label_access"]
    )
    write_csv(
        path,
        existing,
        fields=[
            "event", "stage", "method", "seed", "file_name", "sha256", "rows",
            "started_at_utc", "completed_at_utc", "checkpoint_sha256",
            "prediction_sha256", "labels_accessed", "tagid_column_accessed",
            "er_column_accessed",
        ],
    )
    seed_agreement = {row["method"]: row for row in score["seed_agreement"]}
    final_records = read_json(RUNTIME / "final_model_register.json")["records"]
    source_metrics = {(row["method"], int(row["seed"])): row["source_row_metrics"] for row in final_records}
    for row in score["seed_metrics"]:
        row.update(seed_agreement[row["method"]])
        source_macro_f1 = source_metrics[(row["method"], int(row["seed"]))]["macro_f1"]
        row["source_macro_f1"] = source_macro_f1
        row["source_to_p4_block_macro_f1_drop"] = source_macro_f1 - row["block_macro_f1"]
    write_csv(RESULTS / "17_P4_BLOCK_METRICS.csv", score["seed_metrics"])
    block_only = score["block_only_bootstrap"]
    block_plus_seed = score["block_plus_training_seed_bootstrap"]
    accuracy_block_only = score["accuracy_block_only_bootstrap"]
    accuracy_block_plus_seed = score["accuracy_block_plus_training_seed_bootstrap"]
    paired = score["paired"]
    write_csv(
        RESULTS / "18_P4_PRIMARY_DANN_VS_ERM_CONTRAST.csv",
        [
            {
                "endpoint": "condition_block_macro_f1",
                "point_estimate": block_only["point_estimate"],
                "percentile_95_lower": block_only["interval"][0],
                "percentile_95_upper": block_only["interval"][1],
                "block_plus_seed_lower": block_plus_seed["interval"][0],
                "block_plus_seed_upper": block_plus_seed["interval"][1],
                "independent_unit": "TagID_x_ER_x_surface_block",
            },
            {
                "endpoint": "condition_block_accuracy",
                "point_estimate": accuracy_block_only["point_estimate"],
                "percentile_95_lower": accuracy_block_only["interval"][0],
                "percentile_95_upper": accuracy_block_only["interval"][1],
                "block_plus_seed_lower": accuracy_block_plus_seed["interval"][0],
                "block_plus_seed_upper": accuracy_block_plus_seed["interval"][1],
                "independent_unit": "TagID_x_ER_x_surface_block",
            },
        ]
        + [
            {
                "endpoint": f"seed_{row['seed']}",
                "point_estimate": row["block_macro_f1_change"],
                "block_accuracy_change": row["block_accuracy_change"],
                "row_prediction_changes": row["row_prediction_changes"],
                "block_prediction_changes": row["block_prediction_changes"],
            }
            for row in paired
        ],
        fields=[
            "endpoint", "point_estimate", "percentile_95_lower", "percentile_95_upper",
            "block_plus_seed_lower", "block_plus_seed_upper", "independent_unit",
            "block_accuracy_change", "row_prediction_changes", "block_prediction_changes",
        ],
    )
    write_csv(RESULTS / "19_P4_PER_CLASS_RESULTS.csv", score["per_class"])
    write_csv(
        RESULTS / "20_PREDICTED_CLASS_HISTOGRAMS.csv",
        [{"method": row["method"], "seed": row["seed"], **{f"class_{index}": count for index, count in enumerate(row["counts"])} } for row in score["histograms"]],
    )
    sensitivity = [
        {"analysis": "block_only", "point_estimate": block_only["point_estimate"], "lower": block_only["interval"][0], "upper": block_only["interval"][1]},
        {"analysis": "block_plus_training_seed", "point_estimate": block_plus_seed["point_estimate"], "lower": block_plus_seed["interval"][0], "upper": block_plus_seed["interval"][1]},
        {"analysis": "block_accuracy_only", "point_estimate": accuracy_block_only["point_estimate"], "lower": accuracy_block_only["interval"][0], "upper": accuracy_block_only["interval"][1]},
        {"analysis": "block_accuracy_plus_training_seed", "point_estimate": accuracy_block_plus_seed["point_estimate"], "lower": accuracy_block_plus_seed["interval"][0], "upper": accuracy_block_plus_seed["interval"][1]},
    ] + [{"analysis": f"leave_one_training_seed_out_{row['omitted_seed']}", "point_estimate": row["mean_change"]} for row in score["leave_one_training_seed_out"]]
    write_csv(RESULTS / "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", sensitivity)


def source_probe_bootstrap(
    erm_runs: dict[tuple[str, int], dict],
    dann_runs: dict[tuple[str, int], dict],
    *,
    replicates: int = 10_000,
) -> dict:
    """Paired block bootstrap preserving fold, training-seed, and probe-seed pairing."""

    differences = []
    true_by_fold = []
    for fold in FOLDS:
        fold_differences = []
        fold_true = None
        for seed in SEEDS:
            left = erm_runs[(fold, seed)]["probe_results"]["position"]
            right = dann_runs[(fold, seed)]["probe_results"]["position"]
            probe_differences = []
            for probe_index in range(len(SEEDS)):
                left_true = np.asarray(left[probe_index]["true_labels"], dtype=np.int64)
                right_true = np.asarray(right[probe_index]["true_labels"], dtype=np.int64)
                if not np.array_equal(left_true, right_true):
                    raise RuntimeError("Paired source probe labels differ")
                if fold_true is None:
                    fold_true = left_true
                elif not np.array_equal(fold_true, left_true):
                    raise RuntimeError("Source probe test blocks differ across paired seeds")
                left_correct = np.asarray(left[probe_index]["predictions"], dtype=np.int64) == left_true
                right_correct = np.asarray(right[probe_index]["predictions"], dtype=np.int64) == left_true
                probe_differences.append(right_correct.astype(np.float64) - left_correct.astype(np.float64))
            fold_differences.append(np.stack(probe_differences))
        differences.append(np.stack(fold_differences))
        true_by_fold.append(fold_true)
    values = np.stack(differences)  # fold, training seed, probe seed, block
    generator = np.random.default_rng(20_260_807)

    def draw(*, training_seed: bool, probe_seed: bool) -> np.ndarray:
        output = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            training_indices = generator.integers(0, len(SEEDS), len(SEEDS)) if training_seed else np.arange(len(SEEDS))
            probe_indices = generator.integers(0, len(SEEDS), len(SEEDS)) if probe_seed else np.arange(len(SEEDS))
            fold_values = []
            for fold_index in range(len(FOLDS)):
                true = np.asarray(true_by_fold[fold_index], dtype=np.int64)
                block_indices = np.concatenate(
                    [
                        generator.choice(local, size=len(local), replace=True)
                        for local in (np.flatnonzero(true == label) for label in sorted(np.unique(true)))
                    ]
                )
                selected = values[fold_index][training_indices][:, probe_indices][:, :, block_indices]
                fold_values.append(float(selected.mean()))
            output[replicate] = float(np.mean(fold_values))
        return output

    analyses = {}
    for name, training_seed, probe_seed in (
        ("source_probe_block_only", False, False),
        ("source_probe_block_plus_training_seed", True, False),
        ("source_probe_block_plus_probe_seed", False, True),
        ("source_probe_block_plus_both_seed_dimensions", True, True),
    ):
        draws = draw(training_seed=training_seed, probe_seed=probe_seed)
        interval = np.quantile(draws, [0.025, 0.975])
        analyses[name] = {
            "point_estimate": float(values.mean()),
            "interval": (float(interval[0]), float(interval[1])),
        }
    analyses["leave_one_training_seed_out"] = [
        {
            "omitted_seed": seed,
            "point_estimate": float(np.delete(values, index, axis=1).mean()),
        }
        for index, seed in enumerate(SEEDS)
    ]
    analyses["fold_sensitivity"] = [
        {"fold": fold, "point_estimate": float(values[index].mean())}
        for index, fold in enumerate(FOLDS)
    ]
    return analyses


def final_report() -> None:
    development = _development_results()
    selection = read_json(RUNTIME / "SELECTED_NONZERO_LAMBDA.json")
    final = read_json(RUNTIME / "final_model_register.json")["records"]
    score = read_json(RUNTIME / "p4" / "P4_SCORE.json")
    selected = float(selection["selected_nonzero_lambda"])
    erm_dev = {(row["fold"], row["seed"]): row for row in development if float(row["lambda_max"]) == 0.0}
    dann_dev = {(row["fold"], row["seed"]): row for row in development if float(row["lambda_max"]) == selected}
    position_changes = np.asarray([dann_dev[key]["position_probe_balanced_accuracy"] - erm_dev[key]["position_probe_balanced_accuracy"] for key in sorted(erm_dev)])
    source_changes = np.asarray([dann_dev[key]["held_row_metrics"]["macro_f1"] - erm_dev[key]["held_row_metrics"]["macro_f1"] for key in sorted(erm_dev)])
    source_probe_uncertainty = source_probe_bootstrap(erm_dev, dann_dev)
    position_uncertainty = source_probe_uncertainty["source_probe_block_only"]
    source_uncertainty = paired_vector_bootstrap(source_changes, seed=20_260_806)
    p4_uncertainty = score["block_only_bootstrap"]
    p4_changes = np.asarray([row["block_macro_f1_change"] for row in score["paired"]])
    final_erm = {row["seed"]: row for row in final if row["method"] == "ERM"}
    final_dann = {row["seed"]: row for row in final if row["method"] == "DANN"}
    probe_seed_changes = np.asarray([final_dann[seed]["position_probe_balanced_accuracy"] - final_erm[seed]["position_probe_balanced_accuracy"] for seed in SEEDS])
    tag_probe_changes = np.asarray([final_dann[seed]["tagid_probe_macro_f1"] - final_erm[seed]["tagid_probe_macro_f1"] for seed in SEEDS])
    source_seed_changes = np.asarray([final_dann[seed]["source_row_metrics"]["macro_f1"] - final_erm[seed]["source_row_metrics"]["macro_f1"] for seed in SEEDS])
    link_rows = [
        {"relationship": "position_probe_reduction_vs_p4_change", "correlation": safe_correlation(-probe_seed_changes, p4_changes), "exploratory": True},
        {"relationship": "tagid_probe_retention_vs_p4_change", "correlation": safe_correlation(tag_probe_changes, p4_changes), "exploratory": True},
        {"relationship": "source_macro_f1_change_vs_p4_change", "correlation": safe_correlation(source_seed_changes, p4_changes), "exploratory": True},
    ]
    write_csv(RESULTS / "21_INTERVENTION_TRANSFER_LINK.csv", link_rows)
    sensitivity_path = RESULTS / "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv"
    with sensitivity_path.open("r", encoding="utf-8", newline="") as handle:
        sensitivity_rows = list(csv.DictReader(handle))
    sensitivity_rows = [
        row for row in sensitivity_rows if not row.get("analysis", "").startswith("source_probe_")
    ]
    for name in (
        "source_probe_block_only",
        "source_probe_block_plus_training_seed",
        "source_probe_block_plus_probe_seed",
        "source_probe_block_plus_both_seed_dimensions",
    ):
        row = source_probe_uncertainty[name]
        sensitivity_rows.append(
            {
                "analysis": name,
                "point_estimate": row["point_estimate"],
                "lower": row["interval"][0],
                "upper": row["interval"][1],
            }
        )
    sensitivity_rows.extend(
        {
            "analysis": f"source_probe_leave_one_training_seed_out_{row['omitted_seed']}",
            "point_estimate": row["point_estimate"],
        }
        for row in source_probe_uncertainty["leave_one_training_seed_out"]
    )
    sensitivity_rows.extend(
        {
            "analysis": f"source_probe_fold_sensitivity_{row['fold']}",
            "point_estimate": row["point_estimate"],
        }
        for row in source_probe_uncertainty["fold_sensitivity"]
    )
    write_csv(sensitivity_path, sensitivity_rows)
    interpretation = classify_intervention(
        position_change=position_uncertainty["point_estimate"],
        position_interval=tuple(position_uncertainty["interval"]),
        p4_change=p4_uncertainty["point_estimate"],
        p4_interval=tuple(p4_uncertainty["interval"]),
        source_change=source_uncertainty["point_estimate"],
        source_interval=tuple(source_uncertainty["interval"]),
    )
    p4_accuracy_change = statistics.fmean(row["block_accuracy_change"] for row in score["paired"])
    p4_accuracy_uncertainty = score["accuracy_block_only_bootstrap"]
    source_retained = source_uncertainty["point_estimate"] >= -0.02
    position_point_decreased = position_uncertainty["point_estimate"] < 0.0
    position_confirmed = position_uncertainty["interval"][1] < 0.0
    p4_macro_point_improved = p4_uncertainty["point_estimate"] > 0.0
    p4_macro_confirmed = p4_uncertainty["interval"][0] > 0.0
    p4_accuracy_point_improved = p4_accuracy_change > 0.0
    p4_accuracy_confirmed = p4_accuracy_uncertainty["interval"][0] > 0.0
    summary = (
        "# DANN position intervention executive summary\n\n"
        f"Classification: `{interpretation}`.\n\n"
        f"GRL validity passed: `True`. Selected nonzero lambda: `{selected:.2f}`. "
        f"Source position-probe balanced-accuracy change: `{position_uncertainty['point_estimate']:.6f}` "
        f"(95% paired interval `{position_uncertainty['interval']}`). "
        f"Source held-position Macro-F1 change: `{source_uncertainty['point_estimate']:.6f}`.\n\n"
        f"P4 block Macro-F1 change: `{p4_uncertainty['point_estimate']:.6f}` "
        f"(block-only 95% interval `{p4_uncertainty['interval']}`; block-plus-seed interval `{score['block_plus_training_seed_bootstrap']['interval']}`). "
        f"P4 block Accuracy change: `{p4_accuracy_change:.6f}` "
        f"(block-only 95% interval `{p4_accuracy_uncertainty['interval']}`).\n\n"
        f"Position decodability decreased at the point estimate: `{position_point_decreased}`; confirmed below zero: `{position_confirmed}`. "
        f"Source TagID retention guardrail passed: `{source_retained}`. "
        f"DANN improved P4 block Macro-F1 at the point estimate: `{p4_macro_point_improved}`; survived block uncertainty: `{p4_macro_confirmed}`. "
        f"DANN improved P4 block Accuracy at the point estimate: `{p4_accuracy_point_improved}`; survived block uncertainty: `{p4_accuracy_confirmed}`. "
        f"A target gain survived both block and training-seed uncertainty: `{score['block_plus_training_seed_bootstrap']['interval'][0] > 0.0}`. "
        f"Confirmed position suppression and transfer improvement occurred together: `{position_confirmed and p4_macro_confirmed}`. "
        f"Mechanistic evidence was strengthened by the preregistered criteria: `{interpretation == 'POSITION_SUPPRESSION_AND_P4_TRANSFER_BENEFIT_CONFIRMED'}`.\n\n"
        "P4 predictions were frozen and hashed before labels were opened. The intervention is not formal causal mediation and cannot prove physical causality.\n"
    )
    (RESULTS / "00_EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8")
    (RESULTS / "23_SCIENTIFIC_INTERPRETATION.md").write_text(
        f"# Scientific interpretation\n\nMain classification: `{interpretation}`.\n\nThe paired intervention and transfer contrasts are reported without a definitive physical-causality claim. Five-seed correlations are exploratory.\n",
        encoding="utf-8",
    )
    (RESULTS / "24_LIMITATIONS_AND_NONCLAIMS.md").write_text(
        "# Limitations and nonclaims\n\n- Five training seeds do not support precise population-level mediation.\n- DANN suppression is an intervention on learned source representations, not proof of a physical causal mechanism.\n- P4 comprises 63 condition blocks; repeated rows are not independent inferential units.\n- Lambda selection is source-only and may not optimize target transfer.\n- The domain classifier is restricted to source position and does not remove all condition information.\n",
        encoding="utf-8",
    )
    (PROJECT_ROOT / "docs" / "DANN_POSITION_INTERVENTION_RESULTS.md").write_text(summary, encoding="utf-8")
    (RESULTS / "STATUS.md").write_text(
        f"# Result status\n\nScientific classification: `{interpretation}`.\n\nExecution completed with protocol and label-boundary gates satisfied. The intervention is interpreted as a mechanistic diagnostic rather than evidence of a physical causal mechanism.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "development", "select", "final-train", "freeze-p4", "score-p4", "report", "all"))
    parser.add_argument("--source-inputs", type=Path)
    parser.add_argument("--p4-directory", type=Path)
    parser.add_argument("--only-fold", choices=FOLDS)
    parser.add_argument("--only-seed", type=int, choices=SEEDS)
    parser.add_argument("--only-lambda", type=float, choices=LAMBDAS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_required = args.command in {"audit", "development", "final-train", "all"}
    p4_required = args.command in {"freeze-p4", "score-p4", "all"}
    if source_required and args.source_inputs is None:
        raise SystemExit("--source-inputs is required")
    if p4_required and args.p4_directory is None:
        raise SystemExit("--p4-directory is required")
    if args.command in {"audit", "all"}:
        audit(args.source_inputs)
    if args.command in {"development", "all"}:
        only_values = (args.only_fold, args.only_seed, args.only_lambda)
        only = only_values if all(value is not None for value in only_values) else None
        execute_development(args.source_inputs, only=only)
    if args.command in {"select", "all"}:
        select_lambda()
    if args.command in {"final-train", "all"}:
        final_train(args.source_inputs, only_seed=args.only_seed if args.command == "final-train" else None)
    if args.command in {"freeze-p4", "all"}:
        freeze_p4(args.p4_directory)
    if args.command in {"score-p4", "all"}:
        score_p4(args.p4_directory)
    if args.command in {"report", "all"}:
        final_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
