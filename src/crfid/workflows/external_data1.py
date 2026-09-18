"""External Data1 EV0-through-EV3R-R1 reproduction workflow."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ProtocolViolation
from ..external_data1.aggregation import (
    grouped_metrics,
    metric_bundle,
    summarize_cnn_runs,
)
from ..external_data1.artifacts import (
    array_sha256,
    canonical_json_sha256,
    public_manifest,
    write_csv,
    write_json,
    write_prediction_bundle,
)
from ..external_data1.baselines import (
    majority_tie_decision,
    predict_majority,
)
from ..external_data1.cnn_portability import (
    EPOCHS,
    SEEDS,
    infer_cnn,
    prepare_representation,
    save_and_reload_checkpoint,
    train_cnn,
    transform_test,
)
from ..external_data1.identity import validate_loaded_identity
from ..external_data1.loader import load_authoritative_data1
from ..external_data1.pca_logreg import (
    fit_and_save,
    predict_from_saved_state,
    specification as pca_specification,
)
from ..external_data1.reporting import write_results_report
from ..external_data1.schema import (
    CLASS_ORDER,
    DATASET_ID,
    EXPECTED_CLASS_COUNTS,
    PUBLIC_BRANCH_NAME,
    SCIENTIFIC_CLAIM_TYPE,
)
from ..external_data1.splits import (
    EXCLUSION_REASON,
    split_manifest_rows,
    validate_frozen_split,
)
from ..governance.external_claims import (
    ALLOWED_CONCLUSIONS,
    PRIMARY_CLAIM,
    validate_external_claim,
)
from ..governance.external_data_access import authorize_external_data
from ..governance.external_integrity import validate_external_mode
from ..governance.external_label_access import ExternalLabelVault
from ..governance.integrity import sha256_file
from ..protocols.external_data1 import validate_protocol
from .common import WorkflowPlan, build_plan


ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_AGGREGATE_SHA256 = (
    "886698bfbc5fca7d440842ee4e2778f6ae5549db6563233c6e232ad2ff3a3cd7"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path.name}")
    return payload


def _relative_artifact(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _environment_identity() -> dict[str, Any]:
    distributions = (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "pandas",
        "joblib",
        "pytest",
        "PyYAML",
    )
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "environment_id": "CRFID_STRICT_DG_WINDOWS_CPU_V1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "compute_platform": "cpu",
        "locked": True,
    }


def _write_ev0(
    output_root: Path,
    data: Any,
    identity: dict[str, object],
    authority: dict[str, Any],
) -> None:
    root = output_root / "ev0"
    raw_rows = [
        {
            "logical_file": item.raw_file.path.name,
            "set_id": item.set_id,
            "size_bytes": item.raw_file.size_bytes,
            "sha256": item.raw_file.sha256,
            "row_count": len(item.labels),
            "signal_length": item.inputs.shape[1],
        }
        for item in data.sets
    ]
    write_csv(
        root / "raw_file_manifest.csv",
        raw_rows,
        ["logical_file", "set_id", "size_bytes", "sha256", "row_count", "signal_length"],
    )
    write_json(
        root / "data_identity.json",
        {
            **identity,
            "dataset_id": DATASET_ID,
            "raw_aggregate_sha256": data.raw_aggregate_sha256,
            "authority_verdict": authority["authority_verdict"],
        },
    )
    write_json(
        root / "class_counts.json",
        {
            "class_order": list(CLASS_ORDER),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "sample_count": int(sum(EXPECTED_CLASS_COUNTS)),
        },
    )
    write_json(
        root / "schema_report.json",
        {
            "file_count_valid": len(data.sets) == 9,
            "row_count_valid": identity["sample_count"] == 12250,
            "feature_count_valid": identity["signal_length"] == 1601,
            "finite_values_valid": True,
            "labels_valid": True,
        },
    )
    write_json(
        root / "pipeline_custody.json",
        {
            "intended_components": [
                "training_only_preprocessing",
                "majority_class_baseline",
                "pca_logistic_regression",
                "four_class_groupnorm_cnn",
                "classification_metrics",
            ],
            "model_execution": False,
            "metric_execution": False,
            "other_dataset_access": False,
        },
    )
    (root / "EV0_REPORT.md").write_text(
        "# EV0 data and pipeline custody\n\n"
        "The exact nine-file raw bundle passed file, row, feature, label and hash "
        "checks. No model, inference or performance metric was executed in EV0.\n",
        encoding="utf-8",
    )
    write_json(
        root / "EV0_INTEGRITY.json",
        {
            "verdict": "PASS_EV0_DATA_AND_PIPELINE_CUSTODY",
            "raw_aggregate_sha256": data.raw_aggregate_sha256,
            "model_execution": False,
            "metric_access": False,
        },
    )


def _write_ev1(output_root: Path, authority: dict[str, Any]) -> None:
    root = output_root / "ev1"
    candidate_rows = [
        {
            "candidate_id": "external_data1_raw_bundle",
            "relationship": "independent_acquisition",
            "selection": "selected",
            "eligibility": "PIPELINE_PORTABILITY_ONLY",
        },
        {
            "candidate_id": "source_family_copy",
            "relationship": "source_dataset_copy",
            "selection": "rejected",
            "eligibility": "NOT_EXTERNAL",
        },
        {
            "candidate_id": "raw_bundle_mirror",
            "relationship": "byte_identical_mirror",
            "selection": "excluded",
            "eligibility": "NOT_SEPARATE_AUTHORITY",
        },
        {
            "candidate_id": "processed_derivative",
            "relationship": "processed_component",
            "selection": "excluded",
            "eligibility": "NOT_CANONICAL_RAW",
        },
    ]
    write_csv(
        root / "candidate_registry.csv",
        candidate_rows,
        ["candidate_id", "relationship", "selection", "eligibility"],
    )
    write_json(
        root / "selected_data_authority.json",
        {
            "dataset_id": DATASET_ID,
            "authority_verdict": authority["authority_verdict"],
            "raw_aggregate_sha256": authority["raw_aggregate_sha256"],
            "file_count": 9,
            "sample_count": 12250,
        },
    )
    write_json(
        root / "acquisition_independence.json",
        {
            "classification": authority["acquisition_status"],
            "selected_raw_bundle_used_in_source_training": False,
            "exact_source_file_overlap_count": 0,
            "basis": [
                "separate documented acquisition lineage",
                "nine-set four-class organization",
                "no exact source-file overlap",
            ],
        },
    )
    write_json(
        root / "axis_identity_report.json",
        {
            "documented_frequency_range_ghz": authority[
                "documented_frequency_range_ghz"
            ],
            "signal_point_count": 1601,
            "physical_axis_encoded_in_csv": False,
            "status": authority["frequency_axis_status"],
            "resampling_or_mapping_frozen": False,
        },
    )
    write_json(
        root / "class_mapping_report.json",
        {
            "local_class_order": list(CLASS_ORDER),
            "status": authority["class_mapping_status"],
            "direct_external_model_validation": False,
        },
    )
    compatibility_rows = [
        {"question": "independent_acquisition", "result": "PASS"},
        {"question": "four_class_local_task", "result": "PASS"},
        {"question": "pipeline_portability", "result": "ELIGIBLE"},
        {"question": "exact_axis_identity", "result": "UNRESOLVED"},
        {"question": "cross_dataset_class_mapping", "result": "ABSENT"},
        {"question": "direct_external_model_validation", "result": "INELIGIBLE"},
    ]
    write_csv(
        root / "compatibility_matrix.csv",
        compatibility_rows,
        ["question", "result"],
    )
    (root / "EV1_REPORT.md").write_text(
        "# EV1 compatibility and authority audit\n\n"
        "Verdict: `PASS_EV1_PIPELINE_PORTABILITY_ONLY`. The selected four-class "
        "bundle is independently acquired, but exact physical-axis identity and "
        "a cross-dataset class mapping are unresolved.\n",
        encoding="utf-8",
    )
    write_json(
        root / "EV1_INTEGRITY.json",
        {
            "verdict": "PASS_EV1_PIPELINE_PORTABILITY_ONLY",
            "eligibility": "PIPELINE_PORTABILITY_ONLY",
            "direct_external_model_validation": False,
        },
    )


def _write_ev2r(
    output_root: Path,
    data: Any,
    protocol: dict[str, Any],
    authority: dict[str, Any],
    config_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    root = output_root / "ev2r_protocol"
    validate_protocol(protocol)
    protocol_sha256 = canonical_json_sha256(protocol)
    write_json(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(protocol_sha256 + "\n", encoding="ascii")
    split_rows = split_manifest_rows(data)
    public_split_rows = [
        {
            key: row[key]
            for key in (
                "split_id",
                "split_role",
                "measurement_set_id",
                "row_index",
                "sample_id",
                "exact_signal_sha256",
                "raw_file_sha256",
            )
        }
        for row in split_rows
    ]
    write_csv(
        root / "split_manifest.csv",
        public_split_rows,
        list(public_split_rows[0]),
    )
    candidate_registry = {
        "schema_version": 1,
        "candidate_methods": protocol["candidate_methods"],
        "cnn_seeds": protocol["cnn_seeds"],
        "learned_run_count": 11,
        "context_baseline_count": 1,
        "frozen_before_ev3r": True,
    }
    write_json(root / "candidate_registry.json", candidate_registry)
    write_json(root / "class_order.json", {"class_order": list(CLASS_ORDER)})
    write_json(root / "majority_tie_rule.json", protocol["majority_tie_rule"])
    write_json(root / "preprocessing_specification.json", protocol["preprocessing"])
    write_json(
        root / "model_specification.json",
        {"pca_logistic": protocol["pca_logistic"], "cnn": protocol["cnn"]},
    )
    write_json(
        root / "seed_registry.json",
        {"cnn_seeds": list(SEEDS), "fixed_final_epochs": protocol["cnn_epochs"]},
    )
    write_json(
        root / "label_access_contract.json",
        _read_json(ROOT / "configs" / "external_data1" / "label_access.yaml"),
    )
    module_paths = sorted(
        path
        for path in (ROOT / "src" / "crfid" / "external_data1").glob("*.py")
    )
    module_paths.extend(
        [
            ROOT / "src" / "crfid" / "protocols" / "external_data1.py",
            ROOT / "src" / "crfid" / "governance" / "external_data_access.py",
            ROOT / "src" / "crfid" / "governance" / "external_label_access.py",
            ROOT / "src" / "crfid" / "governance" / "external_claims.py",
            ROOT / "src" / "crfid" / "governance" / "external_integrity.py",
            Path(__file__),
        ]
    )
    write_csv(
        root / "code_manifest.csv",
        [
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in module_paths
        ],
        ["relative_path", "sha256"],
    )
    write_csv(
        root / "config_manifest.csv",
        [
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in config_paths
        ],
        ["relative_path", "sha256"],
    )
    write_json(root / "environment.json", _environment_identity())
    (root / "EV2R_PROTOCOL_FREEZE.md").write_text(
        "# EV2R/EV2R-A1 protocol freeze\n\n"
        f"Protocol SHA-256: `{protocol_sha256}`.\n\n"
        "Train: set 3. Test: sets 4–9. Sets 1–2 are accounted for as "
        "different documented applications. The four-way majority tie selects "
        "class 0 by frozen class order. No EV3R result was used to form this freeze.\n",
        encoding="utf-8",
    )
    return protocol_sha256, candidate_registry


def _reference_paths(reference_root: Path) -> dict[str, Path]:
    paths = {
        "EV3R_R1_MAJORITY_BASELINE": reference_root
        / "06_internal_portability_benchmark"
        / "ev3r_r1"
        / "EV3R_R1_MAJORITY_BASELINE_PREDICTIONS.csv",
        "EV3R_R1_REF_SPLIT00": reference_root
        / "05_external_execution"
        / "ev3r_r1_runs"
        / "reference_pca_logreg"
        / "predictions.csv",
    }
    for representation, prefix in (
        ("raw_1dcnn", "RAW"),
        ("first_difference_1dcnn", "DIFF"),
    ):
        for seed in SEEDS:
            paths[f"EV3R_R1_{prefix}_SPLIT00_SEED{seed}"] = (
                reference_root
                / "05_external_execution"
                / "ev3r_r1_runs"
                / representation
                / f"seed_{seed}"
                / "predictions.csv"
            )
    return paths


def _read_reference_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    truth: list[int] = []
    prediction: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            truth.append(int(row["true_label"]))
            prediction.append(int(row["predicted_label"]))
    return np.asarray(truth, dtype=np.int64), np.asarray(prediction, dtype=np.int64)


def _reference_metric_records(reference_root: Path) -> list[dict[str, Any]]:
    rows = []
    majority = _read_json(
        reference_root
        / "06_internal_portability_benchmark"
        / "ev3r_r1"
        / "EV3R_R1_MAJORITY_BASELINE_METRICS.json"
    )
    rows.append(
        {
            "run_id": majority["run_id"],
            "method": "majority_class_baseline",
            "seed": None,
            "metrics": majority["pooled_metrics"],
        }
    )
    reference = _read_json(
        reference_root
        / "05_external_execution"
        / "ev3r_r1_runs"
        / "reference_pca_logreg"
        / "run_result.json"
    )
    rows.append(
        {
            "run_id": reference["run_id"],
            "method": "pca_logistic_regression",
            "seed": 42,
            "metrics": reference["pooled_metrics"],
        }
    )
    for directory, method in (
        ("raw_1dcnn", "raw_signal_cnn"),
        ("first_difference_1dcnn", "first_difference_cnn"),
    ):
        for seed in SEEDS:
            record = _read_json(
                reference_root
                / "05_external_execution"
                / "ev3r_r1_runs"
                / directory
                / f"seed_{seed}"
                / "run_result.json"
            )
            rows.append(
                {
                    "run_id": record["run_id"],
                    "method": method,
                    "seed": seed,
                    "metrics": record["pooled_metrics"],
                }
            )
    return rows


def _freeze_comparison_inputs(
    output_root: Path, reference_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    root = output_root / "comparison"
    reference_records = _reference_metric_records(reference_root)
    reference_paths = _reference_paths(reference_root)
    write_json(
        root / "authoritative_reference.json",
        {
            "historical_stage": "EV3R-R1",
            "historical_aggregate_sha256": HISTORICAL_AGGREGATE_SHA256,
            "runs": reference_records,
            "scientific_conclusions": [
                "DATA1_LEARNABILITY_CLEAR",
                "CNN_PIPELINE_PORTABILITY_SUPPORTED",
                "FIRST_DIFFERENCE_PORTABILITY_NOT_SUPPORTED",
            ],
        },
    )
    write_json(
        root / "comparison_policy.json",
        {
            "frozen_before_reproduction_metrics": True,
            "exact_match": [
                "raw_bundle_sha256",
                "split_roles",
                "class_counts",
                "candidate_registry",
                "majority_tie_rule",
                "per_run_predicted_labels",
                "confusion_matrices",
            ],
            "metric_absolute_tolerance": 1e-12,
            "expected_determinism": "CPU predictions under the locked environment",
            "container_hash_nondeterminism": [
                "npz_container_bytes",
                "torch_checkpoint_container_bytes",
                "runtime_fields",
            ],
            "historical_aggregate_policy": (
                "recompute the frozen historical manifest aggregate exactly; "
                "the differently structured public package is not byte-identical"
            ),
        },
    )
    return reference_records, reference_paths


def _score_frozen_prediction(
    *,
    output_root: Path,
    vault: ExternalLabelVault,
    run_id: str,
    method: str,
    seed: int | None,
    fixed_epoch: int | None,
    predictions: np.ndarray,
    scores: np.ndarray,
    sample_ids: tuple[str, ...],
    groups: tuple[str, ...],
    prediction_relative: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction_path = output_root / prediction_relative
    artifact = write_prediction_bundle(
        prediction_path,
        relative_path=prediction_relative,
        sample_ids=sample_ids,
        measurement_sets=groups,
        predictions=predictions,
        scores=scores,
    )
    vault.freeze_predictions(run_id, prediction_path)
    truth = vault.labels_for_scoring(run_id)
    metrics = metric_bundle(truth, predictions)
    per_group = grouped_metrics(truth, predictions, groups)
    return (
        {
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "fixed_epoch": fixed_epoch,
            "metrics": metrics,
            "per_measurement_set_metrics": per_group,
            "prediction_artifact": artifact,
        },
        {"truth": truth, "prediction": predictions},
    )


def _write_run_outputs(
    output_root: Path,
    runs: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    root = output_root / "ev3r"
    write_csv(
        root / "execution_manifest.csv",
        execution_rows,
        [
            "run_id",
            "method",
            "seed",
            "representation",
            "train_count",
            "test_count",
            "train_class_counts",
            "test_class_counts",
            "feature_length",
            "preprocessing_state_path",
            "preprocessing_sha256",
            "estimator_or_checkpoint_path",
            "estimator_or_checkpoint_sha256",
            "prediction_path",
            "prediction_sha256",
            "training_history_path",
            "source_weights_used",
            "test_selection",
        ],
    )
    metric_rows = []
    recall_rows = []
    for run in runs:
        metrics = run["metrics"]
        metric_rows.append(
            {
                "run_id": run["run_id"],
                "method": run["method"],
                "seed": "" if run["seed"] is None else run["seed"],
                "fixed_epoch": ""
                if run["fixed_epoch"] is None
                else run["fixed_epoch"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        for label in range(4):
            recall_rows.append(
                {
                    "run_id": run["run_id"],
                    "method": run["method"],
                    "seed": "" if run["seed"] is None else run["seed"],
                    "class_label": label,
                    "recall": metrics["per_class_recall"][str(label)],
                    "support": metrics["support"][str(label)],
                }
            )
        write_json(
            root / "confusion_matrices" / f"{run['run_id']}.json",
            {
                "run_id": run["run_id"],
                "labels": [0, 1, 2, 3],
                "pooled": metrics["confusion_matrix"],
                "per_measurement_set": {
                    group: values["confusion_matrix"]
                    for group, values in run["per_measurement_set_metrics"].items()
                },
            },
        )
    write_csv(
        root / "per_run_metrics.csv",
        metric_rows,
        [
            "run_id",
            "method",
            "seed",
            "fixed_epoch",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
        ],
    )
    write_csv(
        root / "class_recall.csv",
        recall_rows,
        ["run_id", "method", "seed", "class_label", "recall", "support"],
    )
    method_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for method in (
        "majority_class_baseline",
        "pca_logistic_regression",
        "raw_signal_cnn",
        "first_difference_cnn",
    ):
        selected = [run for run in runs if run["method"] == method]
        if len(selected) == 1:
            metrics = selected[0]["metrics"]
            summary = {
                name: {
                    "count": 1,
                    "mean": float(metrics[name]),
                    "sample_standard_deviation": 0.0,
                    "population_standard_deviation": 0.0,
                    "minimum": float(metrics[name]),
                    "maximum": float(metrics[name]),
                }
                for name in ("accuracy", "macro_f1", "balanced_accuracy")
            }
        else:
            summary = summarize_cnn_runs(selected)
        aggregate[method] = summary
        method_rows.append(
            {
                "method": method,
                "run_count": len(selected),
                "accuracy_mean": summary["accuracy"]["mean"],
                "accuracy_sample_std": summary["accuracy"][
                    "sample_standard_deviation"
                ],
                "macro_f1_mean": summary["macro_f1"]["mean"],
                "macro_f1_sample_std": summary["macro_f1"][
                    "sample_standard_deviation"
                ],
                "balanced_accuracy_mean": summary["balanced_accuracy"]["mean"],
                "balanced_accuracy_sample_std": summary["balanced_accuracy"][
                    "sample_standard_deviation"
                ],
            }
        )
    write_csv(
        root / "per_method_metrics.csv",
        method_rows,
        [
            "method",
            "run_count",
            "accuracy_mean",
            "accuracy_sample_std",
            "macro_f1_mean",
            "macro_f1_sample_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_sample_std",
        ],
    )
    write_json(
        root / "aggregate_metrics.json",
        {
            "methods": aggregate,
            "seed_weighting": "equal",
            "historical_standard_deviation": "sample_ddof_1",
            "ev4_additional_standard_deviation": "population_ddof_0",
        },
    )
    (root / "EV3R_RESULTS.md").write_text(
        "# EV3R-R1 reproduced results\n\n"
        f"Majority Macro-F1: {aggregate['majority_class_baseline']['macro_f1']['mean']:.12f}.\n\n"
        f"PCA-logistic Macro-F1: {aggregate['pca_logistic_regression']['macro_f1']['mean']:.12f}.\n\n"
        f"Raw CNN mean Macro-F1: {aggregate['raw_signal_cnn']['macro_f1']['mean']:.12f}.\n\n"
        f"First-difference CNN mean Macro-F1: {aggregate['first_difference_cnn']['macro_f1']['mean']:.12f}.\n",
        encoding="utf-8",
    )
    return aggregate


def _verify_historical_aggregate(reference_root: Path) -> dict[str, Any]:
    manifest = _read_json(reference_root / "EV3R_R1_ARTIFACT_HASHES.json")
    failures = []
    lines = []
    for entry in manifest["entries"]:
        path = reference_root / entry["relative_path"]
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
            failures.append(entry["relative_path"])
        lines.append(
            f"{entry['relative_path']}\t{entry['size_bytes']}\t{entry['sha256']}"
        )
    recomputed = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {
        "declared_sha256": manifest["aggregate_sha256"],
        "recomputed_sha256": recomputed,
        "expected_sha256": HISTORICAL_AGGREGATE_SHA256,
        "entry_count": len(manifest["entries"]),
        "entry_failures": failures,
        "exact_match": (
            not failures
            and recomputed
            == manifest["aggregate_sha256"]
            == HISTORICAL_AGGREGATE_SHA256
        ),
    }


def _compare_reproduction(
    output_root: Path,
    reference_root: Path,
    reference_records: list[dict[str, Any]],
    reference_paths: dict[str, Path],
    runs: list[dict[str, Any]],
    prediction_arrays: dict[str, np.ndarray],
    data_sha256: str,
) -> dict[str, Any]:
    root = output_root / "comparison"
    reference_by_run = {row["run_id"]: row for row in reference_records}
    prediction_rows = []
    metric_rows = []
    all_predictions_match = True
    all_metrics_match = True
    for run in runs:
        run_id = run["run_id"]
        reference_truth, reference_prediction = _read_reference_prediction(
            reference_paths[run_id]
        )
        prediction = prediction_arrays[run_id]
        prediction_match = bool(np.array_equal(prediction, reference_prediction))
        all_predictions_match &= prediction_match
        prediction_rows.append(
            {
                "run_id": run_id,
                "sample_count": len(prediction),
                "exact_prediction_match": prediction_match,
                "mismatch_count": int(np.count_nonzero(prediction != reference_prediction)),
                "reference_truth_matches_reproduction_truth": bool(
                    np.array_equal(reference_truth, run["_truth"])
                ),
                "reproduced_prediction_array_sha256": array_sha256(prediction),
            }
        )
        reference_metrics = reference_by_run[run_id]["metrics"]
        deltas = {
            name: abs(float(run["metrics"][name]) - float(reference_metrics[name]))
            for name in ("accuracy", "macro_f1", "balanced_accuracy")
        }
        confusion_match = (
            run["metrics"]["confusion_matrix"]
            == reference_metrics["confusion_matrix"]
        )
        metric_match = max(deltas.values()) <= 1e-12 and confusion_match
        all_metrics_match &= metric_match
        metric_rows.append(
            {
                "run_id": run_id,
                "accuracy_absolute_delta": deltas["accuracy"],
                "macro_f1_absolute_delta": deltas["macro_f1"],
                "balanced_accuracy_absolute_delta": deltas["balanced_accuracy"],
                "confusion_matrix_exact_match": confusion_match,
                "tolerance_match": metric_match,
            }
        )
    write_csv(
        root / "prediction_comparison.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    write_csv(root / "metric_comparison.csv", metric_rows, list(metric_rows[0]))
    historical = _verify_historical_aggregate(reference_root)
    summary_rows = [
        {
            "comparison": "raw_data_aggregate",
            "result": "EXACT_MATCH"
            if data_sha256
            == "8ca2f55b35ce774d58371faf387fba1a1e14aa2497635db8b895231aa5a40ff8"
            else "MISMATCH",
        },
        {
            "comparison": "per_run_predictions",
            "result": "EXACT_MATCH" if all_predictions_match else "MISMATCH",
        },
        {
            "comparison": "per_run_metrics",
            "result": "TOLERANCE_MATCH" if all_metrics_match else "MISMATCH",
        },
        {
            "comparison": "historical_ev3r_r1_aggregate_self_integrity",
            "result": "EXACT_MATCH" if historical["exact_match"] else "MISMATCH",
        },
    ]
    write_csv(
        root / "reproduced_vs_authoritative.csv",
        summary_rows,
        ["comparison", "result"],
    )
    write_json(root / "historical_aggregate_verification.json", historical)
    comparison = {
        "all_prediction_arrays_exact": all_predictions_match,
        "all_metrics_within_tolerance": all_metrics_match,
        "historical_aggregate_exact": historical["exact_match"],
        "historical_aggregate_sha256": HISTORICAL_AGGREGATE_SHA256,
    }
    (root / "REPRODUCTION_COMPARISON.md").write_text(
        "# Reproduction comparison\n\n"
        f"- Per-run prediction arrays exact: `{all_predictions_match}`\n"
        f"- Metrics within 1e-12: `{all_metrics_match}`\n"
        f"- Historical aggregate independently recomputed: `{historical['exact_match']}`\n\n"
        "The public artifact aggregate is intentionally not byte-identical to the "
        "historical branch because the package structure and serialization containers "
        "differ; scientific arrays and metrics are the comparison targets.\n",
        encoding="utf-8",
    )
    return comparison


def _execute(config: dict[str, Any]) -> dict[str, Any]:
    validate_external_mode(str(config["mode"]))
    authorize_external_data(str(config["dataset_id"]), str(config["data_root"]))
    if bool(config.get("execute_ev4_in_project")):
        raise ProtocolViolation("EV4 must be executed by the independent private verifier")
    output_root = Path(config["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    authority_path = ROOT / str(config["data_authority_path"])
    protocol_path = ROOT / str(config["protocol_path"])
    label_path = ROOT / str(config["label_access_path"])
    authority = _read_json(authority_path)
    protocol = _read_json(protocol_path)
    config_paths = [
        ROOT / "configs" / "external_data1" / "canonical.yaml",
        authority_path,
        protocol_path,
        label_path,
    ]
    started = time.perf_counter()
    data = load_authoritative_data1(
        config["data_root"],
        expected_file_sha256=authority["expected_file_sha256"],
        expected_aggregate_sha256=authority["raw_aggregate_sha256"],
    )
    identity = validate_loaded_identity(data.sets)
    _write_ev0(output_root, data, identity, authority)
    _write_ev1(output_root, authority)
    split = validate_frozen_split(data)
    protocol_sha256, candidate_registry = _write_ev2r(
        output_root, data, protocol, authority, config_paths
    )
    reference_root = Path(config["reference_root"])
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)
    reference_records, reference_paths = _freeze_comparison_inputs(
        output_root, reference_root
    )
    vault = ExternalLabelVault(
        data.sealed_test_labels,
        data.test_features.sample_ids,
        protocol_sha256=protocol_sha256,
    )
    candidate_registry_sha256 = vault.freeze_candidate_registry(candidate_registry)
    label_root = output_root / "label_access"
    write_json(
        label_root / "test_access_authorization.json",
        {
            "protocol_sha256": protocol_sha256,
            "candidate_registry_sha256": candidate_registry_sha256,
            "prediction_first_scoring": True,
            "single_final_metric_access_per_run": True,
            "historical_manifest_label_disclosure": True,
            "historical_selection_influence": False,
        },
    )
    runs: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    train_counts = [int(np.count_nonzero(data.train.labels == label)) for label in CLASS_ORDER]
    test_counts = [
        int(np.count_nonzero(data.sealed_test_labels == label)) for label in CLASS_ORDER
    ]

    tie = majority_tie_decision(data.train.labels)
    run_id = "EV3R_R1_MAJORITY_BASELINE"
    tie_sha256 = canonical_json_sha256(tie)
    vault.register_frozen_model(
        run_id,
        method="majority_class_baseline",
        model_sha256=tie_sha256,
        preprocessing_sha256=None,
    )
    prediction, scores = predict_majority(data.test_features.sample_count)
    scored, arrays = _score_frozen_prediction(
        output_root=output_root,
        vault=vault,
        run_id=run_id,
        method="majority_class_baseline",
        seed=None,
        fixed_epoch=None,
        predictions=prediction,
        scores=scores,
        sample_ids=data.test_features.sample_ids,
        groups=data.test_features.measurement_sets,
        prediction_relative=f"ev3r/predictions/{run_id}.npz",
    )
    scored.update(
        {
            "representation": "constant_class",
            "feature_length": 1601,
            "model": tie,
        }
    )
    scored["_truth"] = arrays["truth"]
    runs.append(scored)
    prediction_arrays[run_id] = prediction
    execution_rows.append(
        {
            "run_id": run_id,
            "method": "majority_class_baseline",
            "seed": "",
            "representation": "constant_class",
            "train_count": 5600,
            "test_count": 1850,
            "train_class_counts": json.dumps(train_counts),
            "test_class_counts": json.dumps(test_counts),
            "feature_length": 1601,
            "preprocessing_state_path": "",
            "preprocessing_sha256": "",
            "estimator_or_checkpoint_path": "ev2r_protocol/majority_tie_rule.json",
            "estimator_or_checkpoint_sha256": tie_sha256,
            "prediction_path": scored["prediction_artifact"]["relative_path"],
            "prediction_sha256": scored["prediction_artifact"]["file_sha256"],
            "training_history_path": "",
            "source_weights_used": False,
            "test_selection": False,
        }
    )

    run_id = "EV3R_R1_REF_SPLIT00"
    pca_state_relative = "ev3r/estimators/pca_logistic_state.npz"
    pca_state_path = output_root / pca_state_relative
    pca_started = time.perf_counter()
    pca_metadata = fit_and_save(data.train.inputs, data.train.labels, pca_state_path)
    pca_fit_seconds = time.perf_counter() - pca_started
    pca_sha256 = sha256_file(pca_state_path)
    vault.register_preprocessing(
        "pca_logistic_regression", pca_sha256, fit_partition="set_3"
    )
    vault.register_frozen_model(
        run_id,
        method="pca_logistic_regression",
        model_sha256=pca_sha256,
        preprocessing_sha256=pca_sha256,
    )
    prediction, scores = predict_from_saved_state(
        pca_state_path, data.test_features.inputs
    )
    scored, arrays = _score_frozen_prediction(
        output_root=output_root,
        vault=vault,
        run_id=run_id,
        method="pca_logistic_regression",
        seed=42,
        fixed_epoch=None,
        predictions=prediction,
        scores=scores,
        sample_ids=data.test_features.sample_ids,
        groups=data.test_features.measurement_sets,
        prediction_relative=f"ev3r/predictions/{run_id}.npz",
    )
    scored.update(
        {
            "representation": "raw",
            "feature_length": 1601,
            "preprocessing_state": {
                "relative_path": pca_state_relative,
                "sha256": pca_sha256,
                "fit_partition": "set_3",
            },
            "estimator": {
                "relative_path": pca_state_relative,
                "sha256": pca_sha256,
                "reload_verified": True,
                "retained_components": pca_metadata.retained_components,
                "explained_variance_ratio_sum": pca_metadata.explained_variance_ratio_sum,
                "converged": pca_metadata.converged,
                "n_iter": list(pca_metadata.n_iter),
                "warnings": list(pca_metadata.warnings),
                "specification": pca_specification(),
            },
            "fit_runtime_seconds": pca_fit_seconds,
        }
    )
    scored["_truth"] = arrays["truth"]
    runs.append(scored)
    prediction_arrays[run_id] = prediction
    execution_rows.append(
        {
            "run_id": run_id,
            "method": "pca_logistic_regression",
            "seed": 42,
            "representation": "raw",
            "train_count": 5600,
            "test_count": 1850,
            "train_class_counts": json.dumps(train_counts),
            "test_class_counts": json.dumps(test_counts),
            "feature_length": 1601,
            "preprocessing_state_path": pca_state_relative,
            "preprocessing_sha256": pca_sha256,
            "estimator_or_checkpoint_path": pca_state_relative,
            "estimator_or_checkpoint_sha256": pca_sha256,
            "prediction_path": scored["prediction_artifact"]["relative_path"],
            "prediction_sha256": scored["prediction_artifact"]["file_sha256"],
            "training_history_path": "",
            "source_weights_used": False,
            "test_selection": False,
        }
    )

    for representation, method, prefix in (
        ("raw", "raw_signal_cnn", "RAW"),
        ("first_difference", "first_difference_cnn", "DIFF"),
    ):
        preprocessing_relative = f"ev3r/preprocessing_states/{representation}.npz"
        transformed_train, state = prepare_representation(
            data.train.inputs,
            representation,
            output_root / preprocessing_relative,
        )
        preprocessing_sha256 = sha256_file(output_root / preprocessing_relative)
        vault.register_preprocessing(
            method, preprocessing_sha256, fit_partition="set_3"
        )
        transformed_test = transform_test(data.test_features.inputs, state)
        for seed in SEEDS:
            fixed_epoch = EPOCHS[seed]
            run_id = f"EV3R_R1_{prefix}_SPLIT00_SEED{seed}"
            history_records: list[dict[str, object]] = []
            model, training = train_cnn(
                transformed_train,
                data.train.labels,
                seed=seed,
                epochs=fixed_epoch,
                progress=history_records.append,
            )
            history_relative = f"ev3r/training_histories/{run_id}.json"
            write_json(
                output_root / history_relative,
                {
                    "run_id": run_id,
                    "seed": seed,
                    "fixed_epoch": fixed_epoch,
                    "test_access_during_training": False,
                    "epochs": history_records,
                },
            )
            checkpoint_relative = f"ev3r/checkpoints/{run_id}.pt"
            reloaded_model, checkpoint = save_and_reload_checkpoint(
                output_root / checkpoint_relative,
                model,
                run_id=run_id,
                representation=representation,
                seed=seed,
                fixed_epoch=fixed_epoch,
                preprocessing_sha256=preprocessing_sha256,
            )
            vault.register_frozen_model(
                run_id,
                method=method,
                model_sha256=str(checkpoint["model_state_sha256"]),
                preprocessing_sha256=preprocessing_sha256,
            )
            prediction, scores = infer_cnn(reloaded_model, transformed_test)
            scored, arrays = _score_frozen_prediction(
                output_root=output_root,
                vault=vault,
                run_id=run_id,
                method=method,
                seed=seed,
                fixed_epoch=fixed_epoch,
                predictions=prediction,
                scores=scores,
                sample_ids=data.test_features.sample_ids,
                groups=data.test_features.measurement_sets,
                prediction_relative=f"ev3r/predictions/{run_id}.npz",
            )
            scored.update(
                {
                    "representation": representation,
                    "feature_length": state.feature_count,
                    "preprocessing_state": {
                        "relative_path": preprocessing_relative,
                        "sha256": preprocessing_sha256,
                        "fit_partition": "set_3",
                    },
                    "checkpoint": {
                        "relative_path": checkpoint_relative,
                        **checkpoint,
                    },
                    "training_history": {
                        "relative_path": history_relative,
                        "sha256": sha256_file(output_root / history_relative),
                    },
                    "training_runtime_seconds": training[
                        "training_runtime_seconds"
                    ],
                    "initial_model_state_sha256": training[
                        "initial_model_state_sha256"
                    ],
                    "source_weights_used": False,
                    "test_metric_during_training": False,
                    "test_selected_checkpoint": False,
                }
            )
            scored["_truth"] = arrays["truth"]
            runs.append(scored)
            prediction_arrays[run_id] = prediction
            execution_rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "seed": seed,
                    "representation": representation,
                    "train_count": 5600,
                    "test_count": 1850,
                    "train_class_counts": json.dumps(train_counts),
                    "test_class_counts": json.dumps(test_counts),
                    "feature_length": state.feature_count,
                    "preprocessing_state_path": preprocessing_relative,
                    "preprocessing_sha256": preprocessing_sha256,
                    "estimator_or_checkpoint_path": checkpoint_relative,
                    "estimator_or_checkpoint_sha256": checkpoint[
                        "file_sha256"
                    ],
                    "prediction_path": scored["prediction_artifact"][
                        "relative_path"
                    ],
                    "prediction_sha256": scored["prediction_artifact"][
                        "file_sha256"
                    ],
                    "training_history_path": history_relative,
                    "source_weights_used": False,
                    "test_selection": False,
                }
            )

    write_csv(
        label_root / "label_access_log.csv",
        vault.log_rows,
        [
            "sequence",
            "run_id",
            "event",
            "prediction_sha256",
            "selection_influence",
        ],
    )
    metric_rows = [
        row
        for row in vault.log_rows
        if row["event"] == "TEST_LABELS_UNSEALED_FOR_FINAL_SCORING"
    ]
    write_csv(
        label_root / "metric_access_log.csv",
        metric_rows,
        [
            "sequence",
            "run_id",
            "event",
            "prediction_sha256",
            "selection_influence",
        ],
    )
    (label_root / "TEST_LABEL_GOVERNANCE.md").write_text(
        "# Test-label governance\n\n"
        "The historical split manifest encoded held-out labels before execution, "
        "but its access ledger records no fitting or selection influence. In this "
        "reproduction, each final scoring access additionally required a serialized "
        "and hashed prediction artifact, with one access per run.\n",
        encoding="utf-8",
    )
    aggregate = _write_run_outputs(output_root, runs, execution_rows)
    comparison = _compare_reproduction(
        output_root,
        reference_root,
        reference_records,
        reference_paths,
        runs,
        prediction_arrays,
        data.raw_aggregate_sha256,
    )
    ev3r_integrity = {
        "learned_runs_expected": 11,
        "learned_runs_completed": 11,
        "context_baseline_completed": True,
        "test_metric_during_training": False,
        "test_selected_checkpoint": False,
        "source_weights_used": False,
        "prediction_first_scored_runs": vault.scored_run_count,
        "prediction_comparison_exact": comparison["all_prediction_arrays_exact"],
        "metric_comparison_pass": comparison["all_metrics_within_tolerance"],
    }
    write_json(output_root / "ev3r" / "EV3R_INTEGRITY.json", ev3r_integrity)
    conclusions = {
        "DATA1_LEARNABILITY_CLEAR",
        "CNN_PIPELINE_PORTABILITY_SUPPORTED",
        "FIRST_DIFFERENCE_PORTABILITY_NOT_SUPPORTED",
    }
    validate_external_claim(SCIENTIFIC_CLAIM_TYPE, PRIMARY_CLAIM, conclusions)
    methods = [
        {
            "method": method,
            "accuracy": values["accuracy"]["mean"],
            "macro_f1": values["macro_f1"]["mean"],
        }
        for method, values in aggregate.items()
    ]
    result = {
        "status": "EV3R_R1_REPRODUCTION_COMPLETE_AWAITING_INDEPENDENT_EV4",
        "public_branch_name": PUBLIC_BRANCH_NAME,
        "scientific_claim_type": SCIENTIFIC_CLAIM_TYPE,
        "primary_claim": PRIMARY_CLAIM,
        "conclusions": sorted(conclusions),
        "primary_verdict": "PENDING_EV4",
        "ev4_verdict": "PENDING_EV4",
        "methods": methods,
        "comparison": comparison,
        "split": split,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_results_report(output_root / "ev3r" / "EV3R_RESULTS.md", result)
    return result


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    if not execute:
        return build_plan(config, SCIENTIFIC_CLAIM_TYPE, execute)
    return _execute(config)

