"""Execution, paired analysis, and audit for the frozen-C1 linear-readout control."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p4_factor_aware.artifacts import write_csv, write_json, write_text
from p4_factor_aware.constants import (
    CHECKPOINT_FILE_HASHES,
    CHECKPOINT_STATE_HASHES,
    PREPROCESSING_FILES,
    QUERY_FOLDS,
)
from p4_factor_aware.data import GovernedP4, load_governed_p4
from p4_factor_aware.metrics import confusion_matrix, metrics_from_confusion
from p4_factor_aware.models import (
    extract_embeddings,
    load_frozen_model,
    load_preprocessing,
    model_state_sha256,
    preprocess,
)
from p4_factor_aware.protocol import ProtocolViolation, QueryLabelSeal
from p4_large_calibration.statistics import (
    aggregate_metric_from_histograms,
    block_layout,
    extended_metrics,
    hierarchical_bootstrap,
    percentile_interval,
    prediction_histograms_by_block,
)

from . import REPOSITORY_ROOT
from .constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINT_RELATIVE,
    CLAIM_POSITIVE,
    CLAIM_ZERO,
    EXPERIMENT_ID,
    MAJOR_BUDGETS,
    METHOD_LINEAR,
    METHOD_PROTOTYPE,
    METHOD_ZERO,
    PARENT_ARTIFACT_HASHES,
    PARENT_BOOTSTRAP_ACCURACY_SHA256,
    PARENT_BOOTSTRAP_BLOCK_SHA256,
    PARENT_BOOTSTRAP_MACRO_SHA256,
    PARENT_BOOTSTRAP_SOURCE_SHA256,
    PARENT_BOOTSTRAP_SUPPORT_SHA256,
    PARENT_EXECUTION_COMMIT,
    PARENT_FINAL_COMMIT,
    PARENT_PATH_NORMALIZED_JSON_HASHES,
    PARENT_PREDICTION_ARRAY_SHA256,
    PARENT_PREREGISTRATION_SHA256,
    PARENT_QUERY_TRUTH_ARRAY_SHA256,
    PARENT_RESULTS_RELATIVE,
    POSITIVE_BUDGETS,
    PRACTICAL_DELTA,
    PREREGISTRATION_RELATIVE,
    PREREGISTRATION_SHA256,
    PREPROCESSING_RELATIVE,
    SELECTED_LAMBDA,
    SOURCE_SEEDS,
    SOURCE_SELECTION_HASHES,
    SOURCE_SELECTION_RELATIVE,
    SOURCE_SELECTION_SUMMARY,
    SUPPORT_SEEDS,
    TOTAL_BUDGETS,
    budget_label,
)
from .linear import fit_linear_readout, linear_predict
from .manifests import InheritedPlan, load_inherited_plans


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
        encoding="utf-8",
        env=environment,
    ).strip()


def _runtime_identity() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_executable_sha256": sha256_file(sys.executable),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "compute_platform": "cpu",
    }


def _verify_preregistration(repository_root: Path) -> dict[str, object]:
    path = repository_root / PREREGISTRATION_RELATIVE
    observed = sha256_file(path)
    if observed != PREREGISTRATION_SHA256:
        raise ProtocolViolation("PREREGISTRATION_HASH_MISMATCH")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or float(payload["primary_readout"]["regularization"]["selected_lambda"])
        != SELECTED_LAMBDA
        or tuple(payload["data"]["budgets"]) != TOTAL_BUDGETS
    ):
        raise ProtocolViolation("PREREGISTRATION_CONTENT_MISMATCH")
    return payload


def _path_normalized_json_sha256(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: ("<WORKTREE>" if key == "worktree" and isinstance(item, str) else normalize(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    encoded = json.dumps(
        normalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_parent_artifacts(repository_root: Path) -> dict[str, object]:
    parent = repository_root / PARENT_RESULTS_RELATIVE
    rows = []
    for filename, historical_sha256 in PARENT_ARTIFACT_HASHES.items():
        path = parent / filename
        if not path.is_file():
            raise ProtocolViolation(f"PARENT_ARTIFACT_MISSING:{filename}")
        if filename in PARENT_PATH_NORMALIZED_JSON_HASHES:
            observed = _path_normalized_json_sha256(path)
            expected = PARENT_PATH_NORMALIZED_JSON_HASHES[filename]
            hash_mode = "PATH_NORMALIZED_JSON"
        else:
            observed = sha256_file(path)
            expected = historical_sha256
            hash_mode = "RAW_FILE"
        if observed != expected:
            raise ProtocolViolation(f"PARENT_ARTIFACT_HASH_MISMATCH:{filename}")
        rows.append(
            {
                "relative_path": str(path.relative_to(repository_root)).replace("\\", "/"),
                "sha256": observed,
                "historical_raw_sha256": historical_sha256,
                "hash_mode": hash_mode,
                "size_bytes": path.stat().st_size,
            }
        )
    with (parent / "24_FINAL_STATUS.json").open(encoding="utf-8") as handle:
        status = json.load(handle)
    if (
        status.get("classification") != "WEAK_OR_UNRELIABLE_TARGET_CALIBRATION_GAIN"
        or status.get("preregistration_sha256") != PARENT_PREREGISTRATION_SHA256
        or status.get("execution_code_commit") != PARENT_EXECUTION_COMMIT
    ):
        raise ProtocolViolation("PARENT_STATUS_IDENTITY_MISMATCH")
    return {"parent_status": status, "artifact_rows": rows}


def _source_regularization_freeze(archive_repository: Path) -> dict[str, object]:
    root = archive_repository / SOURCE_SELECTION_RELATIVE
    for filename, expected in SOURCE_SELECTION_HASHES.items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolViolation(f"SOURCE_SELECTION_HASH_MISMATCH:{filename}")
    with (root / "FS4_CANDIDATE_SUMMARY.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method_kind"] == "SOURCE_ANCHORED_HEAD"]
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["lambda"])].append(row)
    summary = []
    for value in sorted(grouped):
        selected = grouped[value]
        if {int(row["shot_count"]) for row in selected} != {1, 3, 5}:
            raise ProtocolViolation("SOURCE_SELECTION_SHOT_STRUCTURE_MISMATCH")
        summary.append(
            {
                "lambda": value,
                "minimum_worst_held_position_macro_f1": min(float(row["worst_held_position_mean_macro_f1"]) for row in selected),
                "mean_overall_macro_f1": float(np.mean([float(row["overall_mean_macro_f1"]) for row in selected])),
                "minimum_worst_class_mean_recall": min(float(row["worst_class_mean_recall"]) for row in selected),
                "maximum_zero_recall_class_frequency": max(float(row["zero_recall_class_frequency"]) for row in selected),
                "all_optimizations_completed_finite": all(row["all_optimizations_completed_finite"] == "True" for row in selected),
            }
        )
    winner = max(
        summary,
        key=lambda row: (
            row["minimum_worst_held_position_macro_f1"],
            row["mean_overall_macro_f1"],
            row["minimum_worst_class_mean_recall"],
            -row["maximum_zero_recall_class_frequency"],
            row["lambda"],
        ),
    )
    if float(winner["lambda"]) != SELECTED_LAMBDA:
        raise ProtocolViolation("SOURCE_SELECTION_RESULT_MISMATCH")
    for declared, observed in zip(SOURCE_SELECTION_SUMMARY, summary, strict=True):
        if any(abs(float(declared[key]) - float(observed[key])) > 1e-15 for key in declared):
            raise ProtocolViolation("SOURCE_SELECTION_SUMMARY_MISMATCH")
    return {
        "phase": "PRE_P4_SOURCE_ONLY_REGULARIZATION_FREEZE",
        "candidate_summary_sha256": SOURCE_SELECTION_HASHES["FS4_CANDIDATE_SUMMARY.csv"],
        "selection_population": "frozen P1-P3 FS4 pseudo-target summaries",
        "selection_rule": "preregistered lexicographic global rule across 1/3/5 shots",
        "candidate_rows": summary,
        "selected_lambda": SELECTED_LAMBDA,
        "same_value_all_positive_budgets": True,
        "p4_metrics_used": False,
    }


def _checkpoint_preflight(archive_repository: Path) -> bool:
    for seed in SOURCE_SEEDS:
        path = archive_repository / CHECKPOINT_RELATIVE / f"seed_{seed}/FINAL_CHECKPOINT.pt"
        if not path.is_file() or sha256_file(path) != CHECKPOINT_FILE_HASHES[seed]:
            return False
    for filename, expected in PREPROCESSING_FILES.items():
        path = archive_repository / PREPROCESSING_RELATIVE / filename
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def _load_parent_predictions(repository_root: Path) -> np.ndarray:
    path = repository_root / PARENT_RESULTS_RELATIVE / "per_run_predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        if (
            tuple(payload["budgets"].tolist()) != TOTAL_BUDGETS
            or tuple(payload["source_seeds"].tolist()) != SOURCE_SEEDS
            or tuple(payload["support_seeds"].tolist()) != SUPPORT_SEEDS
        ):
            raise ProtocolViolation("PARENT_PREDICTION_AXES_MISMATCH")
        predictions = np.ascontiguousarray(payload["predictions"], dtype=np.int8)
    if predictions.shape != (10, 5, 20, 3150) or array_sha256(predictions) != PARENT_PREDICTION_ARRAY_SHA256:
        raise ProtocolViolation("PARENT_PREDICTION_ARRAY_MISMATCH")
    return predictions


def _preflight(
    *, repository_root: Path, archive_repository: Path, data_directory: Path
) -> tuple[GovernedP4, dict[tuple[int, int], InheritedPlan], list[dict[str, object]], np.ndarray, dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    preregistration = _verify_preregistration(repository_root)
    parent_binding = _verify_parent_artifacts(repository_root)
    source_freeze = _source_regularization_freeze(archive_repository)
    data = load_governed_p4(data_directory)
    plans, support_audit = load_inherited_plans(repository_root=repository_root, data=data, sha256_file=sha256_file)
    parent_predictions = _load_parent_predictions(repository_root)
    fitter_parameters = set(inspect.signature(fit_linear_readout).parameters)
    gates = [
        {"gate_id": "G01_PREREGISTRATION_HASH", "passed": True, "evidence": PREREGISTRATION_SHA256},
        {"gate_id": "G02_PARENT_ARTIFACT_BINDING", "passed": True, "evidence": f"{len(PARENT_ARTIFACT_HASHES)} parent artifacts hash-match"},
        {"gate_id": "G03_GOVERNED_P4_STRUCTURE", "passed": len(data.signals) == 3150 and len(data._groups) == 63, "evidence": "3150 rows; 63 complete blocks"},
        {"gate_id": "G04_EXACT_PARENT_FOLDS", "passed": True, "evidence": PARENT_ARTIFACT_HASHES["06_OUTER_QUERY_FOLD_MANIFEST.csv"]},
        {"gate_id": "G05_EXACT_PARENT_SUPPORTS", "passed": len(plans) == 60 and len(support_audit) == 540, "evidence": "60 plans; 540 budget validations"},
        {"gate_id": "G06_SUPPORT_QUERY_ROW_DISJOINT", "passed": all(bool(row["row_disjoint"]) for row in support_audit), "evidence": "all inherited prefixes"},
        {"gate_id": "G07_SUPPORT_QUERY_BLOCK_DISJOINT", "passed": all(bool(row["block_disjoint"]) for row in support_audit), "evidence": "all inherited prefixes"},
        {"gate_id": "G08_SUPPORT_QUERY_SIGNAL_DISJOINT", "passed": all(bool(row["exact_signal_disjoint"]) for row in support_audit), "evidence": "all inherited prefixes"},
        {"gate_id": "G09_FROZEN_CHECKPOINT_PREPROCESSING_HASHES", "passed": _checkpoint_preflight(archive_repository), "evidence": "five checkpoints and two preprocessing arrays"},
        {"gate_id": "G10_SOURCE_ONLY_REGULARIZATION", "passed": source_freeze["selected_lambda"] == SELECTED_LAMBDA and not source_freeze["p4_metrics_used"], "evidence": "lambda=100 from frozen P1-P3 FS4 summary"},
        {"gate_id": "G11_QUERY_OBJECT_ABSENT_FROM_FITTER", "passed": not any(name.startswith("query") for name in fitter_parameters), "evidence": sorted(fitter_parameters)},
        {"gate_id": "G12_PARENT_PROTOTYPE_PREDICTIONS", "passed": array_sha256(parent_predictions) == PARENT_PREDICTION_ARRAY_SHA256, "evidence": PARENT_PREDICTION_ARRAY_SHA256},
        {"gate_id": "G13_FIXED_BUDGETS_AND_SEEDS", "passed": len(TOTAL_BUDGETS) == 10 and len(SOURCE_SEEDS) == 5 and len(SUPPORT_SEEDS) == 20, "evidence": "10 budgets x 5 source x 20 support"},
        {"gate_id": "G14_PARENT_BOOTSTRAP_BINDING", "passed": sha256_file(repository_root / PARENT_RESULTS_RELATIVE / "bootstrap_samples.npz") == PARENT_ARTIFACT_HASHES["bootstrap_samples.npz"], "evidence": PARENT_ARTIFACT_HASHES["bootstrap_samples.npz"]},
    ]
    failed = [row for row in gates if not row["passed"]]
    if failed:
        raise ProtocolViolation(f"PREEXECUTION_GATE_FAILURE:{failed}")
    return data, plans, support_audit, parent_predictions, preregistration, parent_binding, source_freeze, gates


def _code_identity(repository_root: Path) -> dict[str, object]:
    branch = _git(repository_root, "branch", "--show-current")
    commit = _git(repository_root, "rev-parse", "HEAD")
    tree = _git(repository_root, "rev-parse", "HEAD^{tree}")
    clean = not bool(_git(repository_root, "status", "--porcelain"))
    if branch != "codex/p4-trainable-linear-readout-v1" or not clean:
        raise ProtocolViolation("FULL_EXECUTION_REQUIRES_CLEAN_ISOLATED_BRANCH")
    if _git(repository_root, "merge-base", "--is-ancestor", PARENT_FINAL_COMMIT, "HEAD") != "":
        pass
    return {"branch": branch, "commit": commit, "tree": tree, "clean_before_execution": clean, "worktree": str(repository_root)}


def _write_preexecution_artifacts(
    *, repository_root: Path, output_directory: Path, code_identity: dict[str, object], preregistration: dict[str, object], parent_binding: dict[str, object], source_freeze: dict[str, object], support_audit: list[dict[str, object]], gates: list[dict[str, object]], data: GovernedP4
) -> None:
    code_files = [
        "workflows/13_p4_trainable_linear_readout/run.py",
        "workflows/13_p4_trainable_linear_readout/plot_results.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/__init__.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/constants.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/linear.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/manifests.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/study.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/sampling.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/statistics.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/artifacts.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/constants.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/data.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/metrics.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/models.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/protocol.py",
        "tests/test_p4_trainable_linear_readout.py",
        str(PREREGISTRATION_RELATIVE).replace("\\", "/"),
    ]
    code_rows = []
    for relative in code_files:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing executed code file: {relative}")
        code_rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(output_directory / "00_EXECUTED_CODE_MANIFEST.csv", code_rows)
    write_json(
        output_directory / "01_EXACT_CONFIGURATION.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "claim_type": CLAIM_POSITIVE,
            "code_identity": code_identity,
            "runtime": _runtime_identity(),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "parent_final_commit": PARENT_FINAL_COMMIT,
            "parent_execution_commit": PARENT_EXECUTION_COMMIT,
            "p4_source_hashes": data.source_hashes,
            "budgets": TOTAL_BUDGETS,
            "source_seeds": SOURCE_SEEDS,
            "support_seeds": SUPPORT_SEEDS,
            "selected_lambda": SELECTED_LAMBDA,
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "methods": {"zero": METHOD_ZERO, "prototype": METHOD_PROTOTYPE, "linear": METHOD_LINEAR},
        },
    )
    write_text(output_directory / "02_PARENT_RESULT_CONTEXT.md", "# Parent result context\n\nThe linear-readout study inherits the frozen P4 calibration fold/support plan, predictions, bootstrap samples, metrics, and query-label bindings. Public execution is hash-bound to those scientific artifacts only; release/audit packaging is not part of the runtime contract.\n")
    write_json(output_directory / "03_PREREGISTRATION_BINDING.json", {"relative_path": str(PREREGISTRATION_RELATIVE).replace("\\", "/"), "sha256": PREREGISTRATION_SHA256, "sealed_before_new_p4_fit": True, "payload": preregistration})
    write_json(output_directory / "04_INHERITED_ARTIFACT_BINDING.json", parent_binding)
    write_json(output_directory / "05_SOURCE_REGULARIZATION_FREEZE.json", source_freeze)
    write_csv(output_directory / "06_SUPPORT_MANIFEST_VALIDATION.csv", support_audit)
    write_csv(output_directory / "07_PREEXECUTION_VALIDITY_AUDIT.csv", gates)


@dataclass
class SealRecord:
    budget: int
    source_seed: int
    support_seed: int | None
    fold: int
    query_indices: np.ndarray
    prediction_sha256: str
    seal: QueryLabelSeal


def _determinism_probe(
    *, data: GovernedP4, plans: dict[tuple[int, int], InheritedPlan], archive_repository: Path
) -> dict[str, object]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    inputs = preprocess(data.signals, mean, scale)
    model, _ = load_frozen_model(archive_repository / CHECKPOINT_RELATIVE / "seed_42/FINAL_CHECKPOINT.pt", 42)
    embeddings = extract_embeddings(model, inputs)
    indices = plans[(0, SUPPORT_SEEDS[0])].indices_for_budget(7)
    arguments = {
        "support_embeddings": embeddings[indices],
        "support_labels": data._labels[indices],
        "anchor_weight": model.network[-1].weight,
        "anchor_bias": model.network[-1].bias,
    }
    first = fit_linear_readout(**arguments)
    second = fit_linear_readout(**arguments)
    query_indices = data.query_view(0).indices
    first_prediction = linear_predict(embeddings[query_indices], first.weight, first.bias)
    second_prediction = linear_predict(embeddings[query_indices], second.weight, second.bias)
    passed = bool(
        first.diagnostics["adapted_head_sha256"] == second.diagnostics["adapted_head_sha256"]
        and np.array_equal(first_prediction, second_prediction)
        and first.diagnostics["optimizer_completed_finite"]
        and second.diagnostics["optimizer_completed_finite"]
    )
    if not passed:
        raise ProtocolViolation("DETERMINISTIC_LINEAR_REFIT_FAILURE")
    return {
        "passed": passed,
        "budget": 7,
        "source_seed": 42,
        "support_seed": SUPPORT_SEEDS[0],
        "fold": 0,
        "head_sha256": first.diagnostics["adapted_head_sha256"],
        "prediction_sha256": array_sha256(first_prediction.astype(np.int8)),
    }


def _generate_predictions(
    *, data: GovernedP4, plans: dict[tuple[int, int], InheritedPlan], parent_predictions: np.ndarray, archive_repository: Path, output_directory: Path
) -> tuple[np.ndarray, list[SealRecord], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    inputs = preprocess(data.signals, mean, scale)
    predictions = np.full((10, 5, 20, 3150), -1, dtype=np.int8)
    predictions[0] = parent_predictions[0]
    records: list[SealRecord] = []
    freeze_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for source_index, source_seed in enumerate(SOURCE_SEEDS):
        checkpoint_path = archive_repository / CHECKPOINT_RELATIVE / f"seed_{source_seed}/FINAL_CHECKPOINT.pt"
        model, payload = load_frozen_model(checkpoint_path, source_seed)
        state_before = model_state_sha256(model)
        embeddings = extract_embeddings(model, inputs)
        anchor_weight = model.network[-1].weight
        anchor_bias = model.network[-1].bias
        for fold in sorted(QUERY_FOLDS):
            query_indices = data.query_view(fold).indices
            query_embeddings = embeddings[query_indices]
            zero_prediction = parent_predictions[0, source_index, 0, query_indices].astype(np.int64)
            zero_seal = QueryLabelSeal(data._labels[query_indices])
            zero_hash = zero_seal.freeze_predictions(zero_prediction)
            records.append(SealRecord(0, source_seed, None, fold, query_indices.copy(), zero_hash, zero_seal))
            freeze_rows.append({"budget": 0, "claim_type": CLAIM_ZERO, "source_seed": source_seed, "support_seed": "", "fold": fold, "query_rows": len(query_indices), "prediction_sha256": zero_hash, "state_at_registration": zero_seal.state, "labels_opened": False})
            for support_index, support_seed in enumerate(SUPPORT_SEEDS):
                plan = plans[(fold, support_seed)]
                for budget_index, budget in enumerate(POSITIVE_BUDGETS, start=1):
                    support_indices = plan.indices_for_budget(budget)
                    fit = fit_linear_readout(
                        support_embeddings=embeddings[support_indices],
                        support_labels=data._labels[support_indices],
                        anchor_weight=anchor_weight,
                        anchor_bias=anchor_bias,
                    )
                    prediction = linear_predict(query_embeddings, fit.weight, fit.bias)
                    predictions[budget_index, source_index, support_index, query_indices] = prediction.astype(np.int8)
                    seal = QueryLabelSeal(data._labels[query_indices])
                    prediction_hash = seal.freeze_predictions(prediction)
                    records.append(SealRecord(budget, source_seed, support_seed, fold, query_indices.copy(), prediction_hash, seal))
                    freeze_rows.append({"budget": budget, "claim_type": CLAIM_POSITIVE, "source_seed": source_seed, "support_seed": support_seed, "fold": fold, "query_rows": len(query_indices), "prediction_sha256": prediction_hash, "state_at_registration": seal.state, "labels_opened": False})
                    fit_rows.append({"budget": budget, "source_seed": source_seed, "support_seed": support_seed, "fold": fold, **fit.diagnostics})
        state_after = model_state_sha256(model)
        model_rows.append(
            {
                "source_seed": source_seed,
                "checkpoint_file_sha256": sha256_file(checkpoint_path),
                "declared_state_sha256": payload["model_state_sha256"],
                "state_before_sha256": state_before,
                "state_after_sha256": state_after,
                "encoder_unchanged": state_before == state_after == CHECKPOINT_STATE_HASHES[source_seed],
                "all_model_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
                "all_model_gradients_none": all(parameter.grad is None for parameter in model.parameters()),
            }
        )
        print(json.dumps({"event": "LINEAR_SOURCE_SEED_FROZEN", "source_seed": source_seed, "fit_units_complete": len(fit_rows)}), flush=True)
    if np.any(predictions < 0) or len(fit_rows) != 2700 or len(records) != 2715:
        raise ProtocolViolation("INCOMPLETE_LINEAR_PREDICTION_MATRIX")
    if not all(bool(row["optimizer_completed_finite"]) for row in fit_rows):
        raise ProtocolViolation("LINEAR_OPTIMIZATION_GATE_FAILURE")
    if not all(record.seal.state == "PREDICTIONS_FROZEN" for record in records):
        raise ProtocolViolation("QUERY_LABEL_OPENED_BEFORE_GLOBAL_FREEZE")
    bundle_path = output_directory / "per_run_linear_predictions.npz"
    np.savez_compressed(
        bundle_path,
        budgets=np.asarray(TOTAL_BUDGETS, dtype=np.int64),
        source_seeds=np.asarray(SOURCE_SEEDS, dtype=np.int64),
        support_seeds=np.asarray(SUPPORT_SEEDS, dtype=np.int64),
        predictions=predictions,
    )
    write_csv(output_directory / "08_LINEAR_FIT_DIAGNOSTICS.csv", fit_rows)
    write_csv(output_directory / "09_PREDICTION_FREEZE_MANIFEST.csv", freeze_rows)
    write_json(
        output_directory / "09A_LINEAR_PREDICTION_BUNDLE_BINDING.json",
        {
            "relative_path": bundle_path.name,
            "sha256": sha256_file(bundle_path),
            "prediction_array_sha256": array_sha256(predictions),
            "shape": predictions.shape,
            "dtype": str(predictions.dtype),
            "registered_prediction_units": len(records),
            "all_predictions_frozen_before_any_query_label_open": True,
        },
    )
    return predictions, records, fit_rows, freeze_rows, model_rows


def _open_truth_after_freeze(
    *, repository_root: Path, data: GovernedP4, records: list[SealRecord]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not records or not all(record.seal.state == "PREDICTIONS_FROZEN" for record in records):
        raise ProtocolViolation("GLOBAL_PREDICTION_FREEZE_INCOMPLETE")
    truth = np.full(3150, -1, dtype=np.int64)
    audit_rows = []
    for record in records:
        labels = record.seal.open_labels()
        existing = truth[record.query_indices]
        if np.any((existing >= 0) & (existing != labels)):
            raise ProtocolViolation("INCONSISTENT_QUERY_LABEL_CUSTODY")
        truth[record.query_indices] = labels
        audit_rows.append(
            {
                "budget": record.budget,
                "source_seed": record.source_seed,
                "support_seed": "" if record.support_seed is None else record.support_seed,
                "fold": record.fold,
                "prediction_sha256": record.prediction_sha256,
                "prediction_frozen_before_open": True,
                "label_access_phase": "FINAL_SCORING_ONLY",
                "seal_state_after_open": record.seal.state,
            }
        )
    if np.any(truth < 0) or array_sha256(truth) != PARENT_QUERY_TRUTH_ARRAY_SHA256:
        raise ProtocolViolation("QUERY_TRUTH_RECONSTRUCTION_MISMATCH")
    inherited_path = repository_root / PARENT_RESULTS_RELATIVE / "query_truth_int64.npy"
    inherited_truth = np.load(inherited_path, allow_pickle=False)
    if not np.array_equal(truth, inherited_truth):
        raise ProtocolViolation("PARENT_QUERY_TRUTH_CONTENT_MISMATCH")
    return truth, audit_rows


def _point_units(values: np.ndarray, budget_index: int) -> np.ndarray:
    return values[budget_index, :, 0] if budget_index == 0 else values[budget_index].reshape(-1)


def _analyse(
    *, data: GovernedP4, truth: np.ndarray, linear_predictions: np.ndarray, prototype_predictions: np.ndarray, fit_rows: list[dict[str, object]], support_audit: list[dict[str, object]], output_directory: Path, bootstrap_replicates: int
) -> dict[str, object]:
    block_keys, block_indices, block_truth = block_layout(data)
    linear_hist = prediction_histograms_by_block(linear_predictions, block_indices)
    prototype_hist = prediction_histograms_by_block(prototype_predictions, block_indices)
    linear_macro, linear_accuracy = aggregate_metric_from_histograms(linear_hist)
    prototype_macro, prototype_accuracy = aggregate_metric_from_histograms(prototype_hist)
    linear_boot = hierarchical_bootstrap(linear_hist, block_truth, replicates=bootstrap_replicates, seed=BOOTSTRAP_SEED)
    prototype_boot = hierarchical_bootstrap(prototype_hist, block_truth, replicates=bootstrap_replicates, seed=BOOTSTRAP_SEED)
    if not (
        np.array_equal(linear_boot.block_weights_sha256_payload, prototype_boot.block_weights_sha256_payload)
        and np.array_equal(linear_boot.source_counts, prototype_boot.source_counts)
        and np.array_equal(linear_boot.support_counts, prototype_boot.support_counts)
    ):
        raise ProtocolViolation("PAIRED_BOOTSTRAP_DRAW_MISMATCH")
    if (
        array_sha256(prototype_boot.macro_f1) != PARENT_BOOTSTRAP_MACRO_SHA256
        or array_sha256(prototype_boot.accuracy) != PARENT_BOOTSTRAP_ACCURACY_SHA256
        or array_sha256(prototype_boot.block_weights_sha256_payload) != PARENT_BOOTSTRAP_BLOCK_SHA256
        or array_sha256(prototype_boot.source_counts) != PARENT_BOOTSTRAP_SOURCE_SHA256
        or array_sha256(prototype_boot.support_counts) != PARENT_BOOTSTRAP_SUPPORT_SHA256
    ):
        raise ProtocolViolation("PARENT_BOOTSTRAP_REPLAY_MISMATCH")

    coverage: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for budget in POSITIVE_BUDGETS:
        selected = [row for row in support_audit if int(row["budget"]) == budget]
        coverage[budget] = (
            float(np.mean([float(row["unique_condition_blocks"]) for row in selected])),
            float(np.mean([float(row["unique_exact_signals"]) for row in selected])),
        )
    fit_by_budget: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in fit_rows:
        fit_by_budget[int(row["budget"])].append(row)

    primary_rows = []
    uncertainty_rows = []
    train_query_rows = []
    response_rows = []
    for budget_index, budget in enumerate(TOTAL_BUDGETS):
        linear_units = _point_units(linear_macro, budget_index)
        prototype_units = _point_units(prototype_macro, budget_index)
        linear_accuracy_units = _point_units(linear_accuracy, budget_index)
        prototype_accuracy_units = _point_units(prototype_accuracy, budget_index)
        linear_point = float(linear_units.mean())
        prototype_point = float(prototype_units.mean())
        linear_acc_point = float(linear_accuracy_units.mean())
        prototype_acc_point = float(prototype_accuracy_units.mean())
        linear_ci = percentile_interval(linear_boot.macro_f1[:, budget_index])
        prototype_ci = percentile_interval(prototype_boot.macro_f1[:, budget_index])
        linear_acc_ci = percentile_interval(linear_boot.accuracy[:, budget_index])
        delta_macro_samples = linear_boot.macro_f1[:, budget_index] - prototype_boot.macro_f1[:, budget_index]
        delta_accuracy_samples = linear_boot.accuracy[:, budget_index] - prototype_boot.accuracy[:, budget_index]
        versus_zero_samples = linear_boot.macro_f1[:, budget_index] - linear_boot.macro_f1[:, 0]
        delta_macro_ci = percentile_interval(delta_macro_samples)
        delta_accuracy_ci = percentile_interval(delta_accuracy_samples)
        versus_zero_ci = percentile_interval(versus_zero_samples)
        support_values = fit_by_budget.get(budget, [])
        support_macro = float(np.mean([float(row["support_macro_f1"]) for row in support_values])) if support_values else float("nan")
        support_accuracy = float(np.mean([float(row["support_accuracy"]) for row in support_values])) if support_values else float("nan")
        condition_blocks, exact_signals = coverage[budget]
        primary_rows.append(
            {
                "budget": budget,
                "budget_label": budget_label(budget),
                "claim_type": CLAIM_ZERO if budget == 0 else CLAIM_POSITIVE,
                "independent_condition_blocks_mean": condition_blocks,
                "unique_exact_signals_mean": exact_signals,
                "prototype_macro_f1": prototype_point,
                "prototype_macro_f1_ci_95_lower": prototype_ci[0],
                "prototype_macro_f1_ci_95_upper": prototype_ci[1],
                "linear_macro_f1": linear_point,
                "linear_macro_f1_ci_95_lower": linear_ci[0],
                "linear_macro_f1_ci_95_upper": linear_ci[1],
                "macro_f1_linear_minus_prototype": linear_point - prototype_point,
                "macro_f1_linear_minus_prototype_ci_95_lower": delta_macro_ci[0],
                "macro_f1_linear_minus_prototype_ci_95_upper": delta_macro_ci[1],
                "macro_f1_linear_minus_zero": linear_point - float(linear_macro[0, :, 0].mean()),
                "macro_f1_linear_minus_zero_ci_95_lower": versus_zero_ci[0],
                "macro_f1_linear_minus_zero_ci_95_upper": versus_zero_ci[1],
                "linear_accuracy": linear_acc_point,
                "linear_accuracy_ci_95_lower": linear_acc_ci[0],
                "linear_accuracy_ci_95_upper": linear_acc_ci[1],
                "prototype_accuracy": prototype_acc_point,
                "accuracy_linear_minus_prototype": linear_acc_point - prototype_acc_point,
                "accuracy_linear_minus_prototype_ci_95_lower": delta_accuracy_ci[0],
                "accuracy_linear_minus_prototype_ci_95_upper": delta_accuracy_ci[1],
                "linear_crossed_unit_macro_f1_sd": float(np.std(linear_units, ddof=1)),
                "linear_source_seed_macro_f1_sd": float(np.std(linear_macro[budget_index].mean(axis=1), ddof=1)),
                "linear_support_seed_macro_f1_sd": 0.0 if budget == 0 else float(np.std(linear_macro[budget_index].mean(axis=0), ddof=1)),
                "support_macro_f1_mean": support_macro,
                "support_accuracy_mean": support_accuracy,
                "support_minus_query_macro_f1": support_macro - linear_point if support_values else float("nan"),
            }
        )
        for contrast, estimate, samples in (
            ("linear", linear_point, linear_boot.macro_f1[:, budget_index]),
            ("prototype", prototype_point, prototype_boot.macro_f1[:, budget_index]),
            ("linear_minus_prototype", linear_point - prototype_point, delta_macro_samples),
            ("linear_minus_zero", linear_point - float(linear_macro[0, :, 0].mean()), versus_zero_samples),
        ):
            interval = percentile_interval(samples)
            uncertainty_rows.append({"budget": budget, "metric": "macro_f1", "contrast": contrast, "estimate": estimate, "ci_95_lower": interval[0], "ci_95_upper": interval[1], "replicates": bootstrap_replicates, "scheme": "PAIRED_TAGID_BLOCK_X_SOURCE_X_SUPPORT"})
        if budget > 0:
            train_query_rows.append(
                {
                    "budget": budget,
                    "fit_unit_count": len(support_values),
                    "support_macro_f1_mean": support_macro,
                    "support_macro_f1_sd": float(np.std([float(row["support_macro_f1"]) for row in support_values], ddof=1)),
                    "support_accuracy_mean": support_accuracy,
                    "held_query_macro_f1": linear_point,
                    "held_query_macro_f1_ci_95_lower": linear_ci[0],
                    "held_query_macro_f1_ci_95_upper": linear_ci[1],
                    "support_minus_query_macro_f1": support_macro - linear_point,
                    "mean_optimizer_iterations": float(np.mean([float(row["optimizer_iterations"]) for row in support_values])),
                    "max_optimizer_iterations": int(max(int(row["optimizer_iterations"]) for row in support_values)),
                    "all_optimizations_completed_finite": all(bool(row["optimizer_completed_finite"]) for row in support_values),
                }
            )
    for current_index in range(1, len(TOTAL_BUDGETS)):
        previous_index = current_index - 1
        samples = linear_boot.macro_f1[:, current_index] - linear_boot.macro_f1[:, previous_index]
        interval = percentile_interval(samples)
        response_rows.append(
            {
                "from_budget": TOTAL_BUDGETS[previous_index],
                "to_budget": TOTAL_BUDGETS[current_index],
                "linear_macro_f1_change": primary_rows[current_index]["linear_macro_f1"] - primary_rows[previous_index]["linear_macro_f1"],
                "linear_macro_f1_change_ci_95_lower": interval[0],
                "linear_macro_f1_change_ci_95_upper": interval[1],
            }
        )

    per_run_rows = []
    per_class_accumulator: dict[tuple[int, int, str], list[dict[str, float]]] = defaultdict(list)
    confusion_accumulator: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for method, predictions in ((METHOD_PROTOTYPE, prototype_predictions), (METHOD_LINEAR, linear_predictions)):
        for budget_index, budget in enumerate(TOTAL_BUDGETS):
            if budget == 0 and method == METHOD_LINEAR:
                continue
            source_range = range(5)
            support_range = range(1) if budget == 0 else range(20)
            for source_index in source_range:
                for support_index in support_range:
                    prediction = predictions[budget_index, source_index, support_index].astype(np.int64)
                    metrics = extended_metrics(truth, prediction)
                    per_run_rows.append(
                        {
                            "budget": budget,
                            "method": METHOD_ZERO if budget == 0 else method,
                            "source_seed": SOURCE_SEEDS[source_index],
                            "support_seed": "" if budget == 0 else SUPPORT_SEEDS[support_index],
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "worst_class_f1": metrics["worst_class_f1"],
                            "prediction_sha256": array_sha256(prediction.astype(np.int8)),
                        }
                    )
                    confusion_accumulator[(budget, METHOD_ZERO if budget == 0 else method)].append(np.asarray(metrics["confusion"], dtype=np.int64))
                    for class_index in range(7):
                        per_class_accumulator[(budget, class_index, METHOD_ZERO if budget == 0 else method)].append({"precision": float(metrics["precision"][class_index]), "recall": float(metrics["recall"][class_index]), "f1": float(metrics["f1"][class_index])})

    per_class_rows = []
    for budget in TOTAL_BUDGETS:
        for class_index in range(7):
            zero_key = METHOD_ZERO if budget == 0 else METHOD_PROTOTYPE
            prototype_values = per_class_accumulator[(budget, class_index, zero_key)]
            linear_values = prototype_values if budget == 0 else per_class_accumulator[(budget, class_index, METHOD_LINEAR)]
            per_class_rows.append(
                {
                    "budget": budget,
                    "class_index": class_index,
                    "TagID": class_index + 1,
                    "prototype_f1_mean": float(np.mean([row["f1"] for row in prototype_values])),
                    "linear_f1_mean": float(np.mean([row["f1"] for row in linear_values])),
                    "linear_minus_prototype_f1": float(np.mean([row["f1"] for row in linear_values]) - np.mean([row["f1"] for row in prototype_values])),
                    "prototype_recall_mean": float(np.mean([row["recall"] for row in prototype_values])),
                    "linear_recall_mean": float(np.mean([row["recall"] for row in linear_values])),
                    "prototype_precision_mean": float(np.mean([row["precision"] for row in prototype_values])),
                    "linear_precision_mean": float(np.mean([row["precision"] for row in linear_values])),
                    "unit_count": len(linear_values),
                }
            )

    confusion_rows = []
    for budget in MAJOR_BUDGETS:
        for method in (METHOD_PROTOTYPE, METHOD_LINEAR):
            matrices = np.stack(confusion_accumulator[(budget, method)]).astype(np.float64)
            mean_matrix = matrices.mean(axis=0)
            normalized = mean_matrix / mean_matrix.sum(axis=1, keepdims=True)
            for true_class in range(7):
                for predicted_class in range(7):
                    confusion_rows.append({"budget": budget, "method": method, "true_class": true_class, "predicted_class": predicted_class, "mean_count_per_unit": float(mean_matrix[true_class, predicted_class]), "row_normalized": float(normalized[true_class, predicted_class]), "unit_count": len(matrices)})

    condition_rows = []
    block_rows = []
    for budget in MAJOR_BUDGETS:
        budget_index = TOTAL_BUDGETS.index(budget)
        for factor_name, factor_values in (("ER", data.er), ("surface", data.surface)):
            for level in range(3):
                mask = np.asarray(factor_values == level)
                record: dict[str, object] = {"budget": budget, "factor": factor_name, "level": level, "rows_per_unit": int(mask.sum())}
                method_values = {}
                for label, predictions in (("prototype", prototype_predictions), ("linear", linear_predictions)):
                    values = []
                    for source_index in range(5):
                        for support_index in range(20):
                            values.append(float(extended_metrics(truth[mask], predictions[budget_index, source_index, support_index, mask])["macro_f1"]))
                    method_values[label] = values
                    record[f"{label}_macro_f1_mean"] = float(np.mean(values))
                    record[f"{label}_macro_f1_sd"] = float(np.std(values, ddof=1))
                record["linear_minus_prototype_macro_f1"] = float(np.mean(method_values["linear"]) - np.mean(method_values["prototype"]))
                condition_rows.append(record)
        for block_index, (key, indices) in enumerate(zip(block_keys, block_indices, strict=True)):
            true_class = key[0]
            prototype_values = (prototype_predictions[budget_index][..., indices] == true_class).mean(axis=-1).reshape(-1)
            linear_values = (linear_predictions[budget_index][..., indices] == true_class).mean(axis=-1).reshape(-1)
            block_rows.append({"budget": budget, "block_index": block_index, "TagID": true_class + 1, "ER": key[1], "surface": key[2], "rows_per_unit": len(indices), "prototype_accuracy_mean": float(prototype_values.mean()), "linear_accuracy_mean": float(linear_values.mean()), "linear_minus_prototype_accuracy": float(linear_values.mean() - prototype_values.mean()), "linear_accuracy_sd": float(np.std(linear_values, ddof=1))})

    bootstrap_path = output_directory / "bootstrap_samples.npz"
    np.savez_compressed(
        bootstrap_path,
        budgets=np.asarray(TOTAL_BUDGETS, dtype=np.int64),
        prototype_macro_f1=prototype_boot.macro_f1,
        linear_macro_f1=linear_boot.macro_f1,
        linear_minus_prototype_macro_f1=linear_boot.macro_f1 - prototype_boot.macro_f1,
        prototype_accuracy=prototype_boot.accuracy,
        linear_accuracy=linear_boot.accuracy,
        linear_minus_prototype_accuracy=linear_boot.accuracy - prototype_boot.accuracy,
        block_weights=linear_boot.block_weights_sha256_payload,
        source_resample_counts=linear_boot.source_counts,
        support_resample_counts=linear_boot.support_counts,
    )
    write_json(
        output_directory / "21_BOOTSTRAP_BINDING.json",
        {
            "relative_path": bootstrap_path.name,
            "sha256": sha256_file(bootstrap_path),
            "prototype_macro_f1_sha256": array_sha256(prototype_boot.macro_f1),
            "linear_macro_f1_sha256": array_sha256(linear_boot.macro_f1),
            "paired_macro_f1_difference_sha256": array_sha256(linear_boot.macro_f1 - prototype_boot.macro_f1),
            "block_weights_sha256": array_sha256(linear_boot.block_weights_sha256_payload),
            "source_counts_sha256": array_sha256(linear_boot.source_counts),
            "support_counts_sha256": array_sha256(linear_boot.support_counts),
            "replicates": bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "parent_bootstrap_reproduced_exactly": True,
        },
    )
    write_csv(output_directory / "12_PER_RUN_METRICS.csv", per_run_rows)
    write_csv(output_directory / "13_PRIMARY_RESULTS.csv", primary_rows)
    write_csv(output_directory / "14_PAIRED_UNCERTAINTY.csv", uncertainty_rows)
    write_csv(output_directory / "15_PER_CLASS_COMPARISON.csv", per_class_rows)
    write_csv(output_directory / "16_CONDITION_COMPARISON.csv", condition_rows)
    write_csv(output_directory / "17_BLOCK_COMPARISON.csv", block_rows)
    write_csv(output_directory / "18_CONFUSION_MATRICES.csv", confusion_rows)
    write_csv(output_directory / "19_TRAIN_VS_QUERY.csv", train_query_rows)
    write_csv(output_directory / "20_LABEL_BUDGET_RESPONSE.csv", response_rows)

    rescue = any(
        int(row["budget"]) >= 35
        and float(row["macro_f1_linear_minus_prototype"]) >= PRACTICAL_DELTA
        and float(row["macro_f1_linear_minus_prototype_ci_95_lower"]) > 0
        and float(row["macro_f1_linear_minus_zero"]) >= PRACTICAL_DELTA
        and float(row["macro_f1_linear_minus_zero_ci_95_lower"]) > 0
        for row in primary_rows[1:]
    )
    partial = any(
        float(row["macro_f1_linear_minus_prototype_ci_95_lower"]) > 0
        or (
            float(row["macro_f1_linear_minus_zero"]) >= PRACTICAL_DELTA
            and float(row["macro_f1_linear_minus_zero_ci_95_lower"]) > 0
        )
        for row in primary_rows[1:]
    )
    if rescue:
        classification = "LINEAR_READOUT_RESCUES_TARGET_CALIBRATION"
    elif partial:
        classification = "PARTIAL_LINEAR_READOUT_BENEFIT"
    else:
        classification = "REPRESENTATION_BOTTLENECK_SUPPORTED"
    final_train = next(row for row in train_query_rows if int(row["budget"]) == 500)
    max_linear = max(float(row["linear_macro_f1"]) for row in primary_rows[1:])
    if classification == "LINEAR_READOUT_RESCUES_TARGET_CALIBRATION":
        next_decision = "NO_ENCODER_FINETUNING_JUSTIFIED"
    elif classification == "REPRESENTATION_BOTTLENECK_SUPPORTED":
        next_decision = "ENCODER_FINETUNING_DIAGNOSTIC_JUSTIFIED"
    elif classification == "PARTIAL_LINEAR_READOUT_BENEFIT" and max_linear < 0.30 and float(final_train["support_minus_query_macro_f1"]) >= 0.20:
        next_decision = "ENCODER_FINETUNING_DIAGNOSTIC_JUSTIFIED"
    else:
        next_decision = "NO_ENCODER_FINETUNING_JUSTIFIED"
    return {
        "classification": classification,
        "next_experiment_decision": next_decision,
        "primary_rows": primary_rows,
        "train_query_rows": train_query_rows,
        "response_rows": response_rows,
        "per_class_rows": per_class_rows,
        "condition_rows": condition_rows,
        "confusion_rows": confusion_rows,
        "linear_bootstrap_sha256": array_sha256(linear_boot.macro_f1),
        "paired_bootstrap_sha256": array_sha256(linear_boot.macro_f1 - prototype_boot.macro_f1),
    }


def _main_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Budget | Blocks | Prototype Macro-F1 | Linear Macro-F1 | Delta linear-prototype [95% CI] | Linear Accuracy | Delta linear-zero |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['budget'])} | {float(row['independent_condition_blocks_mean']):.1f} | "
            f"{float(row['prototype_macro_f1']):.4f} | {float(row['linear_macro_f1']):.4f} | "
            f"{float(row['macro_f1_linear_minus_prototype']):+.4f} "
            f"[{float(row['macro_f1_linear_minus_prototype_ci_95_lower']):.4f}, {float(row['macro_f1_linear_minus_prototype_ci_95_upper']):.4f}] | "
            f"{float(row['linear_accuracy']):.4f} | {float(row['macro_f1_linear_minus_zero']):+.4f} |"
        )
    return "\n".join(lines)


def _write_reports(
    *, output_directory: Path, analysis: dict[str, object], code_identity: dict[str, object], determinism_probe: dict[str, object], gates: list[dict[str, object]]
) -> None:
    rows = analysis["primary_rows"]
    positive = rows[1:]
    best_difference = max(positive, key=lambda row: float(row["macro_f1_linear_minus_prototype"]))
    final = next(row for row in rows if int(row["budget"]) == 500)
    train_final = next(row for row in analysis["train_query_rows"] if int(row["budget"]) == 500)
    class_final = [row for row in analysis["per_class_rows"] if int(row["budget"]) == 500]
    weakest = min(class_final, key=lambda row: float(row["linear_f1_mean"]))
    tag1 = next(row for row in class_final if int(row["TagID"]) == 1)
    tag6 = next(row for row in class_final if int(row["TagID"]) == 6)
    condition_final = [row for row in analysis["condition_rows"] if int(row["budget"]) == 500]
    weakest_condition = min(condition_final, key=lambda row: float(row["linear_macro_f1_mean"]))
    all_increments_positive = all(float(row["linear_macro_f1_change"]) >= 0 for row in analysis["response_rows"])
    interpretation = f"""# Scientific interpretation

Classification: `{analysis['classification']}`.

The largest point difference between the trainable linear readout and the matched
cosine prototype occurred at {int(best_difference['budget'])} labels:
{float(best_difference['macro_f1_linear_minus_prototype']):+.4f} Macro-F1, 95% CI
[{float(best_difference['macro_f1_linear_minus_prototype_ci_95_lower']):.4f},
{float(best_difference['macro_f1_linear_minus_prototype_ci_95_upper']):.4f}]. At 500
labels the linear readout achieved Macro-F1 {float(final['linear_macro_f1']):.4f},
compared with prototype {float(final['prototype_macro_f1']):.4f}. The response was
{'monotonic' if all_increments_positive else 'not monotonic'} across the registered budgets.

At 500 labels, mean support Macro-F1 was {float(train_final['support_macro_f1_mean']):.4f}
while held-condition query Macro-F1 was {float(train_final['held_query_macro_f1']):.4f},
a support-minus-query gap of {float(train_final['support_minus_query_macro_f1']):.4f}.
This contrast distinguishes fitting observed P4 conditions from transfer to unseen
ER-by-surface blocks; it was not used for fitting or model selection.

TagID 1 had linear F1 {float(tag1['linear_f1_mean']):.4f} at 500 labels and changed
{float(tag1['linear_minus_prototype_f1']):+.4f} relative to the prototype. TagID 6 had
linear F1 {float(tag6['linear_f1_mean']):.4f} and changed
{float(tag6['linear_minus_prototype_f1']):+.4f}. The weakest final class was TagID
{int(weakest['TagID'])} (F1 {float(weakest['linear_f1_mean']):.4f}). The weakest
reported physical stratum was {weakest_condition['factor']}={int(weakest_condition['level'])}
(linear Macro-F1 {float(weakest_condition['linear_macro_f1_mean']):.4f}).

The evidence therefore attributes the earlier negative curve according to the sealed
classification rule above. It does not prove a causal representation defect, and it
does not authorize encoder fine-tuning within this branch.
"""
    limitations = """# Limitations

- The diagnostic uses one P4 corpus and the same 63 physical-condition blocks as the parent curve.
- The linear head is source-anchored and conditional on one source-only selected regularisation value.
- Cross-validation evaluates held factor cells, but folds are not independent acquisition campaigns.
- Support labels are raw repeated measurements; independent block and exact-signal coverage remain reported separately.
- Bootstrap intervals generalise over the registered blocks and seeds, not new devices or environments.
- Only linear readout capacity is tested; nonlinear adapters and encoder fine-tuning are outside scope.
"""
    write_text(output_directory / "22_SCIENTIFIC_INTERPRETATION.md", interpretation)
    write_text(output_directory / "23_LIMITATIONS.md", limitations)
    readme = f"""# P4 Trainable Linear-Readout Calibration Curve v1

## A. FINAL CLASSIFICATION

`{analysis['classification']}`

## B. SCIENTIFIC IDENTITY

- Branch: `{code_identity['branch']}`
- Execution commit: `{code_identity['commit']}`
- Preregistration SHA-256: `{PREREGISTRATION_SHA256}`
- Parent result commit: `{PARENT_FINAL_COMMIT}`
- Frozen C1 source seeds: 42--46
- Inherited fold manifest: `{PARENT_ARTIFACT_HASHES['06_OUTER_QUERY_FOLD_MANIFEST.csv']}`
- Inherited support manifest: `{PARENT_ARTIFACT_HASHES['07_SUPPORT_SELECTION_MANIFEST.csv']}`

## C. VALIDITY

All {len(gates)} pre-execution gates passed. All encoder states remained frozen, every
support prefix was row/block/exact-signal disjoint from its query, the regularisation
value was selected from P1--P3 only, and {len(POSITIVE_BUDGETS) * 5 * 20 * 3}
positive-budget predictions were frozen before query-label scoring. The deterministic
refit probe passed with head hash `{determinism_probe['head_sha256']}`.

## D. PRIMARY RESULTS

{_main_table(rows)}

## E. LINEAR VS PROTOTYPE

The largest paired point difference was {float(best_difference['macro_f1_linear_minus_prototype']):+.4f}
at {int(best_difference['budget'])} labels, with 95% CI
[{float(best_difference['macro_f1_linear_minus_prototype_ci_95_lower']):.4f},
{float(best_difference['macro_f1_linear_minus_prototype_ci_95_upper']):.4f}].

## F. LABEL-BUDGET RESPONSE

The linear curve was {'monotonic' if all_increments_positive else 'not monotonic'}.
Its crossed-unit Macro-F1 SD changed from {float(rows[1]['linear_crossed_unit_macro_f1_sd']):.4f}
at 7 labels to {float(final['linear_crossed_unit_macro_f1_sd']):.4f} at 500 labels.

## G. TRAIN VS HELD-CONDITION GAP

At 500 labels: support Macro-F1 {float(train_final['support_macro_f1_mean']):.4f},
held-condition Macro-F1 {float(train_final['held_query_macro_f1']):.4f}, gap
{float(train_final['support_minus_query_macro_f1']):.4f}.

## H. CLASS / CONDITION DIAGNOSTICS

At 500 labels TagID 1 had F1 {float(tag1['linear_f1_mean']):.4f}; TagID 6 had F1
{float(tag6['linear_f1_mean']):.4f}. The weakest class was TagID {int(weakest['TagID'])}.
The weakest reported stratum was {weakest_condition['factor']}={int(weakest_condition['level'])}.

## I. SCIENTIFIC INTERPRETATION

See `22_SCIENTIFIC_INTERPRETATION.md`. Positive budgets are target-assisted and do
not constitute strict source-only domain generalisation.

## J. RELATION TO PREVIOUS CALIBRATION CURVE

The parent folds, supports, encoders, queries, seeds, budgets, metrics, and bootstrap
draws are unchanged. The sole scientific delta is a trainable linear softmax readout
in place of the target-only cosine prototype.

## K. THESIS-READY CONCLUSION

Following the negative frozen-C1 cosine-prototype calibration curve, a matched
block-disjoint control replaced only the target adapter with a source-anchored
trainable linear classifier. The resulting evidence was classified as
`{analysis['classification']}`. At 500 labels the linear readout achieved Macro-F1
{float(final['linear_macro_f1']):.4f}, versus {float(final['prototype_macro_f1']):.4f}
for the matched prototype, while its support-to-held-condition gap was
{float(train_final['support_minus_query_macro_f1']):.4f}. This diagnostic constrains
whether the previous failure is attributable to prototype geometry, frozen
representation geometry, or both, without changing the encoder or using query labels
for adaptation.

## L. NEXT-EXPERIMENT DECISION

`{analysis['next_experiment_decision']}`

Encoder fine-tuning was not executed.
"""
    write_text(output_directory / "README.md", readme)


def finalize_artifacts(output_directory: str | Path) -> dict[str, object]:
    root = Path(output_directory)
    status_path = root / "25_FINAL_STATUS.json"
    if not status_path.is_file():
        raise FileNotFoundError("final status does not exist")
    excluded = {"24_FINAL_AUDIT_MANIFEST.csv", "25_FINAL_STATUS.json"}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in excluded:
            continue
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "manifest_scope": "SCIENTIFIC_OR_REPRODUCIBILITY_ARTIFACT"})
    manifest_path = root / "24_FINAL_AUDIT_MANIFEST.csv"
    write_csv(manifest_path, rows)
    with status_path.open(encoding="utf-8") as handle:
        status = json.load(handle)
    status["final_audit_manifest_sha256"] = sha256_file(manifest_path)
    status["final_audit_manifest_file_count"] = len(rows)
    status["figures_present"] = all(
        (root / "figures" / filename).is_file()
        for filename in ("p4_linear_vs_prototype_calibration.png", "p4_linear_minus_prototype.png", "p4_linear_train_vs_query.png")
    )
    write_json(status_path, status)
    return {"status": "FINAL_ARTIFACT_AUDIT_COMPLETE", "artifact_count": len(rows), "manifest_sha256": status["final_audit_manifest_sha256"], "figures_present": status["figures_present"]}


def run_study(
    *, repository_root: str | Path, archive_repository: str | Path, data_directory: str | Path, output_directory: str | Path, mode: str, bootstrap_replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, object]:
    repository_root = Path(repository_root)
    archive_repository = Path(archive_repository)
    data_directory = Path(data_directory)
    output_directory = Path(output_directory)
    preflight = _preflight(repository_root=repository_root, archive_repository=archive_repository, data_directory=data_directory)
    data, plans, support_audit, parent_predictions, preregistration, parent_binding, source_freeze, gates = preflight
    if mode == "dry-run":
        return {"status": "DRY_RUN_COMPLETE", "gates_passed": len(gates), "support_plan_validations": len(support_audit), "new_p4_predictions": 0}
    determinism_probe = _determinism_probe(data=data, plans=plans, archive_repository=archive_repository)
    if mode == "smoke":
        return {"status": "SMOKE_COMPLETE_WITHOUT_QUERY_SCORING", "gates_passed": len(gates), "determinism_probe": determinism_probe, "query_labels_opened": False}
    if mode != "full":
        raise ValueError("mode must be dry-run, smoke, or full")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ProtocolViolation("FULL_EXECUTION_REQUIRES_PREREGISTERED_BOOTSTRAP_COUNT")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("full output directory must not already contain artifacts")
    output_directory.mkdir(parents=True, exist_ok=True)
    code_identity = _code_identity(repository_root)
    _write_preexecution_artifacts(repository_root=repository_root, output_directory=output_directory, code_identity=code_identity, preregistration=preregistration, parent_binding=parent_binding, source_freeze=source_freeze, support_audit=support_audit, gates=gates, data=data)
    linear_predictions, records, fit_rows, freeze_rows, model_rows = _generate_predictions(data=data, plans=plans, parent_predictions=parent_predictions, archive_repository=archive_repository, output_directory=output_directory)
    if not all(row["encoder_unchanged"] and row["all_model_parameters_frozen"] and row["all_model_gradients_none"] for row in model_rows):
        raise ProtocolViolation("FROZEN_ENCODER_POSTFIT_GATE_FAILURE")
    truth, access_rows = _open_truth_after_freeze(repository_root=repository_root, data=data, records=records)
    write_csv(output_directory / "10_QUERY_LABEL_ACCESS_AUDIT.csv", access_rows)
    write_csv(output_directory / "11_FROZEN_MODEL_INTEGRITY.csv", model_rows)
    analysis = _analyse(data=data, truth=truth, linear_predictions=linear_predictions, prototype_predictions=parent_predictions, fit_rows=fit_rows, support_audit=support_audit, output_directory=output_directory, bootstrap_replicates=bootstrap_replicates)
    _write_reports(output_directory=output_directory, analysis=analysis, code_identity=code_identity, determinism_probe=determinism_probe, gates=gates)
    status = {
        "schema_version": 1,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "classification": analysis["classification"],
        "next_experiment_decision": analysis["next_experiment_decision"],
        "branch": code_identity["branch"],
        "execution_code_commit": code_identity["commit"],
        "execution_code_tree": code_identity["tree"],
        "worktree": code_identity["worktree"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "parent_final_commit": PARENT_FINAL_COMMIT,
        "inherited_fold_manifest_sha256": PARENT_ARTIFACT_HASHES["06_OUTER_QUERY_FOLD_MANIFEST.csv"],
        "inherited_support_manifest_sha256": PARENT_ARTIFACT_HASHES["07_SUPPORT_SELECTION_MANIFEST.csv"],
        "selected_lambda": SELECTED_LAMBDA,
        "budgets": TOTAL_BUDGETS,
        "source_seeds": SOURCE_SEEDS,
        "support_seeds": SUPPORT_SEEDS,
        "bootstrap": {"replicates": bootstrap_replicates, "seed": BOOTSTRAP_SEED},
        "preexecution_gates_passed": len(gates),
        "fit_units": len(fit_rows),
        "prediction_units": len(records),
        "all_predictions_frozen_before_query_scoring": all(not bool(row["labels_opened"]) for row in freeze_rows),
        "determinism_probe": determinism_probe,
        "model_integrity_passed": all(bool(row["encoder_unchanged"]) for row in model_rows),
        "linear_bootstrap_sha256": analysis["linear_bootstrap_sha256"],
        "paired_bootstrap_sha256": analysis["paired_bootstrap_sha256"],
        "primary_results": analysis["primary_rows"],
        "train_vs_query": analysis["train_query_rows"],
        "figures_present": False,
        "final_audit_manifest_sha256": None,
        "final_audit_manifest_file_count": 0,
        "encoder_finetuning_executed": False,
    }
    write_json(output_directory / "25_FINAL_STATUS.json", status)
    finalization = finalize_artifacts(output_directory)
    return {"status": "FULL_EXECUTION_COMPLETE", "classification": analysis["classification"], "next_experiment_decision": analysis["next_experiment_decision"], "fit_units": len(fit_rows), "prediction_units": len(records), "artifact_audit": finalization}
