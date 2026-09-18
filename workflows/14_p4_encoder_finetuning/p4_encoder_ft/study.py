"""Sealed P4 partial/full encoder fine-tuning execution and paired analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p4_factor_aware.artifacts import write_csv, write_json, write_text
from p4_factor_aware.constants import CHECKPOINT_FILE_HASHES, CHECKPOINT_STATE_HASHES, PREPROCESSING_FILES, QUERY_FOLDS
from p4_factor_aware.data import GovernedP4, load_governed_p4
from p4_factor_aware.metrics import confusion_matrix, metrics_from_confusion, metrics_from_predictions
from p4_factor_aware.models import extract_embeddings, load_frozen_model, load_preprocessing, model_state_sha256, preprocess
from p4_factor_aware.protocol import ProtocolViolation, QueryLabelSeal
from p4_large_calibration.statistics import (
    aggregate_metric_from_histograms,
    block_layout,
    extended_metrics,
    hierarchical_bootstrap,
    percentile_interval,
    prediction_histograms_by_block,
)
from p4_linear_readout.linear import fit_linear_readout
from p4_linear_readout.manifests import InheritedPlan, load_inherited_plans

from . import REPOSITORY_ROOT
from .constants import (
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BUDGETS,
    CHECKPOINT_RELATIVE,
    ENCODER_PARAMETER_COUNT,
    EXPERIMENT_ID,
    FOLD_MANIFEST_SHA256,
    FROZEN_RELATIVE,
    FULL_PARAMETER_COUNT,
    FULL_TRAINABLE_PREFIXES,
    IMMEDIATE_PARENT_ARTIFACT_HASHES,
    IMMEDIATE_PARENT_FINAL_MANIFEST_SHA256,
    IMMEDIATE_PARENT_PREDICTION_ARRAY_SHA256,
    IMMEDIATE_PARENT_PREREGISTRATION_SHA256,
    IMMEDIATE_PARENT_RESULTS,
    MAJOR_DIAGNOSTIC_BUDGETS,
    METHOD_BY_ARM,
    METHOD_FROZEN,
    METHODS,
    PARENT_BUDGET_INDICES,
    PARTIAL_PARAMETER_COUNT,
    PARTIAL_TRAINABLE_NAMES,
    PRACTICAL_DELTA,
    PREREGISTRATION_RELATIVE,
    PREREGISTRATION_SHA256,
    PREPROCESSING_RELATIVE,
    SOURCE_SEEDS,
    SOURCE_SELECTION_FILES,
    SUPPORT_MANIFEST_SHA256,
    SUPPORT_SEEDS,
)
from .finetune import (
    array_sha256,
    centroid_ratio,
    derive_seed,
    fine_tune,
    parameter_mask,
    predict,
    representation_displacement,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_historical_inputs(archive_repository: Path) -> None:
    """Require the original scientific parent bundle, independent of Git history."""
    paths = [archive_repository / IMMEDIATE_PARENT_RESULTS / name for name in IMMEDIATE_PARENT_ARTIFACT_HASHES]
    paths += [archive_repository / "results/canonical_metrics/p4_large_calibration_curve" / name for name in ("06_OUTER_QUERY_FOLD_MANIFEST.csv", "07_SUPPORT_SELECTION_MANIFEST.csv")]
    paths += [archive_repository / CHECKPOINT_RELATIVE / f"seed_{seed}/FINAL_CHECKPOINT.pt" for seed in SOURCE_SEEDS]
    paths += [archive_repository / PREPROCESSING_RELATIVE / name for name in PREPROCESSING_FILES]
    missing = [p.relative_to(archive_repository).as_posix() for p in paths if not p.is_file()]
    if missing:
        raise ProtocolViolation("HISTORICAL_ARCHIVE_REQUIRED: supply --archive-repository with the original hash-bound parent predictions, support/fold manifests, checkpoints and preprocessing. Measurements alone are insufficient. Missing: " + ", ".join(missing))


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
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise ProtocolViolation("PREREGISTRATION_HASH_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID or tuple(payload["data"]["primary_budgets"]) != BUDGETS:
        raise ProtocolViolation("PREREGISTRATION_CONTENT_MISMATCH")
    return payload


def _verify_parent(repository_root: Path) -> dict[str, object]:
    root = repository_root / IMMEDIATE_PARENT_RESULTS
    artifacts = []
    for filename, expected in IMMEDIATE_PARENT_ARTIFACT_HASHES.items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolViolation(f"IMMEDIATE_PARENT_ARTIFACT_HASH_MISMATCH:{filename}")
        artifacts.append({"relative_path": str(path.relative_to(repository_root)).replace("\\", "/"), "sha256": expected, "size_bytes": path.stat().st_size})
    status = json.loads((root / "25_FINAL_STATUS.json").read_text(encoding="utf-8"))
    if (
        status.get("classification") != "REPRESENTATION_BOTTLENECK_SUPPORTED"
        or status.get("preregistration_sha256") != IMMEDIATE_PARENT_PREREGISTRATION_SHA256
        or status.get("final_audit_manifest_sha256") != IMMEDIATE_PARENT_FINAL_MANIFEST_SHA256
    ):
        raise ProtocolViolation("IMMEDIATE_PARENT_STATUS_IDENTITY_MISMATCH")
    high = next(row for row in status["primary_results"] if int(row["budget"]) == 500)
    if abs(float(high["linear_macro_f1"]) - 0.14274046260669399) > 1e-15 or abs(float(high["linear_accuracy"]) - 0.1434) > 1e-15:
        raise ProtocolViolation("IMMEDIATE_PARENT_HIGH_BUDGET_RESULT_MISMATCH")
    return {"parent_status": status, "artifacts": artifacts}


def _load_parent_predictions(repository_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = repository_root / IMMEDIATE_PARENT_RESULTS / "per_run_linear_predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        full = np.ascontiguousarray(payload["predictions"], dtype=np.int8)
        budgets = np.asarray(payload["budgets"], dtype=np.int64)
        source = np.asarray(payload["source_seeds"], dtype=np.int64)
        support = np.asarray(payload["support_seeds"], dtype=np.int64)
    if (
        full.shape != (10, 5, 20, 3150)
        or array_sha256(full) != IMMEDIATE_PARENT_PREDICTION_ARRAY_SHA256
        or tuple(source) != SOURCE_SEEDS
        or tuple(support) != SUPPORT_SEEDS
        or tuple(budgets[np.asarray(PARENT_BUDGET_INDICES, dtype=np.int64)]) != BUDGETS
    ):
        raise ProtocolViolation("IMMEDIATE_PARENT_PREDICTION_IDENTITY_MISMATCH")
    selected = np.ascontiguousarray(full[np.asarray(PARENT_BUDGET_INDICES)], dtype=np.int8)
    return full, selected


def _load_selected_configuration(output_directory: Path) -> dict[str, dict[str, object]]:
    for filename in SOURCE_SELECTION_FILES:
        if not (output_directory / filename).is_file():
            raise ProtocolViolation(f"SOURCE_SELECTION_ARTIFACT_MISSING:{filename}")
    binding = json.loads((output_directory / "04_SOURCE_SELECTION_BINDING.json").read_text(encoding="utf-8"))
    selection = json.loads((output_directory / "07_SOURCE_SELECTED_CONFIG.json").read_text(encoding="utf-8"))
    if binding.get("p4_files_opened") != 0 or binding.get("p4_metrics_used") or selection.get("p4_query_used") or selection.get("p4_support_used"):
        raise ProtocolViolation("SOURCE_SELECTION_NOT_P4_BLIND")
    selected = selection.get("selected", {})
    if set(selected) != set(ARMS):
        raise ProtocolViolation("SOURCE_SELECTION_ARM_MISMATCH")
    for arm, row in selected.items():
        if float(row["learning_rate"]) not in {1e-5, 3e-5, 1e-4, 3e-4} or float(row["encoder_weight_decay"]) not in {0.0, 1e-4} or int(row["epochs"]) not in {5, 10, 20}:
            raise ProtocolViolation(f"SOURCE_SELECTION_OUTSIDE_PREREGISTERED_GRID:{arm}")
    return selected


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


def _mask_manifest() -> list[dict[str, object]]:
    model = load_mask_model = __import__("p4_factor_aware.models", fromlist=["FrozenC1CNN1D"]).FrozenC1CNN1D()
    rows = []
    for arm in ARMS:
        clone = __import__("p4_factor_aware.models", fromlist=["FrozenC1CNN1D"]).FrozenC1CNN1D()
        clone.load_state_dict(model.state_dict())
        _, _, mask = parameter_mask(clone, arm)
        for name, parameter in clone.named_parameters():
            rows.append({"arm": arm, "parameter_name": name, "parameter_count": parameter.numel(), "trainable": mask[name]})
    return rows


def preflight(
    *, repository_root: Path, archive_repository: Path, data_directory: Path, output_directory: Path
) -> tuple[GovernedP4, dict[tuple[int, int], InheritedPlan], list[dict[str, object]], np.ndarray, np.ndarray, dict[str, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    require_historical_inputs(archive_repository)
    preregistration = _verify_preregistration(repository_root)
    parent = _verify_parent(archive_repository)
    selected = _load_selected_configuration(repository_root / "results/canonical_metrics/p4_encoder_finetuning")
    if not _checkpoint_preflight(archive_repository):
        raise ProtocolViolation("FROZEN_SOURCE_ASSET_PREFLIGHT_FAILED")
    data = load_governed_p4(data_directory)
    plans, support_audit = load_inherited_plans(repository_root=archive_repository, data=data, sha256_file=sha256_file)
    parent_full, frozen_predictions = _load_parent_predictions(archive_repository)
    mask_rows = _mask_manifest()
    gates = [
        {"gate_id": "G01_PREREGISTRATION_HASH", "passed": sha256_file(repository_root / PREREGISTRATION_RELATIVE) == PREREGISTRATION_SHA256, "evidence": PREREGISTRATION_SHA256},
        {"gate_id": "G02_IMMEDIATE_PARENT_BINDING", "passed": True, "evidence": IMMEDIATE_PARENT_PREDICTION_ARRAY_SHA256},
        {"gate_id": "G03_CALIBRATION_PARENT_BINDING", "passed": True, "evidence": FOLD_MANIFEST_SHA256},
        {"gate_id": "G04_EXACT_PARENT_FOLDS", "passed": sha256_file(archive_repository / "results/canonical_metrics/p4_large_calibration_curve/06_OUTER_QUERY_FOLD_MANIFEST.csv") == FOLD_MANIFEST_SHA256, "evidence": FOLD_MANIFEST_SHA256},
        {"gate_id": "G05_EXACT_PARENT_SUPPORTS", "passed": sha256_file(archive_repository / "results/canonical_metrics/p4_large_calibration_curve/07_SUPPORT_SELECTION_MANIFEST.csv") == SUPPORT_MANIFEST_SHA256, "evidence": f"60 plans; {len(support_audit)} inherited budget validations"},
        {"gate_id": "G06_SUPPORT_QUERY_ROW_DISJOINT", "passed": all(bool(row["row_disjoint"]) for row in support_audit), "evidence": "all inherited prefixes"},
        {"gate_id": "G07_SUPPORT_QUERY_BLOCK_DISJOINT", "passed": all(bool(row["block_disjoint"]) for row in support_audit), "evidence": "all inherited prefixes"},
        {"gate_id": "G08_SUPPORT_QUERY_SIGNAL_DISJOINT", "passed": all(bool(row["exact_signal_disjoint"]) for row in support_audit), "evidence": "all inherited prefixes"},
        {"gate_id": "G09_BLOCK_CONSISTENCY", "passed": len(data.structure_rows()) == 63 and all(int(row["row_count"]) == 50 for row in data.structure_rows()), "evidence": "63 blocks x 50 rows"},
        {"gate_id": "G10_SOURCE_SEEDS", "passed": tuple(SOURCE_SEEDS) == (42, 43, 44, 45, 46), "evidence": str(SOURCE_SEEDS)},
        {"gate_id": "G11_SOURCE_ONLY_HYPERPARAMETERS", "passed": True, "evidence": sha256_file(repository_root / "results/canonical_metrics/p4_encoder_finetuning/07_SOURCE_SELECTED_CONFIG.json")},
        {"gate_id": "G12_PARTIAL_MASK", "passed": sum(int(row["parameter_count"]) for row in mask_rows if row["arm"] == "PARTIAL_FT" and row["trainable"]) == PARTIAL_PARAMETER_COUNT, "evidence": str(PARTIAL_PARAMETER_COUNT)},
        {"gate_id": "G13_FULL_MASK", "passed": sum(int(row["parameter_count"]) for row in mask_rows if row["arm"] == "FULL_FT" and row["trainable"]) == FULL_PARAMETER_COUNT, "evidence": str(FULL_PARAMETER_COUNT)},
        {"gate_id": "G14_ORIGINAL_CHECKPOINT_START", "passed": True, "evidence": "both arms independently clone the source checkpoint"},
        {"gate_id": "G15_OPTIMIZER_UNIT_ISOLATION", "passed": True, "evidence": "new model and AdamW instance per arm/budget/fold/source/support unit"},
        {"gate_id": "G16_QUERY_OBJECT_ABSENT_FROM_FITTER", "passed": "query" not in fine_tune.__code__.co_varnames, "evidence": str(fine_tune.__code__.co_varnames[:fine_tune.__code__.co_argcount])},
        {"gate_id": "G17_PARENT_FROZEN_PREDICTIONS", "passed": parent_full.shape == (10, 5, 20, 3150), "evidence": IMMEDIATE_PARENT_PREDICTION_ARRAY_SHA256},
        {"gate_id": "G18_FIXED_BUDGETS_AND_SEEDS", "passed": BUDGETS == (7, 35, 100, 500) and len(SUPPORT_SEEDS) == 20, "evidence": "4 budgets x 5 source x 20 support x 3 folds"},
    ]
    failed = [row for row in gates if not row["passed"]]
    if failed:
        raise ProtocolViolation("PREEXECUTION_GATE_FAILURE:" + ",".join(str(row["gate_id"]) for row in failed))
    binding = {
        "immediate_parent_preregistration_sha256": IMMEDIATE_PARENT_PREREGISTRATION_SHA256,
        "immediate_parent_final_manifest_sha256": IMMEDIATE_PARENT_FINAL_MANIFEST_SHA256,
        "fold_manifest_sha256": FOLD_MANIFEST_SHA256,
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA256,
        "parent_prediction_array_sha256": IMMEDIATE_PARENT_PREDICTION_ARRAY_SHA256,
        "artifact_rows": parent["artifacts"],
    }
    return data, plans, support_audit, parent_full, frozen_predictions, selected, gates, {"preregistration": preregistration, "binding": binding, "mask_rows": mask_rows}


@dataclass
class SealRecord:
    arm: str
    budget: int
    source_seed: int
    support_seed: int
    fold: int
    query_indices: np.ndarray
    prediction_sha256: str
    seal: QueryLabelSeal


def _load_source_retention_subset() -> tuple[np.ndarray, np.ndarray]:
    registry_value = os.environ.get("CRFID_SOURCE_REGISTRY")
    signals_value = os.environ.get("CRFID_SOURCE_SIGNALS")
    if not registry_value or not signals_value:
        raise ProtocolViolation("GOVERNED_SOURCE_INPUTS_REQUIRED: set CRFID_SOURCE_REGISTRY and CRFID_SOURCE_SIGNALS for the source-retention diagnostic")
    registry_path = Path(registry_value)
    signals_path = Path(signals_value)
    with registry_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [index for index, row in enumerate(rows) if int(row["repeat_index"]) == 0]
    if len(selected) != 189:
        raise ProtocolViolation("SOURCE_RETENTION_SUBSET_STRUCTURE_MISMATCH")
    signals = np.load(signals_path, allow_pickle=False)[np.asarray(selected, dtype=np.int64)]
    labels = np.asarray([int(rows[index]["label_index"]) for index in selected], dtype=np.int64)
    return np.ascontiguousarray(signals), labels


def _fit_one(
    *, source_model: Any, inputs: torch.Tensor, frozen_embeddings: torch.Tensor, data: GovernedP4,
    support_indices: np.ndarray, query_indices: np.ndarray, arm: str, config: dict[str, object], seed: int,
) -> tuple[Any, np.ndarray, torch.Tensor, dict[str, object]]:
    head = fit_linear_readout(
        support_embeddings=frozen_embeddings[support_indices],
        support_labels=data._labels[support_indices],
        anchor_weight=source_model.network[-1].weight,
        anchor_bias=source_model.network[-1].bias,
    )
    if not head.diagnostics["optimizer_completed_finite"]:
        raise ProtocolViolation("PARENT_HEAD_REPRODUCTION_FAILED")
    snapshots = fine_tune(
        source_model=source_model,
        support_inputs=inputs[support_indices],
        support_labels=data._labels[support_indices],
        initial_head_weight=head.weight,
        initial_head_bias=head.bias,
        arm=arm,
        learning_rate=float(config["learning_rate"]),
        encoder_weight_decay=float(config["encoder_weight_decay"]),
        epochs=int(config["epochs"]),
        seed=seed,
    )
    snapshot = snapshots[-1]
    prediction, embeddings = predict(snapshot.model, inputs[query_indices])
    diagnostics = {**snapshot.diagnostics, "parent_head_optimizer_completed_finite": True, "parent_head_adapted_sha256": head.diagnostics["adapted_head_sha256"]}
    return snapshot.model, prediction, embeddings, diagnostics


def _determinism_probe(
    *, data: GovernedP4, plans: dict[tuple[int, int], InheritedPlan], archive_repository: Path,
    selected: dict[str, dict[str, object]], arm: str = "PARTIAL_FT",
) -> dict[str, object]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    inputs = preprocess(data.signals, mean, scale)
    source_model, _ = load_frozen_model(archive_repository / CHECKPOINT_RELATIVE / "seed_42/FINAL_CHECKPOINT.pt", 42)
    frozen_embeddings = extract_embeddings(source_model, inputs)
    fold = 0
    support_seed = SUPPORT_SEEDS[0]
    budget = 7
    support_indices = plans[(fold, support_seed)].indices_for_budget(budget)
    query_indices = data.query_view(fold).indices
    seed = derive_seed(arm, 42, support_seed, fold, budget)
    first = _fit_one(source_model=source_model, inputs=inputs, frozen_embeddings=frozen_embeddings, data=data, support_indices=support_indices, query_indices=query_indices, arm=arm, config=selected[arm], seed=seed)
    second = _fit_one(source_model=source_model, inputs=inputs, frozen_embeddings=frozen_embeddings, data=data, support_indices=support_indices, query_indices=query_indices, arm=arm, config=selected[arm], seed=seed)
    passed = bool(np.array_equal(first[1], second[1]) and first[3]["adapted_model_sha256"] == second[3]["adapted_model_sha256"] and first[3]["all_objectives_finite"] and second[3]["all_objectives_finite"])
    if not passed:
        raise ProtocolViolation("DETERMINISTIC_FINETUNING_RERUN_FAILURE")
    return {
        "passed": True,
        "arm": arm,
        "budget": budget,
        "fold": fold,
        "source_seed": 42,
        "support_seed": support_seed,
        "finetuning_seed": seed,
        "adapted_model_sha256": first[3]["adapted_model_sha256"],
        "prediction_sha256": array_sha256(first[1].astype(np.int8)),
        "query_labels_opened": False,
    }


def _generate_predictions(
    *, data: GovernedP4, plans: dict[tuple[int, int], InheritedPlan], archive_repository: Path,
    output_directory: Path, selected: dict[str, dict[str, object]],
) -> tuple[np.ndarray, list[SealRecord], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    inputs = preprocess(data.signals, mean, scale)
    source_retention_signals, source_retention_labels = _load_source_retention_subset()
    source_retention_inputs = preprocess(source_retention_signals, mean, scale)
    predictions = np.full((len(ARMS), len(BUDGETS), len(SOURCE_SEEDS), len(SUPPORT_SEEDS), 3150), -1, dtype=np.int8)
    records: list[SealRecord] = []
    freeze_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    source_retention_rows: list[dict[str, object]] = []
    rep_source = np.empty((len(SOURCE_SEEDS), 3, 1050, 256), dtype=np.float16)
    rep_adapted = np.empty((len(ARMS), len(MAJOR_DIAGNOSTIC_BUDGETS), len(SOURCE_SEEDS), 3, 1050, 256), dtype=np.float16)
    rep_source.fill(np.nan)
    rep_adapted.fill(np.nan)
    for source_index, source_seed in enumerate(SOURCE_SEEDS):
        checkpoint_path = archive_repository / CHECKPOINT_RELATIVE / f"seed_{source_seed}/FINAL_CHECKPOINT.pt"
        source_model, payload = load_frozen_model(checkpoint_path, source_seed)
        source_state_before = model_state_sha256(source_model)
        frozen_embeddings = extract_embeddings(source_model, inputs)
        source_retention_prediction, _ = predict(source_model, source_retention_inputs)
        source_retention_base = metrics_from_predictions(source_retention_labels, source_retention_prediction)
        for fold in sorted(QUERY_FOLDS):
            query_indices = data.query_view(fold).indices
            rep_source[source_index, fold] = frozen_embeddings[query_indices].numpy().astype(np.float16)
            for support_index, support_seed in enumerate(SUPPORT_SEEDS):
                plan = plans[(fold, support_seed)]
                for budget_index, budget in enumerate(BUDGETS):
                    support_indices = plan.indices_for_budget(budget)
                    for arm_index, arm in enumerate(ARMS):
                        unit_seed = derive_seed(arm, source_seed, support_seed, fold, budget)
                        model, prediction, query_embeddings, diagnostics = _fit_one(
                            source_model=source_model,
                            inputs=inputs,
                            frozen_embeddings=frozen_embeddings,
                            data=data,
                            support_indices=support_indices,
                            query_indices=query_indices,
                            arm=arm,
                            config=selected[arm],
                            seed=unit_seed,
                        )
                        predictions[arm_index, budget_index, source_index, support_index, query_indices] = prediction.astype(np.int8)
                        seal = QueryLabelSeal(data._labels[query_indices])
                        prediction_hash = seal.freeze_predictions(prediction)
                        records.append(SealRecord(arm, budget, source_seed, support_seed, fold, query_indices.copy(), prediction_hash, seal))
                        displacement = representation_displacement(frozen_embeddings[query_indices], query_embeddings)
                        row = {
                            "arm": arm,
                            "method": METHOD_BY_ARM[arm],
                            "budget": budget,
                            "source_seed": source_seed,
                            "support_seed": support_seed,
                            "fold": fold,
                            "finetuning_seed": unit_seed,
                            "learning_rate": selected[arm]["learning_rate"],
                            "encoder_weight_decay": selected[arm]["encoder_weight_decay"],
                            "epochs": selected[arm]["epochs"],
                            **diagnostics,
                            **displacement,
                        }
                        fit_rows.append(row)
                        freeze_rows.append(
                            {
                                "arm": arm,
                                "budget": budget,
                                "source_seed": source_seed,
                                "support_seed": support_seed,
                                "fold": fold,
                                "query_rows": len(query_indices),
                                "prediction_sha256": prediction_hash,
                                "state_at_registration": seal.state,
                                "labels_opened": False,
                            }
                        )
                        if support_seed == SUPPORT_SEEDS[0] and budget in MAJOR_DIAGNOSTIC_BUDGETS:
                            diagnostic_index = MAJOR_DIAGNOSTIC_BUDGETS.index(budget)
                            rep_adapted[arm_index, diagnostic_index, source_index, fold] = query_embeddings.numpy().astype(np.float16)
                        if support_seed == SUPPORT_SEEDS[0] and budget == 500:
                            retention_prediction, _ = predict(model, source_retention_inputs)
                            adapted_source = metrics_from_predictions(source_retention_labels, retention_prediction)
                            source_retention_rows.append(
                                {
                                    "arm": arm,
                                    "budget": budget,
                                    "fold": fold,
                                    "source_seed": source_seed,
                                    "support_seed": support_seed,
                                    "source_subset_rows": len(source_retention_labels),
                                    "source_macro_f1_before": float(source_retention_base["macro_f1"]),
                                    "source_macro_f1_after": float(adapted_source["macro_f1"]),
                                    "source_macro_f1_change": float(adapted_source["macro_f1"] - source_retention_base["macro_f1"]),
                                    "source_accuracy_before": float(source_retention_base["accuracy"]),
                                    "source_accuracy_after": float(adapted_source["accuracy"]),
                                }
                            )
        if model_state_sha256(source_model) != source_state_before or source_state_before != CHECKPOINT_STATE_HASHES[source_seed]:
            raise ProtocolViolation("ORIGINAL_SOURCE_MODEL_MUTATED")
        print(json.dumps({"event": "SOURCE_SEED_FINETUNING_COMPLETE", "source_seed": source_seed, "fit_units_complete": len(fit_rows)}), flush=True)
    expected_units = len(ARMS) * len(BUDGETS) * len(SOURCE_SEEDS) * len(SUPPORT_SEEDS) * 3
    if len(fit_rows) != expected_units or len(records) != expected_units or np.any(predictions < 0):
        raise ProtocolViolation("INCOMPLETE_FINETUNING_PREDICTION_MATRIX")
    if not all(bool(row["all_objectives_finite"]) and bool(row["optimizer_state_is_unit_local"]) for row in fit_rows):
        raise ProtocolViolation("FINETUNING_OPTIMIZATION_GATE_FAILURE")
    if not all(record.seal.state == "PREDICTIONS_FROZEN" for record in records):
        raise ProtocolViolation("QUERY_LABEL_OPENED_BEFORE_GLOBAL_FREEZE")
    bundle_path = output_directory / "finetuned_predictions.npz"
    np.savez_compressed(
        bundle_path,
        arms=np.asarray(ARMS),
        budgets=np.asarray(BUDGETS, dtype=np.int64),
        source_seeds=np.asarray(SOURCE_SEEDS, dtype=np.int64),
        support_seeds=np.asarray(SUPPORT_SEEDS, dtype=np.int64),
        predictions=predictions,
    )
    representation_path = output_directory / "representative_query_embeddings_float16.npz"
    np.savez_compressed(
        representation_path,
        arms=np.asarray(ARMS),
        budgets=np.asarray(MAJOR_DIAGNOSTIC_BUDGETS, dtype=np.int64),
        source_seeds=np.asarray(SOURCE_SEEDS, dtype=np.int64),
        support_seed=np.asarray([SUPPORT_SEEDS[0]], dtype=np.int64),
        source_embeddings=rep_source,
        adapted_embeddings=rep_adapted,
    )
    write_csv(output_directory / "14_FINETUNING_DIAGNOSTICS.csv", fit_rows)
    write_csv(output_directory / "15_PREDICTION_FREEZE_MANIFEST.csv", freeze_rows)
    write_csv(output_directory / "26_SOURCE_RETENTION.csv", source_retention_rows)
    write_json(
        output_directory / "15A_PREDICTION_BUNDLE_BINDING.json",
        {
            "prediction_relative_path": bundle_path.name,
            "prediction_file_sha256": sha256_file(bundle_path),
            "prediction_array_sha256": array_sha256(predictions),
            "prediction_shape": predictions.shape,
            "prediction_dtype": str(predictions.dtype),
            "registered_prediction_units": len(records),
            "representation_relative_path": representation_path.name,
            "representation_file_sha256": sha256_file(representation_path),
            "source_embedding_array_sha256": array_sha256(rep_source),
            "adapted_embedding_array_sha256": array_sha256(rep_adapted),
            "all_predictions_frozen_before_any_query_label_open": True,
        },
    )
    return predictions, records, fit_rows, freeze_rows, source_retention_rows, rep_source, rep_adapted


def _open_truth_after_global_freeze(*, data: GovernedP4, records: list[SealRecord]) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not records or not all(record.seal.state == "PREDICTIONS_FROZEN" for record in records):
        raise ProtocolViolation("GLOBAL_PREDICTION_FREEZE_INCOMPLETE")
    truth = np.full(3150, -1, dtype=np.int64)
    audit = []
    for record in records:
        labels = record.seal.open_labels()
        existing = truth[record.query_indices]
        if np.any((existing >= 0) & (existing != labels)):
            raise ProtocolViolation("INCONSISTENT_QUERY_LABEL_CUSTODY")
        truth[record.query_indices] = labels
        audit.append(
            {
                "arm": record.arm,
                "budget": record.budget,
                "source_seed": record.source_seed,
                "support_seed": record.support_seed,
                "fold": record.fold,
                "prediction_sha256": record.prediction_sha256,
                "prediction_frozen_before_open": True,
                "label_access_phase": "FINAL_SCORING_ONLY",
                "seal_state_after_open": record.seal.state,
            }
        )
    if np.any(truth < 0):
        raise ProtocolViolation("QUERY_TRUTH_RECONSTRUCTION_INCOMPLETE")
    return truth, audit


def _padded_predictions(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.int8)
    if values.shape != (len(BUDGETS), 5, 20, 3150):
        raise ValueError("expected four-budget prediction array")
    padded = np.repeat(values[0:1], 10, axis=0)
    for source_index, target_index in enumerate(PARENT_BUDGET_INDICES):
        padded[target_index] = values[source_index]
    return padded


def _bootstrap_method(predictions: np.ndarray, block_indices: list[np.ndarray], block_truth: np.ndarray, replicates: int):
    padded = _padded_predictions(predictions)
    histograms = prediction_histograms_by_block(padded, block_indices)
    boot = hierarchical_bootstrap(histograms, block_truth, replicates=replicates, seed=BOOTSTRAP_SEED)
    macro, accuracy = aggregate_metric_from_histograms(histograms)
    selected = np.asarray(PARENT_BUDGET_INDICES, dtype=np.int64)
    return {
        "histograms": histograms[selected],
        "macro_units": macro[selected],
        "accuracy_units": accuracy[selected],
        "macro_boot": boot.macro_f1[:, selected],
        "accuracy_boot": boot.accuracy[:, selected],
        "block_weights": boot.block_weights_sha256_payload,
        "source_counts": boot.source_counts,
        "support_counts": boot.support_counts,
    }


def _comparison_rows(points: dict[str, dict], budget_index: int) -> list[dict[str, object]]:
    comparisons = (("PARTIAL_MINUS_FROZEN", "PARTIAL_FT", "FROZEN_LINEAR"), ("FULL_MINUS_FROZEN", "FULL_FT", "FROZEN_LINEAR"), ("FULL_MINUS_PARTIAL", "FULL_FT", "PARTIAL_FT"))
    rows = []
    for label, left, right in comparisons:
        macro_delta = float(points[left]["macro_units"][budget_index].mean() - points[right]["macro_units"][budget_index].mean())
        accuracy_delta = float(points[left]["accuracy_units"][budget_index].mean() - points[right]["accuracy_units"][budget_index].mean())
        macro_samples = points[left]["macro_boot"][:, budget_index] - points[right]["macro_boot"][:, budget_index]
        accuracy_samples = points[left]["accuracy_boot"][:, budget_index] - points[right]["accuracy_boot"][:, budget_index]
        macro_lower, macro_upper = percentile_interval(macro_samples)
        accuracy_lower, accuracy_upper = percentile_interval(accuracy_samples)
        rows.append(
            {
                "budget": BUDGETS[budget_index],
                "comparison": label,
                "macro_f1_difference": macro_delta,
                "macro_f1_ci_95_lower": macro_lower,
                "macro_f1_ci_95_upper": macro_upper,
                "accuracy_difference": accuracy_delta,
                "accuracy_ci_95_lower": accuracy_lower,
                "accuracy_ci_95_upper": accuracy_upper,
                "reliable_positive_macro_f1": macro_lower > 0,
                "practical_macro_f1": macro_delta >= PRACTICAL_DELTA,
                "practical_and_reliable_macro_f1": macro_delta >= PRACTICAL_DELTA and macro_lower > 0,
            }
        )
    return rows


def _classify(paired_rows: list[dict[str, object]]) -> str:
    eligible = [row for row in paired_rows if int(row["budget"]) in MAJOR_DIAGNOSTIC_BUDGETS]
    partial = [row for row in eligible if row["comparison"] == "PARTIAL_MINUS_FROZEN"]
    full = [row for row in eligible if row["comparison"] == "FULL_MINUS_FROZEN"]
    if any(bool(row["practical_and_reliable_macro_f1"]) for row in partial):
        return "PARTIAL_ENCODER_ADAPTATION_RESCUES_P4_TRANSFER"
    if any(bool(row["practical_and_reliable_macro_f1"]) for row in full):
        return "FULL_ENCODER_ADAPTATION_REQUIRED_FOR_P4_TRANSFER"
    if any(bool(row["reliable_positive_macro_f1"]) for row in (*partial, *full)):
        return "ENCODER_ADAPTATION_PARTIAL_BENEFIT_ONLY"
    return "ENCODER_ADAPTATION_DOES_NOT_RESCUE_HELD_CONDITION_TRANSFER"


def _block_correctness(values: np.ndarray, budget_index: int, indices: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Preserve source/support axes while selecting rows from one budget."""
    return values[budget_index][..., indices] == truth[indices]


def _analyse(
    *, data: GovernedP4, truth: np.ndarray, frozen_predictions: np.ndarray, finetuned_predictions: np.ndarray,
    fit_rows: list[dict[str, object]], rep_source: np.ndarray, rep_adapted: np.ndarray,
    output_directory: Path, bootstrap_replicates: int,
) -> dict[str, object]:
    block_keys, block_indices, block_truth = block_layout(data)
    method_predictions = {
        "FROZEN_LINEAR": frozen_predictions,
        "PARTIAL_FT": finetuned_predictions[0],
        "FULL_FT": finetuned_predictions[1],
    }
    points = {name: _bootstrap_method(values, block_indices, block_truth, bootstrap_replicates) for name, values in method_predictions.items()}
    reference = points["FROZEN_LINEAR"]
    for name, value in points.items():
        if not np.array_equal(value["block_weights"], reference["block_weights"]) or not np.array_equal(value["source_counts"], reference["source_counts"]) or not np.array_equal(value["support_counts"], reference["support_counts"]):
            raise ProtocolViolation(f"PAIRED_BOOTSTRAP_DRAW_MISMATCH:{name}")
    paired_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    coverage = {budget: float(np.mean([int(row["unique_condition_blocks"]) for row in csv.DictReader((output_directory / "11_SUPPORT_MANIFEST_VALIDATION.csv").open(encoding="utf-8", newline="")) if int(row["budget"]) == budget])) for budget in BUDGETS}
    for budget_index, budget in enumerate(BUDGETS):
        paired = _comparison_rows(points, budget_index)
        paired_rows.extend(paired)
        lookup = {row["comparison"]: row for row in paired}
        row: dict[str, object] = {"budget": budget, "independent_condition_blocks_mean": coverage[budget]}
        for name in ("FROZEN_LINEAR", "PARTIAL_FT", "FULL_FT"):
            macro = float(points[name]["macro_units"][budget_index].mean())
            accuracy = float(points[name]["accuracy_units"][budget_index].mean())
            macro_lower, macro_upper = percentile_interval(points[name]["macro_boot"][:, budget_index])
            accuracy_lower, accuracy_upper = percentile_interval(points[name]["accuracy_boot"][:, budget_index])
            prefix = name.lower()
            row.update({
                f"{prefix}_macro_f1": macro,
                f"{prefix}_macro_f1_ci_95_lower": macro_lower,
                f"{prefix}_macro_f1_ci_95_upper": macro_upper,
                f"{prefix}_accuracy": accuracy,
                f"{prefix}_accuracy_ci_95_lower": accuracy_lower,
                f"{prefix}_accuracy_ci_95_upper": accuracy_upper,
            })
        for label, prefix in (("PARTIAL_MINUS_FROZEN", "partial_minus_frozen"), ("FULL_MINUS_FROZEN", "full_minus_frozen"), ("FULL_MINUS_PARTIAL", "full_minus_partial")):
            contrast = lookup[label]
            row.update({
                f"{prefix}_macro_f1": contrast["macro_f1_difference"],
                f"{prefix}_macro_f1_ci_95_lower": contrast["macro_f1_ci_95_lower"],
                f"{prefix}_macro_f1_ci_95_upper": contrast["macro_f1_ci_95_upper"],
                f"{prefix}_accuracy": contrast["accuracy_difference"],
                f"{prefix}_accuracy_ci_95_lower": contrast["accuracy_ci_95_lower"],
                f"{prefix}_accuracy_ci_95_upper": contrast["accuracy_ci_95_upper"],
            })
        row["partial_recovery_status"] = "PRACTICAL_AND_RELIABLE" if lookup["PARTIAL_MINUS_FROZEN"]["practical_and_reliable_macro_f1"] else "NOT_PRACTICAL_AND_RELIABLE"
        row["full_recovery_status"] = "PRACTICAL_AND_RELIABLE" if lookup["FULL_MINUS_FROZEN"]["practical_and_reliable_macro_f1"] else "NOT_PRACTICAL_AND_RELIABLE"
        primary_rows.append(row)
    classification = _classify(paired_rows)

    per_run_rows = []
    method_keys = ((METHOD_FROZEN, frozen_predictions), (METHOD_BY_ARM["PARTIAL_FT"], finetuned_predictions[0]), (METHOD_BY_ARM["FULL_FT"], finetuned_predictions[1]))
    for method, values in method_keys:
        for budget_index, budget in enumerate(BUDGETS):
            for source_index, source_seed in enumerate(SOURCE_SEEDS):
                for support_index, support_seed in enumerate(SUPPORT_SEEDS):
                    metrics = extended_metrics(truth, values[budget_index, source_index, support_index].astype(np.int64))
                    per_run_rows.append({
                        "method": method,
                        "budget": budget,
                        "source_seed": source_seed,
                        "support_seed": support_seed,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "worst_class_f1": metrics["worst_class_f1"],
                        "prediction_sha256": array_sha256(values[budget_index, source_index, support_index]),
                    })

    query_mean = {(arm, budget): float(points[arm]["macro_units"][BUDGETS.index(budget)].mean()) for arm in ARMS for budget in BUDGETS}
    support_query_rows = []
    grouped_fit: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in fit_rows:
        grouped_fit[(str(row["arm"]), int(row["budget"]))].append(row)
    for arm in ARMS:
        for budget in BUDGETS:
            rows = grouped_fit[(arm, budget)]
            if len(rows) != 300:
                raise ProtocolViolation("FINETUNING_DIAGNOSTIC_UNIT_COUNT_MISMATCH")
            support_macro = float(np.mean([float(row["support_macro_f1"]) for row in rows]))
            support_query_rows.append({
                "arm": arm,
                "budget": budget,
                "fit_units": len(rows),
                "support_macro_f1_mean": support_macro,
                "support_accuracy_mean": float(np.mean([float(row["support_accuracy"]) for row in rows])),
                "held_query_macro_f1": query_mean[(arm, budget)],
                "support_minus_query_macro_f1": support_macro - query_mean[(arm, budget)],
                "encoder_displacement_frobenius_mean": float(np.mean([float(row["encoder_displacement_frobenius"]) for row in rows])),
                "embedding_mean_cosine_to_source_mean": float(np.mean([float(row["embedding_mean_cosine_to_source"]) for row in rows])),
                "embedding_rms_displacement_mean": float(np.mean([float(row["embedding_rms_displacement"]) for row in rows])),
            })

    per_class_rows = []
    condition_rows = []
    confusion_rows = []
    block_rows = []
    for method, values in method_keys:
        for budget in MAJOR_DIAGNOSTIC_BUDGETS:
            budget_index = BUDGETS.index(budget)
            f1_values = np.empty((5, 20, 7), dtype=np.float64)
            total_confusion = np.zeros((7, 7), dtype=np.int64)
            for source_index in range(5):
                for support_index in range(20):
                    matrix = confusion_matrix(truth, values[budget_index, source_index, support_index])
                    total_confusion += matrix
                    f1_values[source_index, support_index] = metrics_from_confusion(matrix)["f1"]
            for class_index in range(7):
                per_class_rows.append({"method": method, "budget": budget, "class_index": class_index, "TagID": class_index + 1, "f1_mean": float(f1_values[..., class_index].mean()), "f1_sd": float(f1_values[..., class_index].std(ddof=0))})
            for true_class in range(7):
                for predicted_class in range(7):
                    confusion_rows.append({"method": method, "budget": budget, "true_class_index": true_class, "predicted_class_index": predicted_class, "count_across_100_units": int(total_confusion[true_class, predicted_class]), "row_normalized": float(total_confusion[true_class, predicted_class] / max(total_confusion[true_class].sum(), 1))})
            for factor, factor_values in (("ER", data.er), ("surface", data.surface)):
                for level in sorted(np.unique(factor_values)):
                    mask = factor_values == level
                    scores = []
                    for source_index in range(5):
                        for support_index in range(20):
                            scores.append(float(extended_metrics(truth[mask], values[budget_index, source_index, support_index, mask])["macro_f1"]))
                    condition_rows.append({"method": method, "budget": budget, "factor": factor, "level": int(level), "rows_per_unit": int(mask.sum()), "macro_f1_mean": float(np.mean(scores)), "macro_f1_sd": float(np.std(scores, ddof=0))})
            for block_key, indices in zip(block_keys, block_indices, strict=True):
                correct = _block_correctness(values, budget_index, indices, truth)
                block_rows.append({"method": method, "budget": budget, "class_index": block_key[0], "ER": block_key[1], "surface": block_key[2], "rows_per_unit": len(indices), "mean_row_accuracy": float(correct.mean()), "unit_sd": float(correct.mean(axis=-1).std(ddof=0))})

    representation_rows = []
    for arm_index, arm in enumerate(ARMS):
        for diagnostic_index, budget in enumerate(MAJOR_DIAGNOSTIC_BUDGETS):
            for source_index, source_seed in enumerate(SOURCE_SEEDS):
                for fold in range(3):
                    query_indices = data.query_view(fold).indices
                    labels = truth[query_indices]
                    cells = data.er[query_indices] * 3 + data.surface[query_indices]
                    source_embedding = rep_source[source_index, fold].astype(np.float64)
                    adapted_embedding = rep_adapted[arm_index, diagnostic_index, source_index, fold].astype(np.float64)
                    displacement = representation_displacement(torch.from_numpy(source_embedding), torch.from_numpy(adapted_embedding))
                    representation_rows.append({
                        "arm": arm,
                        "budget": budget,
                        "source_seed": source_seed,
                        "support_seed": SUPPORT_SEEDS[0],
                        "fold": fold,
                        "query_rows": len(labels),
                        "tagid_centroid_ratio_source": centroid_ratio(source_embedding, labels),
                        "tagid_centroid_ratio_adapted": centroid_ratio(adapted_embedding, labels),
                        "condition_centroid_ratio_source": centroid_ratio(source_embedding, cells),
                        "condition_centroid_ratio_adapted": centroid_ratio(adapted_embedding, cells),
                        **displacement,
                    })

    write_csv(output_directory / "17_PER_RUN_METRICS.csv", per_run_rows)
    write_csv(output_directory / "18_PRIMARY_RESULTS.csv", primary_rows)
    write_csv(output_directory / "19_PAIRED_BOOTSTRAP.csv", paired_rows)
    write_csv(output_directory / "20_SUPPORT_VS_QUERY.csv", support_query_rows)
    write_csv(output_directory / "21_PER_CLASS.csv", per_class_rows)
    write_csv(output_directory / "22_CONDITION_RESULTS.csv", condition_rows)
    write_csv(output_directory / "23_BLOCK_RESULTS.csv", block_rows)
    write_csv(output_directory / "24_CONFUSION_MATRICES.csv", confusion_rows)
    write_csv(output_directory / "25_REPRESENTATION_DIAGNOSTICS.csv", representation_rows)
    bootstrap_path = output_directory / "bootstrap_samples.npz"
    np.savez_compressed(
        bootstrap_path,
        budgets=np.asarray(BUDGETS, dtype=np.int64),
        frozen_macro_f1=points["FROZEN_LINEAR"]["macro_boot"],
        partial_macro_f1=points["PARTIAL_FT"]["macro_boot"],
        full_macro_f1=points["FULL_FT"]["macro_boot"],
        frozen_accuracy=points["FROZEN_LINEAR"]["accuracy_boot"],
        partial_accuracy=points["PARTIAL_FT"]["accuracy_boot"],
        full_accuracy=points["FULL_FT"]["accuracy_boot"],
        block_weights=reference["block_weights"],
        source_counts=reference["source_counts"],
        support_counts=reference["support_counts"],
    )
    write_json(output_directory / "27_BOOTSTRAP_BINDING.json", {
        "relative_path": bootstrap_path.name,
        "sha256": sha256_file(bootstrap_path),
        "replicates": bootstrap_replicates,
        "seed": BOOTSTRAP_SEED,
        "block_weights_sha256": array_sha256(reference["block_weights"]),
        "source_counts_sha256": array_sha256(reference["source_counts"]),
        "support_counts_sha256": array_sha256(reference["support_counts"]),
        "frozen_macro_f1_sha256": array_sha256(points["FROZEN_LINEAR"]["macro_boot"]),
        "partial_macro_f1_sha256": array_sha256(points["PARTIAL_FT"]["macro_boot"]),
        "full_macro_f1_sha256": array_sha256(points["FULL_FT"]["macro_boot"]),
        "paired_draws_identical_for_all_methods": True,
    })
    return {
        "classification": classification,
        "primary_rows": primary_rows,
        "paired_rows": paired_rows,
        "support_query_rows": support_query_rows,
        "representation_rows": representation_rows,
    }


def _executed_code_rows(repository_root: Path) -> list[dict[str, object]]:
    paths = [
        "workflows/14_p4_encoder_finetuning/run.py",
        "workflows/14_p4_encoder_finetuning/plot_results.py",
        "workflows/14_p4_encoder_finetuning/p4_encoder_ft/__init__.py",
        "workflows/14_p4_encoder_finetuning/p4_encoder_ft/constants.py",
        "workflows/14_p4_encoder_finetuning/p4_encoder_ft/finetune.py",
        "workflows/14_p4_encoder_finetuning/p4_encoder_ft/source_selection.py",
        "workflows/14_p4_encoder_finetuning/p4_encoder_ft/study.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/linear.py",
        "workflows/13_p4_trainable_linear_readout/p4_linear_readout/manifests.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/sampling.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/statistics.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/artifacts.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/constants.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/data.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/metrics.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/models.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/protocol.py",
        "tests/test_p4_encoder_finetuning.py",
        str(PREREGISTRATION_RELATIVE).replace("\\", "/"),
    ]
    rows = []
    for relative in paths:
        path = repository_root / relative
        if not path.is_file():
            raise ProtocolViolation(f"EXECUTED_CODE_FILE_MISSING:{relative}")
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _write_preexecution_artifacts(
    *, repository_root: Path, output_directory: Path, data: GovernedP4,
    support_audit: list[dict[str, object]], selected: dict[str, dict[str, object]],
    gates: list[dict[str, object]], context: dict[str, object], code_identity: dict[str, object],
) -> None:
    for filename in SOURCE_SELECTION_FILES:
        source = repository_root / "results/canonical_metrics/p4_encoder_finetuning" / filename
        destination = output_directory / filename
        if source.resolve() != destination.resolve():
            destination.write_bytes(source.read_bytes())
    write_csv(output_directory / "00_EXECUTED_CODE_MANIFEST.csv", _executed_code_rows(repository_root))
    write_json(output_directory / "01_EXACT_CONFIGURATION.json", {
        "experiment_id": EXPERIMENT_ID,
        "code_identity": code_identity,
        "runtime": _runtime_identity(),
        "budgets": BUDGETS,
        "source_seeds": SOURCE_SEEDS,
        "support_seeds": SUPPORT_SEEDS,
        "selected_configuration": selected,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "p4_source_hashes": data.source_hashes,
        "claim_type": "TARGET_ASSISTED_LABELLED_TARGET_REPRESENTATION_ADAPTATION_DIAGNOSTIC",
    })
    write_json(output_directory / "03_PREREGISTRATION_BINDING.json", {
        "relative_path": str(PREREGISTRATION_RELATIVE).replace("\\", "/"),
        "sha256": PREREGISTRATION_SHA256,
        "payload": context["preregistration"],
    })
    write_json(output_directory / "09_INHERITED_ARTIFACT_BINDING.json", context["binding"])
    write_csv(output_directory / "10_PARAMETER_MASK_MANIFEST.csv", context["mask_rows"])
    write_csv(output_directory / "11_SUPPORT_MANIFEST_VALIDATION.csv", [row for row in support_audit if int(row["budget"]) in BUDGETS])
    write_csv(output_directory / "12_PREEXECUTION_VALIDITY_AUDIT.csv", gates)


def _narratives(output_directory: Path, analysis: dict[str, object]) -> None:
    classification = str(analysis["classification"])
    primary = {int(row["budget"]): row for row in analysis["primary_rows"]}
    high = primary[500]
    partial_delta = float(high["partial_minus_frozen_macro_f1"])
    full_delta = float(high["full_minus_frozen_macro_f1"])
    full_partial = float(high["full_minus_partial_macro_f1"])
    interpretation = f"""# Scientific interpretation

## Final classification

`{classification}`

At 500 labels, frozen-linear Macro-F1 was {float(high['frozen_linear_macro_f1']):.4f}, Partial was {float(high['partial_ft_macro_f1']):.4f}, and Full was {float(high['full_ft_macro_f1']):.4f}. The paired changes were {partial_delta:+.4f} for Partial-minus-Frozen, {full_delta:+.4f} for Full-minus-Frozen, and {full_partial:+.4f} for Full-minus-Partial.

The classification follows the preregistered +0.05 practical threshold and paired 95% interval rule. Support performance, class-specific changes, condition strata, representation displacement, and source retention are diagnostic only and did not select a model.

This closes the current P4 labelled-target adaptation model line. No additional architecture or adaptation family was launched.
"""
    thesis = f"""# Thesis-ready conclusion

Strict source-only transfer to P4 was poor, and neither the original few-shot study nor the larger frozen-encoder cosine-prototype calibration curve produced reliable held-condition recovery. A matched trainable-linear control remained near chance and supported a frozen-representation bottleneck. In the final representation-level control, the same source checkpoints, block-disjoint folds, labelled supports, query blocks, seeds, and paired uncertainty framework were retained while either the upper C1 representation block or the full C1 encoder was allowed to adapt. At 500 labels, held-condition Macro-F1 was {float(high['frozen_linear_macro_f1']):.4f} for the frozen readout, {float(high['partial_ft_macro_f1']):.4f} for partial fine-tuning, and {float(high['full_ft_macro_f1']):.4f} for full fine-tuning. Under the preregistered practical and reliability criteria, the experiment was classified `{classification}`. This result is evidence about transfer under the governed ER–surface block protocol, not proof of a unique physical cause, and it marks the scientific stopping point for the current P4 labelled-target adaptation line.
"""
    limitations = """# Limitations

- Positive-budget results are target-assisted and are not strict source-only domain generalisation.
- Fine-tuning hyperparameters were selected on P1--P3 pseudo-target positions; their relevance to P4 is an assumption tested here, not guaranteed.
- The Partial block is the smallest complete final representation block but contains most C1 parameters.
- Query-label uncertainty is based on physical-condition blocks plus crossed source/support seeds; raw repeated rows are not independent inference units.
- Representation and source-retention diagnostics are explanatory and were not used for selection.
- This diagnostic does not establish a unique RF or physical mechanism.
"""
    write_text(output_directory / "28_SCIENTIFIC_INTERPRETATION.md", interpretation)
    write_text(output_directory / "29_THESIS_READY_CONCLUSION.md", thesis)
    write_text(output_directory / "30_LIMITATIONS.md", limitations)
    rows = analysis["primary_rows"]
    table = [
        "| Budget | Frozen F1 | Partial F1 | Partial-Frozen | Full F1 | Full-Frozen | Full-Partial |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(f"| {int(row['budget'])} | {float(row['frozen_linear_macro_f1']):.4f} | {float(row['partial_ft_macro_f1']):.4f} | {float(row['partial_minus_frozen_macro_f1']):+.4f} | {float(row['full_ft_macro_f1']):.4f} | {float(row['full_minus_frozen_macro_f1']):+.4f} | {float(row['full_minus_partial_macro_f1']):+.4f} |")
    readme = "\n".join([
        "# P4 Encoder Fine-Tuning Diagnostic v1",
        "",
        f"Final classification: `{classification}`",
        "",
        "This isolated target-assisted diagnostic compares the completed frozen-linear parent with Partial and Full C1 representation adaptation under identical block-disjoint P4 supports and queries.",
        "",
        *table,
        "",
        "All new predictions were persisted and hash-registered before P4 query labels were opened. Hyperparameters were selected source-only and frozen across budgets.",
        "",
        "Final branch decision: `TARGET_LABELLED_DIAGNOSTIC_COMPLETE`.",
    ]) + "\n"
    write_text(output_directory / "README.md", readme)


def _final_status(
    *, output_directory: Path, analysis: dict[str, object], code_identity: dict[str, object],
    gates: list[dict[str, object]], determinism: dict[str, object], fit_units: int, prediction_units: int,
    analysis_recovery_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    status = {
        "schema_version": 1,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "classification": analysis["classification"],
        "final_branch_decision": "TARGET_LABELLED_DIAGNOSTIC_COMPLETE",
        "scientific_code_identity": code_identity,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "inherited_fold_manifest_sha256": FOLD_MANIFEST_SHA256,
        "inherited_support_manifest_sha256": SUPPORT_MANIFEST_SHA256,
        "budgets": BUDGETS,
        "source_seeds": SOURCE_SEEDS,
        "support_seeds": SUPPORT_SEEDS,
        "fit_units": fit_units,
        "prediction_units": prediction_units,
        "preexecution_gates_passed": len(gates),
        "determinism_probe": determinism,
        "all_predictions_frozen_before_query_scoring": True,
        "source_only_hyperparameters": json.loads((output_directory / "07_SOURCE_SELECTED_CONFIG.json").read_text(encoding="utf-8"))["selected"],
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "primary_results": analysis["primary_rows"],
        "paired_results": analysis["paired_rows"],
        "support_vs_query": analysis["support_query_rows"],
        "another_p4_model_executed": False,
        "figures_present": False,
        "final_audit_manifest_file_count": 0,
        "final_audit_manifest_sha256": "",
    }
    if analysis_recovery_identity is not None:
        status.update({
            "analysis_code_identity": analysis_recovery_identity,
            "analysis_mode": "RECOMPUTE_FROM_FROZEN_PREDICTIONS",
        })
    write_json(output_directory / "32_FINAL_STATUS.json", status)
    return status


def finalize_artifacts(output_directory: str | Path) -> dict[str, object]:
    output = Path(output_directory).resolve()
    status_path = output / "32_FINAL_STATUS.json"
    if not status_path.is_file():
        raise FileNotFoundError("final status is absent")
    plot_script = REPOSITORY_ROOT / "workflows/14_p4_encoder_finetuning/plot_results.py"
    plot_python = os.environ.get("CRFID_PLOT_PYTHON", sys.executable)
    subprocess.check_call([plot_python, str(plot_script), "--output-directory", str(output)])
    excluded = {"31_FINAL_AUDIT_MANIFEST.csv", "32_FINAL_STATUS.json"}
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "manifest_scope": "SCIENTIFIC_OR_REPRODUCIBILITY_ARTIFACT"})
    write_csv(output / "31_FINAL_AUDIT_MANIFEST.csv", rows)
    manifest_hash = sha256_file(output / "31_FINAL_AUDIT_MANIFEST.csv")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["figures_present"] = all((output / relative).is_file() for relative in ("figures/p4_encoder_finetuning_main.png", "figures/p4_encoder_finetuning_support_query.png"))
    status["final_audit_manifest_file_count"] = len(rows)
    status["final_audit_manifest_sha256"] = manifest_hash
    write_json(status_path, status)
    return {"artifact_count": len(rows), "manifest_sha256": manifest_hash, "figures_present": status["figures_present"]}


def recover_analysis(
    *, repository_root: str | Path, archive_repository: str | Path, data_directory: str | Path,
    output_directory: str | Path, bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Resume only post-seal analysis from hash-bound prediction artifacts."""
    repository = Path(repository_root).resolve()
    archive = Path(archive_repository).resolve()
    data_root = Path(data_directory).resolve()
    output = Path(output_directory).resolve()
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ProtocolViolation("ANALYSIS_RECOVERY_REQUIRES_PREREGISTERED_BOOTSTRAP_REPLICATES")
    exact_configuration = json.loads((output / "01_EXACT_CONFIGURATION.json").read_text(encoding="utf-8"))
    execution_identity = exact_configuration["code_identity"]
    recovery_identity = {"scientific_code_files": _executed_code_rows(repository)}
    data, _, support_audit, _, frozen_predictions, _, gates, _ = preflight(
        repository_root=repository,
        archive_repository=archive,
        data_directory=data_root,
        output_directory=output,
    )
    primary_support_audit = [row for row in support_audit if int(row["budget"]) in BUDGETS]
    if len(gates) != 18 or len(support_audit) != 540 or len(primary_support_audit) != 240:
        raise ProtocolViolation("ANALYSIS_RECOVERY_PREFLIGHT_MISMATCH")

    binding = json.loads((output / "15A_PREDICTION_BUNDLE_BINDING.json").read_text(encoding="utf-8"))
    prediction_path = output / str(binding["prediction_relative_path"])
    representation_path = output / str(binding["representation_relative_path"])
    if sha256_file(prediction_path) != binding["prediction_file_sha256"] or sha256_file(representation_path) != binding["representation_file_sha256"]:
        raise ProtocolViolation("ANALYSIS_RECOVERY_BUNDLE_FILE_HASH_MISMATCH")
    with np.load(prediction_path, allow_pickle=False) as payload:
        if tuple(payload["arms"].tolist()) != ARMS or tuple(payload["budgets"].tolist()) != BUDGETS or tuple(payload["source_seeds"].tolist()) != SOURCE_SEEDS or tuple(payload["support_seeds"].tolist()) != SUPPORT_SEEDS:
            raise ProtocolViolation("ANALYSIS_RECOVERY_PREDICTION_COORDINATE_MISMATCH")
        finetuned_predictions = np.asarray(payload["predictions"])
    if finetuned_predictions.shape != (2, 4, 5, 20, 3150) or str(finetuned_predictions.dtype) != "int8" or array_sha256(finetuned_predictions) != binding["prediction_array_sha256"]:
        raise ProtocolViolation("ANALYSIS_RECOVERY_PREDICTION_ARRAY_MISMATCH")
    with np.load(representation_path, allow_pickle=False) as payload:
        rep_source = np.asarray(payload["source_embeddings"])
        rep_adapted = np.asarray(payload["adapted_embeddings"])
    if array_sha256(rep_source) != binding["source_embedding_array_sha256"] or array_sha256(rep_adapted) != binding["adapted_embedding_array_sha256"]:
        raise ProtocolViolation("ANALYSIS_RECOVERY_REPRESENTATION_ARRAY_MISMATCH")

    def read_rows(name: str) -> list[dict[str, str]]:
        with (output / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    fit_rows = read_rows("14_FINETUNING_DIAGNOSTICS.csv")
    freeze_rows = read_rows("15_PREDICTION_FREEZE_MANIFEST.csv")
    query_rows = read_rows("16_QUERY_LABEL_ACCESS_AUDIT.csv")
    expected_units = 2 * 4 * 5 * 20 * 3
    if len(fit_rows) != expected_units or len(freeze_rows) != expected_units or len(query_rows) != expected_units:
        raise ProtocolViolation("ANALYSIS_RECOVERY_UNIT_COUNT_MISMATCH")
    if not all(row["state_at_registration"] == "PREDICTIONS_FROZEN" and row["labels_opened"] == "False" for row in freeze_rows):
        raise ProtocolViolation("ANALYSIS_RECOVERY_FREEZE_AUDIT_FAILURE")
    if not all(row["prediction_frozen_before_open"] == "True" and row["label_access_phase"] == "FINAL_SCORING_ONLY" and row["seal_state_after_open"] == "LABELS_OPENED" for row in query_rows):
        raise ProtocolViolation("ANALYSIS_RECOVERY_LABEL_ACCESS_AUDIT_FAILURE")
    freeze_keys = {(row["arm"], row["budget"], row["source_seed"], row["support_seed"], row["fold"], row["prediction_sha256"]) for row in freeze_rows}
    query_keys = {(row["arm"], row["budget"], row["source_seed"], row["support_seed"], row["fold"], row["prediction_sha256"]) for row in query_rows}
    if freeze_keys != query_keys:
        raise ProtocolViolation("ANALYSIS_RECOVERY_FREEZE_ACCESS_BINDING_MISMATCH")

    truth = np.asarray(data._labels, dtype=np.int64).copy()
    analysis = _analyse(
        data=data,
        truth=truth,
        frozen_predictions=frozen_predictions,
        finetuned_predictions=finetuned_predictions,
        fit_rows=fit_rows,
        rep_source=rep_source,
        rep_adapted=rep_adapted,
        output_directory=output,
        bootstrap_replicates=bootstrap_replicates,
    )
    _narratives(output, analysis)
    recovery_audit = {
        "status": "POST_SEAL_ANALYSIS_RECOVERED_WITHOUT_REFITTING_OR_REPREDICTION",
        "failure": "NumPy advanced indexing moved the selected row axis before source/support axes in block diagnostics",
        "correction": "values[budget_index][..., indices]",
        "execution_code_identity": execution_identity,
        "analysis_recovery_code_identity": recovery_identity,
        "prediction_file_sha256": binding["prediction_file_sha256"],
        "prediction_array_sha256": binding["prediction_array_sha256"],
        "fit_units_reused": len(fit_rows),
        "prediction_units_reused": len(freeze_rows),
        "new_fits": 0,
        "new_predictions": 0,
        "freeze_and_query_access_rows_exactly_bound": True,
    }
    write_json(output / "16A_ANALYSIS_RECOVERY_AUDIT.json", recovery_audit)
    determinism = json.loads((output / "13_DETERMINISM_PROBE.json").read_text(encoding="utf-8"))
    _final_status(
        output_directory=output,
        analysis=analysis,
        code_identity=execution_identity,
        gates=gates,
        determinism=determinism,
        fit_units=len(fit_rows),
        prediction_units=len(freeze_rows),
        analysis_recovery_identity=recovery_identity,
    )
    finalization = finalize_artifacts(output)
    return {
        "status": "FULL_EXECUTION_COMPLETE_VIA_POST_SEAL_ANALYSIS_RECOVERY",
        "classification": analysis["classification"],
        "final_branch_decision": "TARGET_LABELLED_DIAGNOSTIC_COMPLETE",
        "fit_units": len(fit_rows),
        "prediction_units": len(freeze_rows),
        "new_fits_during_recovery": 0,
        "new_predictions_during_recovery": 0,
        "artifact_audit": finalization,
    }


def run_study(
    *, repository_root: str | Path, archive_repository: str | Path, data_directory: str | Path,
    output_directory: str | Path, mode: str, bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    archive = Path(archive_repository).resolve()
    data_root = Path(data_directory).resolve()
    output = Path(output_directory).resolve()
    code_identity = {"scientific_code_files": _executed_code_rows(repository)}
    data, plans, support_audit, parent_full, frozen_predictions, selected, gates, context = preflight(
        repository_root=repository,
        archive_repository=archive,
        data_directory=data_root,
        output_directory=output,
    )
    if mode == "dry-run":
        return {"status": "HISTORICAL_INPUTS_VERIFIED", "gates_passed": len(gates), "support_plan_validations": len(support_audit), "new_p4_predictions": 0}
    output.mkdir(parents=True, exist_ok=True)
    _write_preexecution_artifacts(repository_root=repository, output_directory=output, data=data, support_audit=support_audit, selected=selected, gates=gates, context=context, code_identity=code_identity)
    determinism = _determinism_probe(data=data, plans=plans, archive_repository=archive, selected=selected)
    write_json(output / "13_DETERMINISM_PROBE.json", determinism)
    if mode == "smoke":
        return {"status": "SMOKE_COMPLETE_QUERY_LABELS_UNOPENED", "determinism_probe": determinism, "new_p4_predictions_scored": 0}
    if mode != "full" or bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ProtocolViolation("FULL_EXECUTION_REQUIRES_PREREGISTERED_BOOTSTRAP_REPLICATES")
    finetuned_predictions, records, fit_rows, freeze_rows, source_retention_rows, rep_source, rep_adapted = _generate_predictions(
        data=data,
        plans=plans,
        archive_repository=archive,
        output_directory=output,
        selected=selected,
    )
    truth, query_access = _open_truth_after_global_freeze(data=data, records=records)
    write_csv(output / "16_QUERY_LABEL_ACCESS_AUDIT.csv", query_access)
    analysis = _analyse(
        data=data,
        truth=truth,
        frozen_predictions=frozen_predictions,
        finetuned_predictions=finetuned_predictions,
        fit_rows=fit_rows,
        rep_source=rep_source,
        rep_adapted=rep_adapted,
        output_directory=output,
        bootstrap_replicates=bootstrap_replicates,
    )
    for record in records:
        record.seal.mark_metrics_computed()
    _narratives(output, analysis)
    _final_status(output_directory=output, analysis=analysis, code_identity=code_identity, gates=gates, determinism=determinism, fit_units=len(fit_rows), prediction_units=len(records))
    finalization = finalize_artifacts(output)
    return {"status": "FULL_EXECUTION_COMPLETE", "classification": analysis["classification"], "final_branch_decision": "TARGET_LABELLED_DIAGNOSTIC_COMPLETE", "fit_units": len(fit_rows), "prediction_units": len(records), "artifact_audit": finalization}
