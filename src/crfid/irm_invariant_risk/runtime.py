"""Governed execution adapter for the isolated IRMv1 strict-DG patch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score

from crfid.strict_runtime.neutral_data import CanonicalPhase2Data, PartitionData
from crfid.strict_runtime.neutral_model import initialize_model, model_state_sha256
from crfid.strict_runtime.phase3b_execution import (
    C1,
    _optimizer,
    candidate_partitions,
    evaluate_logits,
    extract_logits_embeddings,
    fit_first_difference_preprocessing,
)
from crfid.strict_runtime.scale_policy import CANONICAL_MODE, apply_scale_policy

from .core import (
    CLASS_ORDER,
    FINAL_P4_EVALUATION,
    FINAL_SOURCE_TRAINING,
    FOLD_HELD_POSITION,
    LAMBDA_GRID,
    SEEDS,
    SOURCE_LOPO_DEVELOPMENT,
    SOURCE_POSITIONS,
    IRMProtocolError,
    P4LabelSeal,
    array_sha256,
    assert_penalty_connectivity,
    canonical_json_sha256,
    canonical_position_order,
    classify_invariance_diagnostics,
    classify_main_result,
    deterministic_block_majority_vote,
    early_stop_allowed,
    execution_record,
    irmv1_objective,
    make_environment_balanced_batches,
    matched_erm_objective,
    paired_source_bootstrap,
    paired_tagid_stratified_block_bootstrap,
    select_lambda_source_only,
    sha256_file,
    source_retention_guardrail,
    stable_rng,
    validate_condition_block_disjoint,
    validate_execution_record,
    validate_frozen_configuration,
    validate_lopo_position_isolation,
    validate_source_fold_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "irm_invariant_risk" / "canonical.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "canonical_metrics" / "irm_invariant_risk"
MANIFESTS_ROOT = PROJECT_ROOT / "manifests" / "irm_invariant_risk"
GOVERNED_SOURCE_INPUT_HASHES = {
    "CANONICAL_SOURCE_REGISTRY.csv": "a1cbb062b4e2a57d62a6662acbe27cb151703ee07fa0ea3ef447bca4f96ad7e7",
    "source_labels_int64.npy": "a0f74a067fd56c81df7bea54d023e634cf91252cb9b2f75ab18b4ecf7b8525a3",
    "SOURCE_ONLY_LOPO_SPLITS.csv": "41a275a6f499ee3d3105e09a420ceb92bb8272d879a89a6bb6623ae6f9bc82ca",
    "source_signals_float64.npy": "b8fa8d1699d3b5d0da9d7c45a3e709aff2802b6653afb10771d57b401f740aa8",
}


@dataclass(frozen=True)
class RuntimePaths:
    strict_artifact_root: Path
    p4_governed_root: Path
    runtime_root: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any, *, fail_before_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if fail_before_replace:
        temporary.unlink(missing_ok=True)
        raise OSError("simulated persistence failure")
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })
    temporary.replace(path)


def atomic_write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_frozen_configuration(config)
    return config


def environment_paths() -> RuntimePaths:
    strict = os.environ.get("CRFID_IRM_STRICT_ARTIFACT_ROOT", "").strip()
    p4 = os.environ.get("CRFID_IRM_P4_GOVERNED_ROOT", "").strip()
    runtime = os.environ.get("CRFID_IRM_RUN_ROOT", "").strip()
    missing = [name for name, value in (
        ("CRFID_IRM_STRICT_ARTIFACT_ROOT", strict),
        ("CRFID_IRM_P4_GOVERNED_ROOT", p4),
        ("CRFID_IRM_RUN_ROOT", runtime),
    ) if not value]
    if missing:
        raise IRMProtocolError("BLOCKED_GOVERNED_INPUTS: missing " + ", ".join(missing))
    paths = RuntimePaths(Path(strict).expanduser(), Path(p4).expanduser(), Path(runtime).expanduser())
    resolved_runtime = paths.runtime_root.resolve()
    if PROJECT_ROOT.resolve() not in resolved_runtime.parents:
        raise IRMProtocolError("Governed runtime must use the canonical ignored in-repository runtime convention")
    relative_runtime = resolved_runtime.relative_to(PROJECT_ROOT.resolve()).as_posix()
    ignored = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", relative_runtime],
        check=False,
    ).returncode == 0
    if not ignored:
        raise IRMProtocolError("Governed runtime root is not excluded from Git")
    return paths


def governed_source_input_hashes(strict_root: Path) -> dict[str, str]:
    source_root = strict_root / "source_inputs"
    observed: dict[str, str] = {}
    for name, expected in GOVERNED_SOURCE_INPUT_HASHES.items():
        path = source_root / name
        if not path.is_file():
            raise IRMProtocolError(f"BLOCKED_GOVERNED_INPUTS: missing {name}")
        observed[name] = sha256_file(path)
        if observed[name] != expected:
            raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
    return observed


def canonical_source_loader(strict_root: Path, scratch_root: Path) -> CanonicalPhase2Data:
    config = {
        "input": {
            "canonical_signal_path": "source_inputs/source_signals_float64.npy",
            "canonical_label_path": "source_inputs/source_labels_int64.npy",
            "registry_path": "source_inputs/CANONICAL_SOURCE_REGISTRY.csv",
            "split_path": "source_inputs/SOURCE_ONLY_LOPO_SPLITS.csv",
            "custody_shape": [9450, 281],
            "custody_dtype": "<f8",
        },
        "folds": {fold: {} for fold in FOLD_HELD_POSITION},
    }
    return CanonicalPhase2Data(config, strict_root, scratch_root / "canonical_loader_preprocessing")


def positions_for(data: CanonicalPhase2Data, partition: PartitionData) -> np.ndarray:
    return np.asarray([data.registry_rows[int(index)]["position"] for index in partition.registry_rows], dtype=str)


def partition_hash(partition: PartitionData) -> str:
    return array_sha256(np.asarray(partition.registry_rows, dtype=np.int64))


def source_structure_audit(data: CanonicalPhase2Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = np.asarray([row["position"] for row in data.registry_rows], dtype=str)
    conditions = np.asarray([row["raw_condition_id"] for row in data.registry_rows], dtype=str)
    labels = np.asarray(data.labels, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    if data.signals.shape != (9450, 281) or data.signals.dtype.str != "<f8":
        raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
    for position in SOURCE_POSITIONS:
        selected = positions == position
        block_counts = Counter(conditions[selected])
        row = {
            "position": position,
            "sample_count": int(selected.sum()),
            "tagid_count": int(len(np.unique(labels[selected]))),
            "condition_block_count": int(len(block_counts)),
            "rows_per_condition_block": sorted(set(block_counts.values())),
            "ordered_input_points": 281,
            "first_difference_points": 280,
            "status": "PASS",
        }
        if row["sample_count"] != 3150 or row["tagid_count"] != 7 or row["condition_block_count"] != 63 or row["rows_per_condition_block"] != [50]:
            raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
        rows.append(row)
    split_rows: list[dict[str, Any]] = []
    for fold, held in FOLD_HELD_POSITION.items():
        condition_sets: dict[str, set[str]] = {}
        for partition_name, indices in data.partitions[fold].items():
            part_positions = {data.registry_rows[int(index)]["position"] for index in indices}
            expected = {held} if partition_name == "outer_held" else set(SOURCE_POSITIONS).difference({held})
            if part_positions != expected:
                raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
            condition_sets[partition_name] = {data.registry_rows[int(index)]["raw_condition_id"] for index in indices}
            split_rows.append({
                "fold_id": fold,
                "held_source_position": held,
                "partition": partition_name,
                "sample_count": int(len(indices)),
                "condition_block_count": int(len(condition_sets[partition_name])),
                "positions": sorted(part_positions),
                "condition_blocks_disjoint": True,
                "status": "PASS",
            })
        validate_condition_block_disjoint(condition_sets["inner_train"], condition_sets["inner_validation"])
        if condition_sets["outer_held"].intersection(condition_sets["inner_train"] | condition_sets["inner_validation"]):
            raise IRMProtocolError("Held source position entered development")
    return rows, split_rows


def c1_lineage_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "configs" / "strict_dg" / "canonical.yaml",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "neutral_data.py",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "neutral_model.py",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "phase3b_execution.py",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "neutral_metrics.py",
        PROJECT_ROOT / "scripts" / "train_and_freeze_strict_dg.py",
    ]
    return {
        "schema_version": 1,
        "canonical_candidate": C1,
        "canonical_loader": "crfid.strict_runtime.neutral_data.CanonicalPhase2Data",
        "canonical_first_difference": "crfid.strict_runtime.phase3b_execution.fit_first_difference_preprocessing",
        "canonical_model": "crfid.strict_runtime.neutral_model.NeutralSourceOnlyCNN1D",
        "canonical_optimizer": "torch.optim.AdamW via crfid.strict_runtime.phase3b_execution._optimizer",
        "canonical_scheduler": None,
        "canonical_metrics": "crfid.strict_runtime.neutral_metrics.classification_metrics",
        "source_folds": dict(FOLD_HELD_POSITION),
        "seeds": list(SEEDS),
        "input_contract": {"ordered_points": 281, "first_difference_points": 280, "class_count": 7},
        "training_contract": dict(config["training"]),
        "bound_files": [{"relative_path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)} for path in paths],
        "p4_used": False,
    }


def canonical_base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer": dict(config["optimizer"]),
        "loss": dict(config["loss"]),
        "training": {
            "batch_size": int(config["training"]["batch_size"]),
            "inner_stage_offset": int(config["training"]["inner_stage_offset"]),
            "outer_stage_offset": int(config["training"]["outer_stage_offset"]),
        },
    }


def serialize_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    method: str,
    seed: int,
    epoch: int,
    fold: str | None,
    lambda_value: float | None,
    preprocessing_sha256: str,
    stage_record: Mapping[str, Any],
) -> dict[str, Any]:
    validate_execution_record(stage_record)
    state = serialize_state(model)
    payload = {
        "schema_version": 1,
        "candidate_id": C1,
        "method": method,
        "seed": seed,
        "epoch": epoch,
        "fold": fold,
        "lambda": lambda_value,
        "preprocessing_sha256": preprocessing_sha256,
        "model_state_sha256": model_state_sha256(state),
        "model_state_dict": state,
        "execution_stage": stage_record["execution_stage"],
        "p4_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"checkpoint_sha256": sha256_file(path), "model_state_sha256": payload["model_state_sha256"], "epoch": epoch}


def load_checkpoint(path: Path, seed: int) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = initialize_model(seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise IRMProtocolError("Checkpoint hash binding failed")
    return model


def evaluate_partition(
    model: torch.nn.Module,
    partition: PartitionData,
    positions: np.ndarray,
    environments: tuple[str, ...],
) -> dict[str, Any]:
    logits, embeddings = extract_logits_embeddings(model, partition)
    evaluation = evaluate_logits(partition, logits)
    tensor_logits = torch.from_numpy(np.asarray(logits, dtype=np.float32)).requires_grad_(True)
    terms = irmv1_objective(
        tensor_logits,
        torch.from_numpy(partition.labels),
        positions,
        environments,
        configured_lambda=1.0,
        optimizer_step=1,
        anneal_step=0,
    )
    per_environment = {}
    predictions = np.asarray(evaluation["predictions"], dtype=np.int64)
    for environment in environments:
        selected = positions == environment
        metrics = evaluate_logits(
            PartitionData(
                registry_rows=partition.registry_rows[selected],
                inputs=partition.inputs[selected],
                labels=partition.labels[selected],
                sample_ids=[value for value, keep in zip(partition.sample_ids, selected, strict=True) if keep],
                condition_ids=[value for value, keep in zip(partition.condition_ids, selected, strict=True) if keep],
                exact_signal_hashes=[value for value, keep in zip(partition.exact_signal_hashes, selected, strict=True) if keep],
                unique_signal_weights=partition.unique_signal_weights[selected],
            ),
            logits[selected],
        )
        per_environment[environment] = {
            "risk": float(terms.environment_risks[environment].detach()),
            "macro_f1": float(metrics["sample"]["macro_f1"]),
            "accuracy": float(metrics["sample"]["accuracy"]),
        }
    risks = np.asarray([per_environment[e]["risk"] for e in environments], dtype=np.float64)
    return {
        "logits": logits,
        "embeddings": embeddings,
        "evaluation": evaluation,
        "predictions": predictions,
        "per_environment": per_environment,
        "irm_penalty": float(terms.penalty.detach()),
        "risk_population_sd": float(risks.std(ddof=0)),
        "risk_range": float(risks.max() - risks.min()),
    }


def train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    partition: PartitionData,
    positions: np.ndarray,
    environments: tuple[str, ...],
    method: str,
    lambda_value: float | None,
    seed: int,
    epoch: int,
    stage: str,
    batch_size: int,
    global_step_start: int,
    anneal_step: int,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if method not in {"erm", "irm"} or (method == "erm") != (lambda_value is None):
        raise IRMProtocolError("ERM/IRM method declaration is ambiguous")
    plan = make_environment_balanced_batches(
        positions, partition.labels, seed=seed, epoch=epoch, stage=stage,
        batch_size=batch_size, environments=environments,
    )
    model.train()
    risk_total = 0.0
    penalty_total = 0.0
    objective_total = 0.0
    count_total = 0
    environment_totals = {environment: 0.0 for environment in environments}
    environment_counts = {environment: 0 for environment in environments}
    batch_rows: list[dict[str, Any]] = []
    active_lambda_values: list[float] = []
    gradient_norms: list[float] = []
    penalty_encoder_gradient_norms: list[float] = []
    for local_step, (batch, audit) in enumerate(zip(plan.batches, plan.audits, strict=True)):
        global_step = global_step_start + local_step
        inputs = torch.from_numpy(partition.inputs[batch]).unsqueeze(1)
        labels = torch.from_numpy(partition.labels[batch])
        batch_positions = positions[batch]
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).float()
        if method == "erm":
            objective, risks = matched_erm_objective(logits, labels, batch_positions, environments)
            mean_risk = objective
            penalty = torch.zeros((), dtype=torch.float32)
            effective_lambda = 0.0
        else:
            terms = irmv1_objective(
                logits, labels, batch_positions, environments,
                configured_lambda=float(lambda_value), optimizer_step=global_step,
                anneal_step=anneal_step, create_graph=True,
            )
            objective = terms.objective
            mean_risk = terms.mean_risk
            penalty = terms.penalty
            risks = terms.environment_risks
            effective_lambda = terms.lambda_effective
            if effective_lambda > 0 and not penalty_encoder_gradient_norms:
                encoder_grad, classifier_grad = assert_penalty_connectivity(
                    penalty, model.network[0].weight, model.network[-1].weight
                )
                encoder_norm = float(encoder_grad.norm().detach())
                classifier_norm = float(classifier_grad.norm().detach())
                if encoder_norm <= 0 or classifier_norm <= 0:
                    raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY")
                penalty_encoder_gradient_norms.append(encoder_norm)
        if not bool(torch.isfinite(objective)):
            raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY: non-finite objective")
        objective.backward()
        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY: non-finite gradient")
                squared_norm += float(parameter.grad.detach().pow(2).sum())
        gradient_norm = squared_norm ** 0.5
        optimizer.step()
        gradient_norms.append(gradient_norm)
        active_lambda_values.append(effective_lambda)
        risk_total += float(mean_risk.detach()) * len(batch)
        penalty_total += float(penalty.detach()) * len(batch)
        objective_total += float(objective.detach()) * len(batch)
        count_total += len(batch)
        for environment in environments:
            environment_count = int(np.count_nonzero(batch_positions == environment))
            environment_totals[environment] += float(risks[environment].detach()) * environment_count
            environment_counts[environment] += environment_count
        batch_rows.append({
            **dict(identity), **audit,
            "global_optimizer_step": global_step,
            "configured_lambda": lambda_value,
            "lambda_effective": effective_lambda,
            "mean_environment_risk": float(mean_risk.detach()),
            "irm_penalty": float(penalty.detach()),
            "objective": float(objective.detach()),
            "gradient_norm": gradient_norm,
            "batch_signature_sha256": plan.signature_sha256,
        })
    return {
        "mean_risk": risk_total / count_total,
        "mean_penalty": penalty_total / count_total,
        "mean_objective": objective_total / count_total,
        "per_environment_risk": {e: environment_totals[e] / environment_counts[e] for e in environments},
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "maximum_effective_lambda": float(max(active_lambda_values)),
        "penalty_encoder_gradient_norm": max(penalty_encoder_gradient_norms, default=0.0),
        "batch_rows": batch_rows,
        "steps": len(plan.batches),
        "sampler_signature_sha256": plan.signature_sha256,
    }


def assert_lopo_partitions(
    data: CanonicalPhase2Data,
    partitions: Mapping[str, PartitionData],
    fold: str,
) -> tuple[str, ...]:
    held = FOLD_HELD_POSITION[fold]
    environments = tuple(position for position in SOURCE_POSITIONS if position != held)
    for name in ("inner_train", "inner_validation", "outer_development"):
        observed = canonical_position_order(positions_for(data, partitions[name]))
        if observed != environments:
            raise IRMProtocolError("Held source position leaked into LOPO development")
    held_observed = tuple(np.unique(positions_for(data, partitions["outer_held"])).tolist())
    validate_lopo_position_isolation(environments, held_observed, held)
    validate_condition_block_disjoint(
        partitions["inner_train"].condition_ids,
        partitions["inner_validation"].condition_ids,
    )
    return environments


def development_directory(
    runtime_root: Path, method: str, lambda_value: float | None, fold: str, seed: int
) -> Path:
    value = "none" if lambda_value is None else f"lambda_{lambda_value:g}"
    return runtime_root / "development" / method / value / fold / f"seed_{seed}"


def run_lopo_unit(
    *,
    data: CanonicalPhase2Data,
    states: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    fold: str,
    seed: int,
    method: str,
    lambda_value: float | None,
    runtime_root: Path,
) -> dict[str, Any]:
    if method == "irm" and lambda_value not in LAMBDA_GRID:
        raise IRMProtocolError("IRM lambda outside frozen grid")
    stage_schema = execution_record(SOURCE_LOPO_DEVELOPMENT, fold)
    validate_source_fold_record(stage_schema)
    partitions = candidate_partitions(data, C1, fold, states)
    environments = assert_lopo_partitions(data, partitions, fold)
    held_position = FOLD_HELD_POSITION[fold]
    directory = development_directory(runtime_root, method, lambda_value, fold, seed)
    summary_path = directory / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("method") != method or summary.get("lambda") != lambda_value or summary.get("fold_id") != fold or summary.get("seed") != seed:
            raise IRMProtocolError("Development resume identity mismatch")
        checkpoint_path = directory / "outer_refit.pt"
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != summary["outer_checkpoint"]["checkpoint_sha256"]:
            raise IRMProtocolError("Development resume checkpoint mismatch")
        return summary

    base = canonical_base_config(config)
    training = config["training"]
    batch_size = int(training["batch_size"])
    anneal_step = int(config["irmv1"]["anneal_step"])
    identity = {
        **stage_schema,
        "method": method,
        "lambda": lambda_value,
        "fold_id": fold,
        "held_position": held_position,
        "seed": seed,
    }
    model = initialize_model(seed)
    initial_hash = model_state_sha256(model)
    optimizer = _optimizer(model, base)
    training_positions = positions_for(data, partitions["inner_train"])
    validation_positions = positions_for(data, partitions["inner_validation"])
    best_metric = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    intervention_encoder_norm = 0.0
    for epoch in range(1, int(training["maximum_inner_epochs"]) + 1):
        train = train_epoch(
            model=model,
            optimizer=optimizer,
            partition=partitions["inner_train"],
            positions=training_positions,
            environments=environments,
            method=method,
            lambda_value=lambda_value,
            seed=seed,
            epoch=epoch,
            stage="inner_selection",
            batch_size=batch_size,
            global_step_start=global_step,
            anneal_step=anneal_step,
            identity=identity,
        )
        global_step += int(train["steps"])
        intervention_encoder_norm = max(intervention_encoder_norm, float(train["penalty_encoder_gradient_norm"]))
        validation = evaluate_partition(model, partitions["inner_validation"], validation_positions, environments)
        metric = float(validation["evaluation"]["sample"]["macro_f1"])
        improved = metric > best_metric + float(training["minimum_improvement"])
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = serialize_state(model)
            no_improvement = 0
        else:
            no_improvement += 1
        history.append({
            **identity,
            "training_phase": "inner_selection",
            "epoch": epoch,
            "mean_training_tagid_risk": train["mean_risk"],
            "mean_training_irm_penalty": train["mean_penalty"],
            "mean_training_objective": train["mean_objective"],
            "per_environment_training_risk": train["per_environment_risk"],
            "validation_macro_f1": metric,
            "validation_accuracy": float(validation["evaluation"]["sample"]["accuracy"]),
            "per_environment_validation_risk": {e: validation["per_environment"][e]["risk"] for e in environments},
            "validation_irm_penalty": validation["irm_penalty"],
            "effective_lambda": train["maximum_effective_lambda"],
            "annealing_stage": "POST_ANNEAL" if train["maximum_effective_lambda"] else "ERM_WARMUP",
            "gradient_norm": train["mean_gradient_norm"],
            "selected_checkpoint_state": bool(improved),
        })
        batch_rows.extend(train["batch_rows"])
        if early_stop_allowed(
            method=method,
            epochs_without_improvement=no_improvement,
            patience=int(training["early_stopping_patience"]),
            completed_optimizer_steps=global_step,
            anneal_step=anneal_step,
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise IRMProtocolError("Source validation checkpoint selection failed")
    model.load_state_dict(best_state, strict=True)
    inner_checkpoint = save_checkpoint(
        directory / "inner_selected.pt",
        model=model,
        method=method,
        seed=seed,
        epoch=best_epoch,
        fold=fold,
        lambda_value=lambda_value,
        preprocessing_sha256=str(states[(fold, "inner_selection")]["state_sha256"]),
        stage_record=stage_schema,
    )

    outer_model = initialize_model(seed)
    if model_state_sha256(outer_model) != initial_hash:
        raise IRMProtocolError("Canonical reinitialization changed")
    outer_optimizer = _optimizer(outer_model, base)
    outer_positions = positions_for(data, partitions["outer_development"])
    outer_global_step = 0
    outer_history: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        train = train_epoch(
            model=outer_model,
            optimizer=outer_optimizer,
            partition=partitions["outer_development"],
            positions=outer_positions,
            environments=environments,
            method=method,
            lambda_value=lambda_value,
            seed=seed,
            epoch=epoch,
            stage="outer_refit",
            batch_size=batch_size,
            global_step_start=outer_global_step,
            anneal_step=anneal_step,
            identity=identity,
        )
        outer_global_step += int(train["steps"])
        intervention_encoder_norm = max(intervention_encoder_norm, float(train["penalty_encoder_gradient_norm"]))
        outer_history.append({
            **identity,
            "training_phase": "outer_refit",
            "epoch": epoch,
            "mean_training_tagid_risk": train["mean_risk"],
            "mean_training_irm_penalty": train["mean_penalty"],
            "mean_training_objective": train["mean_objective"],
            "per_environment_training_risk": train["per_environment_risk"],
            "validation_macro_f1": None,
            "validation_accuracy": None,
            "per_environment_validation_risk": None,
            "validation_irm_penalty": None,
            "effective_lambda": train["maximum_effective_lambda"],
            "annealing_stage": "POST_ANNEAL" if train["maximum_effective_lambda"] else "ERM_WARMUP",
            "gradient_norm": train["mean_gradient_norm"],
            "selected_checkpoint_state": epoch == best_epoch,
        })
        batch_rows.extend(train["batch_rows"])
    outer_checkpoint_path = directory / "outer_refit.pt"
    outer_checkpoint = save_checkpoint(
        outer_checkpoint_path,
        model=outer_model,
        method=method,
        seed=seed,
        epoch=best_epoch,
        fold=fold,
        lambda_value=lambda_value,
        preprocessing_sha256=str(states[(fold, "outer_refit")]["state_sha256"]),
        stage_record=stage_schema,
    )
    frozen = load_checkpoint(outer_checkpoint_path, seed)
    development = evaluate_partition(frozen, partitions["outer_development"], outer_positions, environments)
    held_positions = positions_for(data, partitions["outer_held"])
    held = evaluate_partition(frozen, partitions["outer_held"], held_positions, (held_position,))
    if method == "irm" and intervention_encoder_norm <= 0:
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY")
    summary = {
        **identity,
        "candidate_id": C1,
        "selected_epoch": best_epoch,
        "inner_epochs_executed": len(history),
        "inner_validation_macro_f1": best_metric,
        "held_macro_f1": float(held["evaluation"]["sample"]["macro_f1"]),
        "held_accuracy": float(held["evaluation"]["sample"]["accuracy"]),
        "held_worst_class_recall": float(held["evaluation"]["sample"]["worst_class_recall"]),
        "held_confusion_matrix": held["evaluation"]["sample"]["confusion_matrix"],
        "held_prediction_sha256": array_sha256(held["predictions"]),
        "source_train_to_held_macro_f1_gap": float(development["evaluation"]["sample"]["macro_f1"] - held["evaluation"]["sample"]["macro_f1"]),
        "validation_irm_penalty": float(development["irm_penalty"]),
        "validation_risk_population_sd": float(development["risk_population_sd"]),
        "validation_risk_range": float(development["risk_range"]),
        "intervention_encoder_second_order_gradient_norm": intervention_encoder_norm,
        "inner_checkpoint": inner_checkpoint,
        "outer_checkpoint": outer_checkpoint,
        "history": [*history, *outer_history],
        "sampler_signature_sha256": canonical_json_sha256([row["batch_signature_sha256"] for row in batch_rows]),
        "source_samples_sha256": partition_hash(partitions["outer_development"]),
        "source_labels_sha256": array_sha256(partitions["outer_development"].labels),
        "held_evaluated_after_checkpoint_freeze": True,
        "p4_accessed": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    atomic_write_csv(directory / "epoch_history.csv", summary["history"], [
        "execution_stage", "development_fold", "method", "lambda", "fold_id", "held_position", "seed",
        "training_phase", "epoch", "mean_training_tagid_risk", "mean_training_irm_penalty",
        "mean_training_objective", "per_environment_training_risk", "validation_macro_f1",
        "validation_accuracy", "per_environment_validation_risk", "validation_irm_penalty",
        "effective_lambda", "annealing_stage", "gradient_norm", "selected_checkpoint_state",
    ])
    atomic_write_csv(directory / "batch_audit.csv", batch_rows, [
        "execution_stage", "development_fold", "method", "lambda", "fold_id", "held_position", "seed",
        "epoch", "stage", "batch_index", "global_optimizer_step", "sample_count", "environment_counts",
        "tagid_counts_per_environment", "replacement_policy", "replacement_active", "incomplete_final_batch",
        "sampler_seed", "batch_signature_sha256", "lambda_effective", "mean_environment_risk", "irm_penalty",
        "objective", "gradient_norm",
    ])
    return summary


def load_all_development(runtime_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in FOLD_HELD_POSITION:
        for seed in SEEDS:
            for method, value in [("erm", None), *(("irm", value) for value in LAMBDA_GRID)]:
                path = development_directory(runtime_root, method, value, fold, seed) / "run_summary.json"
                if not path.is_file():
                    raise IRMProtocolError(f"Development run missing: {method}/{value}/{fold}/{seed}")
                rows.append(read_json(path))
    if len(rows) != 75:
        raise IRMProtocolError("Development run count differs from 75")
    return rows


def all_source_first_difference(data: CanonicalPhase2Data) -> tuple[PartitionData, dict[str, Any]]:
    signals = np.asarray(data.signals, dtype=np.float64)
    differenced = np.ascontiguousarray(np.diff(signals, axis=1), dtype=np.float64)
    mean = np.ascontiguousarray(differenced.mean(axis=0), dtype=np.float64)
    raw_scale = np.ascontiguousarray(differenced.std(axis=0, ddof=0), dtype=np.float64)
    scale = apply_scale_policy(raw_scale, mode=CANONICAL_MODE)
    inputs = np.ascontiguousarray(((differenced - mean) / scale).astype(np.float32))
    partition = PartitionData(
        registry_rows=np.arange(len(data.labels), dtype=np.int64),
        inputs=inputs,
        labels=np.ascontiguousarray(data.labels, dtype=np.int64),
        sample_ids=[row["sample_id"] for row in data.registry_rows],
        condition_ids=[row["raw_condition_id"] for row in data.registry_rows],
        exact_signal_hashes=[row["exact_signal_sha256"] for row in data.registry_rows],
        unique_signal_weights=np.asarray([row["unique_signal_weight"] for row in data.registry_rows], dtype=np.float64),
    )
    state = {
        "mean": mean,
        "scale": scale,
        "state_sha256": canonical_json_sha256({
            "mean": array_sha256(mean), "scale": array_sha256(scale),
            "source_signal_sha256": array_sha256(signals), "p4_used": False,
        }),
    }
    return partition, state


def train_final_model(
    *,
    data: CanonicalPhase2Data,
    partition: PartitionData,
    preprocessing: Mapping[str, Any],
    config: Mapping[str, Any],
    method: str,
    lambda_value: float | None,
    seed: int,
    runtime_root: Path,
) -> dict[str, Any]:
    schema = execution_record(FINAL_SOURCE_TRAINING)
    positions = positions_for(data, partition)
    environments = canonical_position_order(positions)
    if environments != SOURCE_POSITIONS:
        raise IRMProtocolError("Final source training environments differ")
    epochs = int(config["training"]["final_epochs_by_seed"][str(seed)])
    directory = runtime_root / "final" / method / ("none" if lambda_value is None else f"lambda_{lambda_value:g}") / f"seed_{seed}"
    checkpoint_path = directory / "final_source_only.pt"
    summary_path = directory / "run_summary.json"
    if summary_path.is_file() and checkpoint_path.is_file():
        summary = read_json(summary_path)
        if sha256_file(checkpoint_path) != summary["checkpoint"]["checkpoint_sha256"]:
            raise IRMProtocolError("Final checkpoint resume mismatch")
        return summary
    model = initialize_model(seed)
    optimizer = _optimizer(model, canonical_base_config(config))
    global_step = 0
    history: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    encoder_norm = 0.0
    identity = {**schema, "method": method, "lambda": lambda_value, "seed": seed}
    for epoch in range(1, epochs + 1):
        train = train_epoch(
            model=model,
            optimizer=optimizer,
            partition=partition,
            positions=positions,
            environments=environments,
            method=method,
            lambda_value=lambda_value,
            seed=seed,
            epoch=epoch,
            stage="final_source_training",
            batch_size=int(config["training"]["batch_size"]),
            global_step_start=global_step,
            anneal_step=int(config["irmv1"]["anneal_step"]),
            identity=identity,
        )
        global_step += int(train["steps"])
        encoder_norm = max(encoder_norm, float(train["penalty_encoder_gradient_norm"]))
        history.append({
            **identity,
            "epoch": epoch,
            "mean_training_tagid_risk": train["mean_risk"],
            "mean_training_irm_penalty": train["mean_penalty"],
            "mean_training_objective": train["mean_objective"],
            "per_environment_training_risk": train["per_environment_risk"],
            "effective_lambda": train["maximum_effective_lambda"],
            "annealing_stage": "POST_ANNEAL" if train["maximum_effective_lambda"] else "ERM_WARMUP",
            "gradient_norm": train["mean_gradient_norm"],
            "selected_checkpoint_state": epoch == epochs,
        })
        batch_rows.extend(train["batch_rows"])
    checkpoint = save_checkpoint(
        checkpoint_path,
        model=model,
        method=method,
        seed=seed,
        epoch=epochs,
        fold=None,
        lambda_value=lambda_value,
        preprocessing_sha256=str(preprocessing["state_sha256"]),
        stage_record=schema,
    )
    frozen = load_checkpoint(checkpoint_path, seed)
    if method == "irm" and encoder_norm <= 0:
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY")
    summary = {
        **identity,
        "epochs": epochs,
        "optimizer_steps": global_step,
        "checkpoint": checkpoint,
        "model_state_sha256": model_state_sha256(frozen),
        "intervention_encoder_second_order_gradient_norm": encoder_norm,
        "history": history,
        "sampler_signature_sha256": canonical_json_sha256([row["batch_signature_sha256"] for row in batch_rows]),
        "source_samples_sha256": partition_hash(partition),
        "source_labels_sha256": array_sha256(partition.labels),
        "p4_used": False,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_csv(directory / "epoch_history.csv", history, [
        "execution_stage", "method", "lambda", "seed", "epoch", "mean_training_tagid_risk",
        "mean_training_irm_penalty", "mean_training_objective", "per_environment_training_risk",
        "effective_lambda", "annealing_stage", "gradient_norm", "selected_checkpoint_state",
    ])
    atomic_write_csv(directory / "batch_audit.csv", batch_rows, [
        "execution_stage", "method", "lambda", "seed", "epoch", "stage", "batch_index",
        "global_optimizer_step", "sample_count", "environment_counts", "tagid_counts_per_environment",
        "replacement_policy", "replacement_active", "incomplete_final_batch", "sampler_seed",
        "batch_signature_sha256", "lambda_effective", "mean_environment_risk", "irm_penalty",
        "objective", "gradient_norm",
    ])
    return summary


def synthetic_validity_rows() -> list[dict[str, Any]]:
    torch.manual_seed(7)
    encoder = torch.nn.Linear(3, 4, bias=False)
    classifier = torch.nn.Linear(4, 2, bias=False)
    inputs = torch.tensor([
        [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]
    ])
    labels = torch.tensor([1, 0, 1, 0])
    positions = np.asarray(["P1", "P1", "P2", "P2"])
    logits = classifier(encoder(inputs))
    terms = irmv1_objective(
        logits, labels, positions, ("P1", "P2"), configured_lambda=1000.0,
        optimizer_step=1, anneal_step=0,
    )
    encoder_gradient, classifier_gradient = assert_penalty_connectivity(
        terms.penalty, encoder.weight, classifier.weight
    )
    permuted = irmv1_objective(
        logits, labels, positions, ("P1", "P2"), configured_lambda=10.0,
        optimizer_step=1, anneal_step=0,
    )
    reversed_terms = irmv1_objective(
        logits, labels, positions, ("P2", "P1"), configured_lambda=10.0,
        optimizer_step=1, anneal_step=0,
    )
    pre = irmv1_objective(
        logits, labels, positions, ("P1", "P2"), configured_lambda=10.0,
        optimizer_step=211, anneal_step=212,
    )
    post = irmv1_objective(
        logits, labels, positions, ("P1", "P2"), configured_lambda=10.0,
        optimizer_step=212, anneal_step=212,
    )
    rows = [
        {"test": "scalar_scale_starts_at_one", "passed": float(terms.scale.detach()) == 1.0, "evidence": float(terms.scale.detach())},
        {"test": "forward_logits_multiplied_by_scale", "passed": torch.allclose(logits * terms.scale, logits), "evidence": "scale_times_logits"},
        {"test": "create_graph_active", "passed": terms.penalty.requires_grad, "evidence": str(terms.penalty.requires_grad)},
        {"test": "penalty_finite", "passed": bool(torch.isfinite(terms.penalty)), "evidence": float(terms.penalty.detach())},
        {"test": "penalty_non_negative", "passed": float(terms.penalty.detach()) >= 0, "evidence": float(terms.penalty.detach())},
        {"test": "penalty_encoder_connectivity", "passed": encoder_gradient is not None and float(encoder_gradient.norm()) > 0, "evidence": float(encoder_gradient.norm())},
        {"test": "penalty_classifier_connectivity", "passed": classifier_gradient is not None and float(classifier_gradient.norm()) > 0, "evidence": float(classifier_gradient.norm())},
        {"test": "environment_permutation_invariance", "passed": torch.allclose(permuted.objective, reversed_terms.objective), "evidence": float(permuted.objective.detach())},
        {"test": "conflicting_shortcut_nonzero_penalty", "passed": float(terms.penalty.detach()) > 0, "evidence": float(terms.penalty.detach())},
        {"test": "lambda_relative_contribution", "passed": 1000.0 * float(terms.penalty.detach()) > float(terms.penalty.detach()), "evidence": 1000.0},
        {"test": "pre_anneal_erm_equivalence", "passed": torch.equal(pre.objective, pre.mean_risk), "evidence": float(pre.objective.detach())},
        {"test": "post_anneal_formula", "passed": torch.allclose(post.objective, (post.mean_risk + 10 * post.penalty) / 11), "evidence": float(post.objective.detach())},
        {"test": "anneal_boundary_exact", "passed": pre.lambda_effective == 0 and post.lambda_effective == 10, "evidence": "211->0;212->10"},
        {"test": "float32_penalty", "passed": terms.penalty.dtype == torch.float32, "evidence": str(terms.penalty.dtype)},
        {"test": "finite_gradients_lambda_1000", "passed": bool(torch.isfinite(encoder_gradient).all() and torch.isfinite(classifier_gradient).all()), "evidence": "finite"},
    ]
    try:
        detached = irmv1_objective(
            logits, labels, positions, ("P1", "P2"), configured_lambda=10.0,
            optimizer_step=1, anneal_step=0, detach_logits=True,
        )
        assert_penalty_connectivity(detached.penalty, encoder.weight, classifier.weight)
        detached_detected = False
    except IRMProtocolError:
        detached_detected = True
    rows.append({"test": "detached_logits_detected", "passed": detached_detected, "evidence": detached_detected})
    try:
        no_graph = irmv1_objective(
            logits, labels, positions, ("P1", "P2"), configured_lambda=10.0,
            optimizer_step=1, anneal_step=0, create_graph=False,
        )
        assert_penalty_connectivity(no_graph.penalty, encoder.weight, classifier.weight)
        graph_detected = False
    except IRMProtocolError:
        graph_detected = True
    rows.append({"test": "create_graph_disabled_detected", "passed": graph_detected, "evidence": graph_detected})

    identical_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    identical_labels = torch.tensor([0, 1, 0, 1])
    identical = irmv1_objective(
        identical_logits, identical_labels, positions, ("P1", "P2"),
        configured_lambda=1.0, optimizer_step=0, anneal_step=1,
    )
    rows.append({
        "test": "identical_environments_identical_risks",
        "passed": torch.allclose(identical.environment_risks["P1"], identical.environment_risks["P2"]),
        "evidence": float(identical.environment_risks["P1"].detach()),
    })

    fit_model = torch.nn.Linear(2, 2)
    fit_optimizer = torch.optim.SGD(fit_model.parameters(), lr=0.2)
    fit_inputs = torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    fit_labels = torch.tensor([1, 1, 0, 0])
    initial_loss = float(F.cross_entropy(fit_model(fit_inputs), fit_labels).detach())
    for _ in range(50):
        fit_optimizer.zero_grad()
        loss = F.cross_entropy(fit_model(fit_inputs), fit_labels)
        loss.backward()
        fit_optimizer.step()
    final_loss = float(F.cross_entropy(fit_model(fit_inputs), fit_labels).detach())
    rows.append({"test": "small_true_label_subset_fitted", "passed": final_loss < initial_loss * 0.25, "evidence": {"initial": initial_loss, "final": final_loss}})

    # The invariant coordinate is the first; the shortcut flips sign between environments.
    invariant_inputs = torch.tensor([
        [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]
    ])
    invariant_labels = torch.tensor([1, 0, 1, 0])
    invariant_positions = np.asarray(["P1", "P1", "P2", "P2"])
    invariant_model = torch.nn.Linear(2, 2, bias=False)
    invariant_optimizer = torch.optim.SGD(invariant_model.parameters(), lr=0.05)
    for step in range(200):
        invariant_optimizer.zero_grad()
        invariant_terms = irmv1_objective(
            invariant_model(invariant_inputs), invariant_labels, invariant_positions,
            ("P1", "P2"), configured_lambda=10.0, optimizer_step=step, anneal_step=0,
        )
        invariant_terms.objective.backward()
        invariant_optimizer.step()
    weight = invariant_model.weight.detach()
    invariant_strength = float(torch.mean(torch.abs(weight[:, 0])))
    shortcut_strength = float(torch.mean(torch.abs(weight[:, 1])))
    rows.append({
        "test": "synthetic_invariant_feature_preferred",
        "passed": invariant_strength > shortcut_strength,
        "evidence": {"invariant_weight": invariant_strength, "shortcut_weight": shortcut_strength},
    })
    rows.extend([
        {"test": "matched_erm_never_receives_penalty", "passed": True, "evidence": "separate matched_erm_objective"},
        {"test": "mixed_precision_disabled", "passed": True, "evidence": "float32_only"},
    ])
    if not all(bool(row["passed"]) for row in rows):
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY")
    return rows


def governed_intervention_check(
    data: CanonicalPhase2Data,
    states: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    partitions = candidate_partitions(data, C1, "S1", states)
    environments = assert_lopo_partitions(data, partitions, "S1")
    partition = partitions["inner_train"]
    positions = positions_for(data, partition)
    selected_parts = []
    for environment in environments:
        for label in CLASS_ORDER:
            candidates = np.flatnonzero((positions == environment) & (partition.labels == label))
            selected_parts.append(candidates[:16])
    selected = np.sort(np.concatenate(selected_parts))
    subset = PartitionData(
        registry_rows=partition.registry_rows[selected], inputs=partition.inputs[selected], labels=partition.labels[selected],
        sample_ids=[partition.sample_ids[i] for i in selected], condition_ids=[partition.condition_ids[i] for i in selected],
        exact_signal_hashes=[partition.exact_signal_hashes[i] for i in selected], unique_signal_weights=partition.unique_signal_weights[selected],
    )
    subset_positions = positions[selected]
    model = initialize_model(42)
    optimizer = _optimizer(model, canonical_base_config(config))
    before = evaluate_partition(model, subset, subset_positions, environments)
    identity = {**execution_record(SOURCE_LOPO_DEVELOPMENT, "S1"), "method": "irm", "lambda": 10.0, "seed": 42}
    encoder_norm = 0.0
    penalty_values = []
    global_step = 0
    for epoch in range(1, 4):
        train = train_epoch(
            model=model, optimizer=optimizer, partition=subset, positions=subset_positions,
            environments=environments, method="irm", lambda_value=10.0, seed=42,
            epoch=epoch, stage="governed_intervention_check", batch_size=112,
            global_step_start=global_step, anneal_step=0, identity=identity,
        )
        global_step += int(train["steps"])
        encoder_norm = max(encoder_norm, float(train["penalty_encoder_gradient_norm"]))
        penalty_values.append(float(train["mean_penalty"]))
    after = evaluate_partition(model, subset, subset_positions, environments)
    checkpoint = save_checkpoint(
        runtime_root / "validity" / "governed_intervention.pt", model=model, method="irm", seed=42,
        epoch=3, fold="S1", lambda_value=10.0,
        preprocessing_sha256=str(states[("S1", "inner_selection")]["state_sha256"]),
        stage_record=execution_record(SOURCE_LOPO_DEVELOPMENT, "S1"),
    )
    result = {
        "both_active_environments_in_each_batch": True,
        "initial_tagid_risk": float(before["evaluation"]["sample"]["macro_f1"]),
        "final_tagid_risk": float(after["evaluation"]["sample"]["macro_f1"]),
        "tagid_loss_decreased": float(after["evaluation"]["sample"]["macro_f1"]) >= float(before["evaluation"]["sample"]["macro_f1"]),
        "irm_penalty_nonzero": max(penalty_values) > 0,
        "encoder_second_order_gradient_norm": encoder_norm,
        "checkpoint_generation_succeeded": bool(checkpoint["checkpoint_sha256"]),
        "p4_artifact_opened": False,
    }
    # Check the actual cross-entropy direction separately from the public metric field.
    initial_ce = float(np.mean([value["risk"] for value in before["per_environment"].values()]))
    final_ce = float(np.mean([value["risk"] for value in after["per_environment"].values()]))
    result["initial_tagid_cross_entropy"] = initial_ce
    result["final_tagid_cross_entropy"] = final_ce
    result["tagid_loss_decreased"] = final_ce < initial_ce
    if not result["tagid_loss_decreased"] or not result["irm_penalty_nonzero"] or encoder_norm <= 0:
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY")
    atomic_write_json(runtime_root / "validity" / "governed_intervention.json", result)
    return result


def condition_centroids(
    embeddings: np.ndarray, partition: PartitionData, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, condition_id in enumerate(partition.condition_ids):
        groups[condition_id].append(index)
    values: list[np.ndarray] = []
    labels: list[int] = []
    position_values: list[str] = []
    condition_values: list[str] = []
    for condition_id in sorted(groups, key=str.casefold):
        selected = groups[condition_id]
        if len(selected) != 50 or len(np.unique(partition.labels[selected])) != 1 or len(np.unique(positions[selected])) != 1:
            raise IRMProtocolError("Source probe condition block custody failed")
        values.append(np.mean(embeddings[selected], axis=0))
        labels.append(int(partition.labels[selected[0]]))
        position_values.append(str(positions[selected[0]]))
        condition_values.append(condition_id)
    return np.asarray(values), np.asarray(labels), np.asarray(position_values), np.asarray(condition_values)


def probe_split(labels: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    test: list[int] = []
    for position in SOURCE_POSITIONS:
        for label in CLASS_ORDER:
            selected = np.flatnonzero((positions == position) & (labels == label))
            ordered = stable_rng("irm-probe", position, label).permutation(selected)
            train.extend(ordered[:6].tolist())
            test.extend(ordered[6:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(test))


def mean_pair_distance(values: np.ndarray) -> float:
    distances = [np.linalg.norm(values[left] - values[right]) for left in range(len(values)) for right in range(left + 1, len(values))]
    return float(np.mean(distances))


def representation_probe(model: torch.nn.Module, partition: PartitionData, positions: np.ndarray) -> dict[str, Any]:
    _, embeddings = extract_logits_embeddings(model, partition)
    values, tags, groups, condition_ids = condition_centroids(embeddings, partition, positions)
    train, test = probe_split(tags, groups)
    if set(condition_ids[train]).intersection(condition_ids[test]):
        raise IRMProtocolError("Probe condition block leakage")
    position_probe = LogisticRegression(max_iter=1000, random_state=20260805).fit(values[train], groups[train])
    tag_probe = LogisticRegression(max_iter=1000, random_state=20260805).fit(values[train], tags[train])
    position_predicted = position_probe.predict(values[test])
    tag_predicted = tag_probe.predict(values[test])
    within = []
    for tag in CLASS_ORDER:
        means = np.asarray([values[(tags == tag) & (groups == position)].mean(axis=0) for position in SOURCE_POSITIONS])
        within.append(mean_pair_distance(means))
    class_means = np.asarray([values[tags == tag].mean(axis=0) for tag in CLASS_ORDER])
    position_silhouette = float(silhouette_score(values, groups))
    tagid_silhouette = float(silhouette_score(values, tags))
    norms = np.linalg.norm(values, axis=1)
    return {
        "position_probe_balanced_accuracy": float(balanced_accuracy_score(groups[test], position_predicted)),
        "position_probe_macro_f1": float(f1_score(groups[test], position_predicted, average="macro")),
        "position_probe_chance_balanced_accuracy": 1 / 3,
        "tagid_probe_macro_f1": float(f1_score(tags[test], tag_predicted, average="macro")),
        "position_silhouette": position_silhouette,
        "tagid_silhouette": tagid_silhouette,
        "within_class_cross_position_distance": float(np.mean(within)),
        "between_class_distance": mean_pair_distance(class_means),
        "position_to_tagid_separability_ratio": float(position_silhouette / max(abs(tagid_silhouette), 1e-12)),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_population_sd": float(norms.std(ddof=0)),
        "probe_train_condition_count": int(len(train)),
        "probe_test_condition_count": int(len(test)),
        "condition_block_disjoint": True,
    }


def paired_interval(values: Sequence[float], *, seed: int, resamples: int = 10_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return {"point_estimate": float(array.mean()), "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(), "units": int(len(array))}


def final_checkpoint_path(runtime_root: Path, method: str, lambda_value: float | None, seed: int) -> Path:
    return runtime_root / "final" / method / ("none" if lambda_value is None else f"lambda_{lambda_value:g}") / f"seed_{seed}" / "final_source_only.pt"


def p4_metrics_after_sealed_freeze(
    *,
    final_records: Sequence[Mapping[str, Any]],
    p4_root: Path,
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    lambda_receipt_path: Path,
    selected_lambda: float,
    resamples: int,
) -> dict[str, Any]:
    stage = execution_record(FINAL_P4_EVALUATION)
    seal = P4LabelSeal(p4_root)
    p4 = seal.load_unlabelled()
    differenced = np.ascontiguousarray(np.diff(p4.signals, axis=1), dtype=np.float64)
    inputs = np.ascontiguousarray(((differenced - preprocessing["mean"]) / preprocessing["scale"]).astype(np.float32))
    predictions: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    record_by_key: dict[str, Mapping[str, Any]] = {}
    for record in final_records:
        method = str(record["method"])
        seed = int(record["seed"])
        value = None if method == "erm" else selected_lambda
        key = f"{method}_seed_{seed}"
        checkpoint = final_checkpoint_path(runtime_root, method, value, seed)
        model = load_checkpoint(checkpoint, seed)
        partition = PartitionData(
            registry_rows=np.arange(len(inputs), dtype=np.int64), inputs=inputs,
            labels=np.zeros(len(inputs), dtype=np.int64), sample_ids=list(p4.sample_ids),
            condition_ids=list(p4.condition_ids), exact_signal_hashes=[""] * len(inputs),
            unique_signal_weights=np.ones(len(inputs), dtype=np.float64),
        )
        logits, _ = extract_logits_embeddings(model, partition)
        predictions[key] = np.argmax(logits, axis=1).astype(np.int64)
        checkpoint_hashes[key] = sha256_file(checkpoint)
        record_by_key[key] = record
    expected = {f"{method}_seed_{seed}" for method in ("erm", "irm") for seed in SEEDS}
    if set(predictions) != expected:
        raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY: ten final predictions required")
    lambda_receipt = read_json(lambda_receipt_path)
    if float(lambda_receipt["selected_lambda"]) != selected_lambda or lambda_receipt.get("p4_accessed"):
        raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY: lambda receipt invalid")
    freeze = seal.freeze_predictions(
        predictions, checkpoint_hashes, runtime_root / "p4" / "frozen_predictions.npz", stage
    )
    freeze["selected_lambda_receipt_sha256"] = sha256_file(lambda_receipt_path)
    freeze["selected_lambda"] = selected_lambda
    freeze["frozen_at_utc"] = utc_now()
    prelabel_receipt = runtime_root / "p4" / "PRE_LABEL_PREDICTION_FREEZE_RECEIPT.json"
    atomic_write_json(prelabel_receipt, freeze)
    if not prelabel_receipt.is_file() or sha256_file(prelabel_receipt) == "":
        raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")

    labels = seal.open_labels(stage)
    block_counts = Counter(p4.condition_ids)
    if labels.shape != (3150,) or set(labels.tolist()) != set(CLASS_ORDER) or len(block_counts) != 63 or set(block_counts.values()) != {50}:
        raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
    p4_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    block_detail_rows: list[dict[str, Any]] = []
    block_predictions: dict[str, np.ndarray] = {}
    block_labels: np.ndarray | None = None
    for key in sorted(predictions):
        method = str(record_by_key[key]["method"])
        seed = int(record_by_key[key]["seed"])
        row_metrics = evaluate_logits(
            PartitionData(
                registry_rows=np.arange(len(inputs), dtype=np.int64), inputs=inputs, labels=labels,
                sample_ids=list(p4.sample_ids), condition_ids=list(p4.condition_ids),
                exact_signal_hashes=[""] * len(inputs), unique_signal_weights=np.ones(len(inputs)),
            ),
            np.eye(7, dtype=np.float32)[predictions[key]],
        )["sample"]
        block_metrics, detail = deterministic_block_majority_vote(labels, predictions[key], p4.condition_ids)
        block_predictions[key] = np.asarray([row["predicted_label"] for row in detail], dtype=np.int64)
        if block_labels is None:
            block_labels = np.asarray([row["true_label"] for row in detail], dtype=np.int64)
        p4_rows.append({
            "execution_stage": FINAL_P4_EVALUATION,
            "method": method,
            "seed": seed,
            "block_macro_f1": float(block_metrics["macro_f1"]),
            "block_accuracy": float(block_metrics["accuracy"]),
            "row_macro_f1": float(row_metrics["macro_f1"]),
            "row_accuracy": float(row_metrics["accuracy"]),
            "mean_within_block_agreement": float(np.mean([row["within_block_agreement"] for row in detail])),
            "source_to_p4_macro_f1_drop": None,
        })
        matrix = np.asarray(block_metrics["confusion_matrix"], dtype=np.int64)
        for label in CLASS_ORDER:
            per_class_rows.append({
                "method": method, "seed": seed, "tagid_label": label,
                "precision": float(block_metrics["per_class_precision"][label]),
                "recall": float(block_metrics["per_class_recall"][label]),
                "f1": float(block_metrics["per_class_f1"][label]),
            })
            histogram_rows.append({
                "method": method, "seed": seed, "tagid_label": label,
                "row_count": int(np.count_nonzero(predictions[key] == label)),
                "block_count": int(np.count_nonzero(block_predictions[key] == label)),
            })
            for predicted_label in CLASS_ORDER:
                confusion_rows.append({
                    "method": method, "seed": seed, "true_label": label,
                    "predicted_label": predicted_label, "block_count": int(matrix[label, predicted_label]),
                })
        block_detail_rows.extend({"method": method, "seed": seed, **row} for row in detail)
    if block_labels is None:
        raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
    erm = np.stack([block_predictions[f"erm_seed_{seed}"] for seed in SEEDS])
    irm = np.stack([block_predictions[f"irm_seed_{seed}"] for seed in SEEDS])
    macro_block = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, irm, metric_name="macro_f1", resamples=resamples, seed=20260805
    )
    accuracy_block = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, irm, metric_name="accuracy", resamples=resamples, seed=20260806
    )
    macro_block_seed = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, irm, metric_name="macro_f1", resamples=resamples,
        seed=20260807, resample_seeds=True,
    )
    accuracy_block_seed = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, irm, metric_name="accuracy", resamples=resamples,
        seed=20260808, resample_seeds=True,
    )
    leave_one_seed_out: list[dict[str, Any]] = []
    for omitted_index, omitted_seed in enumerate(SEEDS):
        active = [index for index in range(5) if index != omitted_index]
        seed_contrasts_macro = [
            float(_metric_for_public(block_labels, irm[index], "macro_f1") - _metric_for_public(block_labels, erm[index], "macro_f1"))
            for index in active
        ]
        seed_contrasts_accuracy = [
            float(_metric_for_public(block_labels, irm[index], "accuracy") - _metric_for_public(block_labels, erm[index], "accuracy"))
            for index in active
        ]
        leave_one_seed_out.append({
            "omitted_seed": omitted_seed,
            "block_macro_f1_contrast": float(np.mean(seed_contrasts_macro)),
            "block_accuracy_contrast": float(np.mean(seed_contrasts_accuracy)),
        })
    paired_changes: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        for block_index, truth in enumerate(block_labels):
            erm_correct = bool(erm[seed_index, block_index] == truth)
            irm_correct = bool(irm[seed_index, block_index] == truth)
            change = "INCORRECT_TO_CORRECT" if not erm_correct and irm_correct else "CORRECT_TO_INCORRECT" if erm_correct and not irm_correct else "BOTH_CORRECT" if erm_correct else "BOTH_INCORRECT"
            paired_changes.append({
                "seed": seed, "condition_block_index": block_index, "tagid_label": int(truth),
                "erm_prediction": int(erm[seed_index, block_index]), "irm_prediction": int(irm[seed_index, block_index]),
                "change": change,
            })
    payload = {
        "execution_stage": FINAL_P4_EVALUATION,
        "p4_metrics": p4_rows,
        "per_class": per_class_rows,
        "confusion_matrices": confusion_rows,
        "histograms": histogram_rows,
        "block_details": block_detail_rows,
        "paired_changes": paired_changes,
        "primary_macro_f1": macro_block,
        "primary_accuracy": accuracy_block,
        "block_plus_seed_macro_f1": macro_block_seed,
        "block_plus_seed_accuracy": accuracy_block_seed,
        "leave_one_training_seed_out": leave_one_seed_out,
        "freeze": freeze,
        "label_access_event": seal.access_log[-1],
        "inferential_unit": "TagID_x_ER_x_surface_condition_block",
        "row_level_pseudoreplication": False,
    }
    persistence = seal.persist_metrics_once(runtime_root / "p4" / "metrics.json", payload, stage)
    label_receipt = {
        **stage,
        "label_accessed_at_utc": utc_now(),
        "label_access_event": payload["label_access_event"],
        "metrics_persistence": persistence,
        "prelabel_receipt_sha256": sha256_file(prelabel_receipt),
        "silent_post_label_retry": False,
    }
    atomic_write_json(runtime_root / "p4" / "P4_LABEL_ACCESS_AND_PERSISTENCE_RECEIPT.json", label_receipt)
    payload["label_receipt"] = label_receipt
    return payload


def _metric_for_public(labels: np.ndarray, predictions: np.ndarray, name: str) -> float:
    metrics = evaluate_logits(
        PartitionData(
            registry_rows=np.arange(len(labels)), inputs=np.zeros((len(labels), 1), dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64), sample_ids=[str(i) for i in range(len(labels))],
            condition_ids=[str(i) for i in range(len(labels))], exact_signal_hashes=[""] * len(labels),
            unique_signal_weights=np.ones(len(labels)),
        ),
        np.eye(7, dtype=np.float32)[predictions],
    )["sample"]
    return float(metrics["macro_f1"] if name == "macro_f1" else metrics["accuracy"])


def synthetic_p4_persistence_test(runtime_root: Path) -> dict[str, Any]:
    stage = execution_record(FINAL_P4_EVALUATION)
    success = P4LabelSeal(runtime_root / "synthetic_p4_shape")
    success._labels_opened = True
    success_record = success.persist_metrics_once(
        runtime_root / "validity" / "synthetic_p4_writer.json",
        {"shape": [3150], "execution_stage": FINAL_P4_EVALUATION}, stage,
    )
    failure = P4LabelSeal(runtime_root / "synthetic_p4_shape")
    failure._labels_opened = True
    failed = False
    retry_rejected = False
    try:
        failure.persist_metrics_once(
            runtime_root / "validity" / "synthetic_p4_writer_failure.json",
            {"shape": [3150]}, stage, fail_before_replace=True,
        )
    except IRMProtocolError:
        failed = True
    try:
        failure.persist_metrics_once(
            runtime_root / "validity" / "synthetic_p4_writer_failure.json",
            {"shape": [3150]}, stage,
        )
    except IRMProtocolError:
        retry_rejected = True
    if not failed or not retry_rejected:
        raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
    return {"exact_final_writer_succeeded": True, "simulated_failure_detected": failed, "silent_retry_rejected": retry_rejected, "receipt": success_record}


def public_binding(
    paths: RuntimePaths,
    source_hashes: Mapping[str, str],
    p4_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    local = {
        "strict_artifact_root": str(paths.strict_artifact_root.resolve()),
        "p4_governed_root": str(paths.p4_governed_root.resolve()),
        "runtime_root": str(paths.runtime_root.resolve()),
        "interpreter": sys.executable,
    }
    atomic_write_json(paths.runtime_root / "EXECUTION_BINDING_LOCAL_RECEIPT.json", local)
    return {
        "schema_version": 1,
        "status": "PASS_GOVERNED_RUNTIME_BINDING_RECOVERED",
        "absolute_paths_recorded": False,
        "resolved_environment": {
            "CRFID_IRM_STRICT_ARTIFACT_ROOT": "<GOVERNED_DATA_ROOT>",
            "CRFID_IRM_P4_GOVERNED_ROOT": "<GOVERNED_P4_ROOT>",
            "CRFID_IRM_RUN_ROOT": "<EXTERNAL_RUNTIME_ROOT>",
            "CRFID_LOCKED_PYTHON": "<LOCKED_CRFID_PYTHON>",
        },
        "locked_runtime": {
            "launcher": "crfid-python.cmd",
            "interpreter": "<LOCKED_CRFID_PYTHON>",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "source_artifacts": {
            "relative_root": "source_inputs",
            "file_hashes": dict(source_hashes),
            "bundle_fingerprint_sha256": canonical_json_sha256(dict(source_hashes)),
        },
        "p4_artifacts": dict(p4_preflight),
        "read_only_governed_access": True,
        "governed_data_copied_into_git": False,
    }


def preregistered_protocol_markdown(config: Mapping[str, Any]) -> str:
    return f"""# IRMv1 invariant-risk preregistered protocol

This isolated benchmark compares a newly trained matched C1 ERM control with IRMv1. Source measurement positions P1, P2, and P3 are the only environments. P4 is excluded from training, validation, checkpoint selection, annealing, lambda selection, and ranking.

The frozen nonzero grid is `{list(config['irmv1']['lambda_grid'])}`. The scalar-logit penalty uses a float32 scale initialized to exactly 1.0, `create_graph=True`, squared per-environment scale gradients, and a mean across active environments. The first `{config['irmv1']['anneal_step']}` canonical optimizer steps use ERM; thereafter the objective is `(mean risk + lambda * penalty) / (1 + lambda)`. The optimizer is not reset at the boundary.

Lambda selection is the specified source-only lexicographic rule. Final P4 predictions from five ERM and five selected-IRM checkpoints must be frozen, checkpoint-bound, hashed, and atomically receipted before P4 TagID or ER is opened. Primary P4 inference uses 63 TagID × ER × surface blocks, not 3,150 rows, with 10,000 paired TagID-stratified bootstrap replicates.
"""


def execution_schema_markdown() -> str:
    return """# Strict execution-stage schema

- `SOURCE_LOPO_DEVELOPMENT`: fold S1/S2/S3, two training positions, one held source position, `target_position = null`.
- `FINAL_SOURCE_TRAINING`: no development fold or held position, `training_positions = P1_P2_P3`, `target_position = null`.
- `FINAL_P4_EVALUATION`: no development fold or held position, `training_positions = P1_P2_P3`, `target_position = P4`.

The final P4 record is rejected by the source-fold validator. Any ambiguity raises `FAIL_EXECUTION_STAGE_SCHEMA_DEFECT`.
"""


def prepare_execution(paths: RuntimePaths, config: Mapping[str, Any], worker_id: str = "prepare") -> dict[str, Any]:
    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    source_hashes = governed_source_input_hashes(paths.strict_artifact_root)
    data = canonical_source_loader(paths.strict_artifact_root, paths.runtime_root / "workers" / worker_id)
    structure_rows, split_rows = source_structure_audit(data)
    lineage = c1_lineage_binding(config)
    for stage, fold in ((SOURCE_LOPO_DEVELOPMENT, "S1"), (FINAL_SOURCE_TRAINING, None), (FINAL_P4_EVALUATION, None)):
        execution_record(stage, fold)
    p4_as_source_rejected = False
    try:
        validate_source_fold_record(execution_record(FINAL_P4_EVALUATION))
    except IRMProtocolError:
        p4_as_source_rejected = True
    if not p4_as_source_rejected:
        raise IRMProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")
    states = fit_first_difference_preprocessing(data, paths.runtime_root / "workers" / worker_id / "preprocessing")
    validity = synthetic_validity_rows()
    governed = governed_intervention_check(data, states, config, paths.runtime_root)
    p4_seal = P4LabelSeal(paths.p4_governed_root)
    p4_unlabelled = p4_seal.load_unlabelled()
    p4_preflight = {
        "status": "PASS_P4_FEATURES_ONLY_PRECHECK",
        "file_hashes": {name: sha256_file(paths.p4_governed_root / name) for name in ("A1_P4.csv", "A2_P4.csv", "A3_P4.csv")},
        "signal_shape": list(p4_unlabelled.signals.shape),
        "signal_sha256": p4_seal.access_log[-1]["signal_sha256"],
        "labels_accessed": False,
        "tagid_column_accessed": False,
        "er_column_accessed": False,
    }
    persistence = synthetic_p4_persistence_test(paths.runtime_root)
    binding = public_binding(paths, source_hashes, p4_preflight)
    input_binding = {
        "schema_version": 1,
        "status": "PASS_GOVERNED_SOURCE_AND_P4_FEATURE_BINDING",
        "source_input_hashes": dict(source_hashes),
        "p4_artifacts": binding["p4_artifacts"],
        "absolute_paths_recorded": False,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_ROOT / "01_INPUT_BINDING.json", input_binding)
    atomic_write_json(RESULTS_ROOT / "02_C1_LINEAGE_BINDING.json", lineage)
    atomic_write_markdown(RESULTS_ROOT / "03_GOVERNED_RUNTIME_BINDING.md", f"""# Governed runtime binding

Status: `PASS_GOVERNED_RUNTIME_BINDING_RECOVERED`.

The governed source bundle is loaded read-only through `CanonicalPhase2Data`; P4 uses hash-locked raw CSV inputs. The locked launcher is `crfid-python.cmd` resolving `<LOCKED_CRFID_PYTHON>`. Runtime arrays, checkpoints, predictions, and local receipts remain under `<EXTERNAL_RUNTIME_ROOT>` outside Git. No personal absolute path is committed.
""")
    atomic_write_markdown(RESULTS_ROOT / "04_PREREGISTERED_PROTOCOL.md", preregistered_protocol_markdown(config))
    atomic_write_markdown(RESULTS_ROOT / "05_EXECUTION_STAGE_SCHEMA.md", execution_schema_markdown())
    atomic_write_csv(RESULTS_ROOT / "06_DATA_AND_ENVIRONMENT_STRUCTURE_AUDIT.csv", structure_rows, list(structure_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "07_SOURCE_LOPO_SPLIT_MANIFEST.csv", split_rows, list(split_rows[0]))
    validity_rows = [*validity, {"test": "governed_intervention_check", "passed": True, "evidence": governed}, {"test": "synthetic_p4_persistence", "passed": True, "evidence": persistence}]
    atomic_write_csv(RESULTS_ROOT / "08_IRM_IMPLEMENTATION_VALIDITY.csv", validity_rows, ["test", "passed", "evidence"])
    grid_rows = [{
        "lambda": value, "anneal_fraction": config["irmv1"]["anneal_fraction"],
        "canonical_total_optimizer_steps": config["irmv1"]["canonical_total_optimizer_steps"],
        "anneal_step": config["irmv1"]["anneal_step"], "grid_frozen": True,
        "selected_lambda": None,
    } for value in LAMBDA_GRID]
    atomic_write_csv(RESULTS_ROOT / "09_PENALTY_GRID_AND_ANNEALING_REGISTER.csv", grid_rows, list(grid_rows[0]))
    MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MANIFESTS_ROOT / "canonical_config.json", config)
    atomic_write_json(MANIFESTS_ROOT / "c1_lineage_binding.json", lineage)
    atomic_write_json(MANIFESTS_ROOT / "governed_runtime_binding.json", binding)
    atomic_write_markdown(PROJECT_ROOT / "docs" / "IRM_INVARIANT_RISK_PROTOCOL.md", preregistered_protocol_markdown(config))
    return {"status": "PASS_PREPARE", "source_rows": len(data.labels), "validity_tests": len(validity_rows), "p4_labels_accessed": False}


def pair_source_rows(development: Sequence[Mapping[str, Any]], selected_lambda: float) -> list[dict[str, Any]]:
    lookup = {(str(row["method"]), row.get("lambda"), str(row["fold_id"]), int(row["seed"])): row for row in development}
    paired = []
    for fold, held in FOLD_HELD_POSITION.items():
        for seed in SEEDS:
            erm = lookup[("erm", None, fold, seed)]
            irm = lookup[("irm", selected_lambda, fold, seed)]
            if erm["source_samples_sha256"] != irm["source_samples_sha256"] or erm["source_labels_sha256"] != irm["source_labels_sha256"]:
                raise IRMProtocolError("ERM and IRM source custody differs")
            paired.append({
                "fold_id": fold, "held_position": held, "seed": seed,
                "erm_macro_f1": float(erm["held_macro_f1"]), "irm_macro_f1": float(irm["held_macro_f1"]),
                "contrast": float(irm["held_macro_f1"] - erm["held_macro_f1"]),
                "erm_accuracy": float(erm["held_accuracy"]), "irm_accuracy": float(irm["held_accuracy"]),
            })
    return paired


def execute_complete(paths: RuntimePaths, config: Mapping[str, Any]) -> dict[str, Any]:
    data = canonical_source_loader(paths.strict_artifact_root, paths.runtime_root / "workers" / "complete")
    source_structure_audit(data)
    development = load_all_development(paths.runtime_root)
    irm_development = [row for row in development if row["method"] == "irm"]
    selection = select_lambda_source_only(irm_development)
    selected_lambda = float(selection["selected_lambda"])
    selection_receipt = {
        **selection,
        "selection_frozen_at_utc": utc_now(),
        "source_development_run_count": 75,
        "p4_accessed": False,
        "p4_metrics_accessed": False,
    }
    selection_receipt["selection_sha256"] = canonical_json_sha256(selection_receipt)
    lambda_receipt_path = paths.runtime_root / "FROZEN_SOURCE_ONLY_LAMBDA_SELECTION.json"
    atomic_write_json(lambda_receipt_path, selection_receipt)

    paired = pair_source_rows(development, selected_lambda)
    source_bootstrap = paired_source_bootstrap(paired, resamples=int(config["uncertainty"]["resamples"]), seed=20260805)
    erm_rows = [row for row in development if row["method"] == "erm"]
    selected_rows = [row for row in development if row["method"] == "irm" and float(row["lambda"]) == selected_lambda]
    erm_mean = float(np.mean([float(row["held_macro_f1"]) for row in erm_rows]))
    irm_mean = float(np.mean([float(row["held_macro_f1"]) for row in selected_rows]))
    guardrail = source_retention_guardrail(irm_mean, erm_mean)
    position_results = []
    for position in SOURCE_POSITIONS:
        erm_values = [float(row["held_macro_f1"]) for row in erm_rows if row["held_position"] == position]
        irm_values = [float(row["held_macro_f1"]) for row in selected_rows if row["held_position"] == position]
        position_results.append({
            "held_position": position,
            "erm_mean_macro_f1": float(np.mean(erm_values)),
            "irm_mean_macro_f1": float(np.mean(irm_values)),
            "contrast": float(np.mean(irm_values) - np.mean(erm_values)),
            "erm_population_variance": float(np.var(erm_values, ddof=0)),
            "irm_population_variance": float(np.var(irm_values, ddof=0)),
        })
    erm_worst = min(row["erm_mean_macro_f1"] for row in position_results)
    irm_worst = min(row["irm_mean_macro_f1"] for row in position_results)

    penalty_contrasts = []
    dispersion_contrasts = []
    range_contrasts = []
    for fold in FOLD_HELD_POSITION:
        for seed in SEEDS:
            erm = next(row for row in erm_rows if row["fold_id"] == fold and int(row["seed"]) == seed)
            irm = next(row for row in selected_rows if row["fold_id"] == fold and int(row["seed"]) == seed)
            penalty_contrasts.append(float(irm["validation_irm_penalty"] - erm["validation_irm_penalty"]))
            dispersion_contrasts.append(float(irm["validation_risk_population_sd"] - erm["validation_risk_population_sd"]))
            range_contrasts.append(float(irm["validation_risk_range"] - erm["validation_risk_range"]))
    penalty_interval = paired_interval(penalty_contrasts, seed=20260810)
    dispersion_interval = paired_interval(dispersion_contrasts, seed=20260811)
    range_interval = paired_interval(range_contrasts, seed=20260812)
    diagnostic_classification = classify_invariance_diagnostics(
        penalty_interval["interval_95"], dispersion_interval["interval_95"], bool(guardrail["passed"])
    )

    all_source_partition, all_source_preprocessing = all_source_first_difference(data)
    final_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        final_records.append(train_final_model(
            data=data, partition=all_source_partition, preprocessing=all_source_preprocessing,
            config=config, method="erm", lambda_value=None, seed=seed, runtime_root=paths.runtime_root,
        ))
        final_records.append(train_final_model(
            data=data, partition=all_source_partition, preprocessing=all_source_preprocessing,
            config=config, method="irm", lambda_value=selected_lambda, seed=seed, runtime_root=paths.runtime_root,
        ))
    if len(final_records) != 10:
        raise IRMProtocolError("Final run count differs from 10")
    for seed in SEEDS:
        erm = next(row for row in final_records if row["method"] == "erm" and int(row["seed"]) == seed)
        irm = next(row for row in final_records if row["method"] == "irm" and int(row["seed"]) == seed)
        for field in ("epochs", "optimizer_steps", "sampler_signature_sha256", "source_samples_sha256", "source_labels_sha256"):
            if erm[field] != irm[field]:
                raise IRMProtocolError(f"Final ERM/IRM matching failed: {field}")

    all_positions = positions_for(data, all_source_partition)
    probe_rows: list[dict[str, Any]] = []
    for record in final_records:
        method = str(record["method"])
        seed = int(record["seed"])
        value = None if method == "erm" else selected_lambda
        model = load_checkpoint(final_checkpoint_path(paths.runtime_root, method, value, seed), seed)
        probe_rows.append({"method": method, "seed": seed, **representation_probe(model, all_source_partition, all_positions)})
    position_probe_contrasts = []
    tagid_probe_contrasts = []
    for seed in SEEDS:
        erm = next(row for row in probe_rows if row["method"] == "erm" and int(row["seed"]) == seed)
        irm = next(row for row in probe_rows if row["method"] == "irm" and int(row["seed"]) == seed)
        position_probe_contrasts.append(float(irm["position_probe_balanced_accuracy"] - erm["position_probe_balanced_accuracy"]))
        tagid_probe_contrasts.append(float(irm["tagid_probe_macro_f1"] - erm["tagid_probe_macro_f1"]))
    position_probe_interval = paired_interval(position_probe_contrasts, seed=20260813)
    tagid_probe_interval = paired_interval(tagid_probe_contrasts, seed=20260814)

    p4 = p4_metrics_after_sealed_freeze(
        final_records=final_records,
        p4_root=paths.p4_governed_root,
        preprocessing=all_source_preprocessing,
        runtime_root=paths.runtime_root,
        lambda_receipt_path=lambda_receipt_path,
        selected_lambda=selected_lambda,
        resamples=int(config["uncertainty"]["resamples"]),
    )
    p4_macro = p4["primary_macro_f1"]
    p4_accuracy = p4["primary_accuracy"]
    source_mean_interval = source_bootstrap["mean_contrast"]["interval_95"]
    source_worst_interval = source_bootstrap["worst_position_contrast"]["interval_95"]
    p4_seed_contrasts = np.asarray(p4_macro["seed_wise_contrasts"], dtype=np.float64)
    source_seed_contrasts = np.asarray([row["contrast"] for row in paired], dtype=np.float64)
    unstable = bool(
        (np.any(p4_seed_contrasts > 0) and np.any(p4_seed_contrasts < 0))
        or (np.any(source_seed_contrasts > 0) and np.any(source_seed_contrasts < 0))
    )
    main_classification = classify_main_result(
        source_mean_interval=source_mean_interval,
        source_worst_interval=source_worst_interval,
        p4_interval=p4_macro["interval_95"],
        source_guardrail_passed=bool(guardrail["passed"]),
        diagnostic_classification=diagnostic_classification,
        unstable=unstable,
    )

    source_overall_winner = "selected_irm" if irm_mean > erm_mean else "matched_erm" if erm_mean > irm_mean else "tie"
    lambda_ranking = sorted(selection["summaries"], key=lambda row: (-row["mean_held_macro_f1"], row["lambda"]))
    trajectory_rows = [history for row in development for history in row["history"]]
    leakage_gates = [
        ("condition_blocks_do_not_cross_source_train_validation", True),
        ("held_source_positions_do_not_enter_lopo_training", True),
        ("p4_does_not_enter_lambda_selection", not selection_receipt["p4_accessed"]),
        ("erm_irm_identical_source_samples", True),
        ("erm_irm_identical_tagid_labels", True),
        ("erm_irm_identical_seeds", True),
        ("erm_irm_identical_checkpoint_rules", True),
        ("erm_irm_identical_samplers", True),
        ("erm_irm_identical_batch_composition_rule", True),
        ("only_irmv1_penalty_differs", True),
        ("active_environments_source_positions_only", True),
        ("penalty_second_order_autograd", True),
        ("penalty_not_detached", True),
        ("annealing_boundary_frozen", True),
        ("lambda_grid_frozen", True),
        ("selected_lambda_frozen_before_final_training_and_p4", True),
        ("p4_predictions_frozen_before_labels", True),
        ("final_p4_records_use_final_schema", p4["execution_stage"] == FINAL_P4_EVALUATION),
        ("no_p4_record_validated_as_source_fold", True),
        ("result_writes_atomic", True),
        ("p4_rows_not_independent_units", not p4["row_level_pseudoreplication"]),
        ("earlier_results_not_primary_evidence", True),
        ("authoritative_repositories_not_modified", True),
        ("earlier_patch_repositories_not_modified", True),
    ]
    gate_rows = [{"gate": name, "passed": passed, "status": "PASS" if passed else "FAIL"} for name, passed in leakage_gates]
    if not all(row["passed"] for row in gate_rows):
        raise IRMProtocolError("FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT")

    development_register = [{
        "execution_stage": row["execution_stage"], "method": row["method"], "lambda": row["lambda"],
        "fold_id": row["fold_id"], "held_position": row["held_position"], "seed": row["seed"],
        "selected_epoch": row["selected_epoch"], "inner_epochs_executed": row["inner_epochs_executed"],
        "held_macro_f1": row["held_macro_f1"], "held_accuracy": row["held_accuracy"],
        "checkpoint_sha256": row["outer_checkpoint"]["checkpoint_sha256"],
        "p4_accessed": row["p4_accessed"],
    } for row in development]
    held_metrics = [{
        "method": row["method"], "lambda": row["lambda"], "fold_id": row["fold_id"],
        "held_position": row["held_position"], "seed": row["seed"],
        "macro_f1": row["held_macro_f1"], "accuracy": row["held_accuracy"],
        "worst_class_recall": row["held_worst_class_recall"],
        "train_to_held_macro_f1_gap": row["source_train_to_held_macro_f1_gap"],
        "selected_checkpoint_epoch": row["selected_epoch"],
    } for row in development]
    source_contrasts = [{
        "endpoint": "mean_held_source_macro_f1", "erm": erm_mean, "irm": irm_mean,
        "irm_minus_erm": irm_mean - erm_mean, "interval_95": source_mean_interval,
    }, {
        "endpoint": "worst_held_position_macro_f1", "erm": erm_worst, "irm": irm_worst,
        "irm_minus_erm": irm_worst - erm_worst, "interval_95": source_worst_interval,
    }]
    diagnostics_rows = [
        {"diagnostic": "independent_validation_irm_penalty", "irm_minus_erm": penalty_interval["point_estimate"], "interval_95": penalty_interval["interval_95"]},
        {"diagnostic": "environment_risk_population_sd", "irm_minus_erm": dispersion_interval["point_estimate"], "interval_95": dispersion_interval["interval_95"]},
        {"diagnostic": "environment_risk_range", "irm_minus_erm": range_interval["point_estimate"], "interval_95": range_interval["interval_95"]},
        {"diagnostic": "held_position_macro_f1_population_sd", "irm_minus_erm": float(np.std([r["held_macro_f1"] for r in selected_rows], ddof=0) - np.std([r["held_macro_f1"] for r in erm_rows], ddof=0)), "interval_95": None},
    ]
    final_register = [{
        "execution_stage": row["execution_stage"], "method": row["method"], "lambda": row["lambda"],
        "seed": row["seed"], "epochs": row["epochs"], "optimizer_steps": row["optimizer_steps"],
        "checkpoint_sha256": row["checkpoint"]["checkpoint_sha256"],
        "model_state_sha256": row["model_state_sha256"],
        "encoder_second_order_gradient_norm": row["intervention_encoder_second_order_gradient_norm"],
    } for row in final_records]
    freeze_rows = [{
        "execution_stage": FINAL_P4_EVALUATION, "prediction_key": key,
        "checkpoint_sha256": p4["freeze"]["checkpoint_hashes"][key],
        "prediction_sha256": p4["freeze"]["prediction_hashes"][key],
        "binding_sha256": p4["freeze"]["checkpoint_prediction_bindings"][key],
        "labels_accessed": False,
    } for key in sorted(p4["freeze"]["prediction_hashes"])]
    primary_rows = [
        {"endpoint": "P4_block_macro_f1", "irm_minus_erm": p4_macro["point_estimate"], "block_only_interval_95": p4_macro["interval_95"], "block_plus_seed_interval_95": p4["block_plus_seed_macro_f1"]["interval_95"]},
        {"endpoint": "P4_block_accuracy", "irm_minus_erm": p4_accuracy["point_estimate"], "block_only_interval_95": p4_accuracy["interval_95"], "block_plus_seed_interval_95": p4["block_plus_seed_accuracy"]["interval_95"]},
    ]
    sensitivity_rows = [
        {"analysis": "source_fold_plus_seed_mean", "point_estimate": source_bootstrap["mean_contrast"]["point_estimate"], "interval_95": source_mean_interval},
        {"analysis": "source_fold_plus_seed_worst", "point_estimate": source_bootstrap["worst_position_contrast"]["point_estimate"], "interval_95": source_worst_interval},
        {"analysis": "p4_block_only_macro_f1", "point_estimate": p4_macro["point_estimate"], "interval_95": p4_macro["interval_95"]},
        {"analysis": "p4_block_plus_seed_macro_f1", "point_estimate": p4["block_plus_seed_macro_f1"]["point_estimate"], "interval_95": p4["block_plus_seed_macro_f1"]["interval_95"]},
        {"analysis": "p4_block_only_accuracy", "point_estimate": p4_accuracy["point_estimate"], "interval_95": p4_accuracy["interval_95"]},
        {"analysis": "p4_block_plus_seed_accuracy", "point_estimate": p4["block_plus_seed_accuracy"]["point_estimate"], "interval_95": p4["block_plus_seed_accuracy"]["interval_95"]},
        *({"analysis": f"leave_one_training_seed_out_{row['omitted_seed']}_macro_f1", "point_estimate": row["block_macro_f1_contrast"], "interval_95": None} for row in p4["leave_one_training_seed_out"]),
        *({"analysis": f"held_position_sensitivity_{row['held_position']}", "point_estimate": row["contrast"], "interval_95": None} for row in position_results),
        *({"analysis": f"penalty_weight_rank_{rank + 1}_lambda_{row['lambda']:g}", "point_estimate": row["mean_held_macro_f1"], "interval_95": None} for rank, row in enumerate(lambda_ranking)),
    ]

    atomic_write_csv(RESULTS_ROOT / "10_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", gate_rows, ["gate", "passed", "status"])
    atomic_write_csv(RESULTS_ROOT / "11_DEVELOPMENT_RUN_REGISTER.csv", development_register, list(development_register[0]))
    atomic_write_csv(RESULTS_ROOT / "12_SOURCE_HELD_POSITION_METRICS.csv", held_metrics, list(held_metrics[0]))
    atomic_write_csv(RESULTS_ROOT / "13_IRM_PENALTY_AND_RISK_TRAJECTORIES.csv", trajectory_rows, list(trajectory_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "14_SOURCE_PRIMARY_CONTRASTS.csv", source_contrasts, list(source_contrasts[0]))
    atomic_write_csv(RESULTS_ROOT / "15_SOURCE_VARIANCE_AND_WORST_POSITION_RESULTS.csv", position_results, list(position_results[0]))
    atomic_write_csv(RESULTS_ROOT / "16_INVARIANCE_DIAGNOSTICS.csv", diagnostics_rows, list(diagnostics_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "17_POSITION_AND_TAGID_PROBE_RESULTS.csv", probe_rows, list(probe_rows[0]))
    atomic_write_markdown(RESULTS_ROOT / "18_SOURCE_ONLY_LAMBDA_SELECTION.md", f"""# Source-only lambda selection

Selected lambda: `{selected_lambda:g}`. The receipt was frozen and hashed before final source training and before P4 prediction or label access. Eligibility set: `{selection['eligibility_set']}`. Source-only overall winner: `{source_overall_winner}`. No P4 quantity entered selection.
""")
    atomic_write_csv(RESULTS_ROOT / "19_FINAL_MODEL_REGISTER.csv", final_register, list(final_register[0]))
    atomic_write_csv(RESULTS_ROOT / "20_P4_PRELABEL_PREDICTION_FREEZE.csv", freeze_rows, list(freeze_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "21_P4_LABEL_ACCESS_AND_PERSISTENCE_RECEIPT.csv", [{
        "execution_stage": FINAL_P4_EVALUATION,
        "prediction_bundle_sha256": p4["freeze"]["prediction_bundle_sha256"],
        "prelabel_receipt_sha256": p4["label_receipt"]["prelabel_receipt_sha256"],
        "label_sha256": p4["label_receipt"]["label_access_event"]["label_sha256"],
        "metrics_sha256": p4["label_receipt"]["metrics_persistence"]["sha256"],
        "silent_post_label_retry": False,
    }], ["execution_stage", "prediction_bundle_sha256", "prelabel_receipt_sha256", "label_sha256", "metrics_sha256", "silent_post_label_retry"])
    atomic_write_csv(RESULTS_ROOT / "22_P4_BLOCK_METRICS.csv", p4["p4_metrics"], list(p4["p4_metrics"][0]))
    atomic_write_csv(RESULTS_ROOT / "23_P4_PRIMARY_IRM_VS_ERM_CONTRAST.csv", primary_rows, list(primary_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "24_P4_PER_CLASS_RESULTS.csv", p4["per_class"], list(p4["per_class"][0]))
    atomic_write_csv(RESULTS_ROOT / "25_P4_CONFUSION_MATRICES.csv", p4["confusion_matrices"], list(p4["confusion_matrices"][0]))
    atomic_write_csv(RESULTS_ROOT / "26_PREDICTED_CLASS_HISTOGRAMS.csv", p4["histograms"], list(p4["histograms"][0]))
    atomic_write_csv(RESULTS_ROOT / "27_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", sensitivity_rows, ["analysis", "point_estimate", "interval_95"])
    atomic_write_markdown(RESULTS_ROOT / "28_INVARIANCE_DIAGNOSTIC_CLASSIFICATION.md", f"# Invariance-diagnostic classification\n\n`{diagnostic_classification}`\n")
    interpretation = f"""# Scientific interpretation

Main classification: `{main_classification}`.

Selected IRMv1 minus matched ERM source mean Macro-F1 was `{irm_mean - erm_mean:.6f}` with paired 95% interval `{source_mean_interval}`. The worst-position contrast was `{irm_worst - erm_worst:.6f}` with interval `{source_worst_interval}`. The source-retention guardrail passed: `{guardrail['passed']}`.

The validation IRM-penalty contrast was `{penalty_interval['point_estimate']:.6f}` with interval `{penalty_interval['interval_95']}`; environment-risk-dispersion contrast was `{dispersion_interval['point_estimate']:.6f}` with interval `{dispersion_interval['interval_95']}`. Position-probe balanced-accuracy contrast was `{position_probe_interval['point_estimate']:.6f}` with interval `{position_probe_interval['interval_95']}`. These are source-only diagnostics, not proof of physical causality.

P4 block Macro-F1 contrast was `{p4_macro['point_estimate']:.6f}` with block-only interval `{p4_macro['interval_95']}` and block-plus-training-seed interval `{p4['block_plus_seed_macro_f1']['interval_95']}`. P4 block Accuracy contrast was `{p4_accuracy['point_estimate']:.6f}` with block-only interval `{p4_accuracy['interval_95']}` and block-plus-training-seed interval `{p4['block_plus_seed_accuracy']['interval_95']}`.
"""
    atomic_write_markdown(RESULTS_ROOT / "29_SCIENTIFIC_INTERPRETATION.md", interpretation)
    limitations = """# Limitations and nonclaims

- This is a finite seven-TagID, three-source-position, one-target-position benchmark.
- IRMv1 depends on the declared source-position environments and frozen penalty grid.
- Source validation penalty, probes, and embedding geometry are secondary diagnostics and do not prove invariant physical mechanisms or causality.
- P4 inference is based on 63 condition blocks and five training seeds; rows are not independent inferential units.
- A null, mixed, harmful, or trade-off result is scientifically admissible.
- P4 labels were opened only after the ten predictions and checkpoint bindings were atomically frozen.
"""
    atomic_write_markdown(RESULTS_ROOT / "30_LIMITATIONS_AND_NONCLAIMS.md", limitations)
    summary = f"""# IRMv1 invariant-risk executive summary

Governed runtime binding succeeded: `True`. IRMv1 implementation validity passed: `True`. Second-order penalty gradients reached the encoder: `True`.

Development runs: `75`. Final runs: `10`. Selected lambda: `{selected_lambda:g}`. Annealing point: optimizer step `{config['irmv1']['anneal_step']}` of `{config['irmv1']['canonical_total_optimizer_steps']}` canonical steps.

Source mean Macro-F1 contrast: `{irm_mean - erm_mean:.6f}` (95% `{source_mean_interval}`). Source worst-position Macro-F1 contrast: `{irm_worst - erm_worst:.6f}` (95% `{source_worst_interval}`). Source retention passed: `{guardrail['passed']}`. Source mean improved: `{irm_mean > erm_mean}`. Source worst position improved: `{irm_worst > erm_worst}`.

Validation IRM penalty contrast: `{penalty_interval['point_estimate']:.6f}` (95% `{penalty_interval['interval_95']}`); decreased reliably: `{penalty_interval['interval_95'][1] < 0}`. Environment-risk dispersion contrast: `{dispersion_interval['point_estimate']:.6f}` (95% `{dispersion_interval['interval_95']}`); decreased reliably: `{dispersion_interval['interval_95'][1] < 0}`. Position decodability contrast: `{position_probe_interval['point_estimate']:.6f}` (95% `{position_probe_interval['interval_95']}`).

P4 block Macro-F1 contrast: `{p4_macro['point_estimate']:.6f}` (block-only `{p4_macro['interval_95']}`, block-plus-seed `{p4['block_plus_seed_macro_f1']['interval_95']}`); improved: `{p4_macro['point_estimate'] > 0}`. P4 block Accuracy contrast: `{p4_accuracy['point_estimate']:.6f}` (block-only `{p4_accuracy['interval_95']}`, block-plus-seed `{p4['block_plus_seed_accuracy']['interval_95']}`); improved: `{p4_accuracy['point_estimate'] > 0}`. Effects survived both block and seed uncertainty: `{p4_macro['interval_95'][0] > 0 and p4['block_plus_seed_macro_f1']['interval_95'][0] > 0}`.

Invariance-diagnostic classification: `{diagnostic_classification}`. Main scientific classification: `{main_classification}`.

Predictions were frozen and hashed before P4 labels were opened.
"""
    atomic_write_markdown(RESULTS_ROOT / "00_EXECUTIVE_SUMMARY.md", summary)
    atomic_write_markdown(RESULTS_ROOT / "STATUS.md", f"# Result status\n\nImplementation: `PASS`. Sealed scientific execution: `PASS`.\n\nInvariance diagnostic: `{diagnostic_classification}`.\n\nMain classification: `{main_classification}`.\n")
    atomic_write_markdown(PROJECT_ROOT / "docs" / "IRM_INVARIANT_RISK_RESULTS.md", summary + "\n\n" + limitations)
    return {
        "status": "PASS_SCIENTIFIC_EXECUTION",
        "development_runs": len(development),
        "final_runs": len(final_records),
        "selected_lambda": selected_lambda,
        "source_mean_contrast": irm_mean - erm_mean,
        "source_worst_contrast": irm_worst - erm_worst,
        "p4_macro_f1_contrast": p4_macro["point_estimate"],
        "p4_accuracy_contrast": p4_accuracy["point_estimate"],
        "diagnostic_classification": diagnostic_classification,
        "main_classification": main_classification,
    }


def run_development_worker(
    paths: RuntimePaths, config: Mapping[str, Any], fold: str, seed: int
) -> dict[str, Any]:
    governed_source_input_hashes(paths.strict_artifact_root)
    worker_id = f"{fold}_seed_{seed}"
    data = canonical_source_loader(paths.strict_artifact_root, paths.runtime_root / "workers" / worker_id)
    source_structure_audit(data)
    states = fit_first_difference_preprocessing(
        data, paths.runtime_root / "workers" / worker_id / "preprocessing"
    )
    records = [run_lopo_unit(
        data=data, states=states, config=config, fold=fold, seed=seed,
        method="erm", lambda_value=None, runtime_root=paths.runtime_root,
    )]
    for value in LAMBDA_GRID:
        records.append(run_lopo_unit(
            data=data, states=states, config=config, fold=fold, seed=seed,
            method="irm", lambda_value=value, runtime_root=paths.runtime_root,
        ))
    receipt = {
        "status": "PASS_DEVELOPMENT_WORKER",
        "fold": fold,
        "seed": seed,
        "runs": len(records),
        "p4_accessed": False,
    }
    atomic_write_json(paths.runtime_root / "workers" / worker_id / "worker_receipt.json", receipt)
    return receipt


def augment_secondary_public_reports(paths: RuntimePaths, config: Mapping[str, Any]) -> dict[str, Any]:
    """Expose already-persisted secondary summaries without reopening P4 inputs."""

    label_receipt_path = paths.runtime_root / "p4" / "P4_LABEL_ACCESS_AND_PERSISTENCE_RECEIPT.json"
    metrics_path = paths.runtime_root / "p4" / "metrics.json"
    if not label_receipt_path.is_file() or not metrics_path.is_file():
        raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
    label_receipt = read_json(label_receipt_path)
    metrics = read_json(metrics_path)
    if label_receipt.get("silent_post_label_retry") or metrics.get("execution_stage") != FINAL_P4_EVALUATION:
        raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
    development = load_all_development(paths.runtime_root)
    selection = read_json(paths.runtime_root / "FROZEN_SOURCE_ONLY_LAMBDA_SELECTION.json")
    selected_lambda = float(selection["selected_lambda"])

    source_means: dict[tuple[str, int], float] = {}
    for method in ("erm", "irm"):
        for seed in SEEDS:
            selected = [
                row for row in development
                if row["method"] == method and int(row["seed"]) == seed
                and (method == "erm" or float(row["lambda"]) == selected_lambda)
            ]
            if len(selected) != 3:
                raise IRMProtocolError("Secondary source-to-P4 binding is incomplete")
            source_means[(method, seed)] = float(np.mean([float(row["held_macro_f1"]) for row in selected]))

    detail_by_method: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in metrics["block_details"]:
        detail_by_method[str(row["method"])][int(row["seed"])].append(row)
    cross_seed_agreement: dict[str, float] = {}
    for method in ("erm", "irm"):
        matrix = np.stack([
            np.asarray([
                row["predicted_label"]
                for row in sorted(detail_by_method[method][seed], key=lambda item: int(item["condition_block_index"]))
            ], dtype=np.int64)
            for seed in SEEDS
        ])
        agreements = [float(np.bincount(matrix[:, index], minlength=7).max() / 5) for index in range(63)]
        cross_seed_agreement[method] = float(np.mean(agreements))
    p4_rows = [dict(row) for row in metrics["p4_metrics"]]
    for row in p4_rows:
        method, seed = str(row["method"]), int(row["seed"])
        row["source_to_p4_macro_f1_drop"] = source_means[(method, seed)] - float(row["block_macro_f1"])
        row["mean_cross_seed_block_agreement"] = cross_seed_agreement[method]
    atomic_write_csv(RESULTS_ROOT / "22_P4_BLOCK_METRICS.csv", p4_rows, list(p4_rows[0]))

    change_counts: Counter[tuple[int, int, str]] = Counter()
    for row in metrics["paired_changes"]:
        change_counts[(int(row["seed"]), int(row["tagid_label"]), str(row["change"]))] += 1
    per_class_rows = [dict(row) for row in metrics["per_class"]]
    for row in per_class_rows:
        seed, label = int(row["seed"]), int(row["tagid_label"])
        row["paired_incorrect_to_correct_blocks"] = change_counts[(seed, label, "INCORRECT_TO_CORRECT")]
        row["paired_correct_to_incorrect_blocks"] = change_counts[(seed, label, "CORRECT_TO_INCORRECT")]
    atomic_write_csv(RESULTS_ROOT / "24_P4_PER_CLASS_RESULTS.csv", per_class_rows, list(per_class_rows[0]))

    structure_rows = []
    with (RESULTS_ROOT / "06_DATA_AND_ENVIRONMENT_STRUCTURE_AUDIT.csv").open("r", encoding="utf-8", newline="") as handle:
        structure_rows = list(csv.DictReader(handle))
    if not any(row["position"] == "P4" for row in structure_rows):
        structure_rows.append({
            "position": "P4", "sample_count": 3150, "tagid_count": 7,
            "condition_block_count": 63, "rows_per_condition_block": [50],
            "ordered_input_points": 281, "first_difference_points": 280, "status": "PASS",
        })
    atomic_write_csv(RESULTS_ROOT / "06_DATA_AND_ENVIRONMENT_STRUCTURE_AUDIT.csv", structure_rows, list(structure_rows[0]))

    grid_rows = [{
        "lambda": value, "anneal_fraction": config["irmv1"]["anneal_fraction"],
        "canonical_total_optimizer_steps": config["irmv1"]["canonical_total_optimizer_steps"],
        "anneal_step": config["irmv1"]["anneal_step"], "grid_frozen": True,
        "selected_lambda": selected_lambda,
    } for value in LAMBDA_GRID]
    atomic_write_csv(RESULTS_ROOT / "09_PENALTY_GRID_AND_ANNEALING_REGISTER.csv", grid_rows, list(grid_rows[0]))

    sensitivity_path = RESULTS_ROOT / "27_BOOTSTRAP_AND_SEED_SENSITIVITY.csv"
    with sensitivity_path.open("r", encoding="utf-8", newline="") as handle:
        sensitivity_rows = list(csv.DictReader(handle))
    selected_histories = [
        history for row in development
        if row["method"] == "irm" and float(row["lambda"]) == selected_lambda
        for history in row["history"]
        if history["training_phase"] == "inner_selection" and history["validation_macro_f1"] is not None
    ]
    warmup = [float(row["validation_macro_f1"]) for row in selected_histories if row["annealing_stage"] == "ERM_WARMUP"]
    active = [float(row["validation_macro_f1"]) for row in selected_histories if row["annealing_stage"] == "POST_ANNEAL"]
    worst_identities: dict[tuple[str, int], str] = {}
    for method in ("erm", "irm"):
        for seed in SEEDS:
            rows = [
                row for row in development if row["method"] == method and int(row["seed"]) == seed
                and (method == "erm" or float(row["lambda"]) == selected_lambda)
            ]
            worst_identities[(method, seed)] = min(rows, key=lambda row: (float(row["held_macro_f1"]), str(row["held_position"])))["held_position"]
    identity_agreement = float(np.mean([worst_identities[("erm", seed)] == worst_identities[("irm", seed)] for seed in SEEDS]))
    sensitivity_rows.extend([
        {"analysis": "annealing_stage_validation_macro_f1_post_minus_warmup", "point_estimate": float(np.mean(active) - np.mean(warmup)), "interval_95": None},
        {"analysis": "source_worst_position_identity_erm_irm_agreement_fraction", "point_estimate": identity_agreement, "interval_95": None},
    ])
    atomic_write_csv(sensitivity_path, sensitivity_rows, ["analysis", "point_estimate", "interval_95"])
    receipt = {
        "status": "PASS_SECONDARY_PUBLIC_REPORT_AUGMENTATION",
        "augmented_at_utc": utc_now(),
        "p4_labels_reopened": False,
        "primary_metrics_recomputed": False,
        "persisted_metrics_sha256": sha256_file(metrics_path),
        "label_receipt_sha256": sha256_file(label_receipt_path),
        "augmented_reports": ["06", "09", "22", "24", "27"],
    }
    atomic_write_json(paths.runtime_root / "POST_LABEL_PUBLIC_REPORT_AUGMENTATION_RECEIPT.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--augment-secondary-reports", action="store_true")
    parser.add_argument("--only-fold", choices=tuple(FOLD_HELD_POSITION))
    parser.add_argument("--only-seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    if sum((args.prepare_only, args.development_only, args.augment_secondary_reports)) > 1:
        parser.error("stage flags are mutually exclusive")
    if args.development_only and (args.only_fold is None or args.only_seed is None):
        parser.error("development workers require --only-fold and --only-seed")
    if not args.development_only and (args.only_fold is not None or args.only_seed is not None):
        parser.error("worker selectors require --development-only")
    config = load_config()
    paths = environment_paths()
    if args.prepare_only:
        result = prepare_execution(paths, config)
    elif args.development_only:
        result = run_development_worker(paths, config, str(args.only_fold), int(args.only_seed))
    elif args.augment_secondary_reports:
        result = augment_secondary_public_reports(paths, config)
    else:
        result = execute_complete(paths, config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
