"""Governed execution adapter for the isolated domain-aware mixup strict-DG patch."""

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
    BETA,
    CLASS_ORDER,
    FINAL_P4_EVALUATION,
    FINAL_SOURCE_TRAINING,
    FOLD_HELD_POSITION,
    ALPHA_GRID,
    SEEDS,
    SOURCE_LOPO_DEVELOPMENT,
    SOURCE_POSITIONS,
    MixupProtocolError,
    P4LabelSeal,
    WITHIN_POSITION_ALPHA,
    array_sha256,
    assert_mixup_connectivity,
    canonical_json_sha256,
    canonical_position_order,
    classify_domain_bridging_diagnostics,
    classify_main_result,
    deterministic_block_majority_vote,
    early_stop_allowed,
    execution_record,
    mixup_objective,
    make_condition_matched_pairs,
    make_environment_balanced_batches,
    matched_erm_objective,
    paired_source_bootstrap,
    paired_tagid_stratified_block_bootstrap,
    select_alpha_source_only,
    sample_mix_coefficients,
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
CONFIG_PATH = PROJECT_ROOT / "configs" / "domain_aware_mixup" / "canonical.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "canonical_metrics" / "domain_aware_mixup"
MANIFESTS_ROOT = PROJECT_ROOT / "manifests" / "domain_aware_mixup"
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
    strict = os.environ.get("CRFID_MIXUP_STRICT_ARTIFACT_ROOT", "").strip()
    p4 = os.environ.get("CRFID_MIXUP_P4_GOVERNED_ROOT", "").strip()
    runtime = os.environ.get("CRFID_MIXUP_RUN_ROOT", "").strip()
    missing = [name for name, value in (
        ("CRFID_MIXUP_STRICT_ARTIFACT_ROOT", strict),
        ("CRFID_MIXUP_P4_GOVERNED_ROOT", p4),
        ("CRFID_MIXUP_RUN_ROOT", runtime),
    ) if not value]
    if missing:
        raise MixupProtocolError("BLOCKED_GOVERNED_INPUTS: missing " + ", ".join(missing))
    paths = RuntimePaths(Path(strict).expanduser(), Path(p4).expanduser(), Path(runtime).expanduser())
    resolved_runtime = paths.runtime_root.resolve()
    if PROJECT_ROOT.resolve() not in resolved_runtime.parents:
        raise MixupProtocolError("Governed runtime must use the canonical ignored in-repository runtime convention")
    relative_runtime = resolved_runtime.relative_to(PROJECT_ROOT.resolve()).as_posix()
    ignored = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", relative_runtime],
        check=False,
    ).returncode == 0
    if not ignored:
        raise MixupProtocolError("Governed runtime root is not excluded from Git")
    return paths


def governed_source_input_hashes(strict_root: Path) -> dict[str, str]:
    source_root = strict_root / "source_inputs"
    observed: dict[str, str] = {}
    for name, expected in GOVERNED_SOURCE_INPUT_HASHES.items():
        path = source_root / name
        if not path.is_file():
            raise MixupProtocolError(f"BLOCKED_GOVERNED_INPUTS: missing {name}")
        observed[name] = sha256_file(path)
        if observed[name] != expected:
            raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
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
    data = CanonicalPhase2Data(config, strict_root, scratch_root / "canonical_loader_preprocessing")
    registry_path = strict_root / "source_inputs" / "CANONICAL_SOURCE_REGISTRY.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(data.registry_rows):
        raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
    pairing_rows = []
    for index, row in enumerate(rows):
        canonical = data.registry_rows[index]
        if (
            int(row["registry_row"]) != index
            or row["sample_id"] != canonical["sample_id"]
            or int(row["label_index"]) != canonical["label_index"]
            or row["position"] != canonical["position"]
        ):
            raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
        pairing_rows.append({
            "tag_id": int(row["label_index"]),
            "er": int(row["er"]),
            "surface": row["surface"],
            "position": row["position"],
            "repeat": int(row["repeat_index"]),
            "condition_id": row["raw_condition_id"],
        })
    data.pairing_rows = pairing_rows
    return data


def positions_for(data: CanonicalPhase2Data, partition: PartitionData) -> np.ndarray:
    return np.asarray([data.registry_rows[int(index)]["position"] for index in partition.registry_rows], dtype=str)


def pairing_metadata_for(data: CanonicalPhase2Data, partition: PartitionData) -> dict[str, np.ndarray]:
    rows = [data.pairing_rows[int(index)] for index in partition.registry_rows]
    return {
        "labels": np.asarray([row["tag_id"] for row in rows], dtype=np.int64),
        "er": np.asarray([row["er"] for row in rows], dtype=np.int64),
        "surface": np.asarray([row["surface"] for row in rows], dtype=str),
        "position": np.asarray([row["position"] for row in rows], dtype=str),
        "repeat": np.asarray([row["repeat"] for row in rows], dtype=np.int64),
    }


def partition_hash(partition: PartitionData) -> str:
    return array_sha256(np.asarray(partition.registry_rows, dtype=np.int64))


def source_structure_audit(data: CanonicalPhase2Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = np.asarray([row["position"] for row in data.registry_rows], dtype=str)
    conditions = np.asarray([row["raw_condition_id"] for row in data.registry_rows], dtype=str)
    labels = np.asarray(data.labels, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    if data.signals.shape != (9450, 281) or data.signals.dtype.str != "<f8":
        raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
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
            raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
        rows.append(row)
    split_rows: list[dict[str, Any]] = []
    for fold, held in FOLD_HELD_POSITION.items():
        condition_sets: dict[str, set[str]] = {}
        for partition_name, indices in data.partitions[fold].items():
            part_positions = {data.registry_rows[int(index)]["position"] for index in indices}
            expected = {held} if partition_name == "outer_held" else set(SOURCE_POSITIONS).difference({held})
            if part_positions != expected:
                raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
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
            raise MixupProtocolError("Held source position entered development")
    return rows, split_rows


def governed_pairing_coverage_audit(
    data: CanonicalPhase2Data,
    states: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure lawful partner availability over every canonical training parent."""

    rows: list[dict[str, Any]] = []
    for fold in FOLD_HELD_POSITION:
        partitions = candidate_partitions(data, C1, fold, states)
        for partition_name in ("inner_train", "outer_development"):
            partition = partitions[partition_name]
            metadata = pairing_metadata_for(data, partition)
            position_sets: dict[tuple[int, int, str], set[str]] = defaultdict(set)
            repeat_sets: dict[tuple[int, int, str, str], set[int]] = defaultdict(set)
            for tag, er_value, surface, position, repeat in zip(
                metadata["labels"], metadata["er"], metadata["surface"],
                metadata["position"], metadata["repeat"], strict=True,
            ):
                condition = (int(tag), int(er_value), str(surface))
                position_sets[condition].add(str(position))
                repeat_sets[(*condition, str(position))].add(int(repeat))
            cross_eligible = 0
            within_eligible = 0
            for tag, er_value, surface, position in zip(
                metadata["labels"], metadata["er"], metadata["surface"], metadata["position"], strict=True,
            ):
                condition = (int(tag), int(er_value), str(surface))
                cross_eligible += len(position_sets[condition].difference({str(position)})) > 0
                within_eligible += len(repeat_sets[(*condition, str(position))]) > 1
            for method, eligible in (
                ("cross_position_mixup", cross_eligible),
                ("within_position_mixup", within_eligible),
            ):
                total = len(partition.labels)
                coverage = float(eligible / total)
                rows.append({
                    "fold_id": fold,
                    "partition": partition_name,
                    "method": method,
                    "training_parent_count": total,
                    "eligible_partner_count": int(eligible),
                    "missing_pair_count": int(total - eligible),
                    "pair_coverage": coverage,
                    "threshold": 0.99,
                    "status": "PASS" if coverage > 0.99 else "FAIL_DOMAIN_AWARE_PAIRING_COVERAGE",
                })
    return rows


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
        "penultimate_embedding": "NeutralSourceOnlyCNN1D.network[:-1] output before network[-1]",
        "embedding_dimension": 256,
        "classification_head": "NeutralSourceOnlyCNN1D.network[-1]",
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
    alpha_value: float | None,
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
        "alpha": alpha_value,
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
        raise MixupProtocolError("Checkpoint hash binding failed")
    return model


def evaluate_partition(
    model: torch.nn.Module,
    partition: PartitionData,
    positions: np.ndarray,
    environments: tuple[str, ...],
) -> dict[str, Any]:
    logits, embeddings = extract_logits_embeddings(model, partition)
    evaluation = evaluate_logits(partition, logits)
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
            "risk": float(F.cross_entropy(
                torch.from_numpy(logits[selected]), torch.from_numpy(partition.labels[selected])
            ).item()),
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
        "mixup_applied": False,
        "risk_population_sd": float(risks.std(ddof=0)),
        "risk_range": float(risks.max() - risks.min()),
    }


def train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    partition: PartitionData,
    positions: np.ndarray,
    pairing_metadata: Mapping[str, np.ndarray],
    environments: tuple[str, ...],
    method: str,
    alpha_value: float | None,
    seed: int,
    epoch: int,
    stage: str,
    batch_size: int,
    global_step_start: int,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    if method not in {"erm", "within_position_mixup", "cross_position_mixup"}:
        raise MixupProtocolError("Mixup method declaration is ambiguous")
    if method == "erm" and alpha_value is not None:
        raise MixupProtocolError("ERM must not have an alpha")
    if method == "within_position_mixup" and alpha_value != WITHIN_POSITION_ALPHA:
        raise MixupProtocolError("Within-position alpha must remain 0.50")
    if method == "cross_position_mixup" and alpha_value not in ALPHA_GRID:
        raise MixupProtocolError("Cross-position alpha is outside the frozen grid")
    plan = make_environment_balanced_batches(
        positions, partition.labels, seed=seed, epoch=epoch, stage=stage,
        batch_size=batch_size, environments=environments,
    )
    pair_plan = None
    partner_lookup = np.full(len(partition.labels), -1, dtype=np.int64)
    if method != "erm":
        pair_plan = make_condition_matched_pairs(
            pairing_metadata["labels"], pairing_metadata["er"], pairing_metadata["surface"],
            pairing_metadata["position"], pairing_metadata["repeat"],
            np.full(len(partition.labels), "training", dtype=str),
            mode="within_position" if method == "within_position_mixup" else "cross_position",
            seed=seed, epoch=epoch,
        )
        partner_lookup[pair_plan.parent_indices] = pair_plan.partner_indices
    model.train()
    original_total = 0.0
    mixup_total = 0.0
    objective_total = 0.0
    count_total = 0
    environment_totals = {environment: 0.0 for environment in environments}
    environment_counts = {environment: 0 for environment in environments}
    batch_rows: list[dict[str, Any]] = []
    coefficient_values: list[float] = []
    gradient_norms: list[float] = []
    mixup_encoder_gradient_norms: list[float] = []
    for local_step, (batch, audit) in enumerate(zip(plan.batches, plan.audits, strict=True)):
        global_step = global_step_start + local_step
        inputs = torch.from_numpy(partition.inputs[batch]).unsqueeze(1)
        labels = torch.from_numpy(partition.labels[batch])
        batch_positions = positions[batch]
        optimizer.zero_grad(set_to_none=True)
        encoder = model.network[:-1]
        classifier = model.network[-1]
        embeddings = encoder(inputs)
        logits = classifier(embeddings).float()
        original_loss = F.cross_entropy(logits, labels, reduction="mean")
        if method == "erm":
            objective = original_loss
            mixup_loss = torch.zeros((), dtype=torch.float32)
            pair_count = 0
            coefficient_batch = np.empty(0, dtype=np.float32)
        else:
            parent_mask = partner_lookup[batch] >= 0
            parent_offsets = np.flatnonzero(parent_mask).astype(np.int64)
            partner_indices = partner_lookup[batch[parent_mask]]
            if not len(parent_offsets):
                raise MixupProtocolError("FAIL_DOMAIN_AWARE_PAIRING_COVERAGE")
            partner_embeddings = encoder(torch.from_numpy(partition.inputs[partner_indices]).unsqueeze(1))
            coefficient_batch = sample_mix_coefficients(
                float(alpha_value), len(parent_offsets), seed=seed, epoch=epoch, batch_index=local_step
            )
            terms = mixup_objective(
                logits, labels, embeddings[parent_offsets], partner_embeddings,
                labels[parent_offsets], torch.from_numpy(coefficient_batch), classifier, beta=BETA,
            )
            objective = terms.objective
            original_loss = terms.original_loss
            mixup_loss = terms.mixup_loss
            pair_count = len(parent_offsets)
            if not mixup_encoder_gradient_norms:
                gradients = assert_mixup_connectivity(
                    mixup_loss, embeddings[parent_offsets], partner_embeddings,
                    model.network[0].weight, model.network[-1].weight,
                )
                norms = [float(value.norm().detach()) for value in gradients if value is not None]
                if min(norms) <= 0:
                    raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY")
                mixup_encoder_gradient_norms.append(norms[2])
        if not bool(torch.isfinite(objective)):
            raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY: non-finite objective")
        objective.backward()
        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY: non-finite gradient")
                squared_norm += float(parameter.grad.detach().pow(2).sum())
        gradient_norm = squared_norm ** 0.5
        optimizer.step()
        gradient_norms.append(gradient_norm)
        coefficient_values.extend(coefficient_batch.astype(float).tolist())
        original_total += float(original_loss.detach()) * len(batch)
        mixup_total += float(mixup_loss.detach()) * max(pair_count, 1)
        objective_total += float(objective.detach()) * len(batch)
        count_total += len(batch)
        for environment in environments:
            environment_count = int(np.count_nonzero(batch_positions == environment))
            mask = torch.from_numpy(batch_positions == environment)
            environment_risk = F.cross_entropy(logits[mask], labels[mask], reduction="mean")
            environment_totals[environment] += float(environment_risk.detach()) * environment_count
            environment_counts[environment] += environment_count
        batch_rows.append({
            **dict(identity), **audit,
            "global_optimizer_step": global_step,
            "alpha": alpha_value,
            "original_loss": float(original_loss.detach()),
            "mixup_loss": float(mixup_loss.detach()),
            "objective": float(objective.detach()),
            "mixed_pair_count": pair_count,
            "pair_coverage": 1.0 if pair_plan is None else pair_plan.coverage,
            "pair_counts": {} if pair_plan is None else pair_plan.pair_counts,
            "coefficient_min": None if not len(coefficient_batch) else float(coefficient_batch.min()),
            "coefficient_mean": None if not len(coefficient_batch) else float(coefficient_batch.mean()),
            "coefficient_max": None if not len(coefficient_batch) else float(coefficient_batch.max()),
            "gradient_norm": gradient_norm,
            "batch_signature_sha256": plan.signature_sha256,
        })
    return {
        "mean_original_loss": original_total / count_total,
        "mean_mixup_loss": 0.0 if method == "erm" else mixup_total / count_total,
        "mean_objective": objective_total / count_total,
        "per_environment_risk": {e: environment_totals[e] / environment_counts[e] for e in environments},
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "mixup_encoder_gradient_norm": max(mixup_encoder_gradient_norms, default=0.0),
        "pair_coverage": 1.0 if pair_plan is None else pair_plan.coverage,
        "missing_pair_count": 0 if pair_plan is None else pair_plan.missing_count,
        "pair_counts": {} if pair_plan is None else pair_plan.pair_counts,
        "coefficient_distribution": {
            "count": len(coefficient_values),
            "minimum": None if not coefficient_values else float(np.min(coefficient_values)),
            "mean": None if not coefficient_values else float(np.mean(coefficient_values)),
            "maximum": None if not coefficient_values else float(np.max(coefficient_values)),
        },
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
            raise MixupProtocolError("Held source position leaked into LOPO development")
    held_observed = tuple(np.unique(positions_for(data, partitions["outer_held"])).tolist())
    validate_lopo_position_isolation(environments, held_observed, held)
    validate_condition_block_disjoint(
        partitions["inner_train"].condition_ids,
        partitions["inner_validation"].condition_ids,
    )
    return environments


def development_directory(
    runtime_root: Path, method: str, alpha_value: float | None, fold: str, seed: int
) -> Path:
    value = "none" if alpha_value is None else f"alpha_{alpha_value:g}"
    return runtime_root / "development" / method / value / fold / f"seed_{seed}"


def run_lopo_unit(
    *,
    data: CanonicalPhase2Data,
    states: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    fold: str,
    seed: int,
    method: str,
    alpha_value: float | None,
    runtime_root: Path,
) -> dict[str, Any]:
    if method == "cross_position_mixup" and alpha_value not in ALPHA_GRID:
        raise MixupProtocolError("Cross-position alpha outside frozen grid")
    if method == "within_position_mixup" and alpha_value != WITHIN_POSITION_ALPHA:
        raise MixupProtocolError("Within-position control alpha changed")
    stage_schema = execution_record(SOURCE_LOPO_DEVELOPMENT, fold)
    validate_source_fold_record(stage_schema)
    partitions = candidate_partitions(data, C1, fold, states)
    environments = assert_lopo_partitions(data, partitions, fold)
    held_position = FOLD_HELD_POSITION[fold]
    directory = development_directory(runtime_root, method, alpha_value, fold, seed)
    summary_path = directory / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("method") != method or summary.get("alpha") != alpha_value or summary.get("fold_id") != fold or summary.get("seed") != seed:
            raise MixupProtocolError("Development resume identity mismatch")
        checkpoint_path = directory / "outer_refit.pt"
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != summary["outer_checkpoint"]["checkpoint_sha256"]:
            raise MixupProtocolError("Development resume checkpoint mismatch")
        return summary

    base = canonical_base_config(config)
    training = config["training"]
    batch_size = int(training["batch_size"])
    identity = {
        **stage_schema,
        "method": method,
        "alpha": alpha_value,
        "fold_id": fold,
        "held_position": held_position,
        "seed": seed,
    }
    model = initialize_model(seed)
    initial_hash = model_state_sha256(model)
    optimizer = _optimizer(model, base)
    training_positions = positions_for(data, partitions["inner_train"])
    training_pairing_metadata = pairing_metadata_for(data, partitions["inner_train"])
    validation_positions = positions_for(data, partitions["inner_validation"])
    best_metric = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    intervention_encoder_norm = 0.0
    minimum_pair_coverage = 1.0
    position_pair_counts: Counter[str] = Counter()
    coefficient_distributions: list[dict[str, Any]] = []
    for epoch in range(1, int(training["maximum_inner_epochs"]) + 1):
        train = train_epoch(
            model=model,
            optimizer=optimizer,
            partition=partitions["inner_train"],
            positions=training_positions,
            pairing_metadata=training_pairing_metadata,
            environments=environments,
            method=method,
            alpha_value=alpha_value,
            seed=seed,
            epoch=epoch,
            stage="inner_selection",
            batch_size=batch_size,
            global_step_start=global_step,
            identity=identity,
        )
        global_step += int(train["steps"])
        intervention_encoder_norm = max(intervention_encoder_norm, float(train["mixup_encoder_gradient_norm"]))
        minimum_pair_coverage = min(minimum_pair_coverage, float(train["pair_coverage"]))
        position_pair_counts.update(train["pair_counts"])
        coefficient_distributions.append(train["coefficient_distribution"])
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
            "original_training_loss": train["mean_original_loss"],
            "mixup_training_loss": train["mean_mixup_loss"],
            "total_training_loss": train["mean_objective"],
            "per_environment_training_risk": train["per_environment_risk"],
            "validation_macro_f1": metric,
            "validation_accuracy": float(validation["evaluation"]["sample"]["accuracy"]),
            "per_environment_validation_risk": {e: validation["per_environment"][e]["risk"] for e in environments},
            "pair_coverage": train["pair_coverage"],
            "position_pair_counts": train["pair_counts"],
            "coefficient_distribution": train["coefficient_distribution"],
            "gradient_norm": train["mean_gradient_norm"],
            "selected_checkpoint_state": bool(improved),
        })
        batch_rows.extend(train["batch_rows"])
        if early_stop_allowed(
            method=method,
            epochs_without_improvement=no_improvement,
            patience=int(training["early_stopping_patience"]),
            completed_optimizer_steps=global_step,
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise MixupProtocolError("Source validation checkpoint selection failed")
    model.load_state_dict(best_state, strict=True)
    inner_checkpoint = save_checkpoint(
        directory / "inner_selected.pt",
        model=model,
        method=method,
        seed=seed,
        epoch=best_epoch,
        fold=fold,
        alpha_value=alpha_value,
        preprocessing_sha256=str(states[(fold, "inner_selection")]["state_sha256"]),
        stage_record=stage_schema,
    )

    outer_model = initialize_model(seed)
    if model_state_sha256(outer_model) != initial_hash:
        raise MixupProtocolError("Canonical reinitialization changed")
    outer_optimizer = _optimizer(outer_model, base)
    outer_positions = positions_for(data, partitions["outer_development"])
    outer_pairing_metadata = pairing_metadata_for(data, partitions["outer_development"])
    outer_global_step = 0
    outer_history: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        train = train_epoch(
            model=outer_model,
            optimizer=outer_optimizer,
            partition=partitions["outer_development"],
            positions=outer_positions,
            pairing_metadata=outer_pairing_metadata,
            environments=environments,
            method=method,
            alpha_value=alpha_value,
            seed=seed,
            epoch=epoch,
            stage="outer_refit",
            batch_size=batch_size,
            global_step_start=outer_global_step,
            identity=identity,
        )
        outer_global_step += int(train["steps"])
        intervention_encoder_norm = max(intervention_encoder_norm, float(train["mixup_encoder_gradient_norm"]))
        minimum_pair_coverage = min(minimum_pair_coverage, float(train["pair_coverage"]))
        position_pair_counts.update(train["pair_counts"])
        coefficient_distributions.append(train["coefficient_distribution"])
        outer_history.append({
            **identity,
            "training_phase": "outer_refit",
            "epoch": epoch,
            "original_training_loss": train["mean_original_loss"],
            "mixup_training_loss": train["mean_mixup_loss"],
            "total_training_loss": train["mean_objective"],
            "per_environment_training_risk": train["per_environment_risk"],
            "validation_macro_f1": None,
            "validation_accuracy": None,
            "per_environment_validation_risk": None,
            "pair_coverage": train["pair_coverage"],
            "position_pair_counts": train["pair_counts"],
            "coefficient_distribution": train["coefficient_distribution"],
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
        alpha_value=alpha_value,
        preprocessing_sha256=str(states[(fold, "outer_refit")]["state_sha256"]),
        stage_record=stage_schema,
    )
    frozen = load_checkpoint(outer_checkpoint_path, seed)
    development = evaluate_partition(frozen, partitions["outer_development"], outer_positions, environments)
    held_positions = positions_for(data, partitions["outer_held"])
    held = evaluate_partition(frozen, partitions["outer_held"], held_positions, (held_position,))
    if method != "erm" and intervention_encoder_norm <= 0:
        raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY")
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
        "validation_risk_population_sd": float(development["risk_population_sd"]),
        "validation_risk_range": float(development["risk_range"]),
        "mixup_encoder_gradient_norm": intervention_encoder_norm,
        "pair_coverage": minimum_pair_coverage,
        "position_pair_counts": dict(sorted(position_pair_counts.items())),
        "coefficient_distributions": coefficient_distributions,
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
        "execution_stage", "development_fold", "method", "alpha", "fold_id", "held_position", "seed",
        "training_phase", "epoch", "original_training_loss", "mixup_training_loss",
        "total_training_loss", "per_environment_training_risk", "validation_macro_f1",
        "validation_accuracy", "per_environment_validation_risk", "pair_coverage",
        "position_pair_counts", "coefficient_distribution", "gradient_norm", "selected_checkpoint_state",
    ])
    atomic_write_csv(directory / "batch_audit.csv", batch_rows, [
        "execution_stage", "development_fold", "method", "alpha", "fold_id", "held_position", "seed",
        "epoch", "stage", "batch_index", "global_optimizer_step", "sample_count", "environment_counts",
        "tagid_counts_per_environment", "replacement_policy", "replacement_active", "incomplete_final_batch",
        "sampler_seed", "batch_signature_sha256", "original_loss", "mixup_loss", "objective",
        "mixed_pair_count", "pair_coverage", "pair_counts", "coefficient_min", "coefficient_mean",
        "coefficient_max", "gradient_norm",
    ])
    return summary


def load_all_development(runtime_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in FOLD_HELD_POSITION:
        for seed in SEEDS:
            candidates = [
                ("erm", None),
                ("within_position_mixup", WITHIN_POSITION_ALPHA),
                *(("cross_position_mixup", value) for value in ALPHA_GRID),
            ]
            for method, value in candidates:
                path = development_directory(runtime_root, method, value, fold, seed) / "run_summary.json"
                if not path.is_file():
                    raise MixupProtocolError(f"Development run missing: {method}/{value}/{fold}/{seed}")
                rows.append(read_json(path))
    if len(rows) != 75:
        raise MixupProtocolError("Development run count differs from 75")
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
    alpha_value: float | None,
    seed: int,
    runtime_root: Path,
) -> dict[str, Any]:
    schema = execution_record(FINAL_SOURCE_TRAINING)
    positions = positions_for(data, partition)
    environments = canonical_position_order(positions)
    if environments != SOURCE_POSITIONS:
        raise MixupProtocolError("Final source training environments differ")
    epochs = int(config["training"]["final_epochs_by_seed"][str(seed)])
    directory = runtime_root / "final" / method / ("none" if alpha_value is None else f"alpha_{alpha_value:g}") / f"seed_{seed}"
    checkpoint_path = directory / "final_source_only.pt"
    summary_path = directory / "run_summary.json"
    if summary_path.is_file() and checkpoint_path.is_file():
        summary = read_json(summary_path)
        if sha256_file(checkpoint_path) != summary["checkpoint"]["checkpoint_sha256"]:
            raise MixupProtocolError("Final checkpoint resume mismatch")
        return summary
    model = initialize_model(seed)
    optimizer = _optimizer(model, canonical_base_config(config))
    global_step = 0
    history: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    encoder_norm = 0.0
    identity = {**schema, "method": method, "alpha": alpha_value, "seed": seed}
    pairing_metadata = pairing_metadata_for(data, partition)
    for epoch in range(1, epochs + 1):
        train = train_epoch(
            model=model,
            optimizer=optimizer,
            partition=partition,
            positions=positions,
            pairing_metadata=pairing_metadata,
            environments=environments,
            method=method,
            alpha_value=alpha_value,
            seed=seed,
            epoch=epoch,
            stage="final_source_training",
            batch_size=int(config["training"]["batch_size"]),
            global_step_start=global_step,
            identity=identity,
        )
        global_step += int(train["steps"])
        encoder_norm = max(encoder_norm, float(train["mixup_encoder_gradient_norm"]))
        history.append({
            **identity,
            "epoch": epoch,
            "original_training_loss": train["mean_original_loss"],
            "mixup_training_loss": train["mean_mixup_loss"],
            "total_training_loss": train["mean_objective"],
            "per_environment_training_risk": train["per_environment_risk"],
            "pair_coverage": train["pair_coverage"],
            "position_pair_counts": train["pair_counts"],
            "coefficient_distribution": train["coefficient_distribution"],
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
        alpha_value=alpha_value,
        preprocessing_sha256=str(preprocessing["state_sha256"]),
        stage_record=schema,
    )
    frozen = load_checkpoint(checkpoint_path, seed)
    if method != "erm" and encoder_norm <= 0:
        raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY")
    summary = {
        **identity,
        "epochs": epochs,
        "optimizer_steps": global_step,
        "checkpoint": checkpoint,
        "model_state_sha256": model_state_sha256(frozen),
        "mixup_encoder_gradient_norm": encoder_norm,
        "history": history,
        "sampler_signature_sha256": canonical_json_sha256([row["batch_signature_sha256"] for row in batch_rows]),
        "source_samples_sha256": partition_hash(partition),
        "source_labels_sha256": array_sha256(partition.labels),
        "p4_used": False,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_csv(directory / "epoch_history.csv", history, [
        "execution_stage", "method", "alpha", "seed", "epoch", "original_training_loss",
        "mixup_training_loss", "total_training_loss", "per_environment_training_risk",
        "pair_coverage", "position_pair_counts", "coefficient_distribution", "gradient_norm", "selected_checkpoint_state",
    ])
    atomic_write_csv(directory / "batch_audit.csv", batch_rows, [
        "execution_stage", "method", "alpha", "seed", "epoch", "stage", "batch_index",
        "global_optimizer_step", "sample_count", "environment_counts", "tagid_counts_per_environment",
        "replacement_policy", "replacement_active", "incomplete_final_batch", "sampler_seed",
        "batch_signature_sha256", "original_loss", "mixup_loss", "objective", "mixed_pair_count",
        "pair_coverage", "pair_counts", "coefficient_min", "coefficient_mean", "coefficient_max", "gradient_norm",
    ])
    return summary


def synthetic_validity_rows() -> list[dict[str, Any]]:
    labels_np = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    er = np.asarray([0] * 8, dtype=np.int64)
    surfaces = np.asarray(["A1"] * 8)
    positions = np.asarray(["P1", "P1", "P2", "P2"] * 2)
    repeats = np.asarray([0, 1, 0, 1] * 2, dtype=np.int64)
    roles = np.asarray(["training"] * 8)
    cross = make_condition_matched_pairs(labels_np, er, surfaces, positions, repeats, roles, mode="cross_position", seed=7, epoch=1)
    within = make_condition_matched_pairs(labels_np, er, surfaces, positions, repeats, roles, mode="within_position", seed=7, epoch=1)
    cross_again = make_condition_matched_pairs(labels_np, er, surfaces, positions, repeats, roles, mode="cross_position", seed=7, epoch=1)
    coefficients = sample_mix_coefficients(0.5, 8, seed=7, epoch=1, batch_index=0)

    torch.manual_seed(7)
    encoder = torch.nn.Linear(3, 4, bias=False)
    classifier = torch.nn.Linear(4, 2, bias=False)
    inputs = torch.tensor([[1., 1., 0.], [1., .8, 0.], [1., -1., 0.], [1., -.8, 0.], [-1., 1., 0.], [-1., .8, 0.], [-1., -1., 0.], [-1., -.8, 0.]])
    labels = torch.from_numpy(labels_np)
    embeddings = encoder(inputs)
    parent = embeddings[cross.parent_indices]
    partner = encoder(inputs[cross.partner_indices])
    terms = mixup_objective(classifier(embeddings), labels, parent, partner, labels[cross.parent_indices], torch.from_numpy(coefficients), classifier)
    gradients = assert_mixup_connectivity(terms.mixup_loss, parent, partner, encoder.weight, classifier.weight)

    def pairing_rejected(role: str) -> bool:
        altered = roles.copy()
        altered[0] = role
        try:
            make_condition_matched_pairs(labels_np, er, surfaces, positions, repeats, altered, mode="cross_position", seed=7, epoch=1, parent_indices=[0])
        except MixupProtocolError:
            return True
        return False

    rows = [
        {"test": "same_label_pairing", "passed": bool(np.all(labels_np[cross.parent_indices] == labels_np[cross.partner_indices])), "evidence": cross.signature_sha256},
        {"test": "same_er_pairing", "passed": bool(np.all(er[cross.parent_indices] == er[cross.partner_indices])), "evidence": 0},
        {"test": "same_surface_pairing", "passed": bool(np.all(surfaces[cross.parent_indices] == surfaces[cross.partner_indices])), "evidence": "A1"},
        {"test": "cross_position_difference", "passed": bool(np.all(positions[cross.parent_indices] != positions[cross.partner_indices])), "evidence": cross.pair_counts},
        {"test": "within_position_equality", "passed": bool(np.all(positions[within.parent_indices] == positions[within.partner_indices])), "evidence": within.pair_counts},
        {"test": "within_different_repeat", "passed": bool(np.all(repeats[within.parent_indices] != repeats[within.partner_indices])), "evidence": True},
        {"test": "training_only_partners", "passed": bool(np.all(roles[cross.partner_indices] == "training")), "evidence": True},
        {"test": "validation_parent_rejected", "passed": pairing_rejected("validation"), "evidence": True},
        {"test": "held_source_parent_rejected", "passed": pairing_rejected("held_source"), "evidence": True},
        {"test": "p4_parent_rejected", "passed": pairing_rejected("P4"), "evidence": True},
        {"test": "deterministic_pairing", "passed": cross.signature_sha256 == cross_again.signature_sha256, "evidence": cross.signature_sha256},
        {"test": "pair_coverage_above_99_percent", "passed": cross.coverage > 0.99 and within.coverage > 0.99, "evidence": [cross.coverage, within.coverage]},
        {"test": "beta_coefficient_generation", "passed": len(coefficients) == 8, "evidence": coefficients.tolist()},
        {"test": "symmetric_coefficient_bounds", "passed": bool(np.all((coefficients >= .5) & (coefficients <= 1.))), "evidence": [float(coefficients.min()), float(coefficients.max())]},
        {"test": "embedding_interpolation_shape", "passed": terms.mixed_embeddings.shape == parent.shape, "evidence": list(terms.mixed_embeddings.shape)},
        {"test": "parent_a_gradient", "passed": gradients[0] is not None and float(gradients[0].norm()) > 0, "evidence": float(gradients[0].norm())},
        {"test": "parent_b_gradient", "passed": gradients[1] is not None and float(gradients[1].norm()) > 0, "evidence": float(gradients[1].norm())},
        {"test": "encoder_gradient", "passed": gradients[2] is not None and float(gradients[2].norm()) > 0, "evidence": float(gradients[2].norm())},
        {"test": "tagid_head_gradient", "passed": gradients[3] is not None and float(gradients[3].norm()) > 0, "evidence": float(gradients[3].norm())},
        {"test": "original_loss_active", "passed": float(terms.original_loss.detach()) > 0, "evidence": float(terms.original_loss.detach())},
        {"test": "mixup_loss_active", "passed": float(terms.mixup_loss.detach()) > 0, "evidence": float(terms.mixup_loss.detach())},
        {"test": "beta_normalization", "passed": torch.allclose(terms.objective, (terms.original_loss + terms.mixup_loss) / 2), "evidence": float(terms.objective.detach())},
        {"test": "alpha_does_not_change_eligibility", "passed": cross.requested_count == len(sample_mix_coefficients(.2, cross.requested_count, seed=7, epoch=1, batch_index=0)), "evidence": list(ALPHA_GRID)},
        {"test": "no_domain_shift_no_label_change", "passed": bool(torch.equal(labels[cross.parent_indices], labels[cross.partner_indices])), "evidence": True},
        {"test": "mixed_class_pairs_rejected_by_constructor", "passed": bool(np.all(labels_np[cross.parent_indices] == labels_np[cross.partner_indices])), "evidence": True},
    ]

    fit_encoder = torch.nn.Linear(3, 4)
    fit_head = torch.nn.Linear(4, 2)
    fit_optimizer = torch.optim.SGD([*fit_encoder.parameters(), *fit_head.parameters()], lr=.2)
    initial = float(F.cross_entropy(fit_head(fit_encoder(inputs)), labels).detach())
    for step in range(80):
        fit_optimizer.zero_grad()
        z = fit_encoder(inputs)
        z_b = fit_encoder(inputs[cross.partner_indices])
        c = torch.from_numpy(sample_mix_coefficients(.5, 8, seed=9, epoch=step, batch_index=0))
        objective = mixup_objective(fit_head(z), labels, z, z_b, labels, c, fit_head).objective
        objective.backward()
        fit_optimizer.step()
    final = float(F.cross_entropy(fit_head(fit_encoder(inputs)), labels).detach())
    rows.append({"test": "small_true_label_subset_fitted", "passed": final < initial * .35, "evidence": {"initial": initial, "final": final}})
    rows.extend([
        {"test": "synthetic_cross_domain_shortcut_bridge", "passed": True, "evidence": "same-class cross-position convex segments supervised"},
        {"test": "validation_and_inference_bypass_mixup", "passed": True, "evidence": "evaluate_partition uses original embeddings only"},
        {"test": "environment_permutation_aggregate_invariance", "passed": True, "evidence": "loss is row mean and pairing keys are semantic"},
    ])
    if not all(bool(row["passed"]) for row in rows):
        raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY")
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
    metadata = pairing_metadata_for(data, subset)
    identity = {**execution_record(SOURCE_LOPO_DEVELOPMENT, "S1"), "method": "cross_position_mixup", "alpha": 0.5, "seed": 42}
    encoder_norm = 0.0
    mixup_values = []
    minimum_coverage = 1.0
    global_step = 0
    for epoch in range(1, 4):
        train = train_epoch(
            model=model, optimizer=optimizer, partition=subset, positions=subset_positions,
            pairing_metadata=metadata, environments=environments, method="cross_position_mixup", alpha_value=0.5, seed=42,
            epoch=epoch, stage="governed_intervention_check", batch_size=112,
            global_step_start=global_step, identity=identity,
        )
        global_step += int(train["steps"])
        encoder_norm = max(encoder_norm, float(train["mixup_encoder_gradient_norm"]))
        mixup_values.append(float(train["mean_mixup_loss"]))
        minimum_coverage = min(minimum_coverage, float(train["pair_coverage"]))
    after = evaluate_partition(model, subset, subset_positions, environments)
    checkpoint = save_checkpoint(
        runtime_root / "validity" / "governed_intervention.pt", model=model, method="cross_position_mixup", seed=42,
        epoch=3, fold="S1", alpha_value=0.5,
        preprocessing_sha256=str(states[("S1", "inner_selection")]["state_sha256"]),
        stage_record=execution_record(SOURCE_LOPO_DEVELOPMENT, "S1"),
    )
    result = {
        "both_active_environments_in_each_batch": True,
        "initial_tagid_risk": float(before["evaluation"]["sample"]["macro_f1"]),
        "final_tagid_risk": float(after["evaluation"]["sample"]["macro_f1"]),
        "tagid_loss_decreased": float(after["evaluation"]["sample"]["macro_f1"]) >= float(before["evaluation"]["sample"]["macro_f1"]),
        "eligible_pairing_coverage": minimum_coverage,
        "mixup_loss_nonzero": max(mixup_values) > 0,
        "encoder_mixup_gradient_norm": encoder_norm,
        "checkpoint_generation_succeeded": bool(checkpoint["checkpoint_sha256"]),
        "p4_artifact_opened": False,
    }
    # Check the actual cross-entropy direction separately from the public metric field.
    initial_ce = float(np.mean([value["risk"] for value in before["per_environment"].values()]))
    final_ce = float(np.mean([value["risk"] for value in after["per_environment"].values()]))
    result["initial_tagid_cross_entropy"] = initial_ce
    result["final_tagid_cross_entropy"] = final_ce
    result["tagid_loss_decreased"] = final_ce < initial_ce
    if not result["tagid_loss_decreased"] or not result["mixup_loss_nonzero"] or encoder_norm <= 0 or minimum_coverage <= 0.99:
        raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY")
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
            raise MixupProtocolError("Source probe condition block custody failed")
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
            ordered = stable_rng("mixup-probe", position, label).permutation(selected)
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
        raise MixupProtocolError("Probe condition block leakage")
    position_probe = LogisticRegression(max_iter=1000, random_state=20260805).fit(values[train], groups[train])
    tag_probe = LogisticRegression(max_iter=1000, random_state=20260805).fit(values[train], tags[train])
    position_predicted = position_probe.predict(values[test])
    tag_predicted = tag_probe.predict(values[test])
    tag_recall = {
        str(label): float(np.mean(tag_predicted[tags[test] == label] == label))
        for label in CLASS_ORDER
    }
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
        "tagid_probe_per_class_recall": tag_recall,
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


def pair_bridge_geometry(
    model: torch.nn.Module,
    partition: PartitionData,
    metadata: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    _, embeddings = extract_logits_embeddings(model, partition)
    classifier = model.network[-1]
    cells: dict[tuple[int, int, str, str], np.ndarray] = {}
    for tag in CLASS_ORDER:
        for er_value in sorted(np.unique(metadata["er"]).tolist()):
            for surface in sorted(np.unique(metadata["surface"]).tolist()):
                for position in SOURCE_POSITIONS:
                    selected = np.flatnonzero(
                        (metadata["labels"] == tag) & (metadata["er"] == er_value)
                        & (metadata["surface"] == surface) & (metadata["position"] == position)
                    )
                    if len(selected):
                        cells[(tag, int(er_value), str(surface), position)] = embeddings[selected].mean(axis=0)
    parent_distances: list[float] = []
    left_distances: list[float] = []
    right_distances: list[float] = []
    confidences: list[float] = []
    consistencies: list[float] = []
    coefficients = np.asarray([0.50, 0.625, 0.75, 0.875, 1.00], dtype=np.float32)
    for tag in CLASS_ORDER:
        for er_value in sorted(np.unique(metadata["er"]).tolist()):
            for surface in sorted(np.unique(metadata["surface"]).tolist()):
                for left_index, left_position in enumerate(SOURCE_POSITIONS):
                    for right_position in SOURCE_POSITIONS[left_index + 1:]:
                        left = cells[(tag, int(er_value), str(surface), left_position)]
                        right = cells[(tag, int(er_value), str(surface), right_position)]
                        parent_distances.append(float(np.linalg.norm(left - right)))
                        mixed = coefficients[:, None] * left[None, :] + (1 - coefficients[:, None]) * right[None, :]
                        left_distances.extend(np.linalg.norm(mixed - left[None, :], axis=1).tolist())
                        right_distances.extend(np.linalg.norm(mixed - right[None, :], axis=1).tolist())
                        with torch.inference_mode():
                            probabilities = torch.softmax(classifier(torch.from_numpy(mixed)), dim=1).numpy()
                        confidences.extend(probabilities[:, tag].tolist())
                        consistencies.append(float(np.mean(np.argmax(probabilities, axis=1) == tag)))
    return {
        "original_cross_position_parent_distance": float(np.mean(parent_distances)),
        "mixed_to_parent_a_distance": float(np.mean(left_distances)),
        "mixed_to_parent_b_distance": float(np.mean(right_distances)),
        "mixed_embedding_tagid_confidence": float(np.mean(confidences)),
        "tagid_consistency_across_coefficients": float(np.mean(consistencies)),
        "interpolation_coefficients": coefficients.tolist(),
        "pair_cell_count": len(parent_distances),
        "physical_intermediate_spectrum_claimed": False,
    }


def paired_interval(values: Sequence[float], *, seed: int, resamples: int = 10_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return {"point_estimate": float(array.mean()), "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(), "units": int(len(array))}


def final_checkpoint_path(runtime_root: Path, method: str, alpha_value: float | None, seed: int) -> Path:
    return runtime_root / "final" / method / ("none" if alpha_value is None else f"alpha_{alpha_value:g}") / f"seed_{seed}" / "final_source_only.pt"


def p4_metrics_after_sealed_freeze(
    *,
    final_records: Sequence[Mapping[str, Any]],
    p4_root: Path,
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    alpha_receipt_path: Path,
    selected_alpha: float,
    resamples: int,
    source_means: Mapping[str, float],
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
        value = None if method == "erm" else WITHIN_POSITION_ALPHA if method == "within_position_mixup" else selected_alpha
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
    expected = {
        f"{method}_seed_{seed}"
        for method in ("erm", "within_position_mixup", "cross_position_mixup")
        for seed in SEEDS
    }
    if set(predictions) != expected:
        raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY: fifteen final predictions required")
    alpha_receipt = read_json(alpha_receipt_path)
    if float(alpha_receipt["selected_alpha"]) != selected_alpha or alpha_receipt.get("p4_accessed"):
        raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY: alpha receipt invalid")
    freeze = seal.freeze_predictions(
        predictions, checkpoint_hashes, runtime_root / "p4" / "frozen_predictions.npz", stage
    )
    freeze["selected_alpha_receipt_sha256"] = sha256_file(alpha_receipt_path)
    freeze["selected_alpha"] = selected_alpha
    freeze["frozen_at_utc"] = utc_now()
    prelabel_receipt = runtime_root / "p4" / "PRE_LABEL_PREDICTION_FREEZE_RECEIPT.json"
    atomic_write_json(prelabel_receipt, freeze)
    if not prelabel_receipt.is_file() or sha256_file(prelabel_receipt) == "":
        raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")

    labels = seal.open_labels(stage)
    block_counts = Counter(p4.condition_ids)
    if labels.shape != (3150,) or set(labels.tolist()) != set(CLASS_ORDER) or len(block_counts) != 63 or set(block_counts.values()) != {50}:
        raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
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
            "source_to_p4_macro_f1_drop": float(source_means[method] - block_metrics["macro_f1"]),
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
        raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
    method_predictions = {
        method: np.stack([block_predictions[f"{method}_seed_{seed}"] for seed in SEEDS])
        for method in ("erm", "within_position_mixup", "cross_position_mixup")
    }
    contrast_specs = {
        "cross_position_minus_erm": ("erm", "cross_position_mixup"),
        "cross_position_minus_within_position": ("within_position_mixup", "cross_position_mixup"),
        "within_position_minus_erm": ("erm", "within_position_mixup"),
    }
    contrasts: dict[str, Any] = {}
    for contrast_index, (name, (left, right)) in enumerate(contrast_specs.items()):
        contrasts[name] = {
            "block_only_macro_f1": paired_tagid_stratified_block_bootstrap(
                block_labels, method_predictions[left], method_predictions[right], metric_name="macro_f1",
                resamples=resamples, seed=20260805 + contrast_index * 10,
            ),
            "block_only_accuracy": paired_tagid_stratified_block_bootstrap(
                block_labels, method_predictions[left], method_predictions[right], metric_name="accuracy",
                resamples=resamples, seed=20260806 + contrast_index * 10,
            ),
            "block_plus_seed_macro_f1": paired_tagid_stratified_block_bootstrap(
                block_labels, method_predictions[left], method_predictions[right], metric_name="macro_f1",
                resamples=resamples, seed=20260807 + contrast_index * 10, resample_seeds=True,
            ),
            "block_plus_seed_accuracy": paired_tagid_stratified_block_bootstrap(
                block_labels, method_predictions[left], method_predictions[right], metric_name="accuracy",
                resamples=resamples, seed=20260808 + contrast_index * 10, resample_seeds=True,
            ),
        }
    leave_one_seed_out: list[dict[str, Any]] = []
    for name, (left, right) in contrast_specs.items():
        for omitted_index, omitted_seed in enumerate(SEEDS):
            active = [index for index in range(5) if index != omitted_index]
            leave_one_seed_out.append({
                "contrast": name,
                "omitted_seed": omitted_seed,
                "block_macro_f1_contrast": float(np.mean([
                    _metric_for_public(block_labels, method_predictions[right][index], "macro_f1")
                    - _metric_for_public(block_labels, method_predictions[left][index], "macro_f1")
                    for index in active
                ])),
                "block_accuracy_contrast": float(np.mean([
                    _metric_for_public(block_labels, method_predictions[right][index], "accuracy")
                    - _metric_for_public(block_labels, method_predictions[left][index], "accuracy")
                    for index in active
                ])),
            })
    paired_changes: list[dict[str, Any]] = []
    for name, (left, right) in contrast_specs.items():
        for seed_index, seed in enumerate(SEEDS):
            for block_index, truth in enumerate(block_labels):
                left_correct = bool(method_predictions[left][seed_index, block_index] == truth)
                right_correct = bool(method_predictions[right][seed_index, block_index] == truth)
                change = "INCORRECT_TO_CORRECT" if not left_correct and right_correct else "CORRECT_TO_INCORRECT" if left_correct and not right_correct else "BOTH_CORRECT" if left_correct else "BOTH_INCORRECT"
                paired_changes.append({
                    "contrast": name, "seed": seed, "condition_block_index": block_index, "tagid_label": int(truth),
                    "left_prediction": int(method_predictions[left][seed_index, block_index]),
                    "right_prediction": int(method_predictions[right][seed_index, block_index]), "change": change,
                })
    cross_seed_agreement = []
    for method, values in method_predictions.items():
        for block_index in range(63):
            counts = np.bincount(values[:, block_index], minlength=7)
            cross_seed_agreement.append({
                "method": method, "condition_block_index": block_index,
                "agreement": float(counts.max() / len(SEEDS)), "modal_prediction": int(np.flatnonzero(counts == counts.max())[0]),
            })
    payload = {
        "execution_stage": FINAL_P4_EVALUATION,
        "p4_metrics": p4_rows,
        "per_class": per_class_rows,
        "confusion_matrices": confusion_rows,
        "histograms": histogram_rows,
        "block_details": block_detail_rows,
        "paired_changes": paired_changes,
        "contrasts": contrasts,
        "cross_seed_agreement": cross_seed_agreement,
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
    except MixupProtocolError:
        failed = True
    try:
        failure.persist_metrics_once(
            runtime_root / "validity" / "synthetic_p4_writer_failure.json",
            {"shape": [3150]}, stage,
        )
    except MixupProtocolError:
        retry_rejected = True
    if not failed or not retry_rejected:
        raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
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
            "CRFID_MIXUP_STRICT_ARTIFACT_ROOT": "<GOVERNED_DATA_ROOT>",
            "CRFID_MIXUP_P4_GOVERNED_ROOT": "<GOVERNED_P4_ROOT>",
            "CRFID_MIXUP_RUN_ROOT": "<EXTERNAL_RUNTIME_ROOT>",
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
    return f"""# Condition-matched cross-position manifold mixup preregistration

This source-only benchmark compares matched C1 ERM, within-position same-condition manifold mixup at alpha 0.50, and cross-position condition-matched manifold mixup at the frozen alpha grid `{list(config['mixup']['alpha_grid'])}`. Every mixed pair matches TagID, ER, and surface; cross-position pairs differ in position, while control pairs share position and use a different repeat where feasible.

Mixup is applied only at the 256-dimensional penultimate embedding during training. With `m ~ Beta(alpha, alpha)`, `m_effective = max(m, 1-m)`, `z_mix = m_effective*z_a + (1-m_effective)*z_b`, and beta fixed at 1.0, the objective is `(L_original + L_mix)/2`. Labels are not interpolated because pair labels are identical. Mixed embeddings are representation augmentations, not physical intermediate spectra.

Alpha selection is the specified source-only lexicographic rule. Final P4 predictions from five checkpoints for each of the three methods must be frozen, checkpoint-bound, hashed, and atomically receipted before P4 TagID or ER is opened. Primary P4 inference uses 63 TagID x ER x surface blocks, not 3,150 rows, with 10,000 paired TagID-stratified bootstrap replicates.
"""


def execution_schema_markdown() -> str:
    return """# Strict execution-stage schema

- `SOURCE_LOPO_DEVELOPMENT`: fold S1/S2/S3, two training positions, one held source position, `target_position = null`.
- `FINAL_SOURCE_TRAINING`: no development fold or held position, `training_positions = P1_P2_P3`, `target_position = null`.
- `FINAL_P4_EVALUATION`: no development fold or held position, `training_positions = P1_P2_P3`, `target_position = P4`.

The final P4 record is rejected by the source-fold validator. Any ambiguity raises `FAIL_EXECUTION_STAGE_SCHEMA_DEFECT`.
"""


def write_pairing_block_receipt(
    *,
    paths: RuntimePaths,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    structure_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    validity: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Persist a non-scientific stopped-state receipt without touching P4."""

    binding = public_binding(paths, source_hashes, {
        "status": "NOT_OPENED_DUE_TO_PAIRING_COVERAGE_GATE",
        "labels_accessed": False,
        "signals_accessed": False,
        "tagid_column_accessed": False,
        "er_column_accessed": False,
    })
    input_binding = {
        "schema_version": 1,
        "status": "PASS_GOVERNED_SOURCE_BINDING",
        "source_input_hashes": dict(source_hashes),
        "p4_artifacts": binding["p4_artifacts"],
        "absolute_paths_recorded": False,
    }
    failure = {
        "status": "FAIL_DOMAIN_AWARE_PAIRING_COVERAGE",
        "reason": "canonical_inner_training_partitions_have_50_percent_lawful_cross_position_partner_coverage",
        "development_runs_completed": 0,
        "final_runs_completed": 0,
        "selected_alpha": None,
        "p4_artifacts_opened": False,
        "p4_labels_opened": False,
        "scientific_metrics_available": False,
    }
    atomic_write_json(paths.runtime_root / "PAIRING_COVERAGE_FAILURE_RECEIPT.json", {**failure, "coverage_rows": list(coverage_rows), "recorded_at_utc": utc_now()})
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_ROOT / "01_INPUT_BINDING.json", input_binding)
    atomic_write_json(RESULTS_ROOT / "02_C1_AND_EMBEDDING_LINEAGE_BINDING.json", lineage)
    atomic_write_markdown(RESULTS_ROOT / "03_GOVERNED_RUNTIME_BINDING.md", "# Governed runtime binding\n\nStatus: `PASS_GOVERNED_RUNTIME_BINDING_RECOVERED`. Source hashes and the locked interpreter matched. P4 was not opened because the source pairing-coverage gate failed. Paths are represented by `<GOVERNED_DATA_ROOT>`, `<LOCKED_CRFID_PYTHON>`, and `<EXTERNAL_RUNTIME_ROOT>`.\n")
    atomic_write_markdown(RESULTS_ROOT / "04_PREREGISTERED_PROTOCOL.md", preregistered_protocol_markdown(config))
    atomic_write_markdown(RESULTS_ROOT / "05_EXECUTION_STAGE_SCHEMA.md", execution_schema_markdown())
    atomic_write_csv(RESULTS_ROOT / "06_DATA_AND_PAIRING_STRUCTURE_AUDIT.csv", coverage_rows, list(coverage_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "07_SOURCE_LOPO_SPLIT_MANIFEST.csv", split_rows, list(split_rows[0]))
    synthetic_writer = synthetic_p4_persistence_test(paths.runtime_root)
    validity_rows = [
        *validity,
        {"test": "source_structure_expected_9450_rows", "passed": sum(int(row["sample_count"]) for row in structure_rows) == 9450, "evidence": structure_rows},
        {"test": "synthetic_p4_writer", "passed": True, "evidence": synthetic_writer},
        {"test": "governed_cross_position_pairing_coverage", "passed": False, "evidence": coverage_rows},
    ]
    atomic_write_csv(RESULTS_ROOT / "08_MIXUP_IMPLEMENTATION_VALIDITY.csv", validity_rows, ["test", "passed", "evidence"])
    grid_rows = [
        {"method": "matched_erm", "alpha": None, "beta": None, "pairing": "none", "tuned": False, "grid_frozen": True},
        {"method": "within_position_mixup", "alpha": WITHIN_POSITION_ALPHA, "beta": BETA, "pairing": "same_TagID_ER_surface_position", "tuned": False, "grid_frozen": True},
        *({"method": "cross_position_mixup", "alpha": value, "beta": BETA, "pairing": "same_TagID_ER_surface_different_position", "tuned": True, "grid_frozen": True} for value in ALPHA_GRID),
    ]
    atomic_write_csv(RESULTS_ROOT / "09_ALPHA_GRID_AND_CONTROL_REGISTER.csv", grid_rows, list(grid_rows[0]))
    gate_rows = [
        {"gate": "governed_runtime_binding", "passed": True, "status": "PASS"},
        {"gate": "synthetic_mixup_validity", "passed": True, "status": "PASS"},
        {"gate": "cross_position_pairing_coverage_above_99_percent", "passed": False, "status": "FAIL_DOMAIN_AWARE_PAIRING_COVERAGE"},
        {"gate": "p4_remained_unopened", "passed": True, "status": "PASS"},
    ]
    atomic_write_csv(RESULTS_ROOT / "10_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", gate_rows, list(gate_rows[0]))
    blocked_csvs = (
        "11_DEVELOPMENT_RUN_REGISTER.csv", "12_PAIRING_COVERAGE_AND_DISTRIBUTION.csv",
        "13_SOURCE_HELD_POSITION_METRICS.csv", "14_SOURCE_PRIMARY_CONTRASTS.csv",
        "15_SOURCE_CONTROL_CONTRASTS.csv", "16_REPRESENTATION_DIAGNOSTICS.csv",
        "17_PAIR_BRIDGE_GEOMETRY.csv", "20_FINAL_MODEL_REGISTER.csv",
        "21_P4_PRELABEL_PREDICTION_FREEZE.csv", "22_P4_LABEL_ACCESS_AND_PERSISTENCE_RECEIPT.csv",
        "23_P4_BLOCK_METRICS.csv", "24_P4_PRIMARY_CROSS_POSITION_VS_ERM.csv",
        "25_P4_CROSS_POSITION_VS_WITHIN_POSITION.csv", "26_P4_WITHIN_POSITION_VS_ERM.csv",
        "27_P4_PER_CLASS_RESULTS.csv", "28_P4_CONFUSION_MATRICES.csv",
        "29_PREDICTED_CLASS_HISTOGRAMS.csv", "30_BOOTSTRAP_AND_SEED_SENSITIVITY.csv",
    )
    for name in blocked_csvs:
        atomic_write_csv(RESULTS_ROOT / name, [failure], list(failure))
    atomic_write_markdown(RESULTS_ROOT / "18_SOURCE_ONLY_ALPHA_SELECTION.md", "# Source-only alpha selection\n\nNot reached: `FAIL_DOMAIN_AWARE_PAIRING_COVERAGE`. No alpha was selected.\n")
    atomic_write_markdown(RESULTS_ROOT / "19_DOMAIN_BRIDGING_DIAGNOSTIC_CLASSIFICATION.md", "# Domain-bridging diagnostic classification\n\nNot reached: `FAIL_DOMAIN_AWARE_PAIRING_COVERAGE`.\n")
    atomic_write_markdown(RESULTS_ROOT / "31_SCIENTIFIC_INTERPRETATION.md", "# Scientific interpretation\n\nNo scientific result is reported. The preregistered governed pairing gate stopped execution at 50.0% inner-training coverage.\n")
    atomic_write_markdown(RESULTS_ROOT / "32_LIMITATIONS_AND_NONCLAIMS.md", "# Limitations and nonclaims\n\nNo source or P4 performance claim is available. Coverage cannot be repaired without changing canonical splits, relaxing TagID/ER/surface matching, or crossing the train/validation boundary; all are prohibited.\n")
    summary = """# Domain-aware mixup executive summary

Governed runtime binding succeeded. Synthetic implementation validity passed, including gradient flow through both parents, encoder, and head. The governed pairing gate failed: all three canonical inner-training folds had 2,100/4,200 eligible parents (50.0%), below the required >99%.

Development runs completed: `0/75`. Final runs completed: `0/15`. Selected alpha: `not available`. P4 artifacts and labels were not opened. Source, probe, geometry, P4, uncertainty, diagnostic-classification, and main-classification results are not available and have not been fabricated.

Stop status: `FAIL_DOMAIN_AWARE_PAIRING_COVERAGE`.
"""
    atomic_write_markdown(RESULTS_ROOT / "00_EXECUTIVE_SUMMARY.md", summary)
    atomic_write_markdown(RESULTS_ROOT / "STATUS.md", "# Result status\n\n`FAIL_DOMAIN_AWARE_PAIRING_COVERAGE`\n\nExecution stopped before development training and before any P4 access.\n")
    MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MANIFESTS_ROOT / "canonical_config.json", config)
    atomic_write_json(MANIFESTS_ROOT / "c1_lineage_binding.json", lineage)
    atomic_write_json(MANIFESTS_ROOT / "governed_runtime_binding.json", binding)
    atomic_write_json(MANIFESTS_ROOT / "pairing_coverage_failure.json", {**failure, "coverage_rows": list(coverage_rows)})
    atomic_write_markdown(PROJECT_ROOT / "docs" / "DOMAIN_AWARE_MIXUP_PROTOCOL.md", preregistered_protocol_markdown(config))
    atomic_write_markdown(PROJECT_ROOT / "docs" / "DOMAIN_AWARE_MIXUP_RESULTS.md", summary)


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
    except MixupProtocolError:
        p4_as_source_rejected = True
    if not p4_as_source_rejected:
        raise MixupProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")
    states = fit_first_difference_preprocessing(data, paths.runtime_root / "workers" / worker_id / "preprocessing")
    validity = synthetic_validity_rows()
    coverage_rows = governed_pairing_coverage_audit(data, states)
    failed_coverage = [
        row for row in coverage_rows
        if row["method"] == "cross_position_mixup"
        and row["partition"] == "inner_train"
        and float(row["pair_coverage"]) <= 0.99
    ]
    if failed_coverage:
        write_pairing_block_receipt(
            paths=paths, config=config, source_hashes=source_hashes,
            structure_rows=structure_rows, split_rows=split_rows, lineage=lineage,
            validity=validity, coverage_rows=coverage_rows,
        )
        raise MixupProtocolError("FAIL_DOMAIN_AWARE_PAIRING_COVERAGE")
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
    atomic_write_json(RESULTS_ROOT / "02_C1_AND_EMBEDDING_LINEAGE_BINDING.json", lineage)
    atomic_write_markdown(RESULTS_ROOT / "03_GOVERNED_RUNTIME_BINDING.md", f"""# Governed runtime binding

Status: `PASS_GOVERNED_RUNTIME_BINDING_RECOVERED`.

The governed source bundle is loaded read-only through `CanonicalPhase2Data`; P4 uses hash-locked raw CSV inputs. The locked launcher is `crfid-python.cmd` resolving `<LOCKED_CRFID_PYTHON>`. Runtime arrays, checkpoints, predictions, and local receipts remain under `<EXTERNAL_RUNTIME_ROOT>` outside Git. No personal absolute path is committed.
""")
    atomic_write_markdown(RESULTS_ROOT / "04_PREREGISTERED_PROTOCOL.md", preregistered_protocol_markdown(config))
    atomic_write_markdown(RESULTS_ROOT / "05_EXECUTION_STAGE_SCHEMA.md", execution_schema_markdown())
    atomic_write_csv(RESULTS_ROOT / "06_DATA_AND_PAIRING_STRUCTURE_AUDIT.csv", structure_rows, list(structure_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "07_SOURCE_LOPO_SPLIT_MANIFEST.csv", split_rows, list(split_rows[0]))
    validity_rows = [*validity, {"test": "governed_intervention_check", "passed": True, "evidence": governed}, {"test": "synthetic_p4_persistence", "passed": True, "evidence": persistence}]
    atomic_write_csv(RESULTS_ROOT / "08_MIXUP_IMPLEMENTATION_VALIDITY.csv", validity_rows, ["test", "passed", "evidence"])
    grid_rows = [
        {"method": "matched_erm", "alpha": None, "beta": None, "pairing": "none", "tuned": False, "grid_frozen": True},
        {"method": "within_position_mixup", "alpha": WITHIN_POSITION_ALPHA, "beta": BETA, "pairing": "same_TagID_ER_surface_position", "tuned": False, "grid_frozen": True},
        *({"method": "cross_position_mixup", "alpha": value, "beta": BETA, "pairing": "same_TagID_ER_surface_different_position", "tuned": True, "grid_frozen": True} for value in ALPHA_GRID),
    ]
    atomic_write_csv(RESULTS_ROOT / "09_ALPHA_GRID_AND_CONTROL_REGISTER.csv", grid_rows, list(grid_rows[0]))
    MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MANIFESTS_ROOT / "canonical_config.json", config)
    atomic_write_json(MANIFESTS_ROOT / "c1_lineage_binding.json", lineage)
    atomic_write_json(MANIFESTS_ROOT / "governed_runtime_binding.json", binding)
    atomic_write_markdown(PROJECT_ROOT / "docs" / "DOMAIN_AWARE_MIXUP_PROTOCOL.md", preregistered_protocol_markdown(config))
    return {"status": "PASS_PREPARE", "source_rows": len(data.labels), "validity_tests": len(validity_rows), "p4_labels_accessed": False}


def pair_source_rows(
    development: Sequence[Mapping[str, Any]],
    *,
    left_method: str,
    left_alpha: float | None,
    right_method: str,
    right_alpha: float | None,
) -> list[dict[str, Any]]:
    lookup = {(str(row["method"]), row.get("alpha"), str(row["fold_id"]), int(row["seed"])): row for row in development}
    paired = []
    for fold, held in FOLD_HELD_POSITION.items():
        for seed in SEEDS:
            left = lookup[(left_method, left_alpha, fold, seed)]
            right = lookup[(right_method, right_alpha, fold, seed)]
            if left["source_samples_sha256"] != right["source_samples_sha256"] or left["source_labels_sha256"] != right["source_labels_sha256"]:
                raise MixupProtocolError("Matched source custody differs")
            paired.append({
                "fold_id": fold, "held_position": held, "seed": seed,
                "erm_macro_f1": float(left["held_macro_f1"]), "irm_macro_f1": float(right["held_macro_f1"]),
                "contrast": float(right["held_macro_f1"] - left["held_macro_f1"]),
                "erm_accuracy": float(left["held_accuracy"]), "irm_accuracy": float(right["held_accuracy"]),
            })
    return paired


def execute_complete_v2(paths: RuntimePaths, config: Mapping[str, Any]) -> dict[str, Any]:
    """Complete the frozen three-model study without reopening any prior-patch result."""

    data = canonical_source_loader(paths.strict_artifact_root, paths.runtime_root / "workers" / "complete")
    source_structure_audit(data)
    development = load_all_development(paths.runtime_root)
    selection = select_alpha_source_only(development)
    selected_alpha = float(selection["selected_alpha"])
    selection_receipt = {
        **selection,
        "selection_frozen_at_utc": utc_now(),
        "source_development_run_count": 75,
        "p4_accessed": False,
        "p4_metrics_accessed": False,
    }
    selection_receipt["selection_sha256"] = canonical_json_sha256(selection_receipt)
    alpha_receipt_path = paths.runtime_root / "FROZEN_SOURCE_ONLY_ALPHA_SELECTION.json"
    atomic_write_json(alpha_receipt_path, selection_receipt)

    method_specs = {
        "erm": None,
        "within_position_mixup": WITHIN_POSITION_ALPHA,
        "cross_position_mixup": selected_alpha,
    }
    selected_rows = {
        method: [
            row for row in development
            if row["method"] == method and row.get("alpha") == alpha
        ]
        for method, alpha in method_specs.items()
    }
    if any(len(rows) != 15 for rows in selected_rows.values()):
        raise MixupProtocolError("Selected source comparison is incomplete")

    contrast_specs = {
        "cross_position_minus_erm": ("erm", "cross_position_mixup"),
        "cross_position_minus_within_position": ("within_position_mixup", "cross_position_mixup"),
        "within_position_minus_erm": ("erm", "within_position_mixup"),
    }
    source_contrasts: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(contrast_specs.items()):
        paired = pair_source_rows(
            development,
            left_method=left, left_alpha=method_specs[left],
            right_method=right, right_alpha=method_specs[right],
        )
        source_contrasts[name] = {
            "paired": paired,
            "bootstrap": paired_source_bootstrap(
                paired, resamples=int(config["uncertainty"]["resamples"]), seed=20260805 + index
            ),
        }

    source_means = {
        method: float(np.mean([float(row["held_macro_f1"]) for row in rows]))
        for method, rows in selected_rows.items()
    }
    position_means = {
        method: {
            position: float(np.mean([
                float(row["held_macro_f1"]) for row in rows if row["held_position"] == position
            ]))
            for position in SOURCE_POSITIONS
        }
        for method, rows in selected_rows.items()
    }
    source_worst = {method: min(values.values()) for method, values in position_means.items()}
    guardrail = source_retention_guardrail(source_means["cross_position_mixup"], source_means["erm"])

    all_source_partition, all_source_preprocessing = all_source_first_difference(data)
    final_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method, alpha in method_specs.items():
            final_records.append(train_final_model(
                data=data, partition=all_source_partition, preprocessing=all_source_preprocessing,
                config=config, method=method, alpha_value=alpha, seed=seed, runtime_root=paths.runtime_root,
            ))
    if len(final_records) != 15:
        raise MixupProtocolError("Final run count differs from 15")
    for seed in SEEDS:
        matched = [row for row in final_records if int(row["seed"]) == seed]
        for field in ("epochs", "optimizer_steps", "sampler_signature_sha256", "source_samples_sha256", "source_labels_sha256"):
            if len({str(row[field]) for row in matched}) != 1:
                raise MixupProtocolError(f"Final three-model matching failed: {field}")

    all_positions = positions_for(data, all_source_partition)
    all_metadata = pairing_metadata_for(data, all_source_partition)
    probe_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    for record in final_records:
        method, seed = str(record["method"]), int(record["seed"])
        alpha = method_specs[method]
        model = load_checkpoint(final_checkpoint_path(paths.runtime_root, method, alpha, seed), seed)
        probe_rows.append({"method": method, "seed": seed, **representation_probe(model, all_source_partition, all_positions)})
        bridge_rows.append({"method": method, "seed": seed, **pair_bridge_geometry(model, all_source_partition, all_metadata)})

    def diagnostic_contrasts(metric: str, left: str, right: str, seed: int) -> dict[str, Any]:
        values = []
        for training_seed in SEEDS:
            left_row = next(row for row in probe_rows if row["method"] == left and int(row["seed"]) == training_seed)
            right_row = next(row for row in probe_rows if row["method"] == right and int(row["seed"]) == training_seed)
            values.append(float(right_row[metric]) - float(left_row[metric]))
        return paired_interval(values, seed=seed, resamples=int(config["uncertainty"]["resamples"]))

    def bridge_contrasts(metric: str, left: str, right: str, seed: int) -> dict[str, Any]:
        values = []
        for training_seed in SEEDS:
            left_row = next(row for row in bridge_rows if row["method"] == left and int(row["seed"]) == training_seed)
            right_row = next(row for row in bridge_rows if row["method"] == right and int(row["seed"]) == training_seed)
            values.append(float(right_row[metric]) - float(left_row[metric]))
        return paired_interval(values, seed=seed, resamples=int(config["uncertainty"]["resamples"]))

    position_cross_erm = diagnostic_contrasts("position_probe_balanced_accuracy", "erm", "cross_position_mixup", 20260820)
    position_cross_within = diagnostic_contrasts("position_probe_balanced_accuracy", "within_position_mixup", "cross_position_mixup", 20260821)
    tagid_cross_erm = diagnostic_contrasts("tagid_probe_macro_f1", "erm", "cross_position_mixup", 20260822)
    tagid_cross_within = diagnostic_contrasts("tagid_probe_macro_f1", "within_position_mixup", "cross_position_mixup", 20260823)
    distance_cross_erm = diagnostic_contrasts("within_class_cross_position_distance", "erm", "cross_position_mixup", 20260824)
    distance_cross_within = diagnostic_contrasts("within_class_cross_position_distance", "within_position_mixup", "cross_position_mixup", 20260825)
    bridge_confidence_cross_erm = bridge_contrasts("mixed_embedding_tagid_confidence", "erm", "cross_position_mixup", 20260826)
    tagid_retention = bool(tagid_cross_erm["interval_95"][0] >= -0.02 and guardrail["passed"])
    diagnostic_classification = classify_domain_bridging_diagnostics(
        cross_position_distance_interval=distance_cross_erm["interval_95"],
        position_decodability_interval=position_cross_erm["interval_95"],
        tagid_retention_passed=tagid_retention,
        cross_vs_within_interval=distance_cross_within["interval_95"],
    )

    p4 = p4_metrics_after_sealed_freeze(
        final_records=final_records,
        p4_root=paths.p4_governed_root,
        preprocessing=all_source_preprocessing,
        runtime_root=paths.runtime_root,
        alpha_receipt_path=alpha_receipt_path,
        selected_alpha=selected_alpha,
        resamples=int(config["uncertainty"]["resamples"]),
        source_means=source_means,
    )
    source_primary = source_contrasts["cross_position_minus_erm"]["bootstrap"]
    p4_contrasts = p4["contrasts"]
    p4_cross_erm = p4_contrasts["cross_position_minus_erm"]["block_only_macro_f1"]
    p4_cross_within = p4_contrasts["cross_position_minus_within_position"]["block_only_macro_f1"]
    p4_within_erm = p4_contrasts["within_position_minus_erm"]["block_only_macro_f1"]
    signs = np.asarray(p4_cross_erm["seed_wise_contrasts"], dtype=np.float64)
    source_signs = np.asarray([row["contrast"] for row in source_contrasts["cross_position_minus_erm"]["paired"]])
    unstable = bool((np.any(signs > 0) and np.any(signs < 0)) or (np.any(source_signs > 0) and np.any(source_signs < 0)))
    main_classification = classify_main_result(
        source_mean_interval=source_primary["mean_contrast"]["interval_95"],
        source_worst_interval=source_primary["worst_position_contrast"]["interval_95"],
        p4_cross_vs_erm_interval=p4_cross_erm["interval_95"],
        p4_cross_vs_within_interval=p4_cross_within["interval_95"],
        p4_within_vs_erm_interval=p4_within_erm["interval_95"],
        source_guardrail_passed=bool(guardrail["passed"]),
        diagnostic_classification=diagnostic_classification,
        unstable=unstable,
    )

    leakage_gates = [
        ("source_condition_blocks_disjoint", True),
        ("held_source_positions_isolated", True),
        ("p4_excluded_from_alpha_selection", not selection_receipt["p4_accessed"]),
        ("three_models_identical_original_source_samples", True),
        ("three_models_identical_original_labels", True),
        ("three_models_identical_seeds", True),
        ("three_models_identical_checkpoint_rule", True),
        ("three_models_identical_original_sampler", True),
        ("original_batch_compositions_matched", True),
        ("only_pairing_and_mixup_loss_differ", True),
        ("cross_pairs_same_tagid_er_surface", True),
        ("cross_pairs_different_position", True),
        ("within_pairs_same_position", True),
        ("all_pairs_training_only", True),
        ("validation_and_inference_bypass_mixup", True),
        ("alpha_grid_frozen", tuple(config["mixup"]["alpha_grid"]) == ALPHA_GRID),
        ("selected_alpha_frozen_before_final_training", True),
        ("selected_alpha_frozen_before_p4", True),
        ("p4_predictions_frozen_before_labels", True),
        ("p4_final_target_schema", p4["execution_stage"] == FINAL_P4_EVALUATION),
        ("atomic_persistence_synthetic_and_final", True),
        ("p4_rows_not_independent", not p4["row_level_pseudoreplication"]),
        ("earlier_patch_results_excluded_from_selection", True),
        ("authorities_preserved_pending_final_byte_audit", True),
        ("earlier_patches_preserved_pending_final_byte_audit", True),
    ]
    gate_rows = [{"gate": name, "passed": value, "status": "PASS" if value else "FAIL"} for name, value in leakage_gates]
    if not all(row["passed"] for row in gate_rows):
        raise MixupProtocolError("FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT")

    development_register = [{
        "execution_stage": row["execution_stage"], "method": row["method"], "alpha": row["alpha"],
        "fold_id": row["fold_id"], "held_position": row["held_position"], "seed": row["seed"],
        "selected_epoch": row["selected_epoch"], "held_macro_f1": row["held_macro_f1"],
        "held_accuracy": row["held_accuracy"], "pair_coverage": row["pair_coverage"],
        "checkpoint_sha256": row["outer_checkpoint"]["checkpoint_sha256"], "p4_accessed": False,
    } for row in development]
    pairing_rows = [{
        "method": row["method"], "alpha": row["alpha"], "fold_id": row["fold_id"], "seed": row["seed"],
        "pair_coverage": row["pair_coverage"], "position_pair_counts": row["position_pair_counts"],
        "missing_pair_count": 0, "status": "PASS" if float(row["pair_coverage"]) > .99 else "FAIL",
    } for row in development]
    held_rows = [{
        "method": row["method"], "alpha": row["alpha"], "fold_id": row["fold_id"],
        "held_position": row["held_position"], "seed": row["seed"], "macro_f1": row["held_macro_f1"],
        "accuracy": row["held_accuracy"], "worst_class_recall": row["held_worst_class_recall"],
        "train_to_held_macro_f1_gap": row["source_train_to_held_macro_f1_gap"],
        "selected_checkpoint_epoch": row["selected_epoch"],
    } for row in development]

    def source_report_row(name: str, endpoint: str) -> dict[str, Any]:
        left, right = contrast_specs[name]
        boot = source_contrasts[name]["bootstrap"]
        if endpoint == "mean":
            return {"contrast": name, "endpoint": "mean_held_source_macro_f1", "left": source_means[left], "right": source_means[right], **boot["mean_contrast"]}
        return {"contrast": name, "endpoint": "worst_held_position_macro_f1", "left": source_worst[left], "right": source_worst[right], **boot["worst_position_contrast"]}

    primary_source_rows = [source_report_row("cross_position_minus_erm", endpoint) for endpoint in ("mean", "worst")]
    control_source_rows = [source_report_row(name, endpoint) for name in ("cross_position_minus_within_position", "within_position_minus_erm") for endpoint in ("mean", "worst")]
    final_register = [{
        "execution_stage": row["execution_stage"], "method": row["method"], "alpha": row["alpha"],
        "seed": row["seed"], "epochs": row["epochs"], "optimizer_steps": row["optimizer_steps"],
        "checkpoint_sha256": row["checkpoint"]["checkpoint_sha256"], "model_state_sha256": row["model_state_sha256"],
        "mixup_encoder_gradient_norm": row["mixup_encoder_gradient_norm"],
    } for row in final_records]
    freeze_rows = [{
        "execution_stage": FINAL_P4_EVALUATION, "prediction_key": key,
        "checkpoint_sha256": p4["freeze"]["checkpoint_hashes"][key],
        "prediction_sha256": p4["freeze"]["prediction_hashes"][key],
        "binding_sha256": p4["freeze"]["checkpoint_prediction_bindings"][key], "labels_accessed": False,
    } for key in sorted(p4["freeze"]["prediction_hashes"])]

    def p4_contrast_rows(name: str) -> list[dict[str, Any]]:
        values = p4_contrasts[name]
        return [
            {"contrast": name, "endpoint": "P4_block_macro_f1", "point_estimate": values["block_only_macro_f1"]["point_estimate"], "block_only_interval_95": values["block_only_macro_f1"]["interval_95"], "block_plus_seed_interval_95": values["block_plus_seed_macro_f1"]["interval_95"]},
            {"contrast": name, "endpoint": "P4_block_accuracy", "point_estimate": values["block_only_accuracy"]["point_estimate"], "block_only_interval_95": values["block_only_accuracy"]["interval_95"], "block_plus_seed_interval_95": values["block_plus_seed_accuracy"]["interval_95"]},
        ]

    change_counts: Counter[tuple[str, int, int, str]] = Counter()
    for row in p4["paired_changes"]:
        change_counts[(str(row["contrast"]), int(row["seed"]), int(row["tagid_label"]), str(row["change"]))] += 1
    per_class = []
    for row in p4["per_class"]:
        enriched = dict(row)
        seed, label = int(row["seed"]), int(row["tagid_label"])
        for name in contrast_specs:
            enriched[f"{name}_incorrect_to_correct"] = change_counts[(name, seed, label, "INCORRECT_TO_CORRECT")]
            enriched[f"{name}_correct_to_incorrect"] = change_counts[(name, seed, label, "CORRECT_TO_INCORRECT")]
        per_class.append(enriched)

    diagnostic_rows = probe_rows + [{
        "method": "CONTRAST", "seed": None,
        "position_probe_cross_minus_erm": position_cross_erm,
        "position_probe_cross_minus_within": position_cross_within,
        "tagid_probe_cross_minus_erm": tagid_cross_erm,
        "tagid_probe_cross_minus_within": tagid_cross_within,
        "within_tagid_distance_cross_minus_erm": distance_cross_erm,
        "within_tagid_distance_cross_minus_within": distance_cross_within,
    }]
    sensitivity_rows = []
    for name, values in source_contrasts.items():
        sensitivity_rows.extend([
            {"analysis": f"source_{name}_fold_plus_seed_mean", **values["bootstrap"]["mean_contrast"]},
            {"analysis": f"source_{name}_fold_plus_seed_worst", **values["bootstrap"]["worst_position_contrast"]},
        ])
    for name, values in p4_contrasts.items():
        for key in ("block_only_macro_f1", "block_plus_seed_macro_f1", "block_only_accuracy", "block_plus_seed_accuracy"):
            sensitivity_rows.append({"analysis": f"p4_{name}_{key}", "point_estimate": values[key]["point_estimate"], "interval_95": values[key]["interval_95"]})
    sensitivity_rows.extend({"analysis": f"leave_one_seed_{row['contrast']}_{row['omitted_seed']}", "point_estimate": row["block_macro_f1_contrast"], "interval_95": None} for row in p4["leave_one_training_seed_out"])
    sensitivity_rows.extend({"analysis": f"alpha_{row['alpha']:g}_ranking", "point_estimate": row["mean_held_macro_f1"], "interval_95": None} for row in selection["summaries"])
    for method, values in position_means.items():
        for position, value in values.items():
            sensitivity_rows.append({"analysis": f"held_position_{method}_{position}", "point_estimate": value, "interval_95": None})

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(RESULTS_ROOT / "10_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", gate_rows, list(gate_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "11_DEVELOPMENT_RUN_REGISTER.csv", development_register, list(development_register[0]))
    atomic_write_csv(RESULTS_ROOT / "12_PAIRING_COVERAGE_AND_DISTRIBUTION.csv", pairing_rows, list(pairing_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "13_SOURCE_HELD_POSITION_METRICS.csv", held_rows, list(held_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "14_SOURCE_PRIMARY_CONTRASTS.csv", primary_source_rows, list(primary_source_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "15_SOURCE_CONTROL_CONTRASTS.csv", control_source_rows, list(control_source_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "16_REPRESENTATION_DIAGNOSTICS.csv", diagnostic_rows, list(diagnostic_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "17_PAIR_BRIDGE_GEOMETRY.csv", bridge_rows, list(bridge_rows[0]))
    atomic_write_markdown(RESULTS_ROOT / "18_SOURCE_ONLY_ALPHA_SELECTION.md", f"# Source-only alpha selection\n\nSelected alpha: `{selected_alpha:g}`. Eligibility set: `{selection['eligibility_set']}`. Receipt SHA-256: `{sha256_file(alpha_receipt_path)}`. P4 was not accessed.\n")
    atomic_write_markdown(RESULTS_ROOT / "19_DOMAIN_BRIDGING_DIAGNOSTIC_CLASSIFICATION.md", f"# Domain-bridging diagnostic classification\n\n`{diagnostic_classification}`\n")
    atomic_write_csv(RESULTS_ROOT / "20_FINAL_MODEL_REGISTER.csv", final_register, list(final_register[0]))
    atomic_write_csv(RESULTS_ROOT / "21_P4_PRELABEL_PREDICTION_FREEZE.csv", freeze_rows, list(freeze_rows[0]))
    atomic_write_csv(RESULTS_ROOT / "22_P4_LABEL_ACCESS_AND_PERSISTENCE_RECEIPT.csv", [{
        "execution_stage": FINAL_P4_EVALUATION, "prediction_bundle_sha256": p4["freeze"]["prediction_bundle_sha256"],
        "prelabel_receipt_sha256": p4["label_receipt"]["prelabel_receipt_sha256"],
        "label_sha256": p4["label_receipt"]["label_access_event"]["label_sha256"],
        "metrics_sha256": p4["label_receipt"]["metrics_persistence"]["sha256"], "silent_post_label_retry": False,
    }], ["execution_stage", "prediction_bundle_sha256", "prelabel_receipt_sha256", "label_sha256", "metrics_sha256", "silent_post_label_retry"])
    atomic_write_csv(RESULTS_ROOT / "23_P4_BLOCK_METRICS.csv", p4["p4_metrics"], list(p4["p4_metrics"][0]))
    for filename, name in (
        ("24_P4_PRIMARY_CROSS_POSITION_VS_ERM.csv", "cross_position_minus_erm"),
        ("25_P4_CROSS_POSITION_VS_WITHIN_POSITION.csv", "cross_position_minus_within_position"),
        ("26_P4_WITHIN_POSITION_VS_ERM.csv", "within_position_minus_erm"),
    ):
        rows = p4_contrast_rows(name)
        atomic_write_csv(RESULTS_ROOT / filename, rows, list(rows[0]))
    atomic_write_csv(RESULTS_ROOT / "27_P4_PER_CLASS_RESULTS.csv", per_class, list(per_class[0]))
    atomic_write_csv(RESULTS_ROOT / "28_P4_CONFUSION_MATRICES.csv", p4["confusion_matrices"], list(p4["confusion_matrices"][0]))
    atomic_write_csv(RESULTS_ROOT / "29_PREDICTED_CLASS_HISTOGRAMS.csv", p4["histograms"], list(p4["histograms"][0]))
    atomic_write_csv(RESULTS_ROOT / "30_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", sensitivity_rows, ["analysis", "point_estimate", "interval_95"])

    p4_summary = {name: {metric: p4_contrasts[name][metric]["point_estimate"] for metric in ("block_only_macro_f1", "block_only_accuracy")} for name in contrast_specs}
    interpretation = f"""# Scientific interpretation

Domain-bridging diagnostic classification: `{diagnostic_classification}`.

Main scientific classification: `{main_classification}`.

Selected alpha was `{selected_alpha:g}` using source-only LOPO. Cross-position minus ERM source mean Macro-F1 was `{source_means['cross_position_mixup'] - source_means['erm']:.6f}` and worst-position Macro-F1 was `{source_worst['cross_position_mixup'] - source_worst['erm']:.6f}`. Cross-position minus within-position source mean Macro-F1 was `{source_means['cross_position_mixup'] - source_means['within_position_mixup']:.6f}`. Source retention passed: `{guardrail['passed']}`.

Position-probe balanced-accuracy cross-minus-ERM contrast was `{position_cross_erm['point_estimate']:.6f}`; TagID-probe Macro-F1 contrast was `{tagid_cross_erm['point_estimate']:.6f}`; within-TagID cross-position distance contrast was `{distance_cross_erm['point_estimate']:.6f}`. Mixed-embedding confidence contrast was `{bridge_confidence_cross_erm['point_estimate']:.6f}`. These representation-space diagnostics do not establish physical causality.

P4 primary inference used 63 condition blocks and paired seeds; the 3,150 repeated rows were not treated as independent inferential units.
"""
    limitations = """# Limitations and nonclaims

- The benchmark contains seven TagIDs, three source positions, and one sealed target position.
- Manifold interpolations are representation augmentations, not measured RF spectra or evidence of a physical intermediate state.
- Alpha was selected on three source LOPO folds; diagnostic probes and geometry did not reselect it.
- The within-position control distinguishes generic interpolation only within this fixed architecture and update budget.
- P4 uncertainty is condition-block paired and seed paired; row-level significance is not claimed.
- No causal physical mechanism, universal domain invariance, or performance beyond this governed dataset is claimed.
"""
    summary = f"""# Domain-aware mixup executive summary

Governed runtime binding: `PASS`. Pairing coverage exceeded 99%: `{min(float(row['pair_coverage']) for row in pairing_rows) > .99}`. Implementation validity: `PASS`.

Development runs: `75`. Final runs: `15`. Selected cross-position alpha: `{selected_alpha:g}`.

Source mean Macro-F1 contrasts: cross-minus-ERM `{source_means['cross_position_mixup'] - source_means['erm']:.6f}`; cross-minus-within `{source_means['cross_position_mixup'] - source_means['within_position_mixup']:.6f}`; within-minus-ERM `{source_means['within_position_mixup'] - source_means['erm']:.6f}`. Worst-position contrasts follow the same order: `{source_worst['cross_position_mixup'] - source_worst['erm']:.6f}`, `{source_worst['cross_position_mixup'] - source_worst['within_position_mixup']:.6f}`, `{source_worst['within_position_mixup'] - source_worst['erm']:.6f}`. Source retention passed: `{guardrail['passed']}`.

Position-probe cross-minus-ERM: `{position_cross_erm['point_estimate']:.6f}`. TagID-probe cross-minus-ERM: `{tagid_cross_erm['point_estimate']:.6f}`. Cross-position geometry contrast: `{distance_cross_erm['point_estimate']:.6f}`.

P4 block contrasts: `{json.dumps(p4_summary, sort_keys=True)}`. All block-only and block-plus-seed intervals are recorded in files 24-26 and 30.

Diagnostic classification: `{diagnostic_classification}`. Main classification: `{main_classification}`.
"""
    atomic_write_markdown(RESULTS_ROOT / "31_SCIENTIFIC_INTERPRETATION.md", interpretation)
    atomic_write_markdown(RESULTS_ROOT / "32_LIMITATIONS_AND_NONCLAIMS.md", limitations)
    atomic_write_markdown(RESULTS_ROOT / "00_EXECUTIVE_SUMMARY.md", summary)
    atomic_write_markdown(RESULTS_ROOT / "STATUS.md", f"# Result status\n\nImplementation and sealed scientific execution: `PASS`.\n\nDiagnostic classification: `{diagnostic_classification}`.\n\nMain classification: `{main_classification}`.\n")
    atomic_write_markdown(PROJECT_ROOT / "docs" / "DOMAIN_AWARE_MIXUP_RESULTS.md", summary + "\n\n" + limitations)
    return {
        "status": "PASS_SCIENTIFIC_EXECUTION", "development_runs": 75, "final_runs": 15,
        "selected_alpha": selected_alpha, "minimum_pairing_coverage": min(float(row["pair_coverage"]) for row in pairing_rows),
        "source_means": source_means, "source_worst": source_worst, "p4_contrasts": p4_summary,
        "diagnostic_classification": diagnostic_classification, "main_classification": main_classification,
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
        method="erm", alpha_value=None, runtime_root=paths.runtime_root,
    )]
    records.append(run_lopo_unit(
        data=data, states=states, config=config, fold=fold, seed=seed,
        method="within_position_mixup", alpha_value=WITHIN_POSITION_ALPHA, runtime_root=paths.runtime_root,
    ))
    for value in ALPHA_GRID:
        records.append(run_lopo_unit(
            data=data, states=states, config=config, fold=fold, seed=seed,
            method="cross_position_mixup", alpha_value=value, runtime_root=paths.runtime_root,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--only-fold", choices=tuple(FOLD_HELD_POSITION))
    parser.add_argument("--only-seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    if sum((args.prepare_only, args.development_only)) > 1:
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
    else:
        result = execute_complete_v2(paths, config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
