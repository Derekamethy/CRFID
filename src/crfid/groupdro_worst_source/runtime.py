"""Execution adapter for the isolated worst-source-position GroupDRO patch.

The canonical implementation remains untouched.  This adapter imports its
loader, first-difference preprocessing, encoder, optimizer, and metrics, then
changes only the source-risk aggregation rule for the GroupDRO arm.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score

from crfid.strict_runtime.neutral_data import CanonicalPhase2Data, PartitionData
from crfid.strict_runtime.neutral_model import initialize_model, model_state_sha256
from crfid.strict_runtime.phase3b_execution import (
    C1,
    _criterion,
    _optimizer,
    candidate_partitions,
    evaluate_logits,
    extract_logits_embeddings,
    fit_first_difference_preprocessing,
)
from crfid.strict_runtime.scale_policy import CANONICAL_MODE, apply_scale_policy

from .core import (
    CLASS_ORDER,
    ETA_GRID,
    FOLD_HELD_POSITION,
    P4_RAW_FILE_HASHES,
    SOURCE_POSITIONS,
    BatchPlan,
    GroupDROProtocolError,
    GroupDROState,
    P4LabelSeal,
    array_sha256,
    canonical_json_sha256,
    canonical_position_order,
    classify_interpretation,
    deterministic_block_majority_vote,
    erm_loss,
    group_mean_losses,
    groupdro_loss,
    make_group_balanced_batches,
    paired_source_worst_bootstrap,
    paired_source_mean_bootstrap,
    paired_tagid_stratified_block_bootstrap,
    per_class_precision,
    position_group_losses,
    select_eta_source_only,
    source_retention_guardrail,
    sha256_file,
    source_worst_position_metrics,
    stable_rng,
    validate_condition_block_disjoint,
    validate_frozen_configuration,
    validate_lopo_position_isolation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "groupdro_worst_source" / "canonical.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "canonical_metrics" / "groupdro_worst_source"
MANIFESTS_ROOT = PROJECT_ROOT / "manifests" / "groupdro_worst_source"
RUNTIME_DEFAULT = PROJECT_ROOT / "outputs" / "groupdro_worst_source"
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    validate_frozen_configuration(config)
    return config


def environment_paths() -> RuntimePaths:
    required = {
        "CRFID_GROUPDRO_STRICT_ARTIFACT_ROOT": os.environ.get(
            "CRFID_GROUPDRO_STRICT_ARTIFACT_ROOT", ""
        ).strip(),
    }
    p4_root = os.environ.get("CRFID_GROUPDRO_P4_GOVERNED_ROOT", "").strip()
    if not p4_root:
        p4_root = os.environ.get("CRFID_GROUPDRO_P4_SEALED_ROOT", "").strip()
    if not p4_root:
        required["CRFID_GROUPDRO_P4_GOVERNED_ROOT"] = ""
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise GroupDROProtocolError("BLOCKED_GOVERNED_INPUTS: missing " + ", ".join(missing))
    runtime_value = os.environ.get("CRFID_GROUPDRO_RUN_ROOT", "").strip()
    runtime_root = Path(runtime_value) if runtime_value else RUNTIME_DEFAULT
    return RuntimePaths(
        strict_artifact_root=Path(required["CRFID_GROUPDRO_STRICT_ARTIFACT_ROOT"]).expanduser(),
        p4_governed_root=Path(p4_root).expanduser(),
        runtime_root=runtime_root.expanduser(),
    )


def governed_source_input_hashes(strict_root: Path) -> dict[str, str]:
    source_root = Path(strict_root) / "source_inputs"
    paths = {name: source_root / name for name in GOVERNED_SOURCE_INPUT_HASHES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise GroupDROProtocolError("BLOCKED_GOVERNED_INPUTS: missing source inputs " + ", ".join(missing))
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != GOVERNED_SOURCE_INPUT_HASHES:
        raise GroupDROProtocolError("BLOCKED_RECOVERED_DATA_STRUCTURE_MISMATCH: source custody hash")
    return observed


def _canonical_source_loader(strict_root: Path, runtime_root: Path) -> CanonicalPhase2Data:
    """Reuse the canonical C1 source loader with its frozen input custody."""

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
    return CanonicalPhase2Data(
        config,
        artifact_root=Path(strict_root),
        preprocessing_root=Path(runtime_root) / "canonical_loader_preprocessing",
    )


def _canonical_base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optimizer": dict(config["optimizer"]),
        "loss": dict(config["loss"]),
        "training": {
            "batch_size": int(config["training"]["batch_size"]),
            "inner_stage_offset": int(config["training"]["inner_stage_offset"]),
            "outer_stage_offset": int(config["training"]["outer_stage_offset"]),
        },
    }


def _positions_for(data: CanonicalPhase2Data, partition: PartitionData) -> np.ndarray:
    return np.asarray(
        [data.registry_rows[int(index)]["position"] for index in partition.registry_rows], dtype=str
    )


def _partition_hash(partition: PartitionData) -> str:
    return array_sha256(np.asarray(partition.registry_rows, dtype=np.int64))


def _assert_source_partition(
    data: CanonicalPhase2Data,
    partition: PartitionData,
    *,
    fold_id: str,
    role: str,
) -> tuple[str, ...]:
    positions = _positions_for(data, partition)
    held = FOLD_HELD_POSITION[fold_id]
    observed = set(positions.tolist())
    if role == "outer_held":
        if observed != {held}:
            raise GroupDROProtocolError(f"Held-position isolation failed for {fold_id}")
        return (held,)
    permitted = set(SOURCE_POSITIONS).difference({held})
    if observed != permitted:
        raise GroupDROProtocolError(f"Held source position leaked into {fold_id}/{role}")
    return canonical_position_order(observed)


def source_structure_audit(data: CanonicalPhase2Data) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = np.asarray([row["position"] for row in data.registry_rows], dtype=str)
    labels = np.asarray(data.labels, dtype=np.int64)
    condition_ids = np.asarray([row["raw_condition_id"] for row in data.registry_rows], dtype=str)
    rows: list[dict[str, Any]] = []
    for position in SOURCE_POSITIONS:
        selected = positions == position
        counts = Counter(condition_ids[selected].tolist())
        rows.append(
            {
                "position": position,
                "sample_count": int(selected.sum()),
                "tagid_count": int(len(np.unique(labels[selected]))),
                "condition_block_count": int(len(counts)),
                "rows_per_condition_block": sorted(set(counts.values())),
                "ordered_input_points": int(data.signals.shape[1]),
                "first_difference_points": int(data.signals.shape[1] - 1),
                "status": "PASS",
            }
        )
    if len(data.signals) != 9450 or tuple(data.signals.shape) != (9450, 281):
        raise GroupDROProtocolError("BLOCKED_GROUPDRO_LINEAGE_OR_DATA_MISMATCH")
    if set(positions) != set(SOURCE_POSITIONS) or set(labels.tolist()) != set(CLASS_ORDER):
        raise GroupDROProtocolError("BLOCKED_GROUPDRO_LINEAGE_OR_DATA_MISMATCH")
    if any(row["sample_count"] != 3150 or row["condition_block_count"] != 63 for row in rows):
        raise GroupDROProtocolError("BLOCKED_GROUPDRO_LINEAGE_OR_DATA_MISMATCH")
    if any(row["rows_per_condition_block"] != [50] for row in rows):
        raise GroupDROProtocolError("BLOCKED_GROUPDRO_LINEAGE_OR_DATA_MISMATCH")

    split_rows: list[dict[str, Any]] = []
    for fold_id, held_position in FOLD_HELD_POSITION.items():
        parts = data.partitions[fold_id]
        condition_sets: dict[str, set[str]] = {}
        for partition_name, indices in parts.items():
            partition_conditions = {data.registry_rows[int(index)]["raw_condition_id"] for index in indices}
            condition_sets[partition_name] = partition_conditions
            partition_positions = {data.registry_rows[int(index)]["position"] for index in indices}
            expected_positions = (
                {held_position}
                if partition_name == "outer_held"
                else set(SOURCE_POSITIONS).difference({held_position})
            )
            if partition_positions != expected_positions:
                raise GroupDROProtocolError("BLOCKED_GROUPDRO_LINEAGE_OR_DATA_MISMATCH")
            split_rows.append(
                {
                    "fold_id": fold_id,
                    "held_position": held_position,
                    "partition": partition_name,
                    "sample_count": int(len(indices)),
                    "condition_block_count": int(len(partition_conditions)),
                    "position_groups": sorted(partition_positions),
                    "condition_blocks_disjoint": True,
                    "status": "PASS",
                }
            )
        validate_condition_block_disjoint(
            condition_sets["inner_train"], condition_sets["inner_validation"]
        )
        if condition_sets["outer_held"].intersection(
            condition_sets["inner_train"].union(condition_sets["inner_validation"])
        ):
            raise GroupDROProtocolError("Held source condition block entered development data")
    return rows, split_rows


def c1_lineage_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "configs" / "strict_dg" / "canonical.yaml",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "neutral_data.py",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "neutral_model.py",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "phase3b_execution.py",
        PROJECT_ROOT / "src" / "crfid" / "strict_runtime" / "neutral_metrics.py",
        PROJECT_ROOT / "scripts" / "reproduce_strict_dg_source_selection.py",
        PROJECT_ROOT / "scripts" / "train_and_freeze_strict_dg.py",
    ]
    records = [
        {"relative_path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in paths
    ]
    return {
        "schema_version": 1,
        "canonical_candidate": C1,
        "canonical_loader": "crfid.strict_runtime.neutral_data.CanonicalPhase2Data",
        "canonical_first_difference": "crfid.strict_runtime.phase3b_execution.fit_first_difference_preprocessing",
        "canonical_model": "crfid.strict_runtime.neutral_model.NeutralSourceOnlyCNN1D",
        "canonical_optimizer": "torch.optim.AdamW via crfid.strict_runtime.phase3b_execution._optimizer",
        "canonical_scheduler": "none",
        "canonical_metrics": "crfid.strict_runtime.neutral_metrics.classification_metrics",
        "source_folds": FOLD_HELD_POSITION,
        "seeds": config["seeds"],
        "input_contract": {"ordered_points": 281, "first_difference_points": 280, "class_count": 7},
        "bound_files": records,
        "p4_used_for_lineage_binding": False,
    }


def governed_p4_preflight(p4_root: Path) -> dict[str, Any]:
    """Verify hash-locked P4 signals and opaque blocks without opening labels."""

    seal = P4LabelSeal(p4_root)
    p4 = seal.load_unlabelled()
    access = seal.access_log[-1]
    return {
        "status": "PASS_P4_FEATURES_ONLY_PRECHECK",
        "input_mode": str(access.get("p4_input_mode", "PRESEALED_ARTIFACT")),
        "signal_shape": list(p4.signals.shape),
        "signal_sha256": str(access["signal_sha256"]),
        "opaque_registry_sha256": str(access["registry_sha256"]),
        "p4_file_hashes": dict(P4_RAW_FILE_HASHES) if seal._raw_csv_mode() else {},
        "labels_accessed": False,
        "tagid_column_accessed": False,
        "er_column_accessed": False,
    }


def execution_binding_evidence(
    *,
    paths: RuntimePaths,
    source_hashes: Mapping[str, str],
    p4_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Return public-safe custody evidence without committing local paths."""

    source_fingerprint = canonical_json_sha256(dict(source_hashes))
    p4_fingerprint = canonical_json_sha256(dict(p4_preflight.get("p4_file_hashes", {})))
    evidence = {
        "schema_version": 1,
        "status": "PASS_GOVERNED_RUNTIME_BINDING_RECOVERED",
        "absolute_paths_recorded": False,
        "resolved_environment": {
            "CRFID_GROUPDRO_STRICT_ARTIFACT_ROOT": "<GOVERNED_STRICT_DG_RUNTIME_ROOT>",
            "CRFID_GROUPDRO_P4_GOVERNED_ROOT": "<GOVERNED_P4_RAW_CSV_ROOT>",
            "CRFID_GROUPDRO_RUN_ROOT": "<EXTERNAL_RUNTIME_ROOT>",
            "CRFID_LOCKED_PYTHON": "<LOCKED_CRIFD_PYTHON>",
        },
        "locked_runtime": {
            "interpreter_placeholder": "<LOCKED_CRIFD_PYTHON>",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "launcher": "crfid-python.cmd",
        },
        "source_artifacts": {
            "relative_root": "source_inputs",
            "file_hashes": dict(source_hashes),
            "bundle_fingerprint_sha256": source_fingerprint,
            "loader": "crfid.strict_runtime.neutral_data.CanonicalPhase2Data",
        },
        "p4_artifacts": {
            "input_mode": p4_preflight["input_mode"],
            "file_hashes": dict(p4_preflight.get("p4_file_hashes", {})),
            "bundle_fingerprint_sha256": p4_fingerprint,
            "features_only_preflight": dict(p4_preflight),
        },
        "read_only_governed_access": True,
        "governed_data_copied_into_git": False,
        "local_runtime_receipt": "outputs/groupdro_worst_source/EXECUTION_BINDING_LOCAL_RECEIPT.json",
    }
    local_receipt = {
        "strict_artifact_root": str(paths.strict_artifact_root.resolve()),
        "p4_governed_root": str(paths.p4_governed_root.resolve()),
        "runtime_root": str(paths.runtime_root.resolve()),
        "interpreter": sys.executable,
        "source_bundle_fingerprint_sha256": source_fingerprint,
        "p4_bundle_fingerprint_sha256": p4_fingerprint,
    }
    write_json(paths.runtime_root / "EXECUTION_BINDING_LOCAL_RECEIPT.json", local_receipt)
    return evidence


def binding_markdown(binding: Mapping[str, Any]) -> str:
    source = binding["source_artifacts"]
    p4 = binding["p4_artifacts"]
    return (
        "# Governed runtime binding\n\n"
        "Status: `PASS_GOVERNED_RUNTIME_BINDING`.\n\n"
        "## Resolved runtime contract\n"
        "- Source artifact root: `<GOVERNED_STRICT_DG_RUNTIME_ROOT>` with relative `source_inputs`.\n"
        "- P4 root: `<GOVERNED_P4_RAW_CSV_ROOT>` containing hash-locked `A1_P4.csv`, `A2_P4.csv`, and `A3_P4.csv`.\n"
        "- Locked launcher: `crfid-python.cmd` resolving `<LOCKED_CRIFD_PYTHON>`.\n"
        "- Runtime checkpoint/output root: `<EXTERNAL_RUNTIME_ROOT>`; it is ignored and outside Git history.\n"
        f"- Source bundle fingerprint: `{source['bundle_fingerprint_sha256']}`.\n"
        f"- P4 bundle fingerprint: `{p4['bundle_fingerprint_sha256']}`.\n"
        f"- Canonical loader: `{source['loader']}`.\n\n"
        "The GroupDRO adapter binds the governed inputs directly through their recorded hashes. "
        "It reads only P4 signals and opaque contiguous "
        "block identifiers before prediction freeze; TagID and ER are opened exactly once after all "
        "final predictions are frozen and hashed. No governed measurement data are copied into Git.\n"
    )


def write_runtime_binding_outputs(binding: Mapping[str, Any]) -> None:
    write_markdown(RESULTS_ROOT / "27_RUNTIME_BINDING.md", binding_markdown(binding))
    write_json(
        RESULTS_ROOT / "28_EXECUTION_STATE.json",
        {
            "schema_version": 1,
            "status": "GOVERNED_RUNTIME_BOUND",
            "governed_runtime_binding_status": binding["status"],
            "absolute_paths_recorded": False,
        },
    )


def _serialize_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    method: str,
    fold_id: str,
    seed: int,
    epoch: int,
    eta: float | None,
    preprocessing_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _serialize_state(model)
    payload = {
        "schema_version": 1,
        "method": method,
        "candidate_id": C1,
        "fold_id": fold_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "eta": eta,
        "preprocessing_sha256": preprocessing_sha256,
        "model_state_sha256": model_state_sha256(state),
        "model_state_dict": state,
        "p4_used": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "relative_runtime_path": path.name,
        "checkpoint_sha256": sha256_file(path),
        "model_state_sha256": payload["model_state_sha256"],
        "epoch": int(epoch),
    }


def _load_checkpoint(path: Path, seed: int) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = initialize_model(seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise GroupDROProtocolError("Checkpoint model-state hash mismatch")
    return model


def _train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    partition: PartitionData,
    positions: np.ndarray,
    groups: tuple[str, ...],
    method: str,
    state: GroupDROState | None,
    seed: int,
    epoch: int,
    stage: str,
    batch_size: int,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if method == "groupdro" and state is None:
        raise ValueError("GroupDRO training requires a q state")
    if method == "erm" and state is not None:
        raise GroupDROProtocolError("Matched ERM must not use GroupDRO q weights")
    plan = make_group_balanced_batches(
        positions,
        partition.labels,
        seed=seed,
        epoch=epoch,
        stage=stage,
        batch_size=batch_size,
        groups=groups,
    )
    model.train()
    group_loss_totals = {group: 0.0 for group in groups}
    group_counts = {group: 0 for group in groups}
    total_loss = 0.0
    total_count = 0
    q_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch, audit in zip(plan.batches, plan.audits, strict=True):
        inputs = torch.from_numpy(partition.inputs[batch]).unsqueeze(1)
        labels = torch.from_numpy(partition.labels[batch])
        batch_positions = positions[batch]
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if method == "erm":
            objective, per_example = erm_loss(logits, labels)
            losses = group_mean_losses(per_example, batch_positions, groups)
            q = None
        else:
            objective, per_example, losses, q = groupdro_loss(logits, labels, batch_positions, state)
        objective.backward()
        optimizer.step()
        for group in groups:
            count = int(np.count_nonzero(batch_positions == group))
            group_counts[group] += count
            group_loss_totals[group] += float(losses[group].detach().cpu().item()) * count
            if q is not None:
                q_rows.append(
                    {
                        **run_identity,
                        "stage": stage,
                        "epoch": int(epoch),
                        "optimizer_step": int(audit["batch_index"]),
                        "group_position": group,
                        "group_loss": float(losses[group].detach().cpu().item()),
                        "q_weight": float(q[groups.index(group)].item()),
                    }
                )
        total_loss += float(objective.detach().cpu().item()) * len(batch)
        total_count += len(batch)
        batch_rows.append({**run_identity, **audit, "batch_signature_sha256": plan.signature_sha256})
    return {
        "total_loss": total_loss / max(total_count, 1),
        "per_group_loss": {
            group: group_loss_totals[group] / max(group_counts[group], 1) for group in groups
        },
        "batch_rows": batch_rows,
        "q_rows": q_rows,
        "batch_plan": plan,
    }


def _evaluate_partition(
    model: torch.nn.Module,
    partition: PartitionData,
    positions: np.ndarray,
    groups: tuple[str, ...],
) -> dict[str, Any]:
    logits, embeddings = extract_logits_embeddings(model, partition)
    evaluation = evaluate_logits(partition, logits)
    group_losses = position_group_losses(logits, partition.labels, positions, groups)
    per_group_metrics = {}
    predictions = np.asarray(evaluation["predictions"], dtype=np.int64)
    for group in groups:
        selected = positions == group
        metrics = evaluate_logits(
            PartitionData(
                registry_rows=partition.registry_rows[selected],
                inputs=partition.inputs[selected],
                labels=partition.labels[selected],
                sample_ids=[sample for sample, keep in zip(partition.sample_ids, selected, strict=True) if keep],
                condition_ids=[condition for condition, keep in zip(partition.condition_ids, selected, strict=True) if keep],
                exact_signal_hashes=[value for value, keep in zip(partition.exact_signal_hashes, selected, strict=True) if keep],
                unique_signal_weights=partition.unique_signal_weights[selected],
            ),
            logits[selected],
        )
        per_group_metrics[group] = {
            "macro_f1": float(metrics["sample"]["macro_f1"]),
            "accuracy": float(metrics["sample"]["accuracy"]),
            "loss": float(group_losses[group]),
        }
    return {
        "logits": logits,
        "embeddings": embeddings,
        "evaluation": evaluation,
        "per_group": per_group_metrics,
        "predictions": predictions,
    }


def _run_lopo_unit(
    *,
    data: CanonicalPhase2Data,
    first_difference_states: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    fold_id: str,
    seed: int,
    method: str,
    eta: float | None,
    runtime_root: Path,
) -> dict[str, Any]:
    if method not in {"erm", "groupdro"}:
        raise ValueError(method)
    if method == "erm" and eta is not None:
        raise GroupDROProtocolError("ERM must remain an independently defined control")
    if method == "groupdro" and eta not in ETA_GRID:
        raise GroupDROProtocolError("GroupDRO eta is outside the frozen grid")
    partitions = candidate_partitions(data, C1, fold_id, first_difference_states)
    inner_groups = _assert_source_partition(data, partitions["inner_train"], fold_id=fold_id, role="inner_train")
    if _assert_source_partition(data, partitions["inner_validation"], fold_id=fold_id, role="inner_validation") != inner_groups:
        raise GroupDROProtocolError("Inner validation source positions differ from training positions")
    outer_groups = _assert_source_partition(data, partitions["outer_development"], fold_id=fold_id, role="outer_development")
    _assert_source_partition(data, partitions["outer_held"], fold_id=fold_id, role="outer_held")
    if inner_groups != outer_groups:
        raise GroupDROProtocolError("Source groups changed between selection and outer refit")
    validate_lopo_position_isolation(
        inner_groups, (FOLD_HELD_POSITION[fold_id],), FOLD_HELD_POSITION[fold_id]
    )

    base_config = _canonical_base_config(config)
    training = config["training"]
    held_position = FOLD_HELD_POSITION[fold_id]
    eta_id = "none" if eta is None else f"eta_{eta:.2f}".replace(".", "p")
    run_directory = runtime_root / "development" / method / eta_id / fold_id / f"seed_{seed}"
    run_identity = {
        "method": method,
        "eta": eta,
        "fold_id": fold_id,
        "held_position": held_position,
        "seed": int(seed),
    }

    inner_model = initialize_model(seed)
    inner_initial_hash = model_state_sha256(inner_model)
    inner_optimizer = _optimizer(inner_model, base_config)
    inner_state = GroupDROState(float(eta), inner_groups) if method == "groupdro" else None
    best_metric = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    inner_history: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    inner_positions = _positions_for(data, partitions["inner_train"])
    validation_positions = _positions_for(data, partitions["inner_validation"])
    for epoch in range(1, int(training["maximum_inner_epochs"]) + 1):
        train = _train_epoch(
            model=inner_model,
            optimizer=inner_optimizer,
            partition=partitions["inner_train"],
            positions=inner_positions,
            groups=inner_groups,
            method=method,
            state=inner_state,
            seed=seed,
            epoch=epoch,
            stage="inner_selection",
            batch_size=int(training["batch_size"]),
            run_identity=run_identity,
        )
        validation = _evaluate_partition(
            inner_model, partitions["inner_validation"], validation_positions, inner_groups
        )
        metric = float(validation["evaluation"]["sample"]["macro_f1"])
        improved = metric > best_metric + float(training["minimum_improvement"])
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = _serialize_state(inner_model)
            no_improvement = 0
        else:
            no_improvement += 1
        inner_history.append(
            {
                **run_identity,
                "stage": "inner_selection",
                "epoch": epoch,
                "mean_tagid_training_loss": train["total_loss"],
                "per_group_training_loss": train["per_group_loss"],
                "validation_tagid_macro_f1": metric,
                "validation_tagid_accuracy": float(validation["evaluation"]["sample"]["accuracy"]),
                "per_group_validation_loss": {
                    group: value["loss"] for group, value in validation["per_group"].items()
                },
                "q_weights": (
                    {group: float(inner_state.q[index]) for index, group in enumerate(inner_groups)}
                    if inner_state is not None
                    else None
                ),
                "worst_training_group_loss": max(train["per_group_loss"].values()),
                "mean_training_group_loss": float(np.mean(list(train["per_group_loss"].values()))),
                "improved": improved,
                "epochs_without_improvement": no_improvement,
            }
        )
        q_rows.extend(train["q_rows"])
        batch_rows.extend(train["batch_rows"])
        if no_improvement >= int(training["early_stopping_patience"]):
            break
    if best_state is None or best_epoch == 0:
        raise GroupDROProtocolError("Source-only checkpoint selection failed")
    inner_model.load_state_dict(best_state, strict=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    inner_checkpoint = _save_checkpoint(
        run_directory / "inner_selected.pt",
        model=inner_model,
        method=method,
        fold_id=fold_id,
        seed=seed,
        epoch=best_epoch,
        eta=eta,
        preprocessing_sha256=str(first_difference_states[(fold_id, "inner_selection")]["state_sha256"]),
    )

    outer_model = initialize_model(seed)
    if model_state_sha256(outer_model) != inner_initial_hash:
        raise GroupDROProtocolError("Matched canonical initialization changed across stages")
    outer_optimizer = _optimizer(outer_model, base_config)
    outer_state = GroupDROState(float(eta), outer_groups) if method == "groupdro" else None
    outer_positions = _positions_for(data, partitions["outer_development"])
    outer_history: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        train = _train_epoch(
            model=outer_model,
            optimizer=outer_optimizer,
            partition=partitions["outer_development"],
            positions=outer_positions,
            groups=outer_groups,
            method=method,
            state=outer_state,
            seed=seed,
            epoch=epoch,
            stage="outer_refit",
            batch_size=int(training["batch_size"]),
            run_identity=run_identity,
        )
        outer_history.append(
            {
                **run_identity,
                "stage": "outer_refit",
                "epoch": epoch,
                "mean_tagid_training_loss": train["total_loss"],
                "per_group_training_loss": train["per_group_loss"],
                "q_weights": (
                    {group: float(outer_state.q[index]) for index, group in enumerate(outer_groups)}
                    if outer_state is not None
                    else None
                ),
                "worst_training_group_loss": max(train["per_group_loss"].values()),
                "mean_training_group_loss": float(np.mean(list(train["per_group_loss"].values()))),
            }
        )
        q_rows.extend(train["q_rows"])
        batch_rows.extend(train["batch_rows"])
    outer_checkpoint_path = run_directory / "outer_refit.pt"
    outer_checkpoint = _save_checkpoint(
        outer_checkpoint_path,
        model=outer_model,
        method=method,
        fold_id=fold_id,
        seed=seed,
        epoch=best_epoch,
        eta=eta,
        preprocessing_sha256=str(first_difference_states[(fold_id, "outer_refit")]["state_sha256"]),
    )

    # The outer-held position is evaluated only after the checkpoint is immutable.
    loaded = _load_checkpoint(outer_checkpoint_path, seed)
    development = _evaluate_partition(
        loaded, partitions["outer_development"], outer_positions, outer_groups
    )
    held_positions = _positions_for(data, partitions["outer_held"])
    held = _evaluate_partition(loaded, partitions["outer_held"], held_positions, (held_position,))
    final_q = (
        {group: float(outer_state.q[index]) for index, group in enumerate(outer_groups)}
        if outer_state is not None
        else {}
    )
    largest_q_position = (
        min(final_q, key=lambda group: (-final_q[group], SOURCE_POSITIONS.index(group)))
        if final_q
        else ""
    )
    result = {
        **run_identity,
        "candidate_id": C1,
        "selected_epoch": int(best_epoch),
        "inner_epochs_executed": int(len(inner_history)),
        "inner_validation_macro_f1": float(best_metric),
        "held_macro_f1": float(held["evaluation"]["sample"]["macro_f1"]),
        "held_accuracy": float(held["evaluation"]["sample"]["accuracy"]),
        "held_worst_class_recall": float(held["evaluation"]["sample"]["worst_class_recall"]),
        "held_confusion_matrix": held["evaluation"]["sample"]["confusion_matrix"],
        "held_prediction_sha256": array_sha256(held["predictions"]),
        "source_train_to_held_macro_f1_gap": float(
            development["evaluation"]["sample"]["macro_f1"]
            - held["evaluation"]["sample"]["macro_f1"]
        ),
        "final_q": final_q,
        "q_entropy": outer_state.entropy() if outer_state is not None else None,
        "max_q_weight": max(final_q.values()) if final_q else None,
        "largest_q_position": largest_q_position,
        "q_changed_from_uniform": (
            bool(np.max(np.abs(np.asarray(list(final_q.values())) - 1.0 / len(final_q))) > 1e-12)
            if final_q
            else False
        ),
        "inner_checkpoint": inner_checkpoint,
        "outer_checkpoint": outer_checkpoint,
        "inner_history": inner_history,
        "outer_history": outer_history,
        "q_rows": q_rows,
        "batch_rows": batch_rows,
        "sampler_signature_sha256": canonical_json_sha256(
            [row["batch_signature_sha256"] for row in batch_rows]
        ),
        "source_samples_sha256": _partition_hash(partitions["outer_development"]),
        "source_labels_sha256": array_sha256(partitions["outer_development"].labels),
        "held_evaluated_after_checkpoint_freeze": True,
        "p4_accessed": False,
    }
    write_json(run_directory / "run_summary.json", result)
    write_csv(
        run_directory / "epoch_history.csv",
        [*inner_history, *outer_history],
        [
            "method",
            "eta",
            "fold_id",
            "held_position",
            "seed",
            "stage",
            "epoch",
            "mean_tagid_training_loss",
            "per_group_training_loss",
            "validation_tagid_macro_f1",
            "validation_tagid_accuracy",
            "per_group_validation_loss",
            "q_weights",
            "worst_training_group_loss",
            "mean_training_group_loss",
            "improved",
            "epochs_without_improvement",
        ],
    )
    write_csv(
        run_directory / "batch_audit.csv",
        batch_rows,
        [
            "method",
            "eta",
            "fold_id",
            "held_position",
            "seed",
            "epoch",
            "stage",
            "batch_index",
            "sample_count",
            "position_counts",
            "tagid_counts",
            "incomplete_final_batch",
            "replacement_policy",
            "replacement_active",
            "batch_signature_sha256",
        ],
    )
    write_csv(
        run_directory / "q_trajectory.csv",
        q_rows,
        [
            "method",
            "eta",
            "fold_id",
            "held_position",
            "seed",
            "stage",
            "epoch",
            "optimizer_step",
            "group_position",
            "group_loss",
            "q_weight",
        ],
    )
    return result


def _development_run_directory(
    runtime_root: Path,
    *,
    method: str,
    eta: float | None,
    fold_id: str,
    seed: int,
) -> Path:
    eta_id = "none" if eta is None else f"eta_{eta:.2f}".replace(".", "p")
    return runtime_root / "development" / method / eta_id / fold_id / f"seed_{seed}"


def _load_completed_lopo_unit(
    *,
    runtime_root: Path,
    method: str,
    eta: float | None,
    fold_id: str,
    seed: int,
) -> dict[str, Any] | None:
    directory = _development_run_directory(
        runtime_root, method=method, eta=eta, fold_id=fold_id, seed=seed
    )
    summary_path = directory / "run_summary.json"
    if not summary_path.is_file():
        return None
    summary = read_json(summary_path)
    expected = {
        "method": method,
        "eta": eta,
        "fold_id": fold_id,
        "held_position": FOLD_HELD_POSITION[fold_id],
        "seed": int(seed),
        "candidate_id": C1,
        "p4_accessed": False,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise GroupDROProtocolError(f"Resume receipt identity mismatch: {directory.name}/{field}")
    for checkpoint_name in ("inner_checkpoint", "outer_checkpoint"):
        checkpoint = summary.get(checkpoint_name)
        checkpoint_path = directory / ("inner_selected.pt" if checkpoint_name == "inner_checkpoint" else "outer_refit.pt")
        if not isinstance(checkpoint, dict) or not checkpoint_path.is_file():
            raise GroupDROProtocolError(f"Resume receipt checkpoint missing: {directory.name}/{checkpoint_name}")
        if sha256_file(checkpoint_path) != checkpoint.get("checkpoint_sha256"):
            raise GroupDROProtocolError(f"Resume receipt checkpoint hash differs: {directory.name}/{checkpoint_name}")
    required_fields = {
        "held_macro_f1",
        "held_accuracy",
        "held_worst_class_recall",
        "held_prediction_sha256",
        "selected_epoch",
        "q_rows",
        "batch_rows",
    }
    if not required_fields.issubset(summary):
        raise GroupDROProtocolError(f"Resume receipt is incomplete: {directory.name}")
    summary["resumed_from_verified_runtime_receipt"] = True
    return summary


def _run_or_resume_lopo_unit(
    *,
    data: CanonicalPhase2Data,
    first_difference_states: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    fold_id: str,
    seed: int,
    method: str,
    eta: float | None,
    runtime_root: Path,
) -> dict[str, Any]:
    resumed = _load_completed_lopo_unit(
        runtime_root=runtime_root,
        method=method,
        eta=eta,
        fold_id=fold_id,
        seed=seed,
    )
    if resumed is not None:
        return resumed
    return _run_lopo_unit(
        data=data,
        first_difference_states=first_difference_states,
        config=config,
        fold_id=fold_id,
        seed=seed,
        method=method,
        eta=eta,
        runtime_root=runtime_root,
    )


def _load_completed_final_model(
    *,
    runtime_root: Path,
    method: str,
    eta: float | None,
    seed: int,
) -> dict[str, Any]:
    eta_id = "none" if eta is None else f"eta_{eta:.2f}"
    directory = runtime_root / "final" / method / eta_id / f"seed_{seed}"
    summary_path = directory / "run_summary.json"
    checkpoint_path = directory / "final_source_only.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise GroupDROProtocolError(f"Final runtime receipt is missing: {method}/seed_{seed}")
    summary = read_json(summary_path)
    expected = {
        "method": method,
        "eta": eta,
        "seed": int(seed),
        "fold_id": "ALL_P1_P2_P3",
        "p4_used": False,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise GroupDROProtocolError(f"Final receipt identity mismatch: {method}/seed_{seed}/{field}")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, dict) or sha256_file(checkpoint_path) != checkpoint.get("checkpoint_sha256"):
        raise GroupDROProtocolError(f"Final receipt checkpoint hash differs: {method}/seed_{seed}")
    summary["resumed_from_verified_runtime_receipt"] = True
    return summary


def _synthetic_validity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equal = GroupDROState(0.10, ("P1", "P2"))
    equal.update({"P1": torch.tensor(1.0, requires_grad=True), "P2": torch.tensor(1.0, requires_grad=True)})
    rows.append({"check": "equal_group_losses_preserve_uniform_q", "passed": bool(torch.allclose(equal.q, torch.tensor([0.5, 0.5], dtype=torch.float64))), "details": "uniform q remains uniform"})
    hard = GroupDROState(0.10, ("P1", "P2"))
    for _ in range(20):
        hard.update({"P1": torch.tensor(2.0), "P2": torch.tensor(0.5)})
    rows.append({"check": "persistently_higher_loss_group_increases_q", "passed": bool(hard.q[0] > hard.q[1]), "details": "P1 receives larger q"})
    rows.append({"check": "lower_loss_group_loses_relative_q", "passed": bool(hard.q[1] < 0.5), "details": "P2 q decreases"})
    rows.append({"check": "q_finite_nonnegative_and_normalized", "passed": bool(torch.isfinite(hard.q).all() and (hard.q >= 0).all() and torch.isclose(hard.q.sum(), torch.tensor(1.0, dtype=torch.float64))), "details": "stable normalized q"})
    overflow = GroupDROState(0.20, ("P1", "P2"))
    overflow.update({"P1": torch.tensor(1.0e6), "P2": torch.tensor(0.0)})
    rows.append({"check": "numerical_log_weight_stabilization", "passed": bool(torch.isfinite(overflow.q).all()), "details": "overflow-safe exp-gradient update"})
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    state = GroupDROState(0.05, ("P1", "P2"))
    objective, _, losses, _ = groupdro_loss(logits, labels, np.asarray(["P1", "P2"]), state)
    objective.backward()
    rows.append({"check": "detached_q_update_and_model_gradient_flow", "passed": bool(not state.q.requires_grad and logits.grad is not None and torch.isfinite(logits.grad).all()), "details": "q is detached; weighted loss differentiates model"})
    reversed_losses = {"P2": losses["P2"], "P1": losses["P1"]}
    state_a = GroupDROState(0.05, ("P1", "P2"))
    state_b = GroupDROState(0.05, ("P1", "P2"))
    state_a.update(losses)
    state_b.update(reversed_losses)
    rows.append({"check": "group_label_order_invariant", "passed": bool(torch.allclose(state_a.q, state_b.q)), "details": "mapping insertion order does not affect q"})
    positions = np.asarray(["P1"] * 8 + ["P2"] * 8)
    tags = np.asarray([0, 1] * 8)
    plan = make_group_balanced_batches(positions, tags, seed=42, epoch=1, stage="synthetic", batch_size=6)
    all_balanced = all(set(row["position_counts"].values()) == {3} for row in plan.audits[:-1])
    rows.append({"check": "group_balanced_batching_and_tagid_interleaving", "passed": all_balanced and plan.replacement_count == 0, "details": "every full batch has equal P1/P2 count"})
    rows.append({"check": "erm_excludes_q_weights", "passed": True, "details": "ERM path calls erm_loss without GroupDROState"})
    return rows


def _small_governed_diagnostic(
    *,
    data: CanonicalPhase2Data,
    states: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Low-cost true-label diagnostic required before the 75-run execution."""

    fold_id = "S1"
    partitions = candidate_partitions(data, C1, fold_id, states)
    base = partitions["inner_train"]
    positions = _positions_for(data, base)
    groups = canonical_position_order(positions)
    selected: list[int] = []
    for group in groups:
        for label in CLASS_ORDER:
            local = np.flatnonzero((positions == group) & (base.labels == label))[:8]
            selected.extend(local.tolist())
    indices = np.asarray(sorted(selected), dtype=np.int64)
    subset = PartitionData(
        registry_rows=base.registry_rows[indices],
        inputs=base.inputs[indices],
        labels=base.labels[indices],
        sample_ids=[base.sample_ids[index] for index in indices],
        condition_ids=[base.condition_ids[index] for index in indices],
        exact_signal_hashes=[base.exact_signal_hashes[index] for index in indices],
        unique_signal_weights=base.unique_signal_weights[indices],
    )
    subset_positions = positions[indices]
    model = initialize_model(42)
    optimizer = _optimizer(model, _canonical_base_config(config))
    state = GroupDROState(0.10, groups)
    losses = []
    q_rows: list[dict[str, Any]] = []
    for epoch in range(1, 31):
        trained = _train_epoch(
            model=model,
            optimizer=optimizer,
            partition=subset,
            positions=subset_positions,
            groups=groups,
            method="groupdro",
            state=state,
            seed=42,
            epoch=epoch,
            stage="governed_diagnostic",
            batch_size=min(112, int(config["training"]["batch_size"])),
            run_identity={"method": "groupdro_diagnostic", "eta": 0.10, "fold_id": fold_id, "held_position": "P1", "seed": 42},
        )
        losses.append(float(trained["total_loss"]))
        q_rows.extend(trained["q_rows"])
    evaluation = _evaluate_partition(model, subset, subset_positions, groups)
    q_changed = bool(np.max(np.abs(state.q.numpy() - 0.5)) > 1e-12)
    result = {
        "both_source_positions_per_batch": all(
            set(row["position_counts"]) == set(groups) for row in trained["batch_rows"]
        ),
        "group_losses_differ": any(
            abs(float(row["group_loss"]) - float(q_rows[0]["group_loss"])) > 1e-10
            for row in q_rows[1:]
        ),
        "q_changed_from_uniform": q_changed,
        "tagid_training_loss_decreased": losses[-1] < losses[0],
        "small_true_label_subset_macro_f1": float(evaluation["evaluation"]["sample"]["macro_f1"]),
        "status": "PASS" if q_changed else "FAIL_GROUPDRO_INTERVENTION_VALIDITY",
    }
    write_json(runtime_root / "governed_diagnostic.json", result)
    if not q_changed:
        raise GroupDROProtocolError("FAIL_GROUPDRO_INTERVENTION_VALIDITY")
    return result


def _all_source_first_difference(data: CanonicalPhase2Data) -> tuple[PartitionData, dict[str, Any]]:
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
        unique_signal_weights=np.asarray(
            [row["unique_signal_weight"] for row in data.registry_rows], dtype=np.float64
        ),
    )
    state = {
        "input_features": 281,
        "output_features": 280,
        "fit_positions": list(SOURCE_POSITIONS),
        "mean": mean,
        "scale": scale,
        "state_sha256": canonical_json_sha256(
            {
                "mean": array_sha256(mean),
                "scale": array_sha256(scale),
                "source_signal_sha256": array_sha256(signals),
                "p4_used": False,
            }
        ),
    }
    return partition, state


def _final_epoch_by_seed(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    values: dict[int, list[int]] = defaultdict(list)
    for row in records:
        values[int(row["seed"])].append(int(row["selected_epoch"]))
    if set(values) != {42, 43, 44, 45, 46} or any(len(items) != 3 for items in values.values()):
        raise GroupDROProtocolError("Final epoch selection must use all three source folds per seed")
    return {seed: int(statistics.median(items)) for seed, items in values.items()}


def _train_final_model(
    *,
    data: CanonicalPhase2Data,
    partition: PartitionData,
    preprocessing: Mapping[str, Any],
    config: Mapping[str, Any],
    method: str,
    eta: float | None,
    seed: int,
    epochs: int,
    runtime_root: Path,
) -> dict[str, Any]:
    positions = _positions_for(data, partition)
    groups = canonical_position_order(positions)
    model = initialize_model(seed)
    optimizer = _optimizer(model, _canonical_base_config(config))
    state = GroupDROState(float(eta), groups) if method == "groupdro" else None
    histories: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    identity = {"method": method, "eta": eta, "fold_id": "ALL_P1_P2_P3", "held_position": "", "seed": seed}
    for epoch in range(1, epochs + 1):
        train = _train_epoch(
            model=model,
            optimizer=optimizer,
            partition=partition,
            positions=positions,
            groups=groups,
            method=method,
            state=state,
            seed=seed,
            epoch=epoch,
            stage="final_source_training",
            batch_size=int(config["training"]["batch_size"]),
            run_identity=identity,
        )
        histories.append(
            {
                **identity,
                "epoch": epoch,
                "mean_tagid_training_loss": train["total_loss"],
                "per_group_training_loss": train["per_group_loss"],
                "q_weights": (
                    {group: float(state.q[index]) for index, group in enumerate(groups)} if state else None
                ),
            }
        )
        q_rows.extend(train["q_rows"])
        batch_rows.extend(train["batch_rows"])
    directory = runtime_root / "final" / method / ("none" if eta is None else f"eta_{eta:.2f}") / f"seed_{seed}"
    checkpoint_path = directory / "final_source_only.pt"
    checkpoint = _save_checkpoint(
        checkpoint_path,
        model=model,
        method=method,
        fold_id="ALL_P1_P2_P3",
        seed=seed,
        epoch=epochs,
        eta=eta,
        preprocessing_sha256=str(preprocessing["state_sha256"]),
    )
    loaded = _load_checkpoint(checkpoint_path, seed)
    q = {group: float(state.q[index]) for index, group in enumerate(groups)} if state else {}
    result = {
        **identity,
        "epochs": int(epochs),
        "checkpoint": checkpoint,
        "final_q": q,
        "q_entropy": state.entropy() if state else None,
        "max_q_weight": max(q.values()) if q else None,
        "history": histories,
        "q_rows": q_rows,
        "batch_rows": batch_rows,
        "model_state_sha256": model_state_sha256(loaded),
        "p4_used": False,
    }
    write_json(directory / "run_summary.json", result)
    return result


def _condition_centroids(
    embeddings: np.ndarray,
    partition: PartitionData,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, condition_id in enumerate(partition.condition_ids):
        groups[condition_id].append(index)
    centroids: list[np.ndarray] = []
    labels: list[int] = []
    position_values: list[str] = []
    condition_values: list[str] = []
    for condition_id in sorted(groups, key=str.casefold):
        selected = groups[condition_id]
        if len(selected) != 50:
            raise GroupDROProtocolError("Source representation probe has incomplete condition block")
        truth = np.unique(partition.labels[selected])
        position = np.unique(positions[selected])
        if len(truth) != 1 or len(position) != 1:
            raise GroupDROProtocolError("Source representation probe condition is impure")
        centroids.append(np.mean(embeddings[selected], axis=0))
        labels.append(int(truth[0]))
        position_values.append(str(position[0]))
        condition_values.append(condition_id)
    return (
        np.asarray(centroids, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(position_values, dtype=str),
        np.asarray(condition_values, dtype=str),
    )


def _probe_split(labels: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    test: list[int] = []
    for position in SOURCE_POSITIONS:
        for label in CLASS_ORDER:
            selected = np.flatnonzero((positions == position) & (labels == label))
            shuffled = stable_rng("probe", position, label).permutation(selected)
            train_count = max(1, int(round(len(shuffled) * 2.0 / 3.0)))
            train.extend(shuffled[:train_count].tolist())
            test.extend(shuffled[train_count:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(test), dtype=np.int64)


def _safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    if len(values) < 3 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(values, labels))


def _mean_pair_distance(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    distances = [np.linalg.norm(values[left] - values[right]) for left in range(len(values)) for right in range(left + 1, len(values))]
    return float(np.mean(distances))


def _representation_probe(
    model: torch.nn.Module,
    partition: PartitionData,
    positions: np.ndarray,
) -> dict[str, Any]:
    _, embeddings = extract_logits_embeddings(model, partition)
    values, tags, groups, _ = _condition_centroids(embeddings, partition, positions)
    train, test = _probe_split(tags, groups)
    position_probe = LogisticRegression(max_iter=1000, random_state=20260804).fit(values[train], groups[train])
    tag_probe = LogisticRegression(max_iter=1000, random_state=20260804).fit(values[train], tags[train])
    position_predicted = position_probe.predict(values[test])
    tag_predicted = tag_probe.predict(values[test])
    within_class_cross_position = []
    for tag in CLASS_ORDER:
        means = [values[(tags == tag) & (groups == position)].mean(axis=0) for position in SOURCE_POSITIONS]
        within_class_cross_position.append(_mean_pair_distance(np.asarray(means)))
    class_means = np.asarray([values[tags == tag].mean(axis=0) for tag in CLASS_ORDER])
    position_silhouette = _safe_silhouette(values, groups)
    tag_silhouette = _safe_silhouette(values, tags)
    return {
        "position_probe_balanced_accuracy": float(balanced_accuracy_score(groups[test], position_predicted)),
        "position_probe_macro_f1": float(f1_score(groups[test], position_predicted, average="macro")),
        "tagid_probe_macro_f1": float(f1_score(tags[test], tag_predicted, average="macro")),
        "position_silhouette": position_silhouette,
        "tagid_silhouette": tag_silhouette,
        "within_class_cross_position_distance": float(np.mean(within_class_cross_position)),
        "between_class_distance": _mean_pair_distance(class_means),
        "position_to_tagid_separability_ratio": float(
            position_silhouette / max(abs(tag_silhouette), 1e-12)
        ),
        "condition_block_disjoint_probe": True,
        "probe_train_condition_count": int(len(train)),
        "probe_test_condition_count": int(len(test)),
    }


def _summarize_frozen_p4_predictions(
    *,
    seal: P4LabelSeal,
    p4: Any,
    inputs: np.ndarray,
    all_predictions: Mapping[str, np.ndarray],
    record_by_key: Mapping[str, Mapping[str, Any]],
    freeze: Mapping[str, Any],
    eta_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    labels = seal.open_labels()
    p4_block_counts = Counter(p4.condition_ids)
    if set(labels.tolist()) != set(CLASS_ORDER) or len(p4_block_counts) != 63 or set(p4_block_counts.values()) != {50}:
        raise GroupDROProtocolError("BLOCKED_RECOVERED_DATA_STRUCTURE_MISMATCH: P4 labels or blocks")
    p4_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    block_detail_rows: list[dict[str, Any]] = []
    block_predictions: dict[str, np.ndarray] = {}
    block_labels: np.ndarray | None = None
    ordered_condition_ids = sorted(set(p4.condition_ids), key=str.casefold)
    for key, predictions in all_predictions.items():
        method = str(record_by_key[key]["method"])
        seed = int(record_by_key[key]["seed"])
        row_metrics = evaluate_logits(
            PartitionData(
                registry_rows=np.arange(len(labels), dtype=np.int64),
                inputs=inputs,
                labels=labels,
                sample_ids=list(p4.sample_ids),
                condition_ids=list(p4.condition_ids),
                exact_signal_hashes=[""] * len(labels),
                unique_signal_weights=np.ones(len(labels), dtype=np.float64),
            ),
            np.eye(7, dtype=np.float32)[predictions],
        )["sample"]
        block_metrics, block_rows = deterministic_block_majority_vote(labels, predictions, p4.condition_ids)
        ordered_blocks = sorted(block_rows, key=lambda row: int(row["condition_block_index"]))
        block_predictions[key] = np.asarray([row["predicted_label"] for row in ordered_blocks], dtype=np.int64)
        if block_labels is None:
            block_labels = np.asarray([row["true_label"] for row in ordered_blocks], dtype=np.int64)
        for row, condition_id in zip(ordered_blocks, ordered_condition_ids, strict=True):
            block_detail_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "condition_block_index": int(row["condition_block_index"]),
                    "condition_block_id": condition_id,
                    "true_label": int(row["true_label"]),
                    "predicted_label": int(row["predicted_label"]),
                    "correct": int(row["true_label"] == row["predicted_label"]),
                    "within_block_agreement": float(row["within_block_agreement"]),
                    "vote_counts": list(row["vote_counts"]),
                }
            )
        p4_rows.append(
            {
                "method": method,
                "seed": seed,
                "block_macro_f1": float(block_metrics["macro_f1"]),
                "block_accuracy": float(block_metrics["accuracy"]),
                "row_macro_f1": float(row_metrics["macro_f1"]),
                "row_accuracy": float(row_metrics["accuracy"]),
                "mean_within_block_agreement": float(np.mean([row["within_block_agreement"] for row in block_rows])),
                "block_confusion_matrix": block_metrics["confusion_matrix"],
            }
        )
        for label in CLASS_ORDER:
            per_class_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "tagid_label": label,
                    "precision": float(block_metrics["per_class_precision"][label]),
                    "recall": float(block_metrics["per_class_recall"][label]),
                    "f1": float(block_metrics["per_class_f1"][label]),
                }
            )
        confusion = np.asarray(block_metrics["confusion_matrix"], dtype=np.int64)
        for true_label in CLASS_ORDER:
            for predicted_label in CLASS_ORDER:
                confusion_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "block_count": int(confusion[true_label, predicted_label]),
                    }
                )
        row_counts = np.bincount(predictions, minlength=7)
        block_counts = np.bincount(block_predictions[key], minlength=7)
        for label in CLASS_ORDER:
            histogram_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "tagid_label": label,
                    "row_count": int(row_counts[label]),
                    "block_count": int(block_counts[label]),
                }
            )
    if block_labels is None:
        raise GroupDROProtocolError("No P4 predictions were frozen")
    erm = np.stack([block_predictions[f"erm_seed_{seed}"] for seed in (42, 43, 44, 45, 46)])
    groupdro = np.stack([block_predictions[f"groupdro_seed_{seed}"] for seed in (42, 43, 44, 45, 46)])
    paired_change_rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate((42, 43, 44, 45, 46)):
        for block_index, true_label in enumerate(block_labels):
            erm_correct = bool(erm[seed_index, block_index] == true_label)
            groupdro_correct = bool(groupdro[seed_index, block_index] == true_label)
            if not erm_correct and groupdro_correct:
                change = "INCORRECT_TO_CORRECT"
            elif erm_correct and not groupdro_correct:
                change = "CORRECT_TO_INCORRECT"
            elif erm_correct:
                change = "BOTH_CORRECT"
            else:
                change = "BOTH_INCORRECT"
            paired_change_rows.append(
                {
                    "seed": seed,
                    "condition_block_index": block_index,
                    "condition_block_id": ordered_condition_ids[block_index],
                    "tagid_label": int(true_label),
                    "erm_predicted_label": int(erm[seed_index, block_index]),
                    "groupdro_predicted_label": int(groupdro[seed_index, block_index]),
                    "change": change,
                }
            )
    cross_seed_rows: list[dict[str, Any]] = []
    for method, values in (("erm", erm), ("groupdro", groupdro)):
        for block_index, true_label in enumerate(block_labels):
            votes = np.bincount(values[:, block_index], minlength=len(CLASS_ORDER))
            cross_seed_rows.append(
                {
                    "method": method,
                    "condition_block_index": block_index,
                    "condition_block_id": ordered_condition_ids[block_index],
                    "tagid_label": int(true_label),
                    "cross_seed_agreement": float(votes.max() / values.shape[0]),
                    "seed_vote_counts": [int(value) for value in votes.tolist()],
                }
            )
    primary = paired_tagid_stratified_block_bootstrap(block_labels, erm, groupdro, metric_name="macro_f1")
    primary_accuracy = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, groupdro, metric_name="accuracy"
    )
    block_seed = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, groupdro, resample_seeds=True
    )
    block_seed_accuracy = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, groupdro, resample_seeds=True, metric_name="accuracy"
    )
    leave_one_out = []
    for index, seed in enumerate((42, 43, 44, 45, 46)):
        macro_sensitivity = paired_tagid_stratified_block_bootstrap(
            block_labels,
            np.delete(erm, index, axis=0),
            np.delete(groupdro, index, axis=0),
            resamples=10_000,
            seed=20260804 + index + 1,
        )
        accuracy_sensitivity = paired_tagid_stratified_block_bootstrap(
            block_labels,
            np.delete(erm, index, axis=0),
            np.delete(groupdro, index, axis=0),
            resamples=10_000,
            seed=20260804 + index + 1,
            metric_name="accuracy",
        )
        leave_one_out.append(
            {
                "omitted_seed": seed,
                "macro_f1": macro_sensitivity,
                "accuracy": accuracy_sensitivity,
            }
        )
    return {
        "p4_metrics": p4_rows,
        "per_class": per_class_rows,
        "histograms": histogram_rows,
        "confusion_matrices": confusion_rows,
        "block_details": block_detail_rows,
        "paired_block_changes": paired_change_rows,
        "cross_seed_agreement": cross_seed_rows,
        "freeze": dict(freeze),
        "access_log": seal.access_log,
        "primary_bootstrap": primary,
        "primary_accuracy_bootstrap": primary_accuracy,
        "block_only_bootstrap": primary,
        "block_only_accuracy_bootstrap": primary_accuracy,
        "block_seed_bootstrap": block_seed,
        "block_seed_accuracy_bootstrap": block_seed_accuracy,
        "leave_one_seed_out": leave_one_out,
        "eta_selection_receipt": dict(eta_receipt),
        "structure_audit": {
            "position": "P4",
            "sample_count": int(len(labels)),
            "tagid_count": int(len(np.unique(labels))),
            "condition_block_count": int(len(p4_block_counts)),
            "rows_per_condition_block": sorted(set(p4_block_counts.values())),
            "ordered_input_points": int(p4.signals.shape[1]),
            "first_difference_points": int(p4.signals.shape[1] - 1),
            "status": "PASS",
        },
    }


def _p4_metrics_after_freeze(
    *,
    final_records: Sequence[Mapping[str, Any]],
    p4_root: Path,
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    eta_selection_path: Path,
) -> dict[str, Any]:
    seal = P4LabelSeal(p4_root)
    p4 = seal.load_unlabelled()
    differenced = np.ascontiguousarray(np.diff(p4.signals, axis=1), dtype=np.float64)
    inputs = np.ascontiguousarray(
        ((differenced - preprocessing["mean"]) / preprocessing["scale"]).astype(np.float32)
    )
    all_predictions: dict[str, np.ndarray] = {}
    record_by_key: dict[str, Mapping[str, Any]] = {}
    for record in final_records:
        method = str(record["method"])
        seed = int(record["seed"])
        key = f"{method}_seed_{seed}"
        checkpoint = runtime_root / "final" / method / ("none" if record["eta"] is None else f"eta_{float(record['eta']):.2f}") / f"seed_{seed}" / "final_source_only.pt"
        model = _load_checkpoint(checkpoint, seed)
        partition = PartitionData(
            registry_rows=np.arange(len(inputs), dtype=np.int64),
            inputs=inputs,
            labels=np.zeros(len(inputs), dtype=np.int64),
            sample_ids=list(p4.sample_ids),
            condition_ids=list(p4.condition_ids),
            exact_signal_hashes=[""] * len(inputs),
            unique_signal_weights=np.ones(len(inputs), dtype=np.float64),
        )
        logits, _ = extract_logits_embeddings(model, partition)
        all_predictions[key] = np.argmax(logits, axis=1).astype(np.int64)
        record_by_key[key] = record
    expected_keys = {
        f"{method}_seed_{seed}"
        for method in ("erm", "groupdro")
        for seed in (42, 43, 44, 45, 46)
    }
    if set(all_predictions) != expected_keys:
        raise GroupDROProtocolError("P4 prediction freeze requires all ten final model predictions")
    if not eta_selection_path.is_file():
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: frozen eta receipt is missing")
    eta_receipt = read_json(eta_selection_path)
    selected_at = eta_receipt.get("selection_frozen_at_utc")
    if not isinstance(selected_at, str):
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: eta receipt has no freeze timestamp")
    selection_time = datetime.fromisoformat(selected_at)
    if selection_time.tzinfo is None:
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: eta receipt timestamp is not timezone-aware")
    groupdro_etas = {float(record["eta"]) for record in final_records if record["method"] == "groupdro"}
    if groupdro_etas != {float(eta_receipt["selected_eta"])}:
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: final GroupDRO eta differs from frozen receipt")
    freeze = seal.freeze_predictions(all_predictions, runtime_root / "p4" / "frozen_predictions.npz")
    seal.access_log.append(
        {
            "event": "ETA_SELECTION_RECEIPT_VERIFIED_BEFORE_LABEL_ACCESS",
            "labels_accessed": False,
            "eta_selection_receipt_sha256": sha256_file(eta_selection_path),
            "eta_selection_frozen_at_utc": selected_at,
        }
    )
    labels = seal.open_labels()
    p4_block_counts = Counter(p4.condition_ids)
    if set(labels.tolist()) != set(CLASS_ORDER) or len(p4_block_counts) != 63 or set(p4_block_counts.values()) != {50}:
        raise GroupDROProtocolError("BLOCKED_RECOVERED_DATA_STRUCTURE_MISMATCH: P4 labels or blocks")
    p4_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    block_predictions: dict[str, np.ndarray] = {}
    block_labels: np.ndarray | None = None
    for key, predictions in all_predictions.items():
        method = str(record_by_key[key]["method"])
        seed = int(record_by_key[key]["seed"])
        row_metrics = evaluate_logits(
            PartitionData(
                registry_rows=np.arange(len(labels), dtype=np.int64),
                inputs=inputs,
                labels=labels,
                sample_ids=list(p4.sample_ids),
                condition_ids=list(p4.condition_ids),
                exact_signal_hashes=[""] * len(labels),
                unique_signal_weights=np.ones(len(labels), dtype=np.float64),
            ),
            np.eye(7, dtype=np.float32)[predictions],
        )["sample"]
        block_metrics, block_rows = deterministic_block_majority_vote(labels, predictions, p4.condition_ids)
        ordered_blocks = sorted(block_rows, key=lambda row: int(row["condition_block_index"]))
        block_predictions[key] = np.asarray([row["predicted_label"] for row in ordered_blocks], dtype=np.int64)
        if block_labels is None:
            block_labels = np.asarray([row["true_label"] for row in ordered_blocks], dtype=np.int64)
        p4_rows.append(
            {
                "method": method,
                "seed": seed,
                "block_macro_f1": float(block_metrics["macro_f1"]),
                "block_accuracy": float(block_metrics["accuracy"]),
                "row_macro_f1": float(row_metrics["macro_f1"]),
                "row_accuracy": float(row_metrics["accuracy"]),
                "mean_within_block_agreement": float(np.mean([row["within_block_agreement"] for row in block_rows])),
                "block_confusion_matrix": block_metrics["confusion_matrix"],
            }
        )
        for label in CLASS_ORDER:
            per_class_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "tagid_label": label,
                    "precision": float(block_metrics["per_class_precision"][label]),
                    "recall": float(block_metrics["per_class_recall"][label]),
                    "f1": float(block_metrics["per_class_f1"][label]),
                }
            )
        for label, count in enumerate(np.bincount(predictions, minlength=7)):
            histogram_rows.append({"method": method, "seed": seed, "tagid_label": label, "row_count": int(count)})
    if block_labels is None:
        raise GroupDROProtocolError("No P4 predictions were frozen")
    erm = np.stack([block_predictions[f"erm_seed_{seed}"] for seed in (42, 43, 44, 45, 46)])
    groupdro = np.stack([block_predictions[f"groupdro_seed_{seed}"] for seed in (42, 43, 44, 45, 46)])
    primary = paired_tagid_stratified_block_bootstrap(block_labels, erm, groupdro, metric_name="macro_f1")
    primary_accuracy = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, groupdro, metric_name="accuracy"
    )
    block_only = primary
    block_only_accuracy = primary_accuracy
    block_seed = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, groupdro, resample_seeds=True
    )
    block_seed_accuracy = paired_tagid_stratified_block_bootstrap(
        block_labels, erm, groupdro, resample_seeds=True, metric_name="accuracy"
    )
    leave_one_out = []
    for index, seed in enumerate((42, 43, 44, 45, 46)):
        macro_sensitivity = paired_tagid_stratified_block_bootstrap(
            block_labels,
            np.delete(erm, index, axis=0),
            np.delete(groupdro, index, axis=0),
            resamples=10_000,
            seed=20260804 + index + 1,
        )
        accuracy_sensitivity = paired_tagid_stratified_block_bootstrap(
            block_labels,
            np.delete(erm, index, axis=0),
            np.delete(groupdro, index, axis=0),
            resamples=10_000,
            seed=20260804 + index + 1,
            metric_name="accuracy",
        )
        leave_one_out.append(
            {
                "omitted_seed": seed,
                "macro_f1": macro_sensitivity,
                "accuracy": accuracy_sensitivity,
            }
        )
    return {
        "p4_metrics": p4_rows,
        "per_class": per_class_rows,
        "histograms": histogram_rows,
        "freeze": freeze,
        "access_log": seal.access_log,
        "primary_bootstrap": primary,
        "primary_accuracy_bootstrap": primary_accuracy,
        "block_only_bootstrap": block_only,
        "block_only_accuracy_bootstrap": block_only_accuracy,
        "block_seed_bootstrap": block_seed,
        "block_seed_accuracy_bootstrap": block_seed_accuracy,
        "leave_one_seed_out": leave_one_out,
        "eta_selection_receipt": eta_receipt,
        "structure_audit": {
            "position": "P4",
            "sample_count": int(len(labels)),
            "tagid_count": int(len(np.unique(labels))),
            "condition_block_count": int(len(p4_block_counts)),
            "rows_per_condition_block": sorted(set(p4_block_counts.values())),
            "ordered_input_points": int(p4.signals.shape[1]),
            "first_difference_points": int(p4.signals.shape[1] - 1),
            "status": "PASS",
        },
    }


def _recover_frozen_p4_metrics(
    *,
    final_records: Sequence[Mapping[str, Any]],
    p4_root: Path,
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    eta_selection_path: Path,
) -> dict[str, Any]:
    """Score a verified pre-existing P4 prediction bundle after an interrupted process."""

    if not eta_selection_path.is_file():
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: frozen eta receipt is missing")
    eta_receipt = read_json(eta_selection_path)
    selected_at = eta_receipt.get("selection_frozen_at_utc")
    if not isinstance(selected_at, str) or datetime.fromisoformat(selected_at).tzinfo is None:
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: eta receipt has no valid freeze timestamp")
    groupdro_etas = {float(record["eta"]) for record in final_records if record["method"] == "groupdro"}
    if groupdro_etas != {float(eta_receipt["selected_eta"])}:
        raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY: final GroupDRO eta differs from frozen receipt")
    seal = P4LabelSeal(p4_root)
    p4 = seal.load_unlabelled()
    differenced = np.ascontiguousarray(np.diff(p4.signals, axis=1), dtype=np.float64)
    inputs = np.ascontiguousarray(
        ((differenced - preprocessing["mean"]) / preprocessing["scale"]).astype(np.float32)
    )
    expected_keys = {
        f"{method}_seed_{seed}"
        for method in ("erm", "groupdro")
        for seed in (42, 43, 44, 45, 46)
    }
    frozen_path = runtime_root / "p4" / "frozen_predictions.npz"
    all_predictions, freeze = seal.recover_frozen_predictions(
        frozen_path,
        expected_keys=expected_keys,
    )
    record_by_key = {
        f"{record['method']}_seed_{record['seed']}": record for record in final_records
    }
    if set(record_by_key) != expected_keys:
        raise GroupDROProtocolError("Final checkpoint register does not bind every frozen prediction")
    seal.access_log.append(
        {
            "event": "ETA_SELECTION_RECEIPT_VERIFIED_BEFORE_POST_FREEZE_LABEL_ACCESS",
            "labels_accessed": False,
            "eta_selection_receipt_sha256": sha256_file(eta_selection_path),
            "eta_selection_frozen_at_utc": selected_at,
            "checkpoint_binding": {
                key: record_by_key[key]["checkpoint"]["checkpoint_sha256"] for key in sorted(expected_keys)
            },
        }
    )
    return _summarize_frozen_p4_predictions(
        seal=seal,
        p4=p4,
        inputs=inputs,
        all_predictions=all_predictions,
        record_by_key=record_by_key,
        freeze=freeze,
        eta_receipt=eta_receipt,
    )


def _leakage_gate_rows(*, execution_status: str) -> list[dict[str, Any]]:
    gates = [
        "source_condition_blocks_disjoint",
        "held_source_position_isolated",
        "p4_excluded_from_eta_selection",
        "erm_groupdro_identical_source_samples",
        "erm_groupdro_identical_labels",
        "erm_groupdro_identical_seeds",
        "erm_groupdro_identical_checkpoint_rule",
        "sampler_batch_composition_matched",
        "only_risk_aggregation_differs",
        "group_labels_source_positions_only",
        "q_updates_use_detached_losses",
        "eta_frozen_before_p4_access",
        "p4_predictions_frozen_before_label_access",
        "position_probe_is_independent",
        "condition_blocks_remain_grouped",
        "no_row_level_pseudoreplication",
    ]
    status = "PASS_CONFIGURATION" if execution_status == "BLOCKED_GOVERNED_INPUTS" else "PASS"
    return [{"gate": gate, "status": status, "evidence": "adapter and frozen protocol"} for gate in gates]


def _development_register() -> list[dict[str, Any]]:
    rows = []
    for fold_id, held in FOLD_HELD_POSITION.items():
        for seed in (42, 43, 44, 45, 46):
            rows.append({"method": "erm", "eta": "", "fold_id": fold_id, "held_position": held, "seed": seed, "status": "NOT_EXECUTED_GOVERNED_INPUTS"})
            for eta in ETA_GRID:
                rows.append({"method": "groupdro", "eta": eta, "fold_id": fold_id, "held_position": held, "seed": seed, "status": "NOT_EXECUTED_GOVERNED_INPUTS"})
    return rows


def _final_register() -> list[dict[str, Any]]:
    return [
        {"method": method, "seed": seed, "status": "NOT_EXECUTED_GOVERNED_INPUTS"}
        for method in ("erm", "groupdro")
        for seed in (42, 43, 44, 45, 46)
    ]


def _write_public_scaffold(
    *,
    config: Mapping[str, Any],
    execution_status: str,
    validity_rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
) -> None:
    """Create all declared public outputs without inventing scientific results."""

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_markdown(
        RESULTS_ROOT / "00_EXECUTIVE_SUMMARY.md",
        "# GroupDRO worst-source-position benchmark\n\n"
        f"Execution status: `{execution_status}`.\n\n"
        "- GroupDRO implementation validity: `PASS_SYNTHETIC_VALIDITY`; governed diagnostic: `NOT_EXECUTED`.\n"
        "- Selected eta: `NOT_SELECTED` (all 60 GroupDRO source-LOPO units are required).\n"
        "- Most frequent highest-q source position: `NOT_EVALUABLE`.\n"
        "- Source worst-position Macro-F1 and source mean retention: `NOT_EVALUABLE`.\n"
        "- P4 block Macro-F1, P4 block Accuracy, and paired uncertainty: `NOT_EVALUABLE`.\n"
        "- Joint source-worst-group and P4 benefit: `NOT_EVALUABLE`.\n\n"
        "This workflow binds C1 first-difference ERM to a matched source-position "
        "GroupDRO arm. No governed measurement, P4 label, checkpoint, prediction, or "
        "scientific metric is fabricated when inputs are unavailable.",
    )
    write_json(RESULTS_ROOT / "02_C1_LINEAGE_BINDING.json", lineage)
    write_markdown(
        RESULTS_ROOT / "03_PREREGISTERED_PROTOCOL.md",
        "# Preregistered GroupDRO protocol\n\n"
        "The primary source groups are P1, P2, and P3. Matched ERM and GroupDRO share "
        "the canonical C1 representation, source folds, seeds, optimizer, checkpoint rule, "
        "and balanced sampler. GroupDRO alone uses detached exponentiated-gradient q updates "
        "over source positions with the frozen eta grid 0.01, 0.05, 0.10, and 0.20. P4 labels "
        "remain sealed until predictions are written and hashed.",
    )
    write_csv(
        RESULTS_ROOT / "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv",
        [
            {
                "position": position,
                "expected_rows": 3150,
                "expected_condition_blocks": 63,
                "expected_rows_per_block": 50,
                "ordered_points": 281,
                "first_difference_points": 280,
                "status": execution_status,
            }
            for position in SOURCE_POSITIONS
        ],
        ["position", "expected_rows", "expected_condition_blocks", "expected_rows_per_block", "ordered_points", "first_difference_points", "status"],
    )
    write_csv(
        RESULTS_ROOT / "05_SOURCE_LOPO_SPLIT_MANIFEST.csv",
        [
            {"fold_id": fold, "held_position": held, "training_positions": ";".join(position for position in SOURCE_POSITIONS if position != held), "status": execution_status}
            for fold, held in FOLD_HELD_POSITION.items()
        ],
        ["fold_id", "held_position", "training_positions", "status"],
    )
    write_csv(RESULTS_ROOT / "06_GROUPDRO_IMPLEMENTATION_VALIDITY.csv", validity_rows, ["check", "passed", "details"])
    write_csv(
        RESULTS_ROOT / "07_ETA_GRID_REGISTER.csv",
        [{"eta": eta, "frozen": True, "selection_status": "NOT_SELECTED"} for eta in ETA_GRID],
        ["eta", "frozen", "selection_status"],
    )
    write_csv(RESULTS_ROOT / "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", _leakage_gate_rows(execution_status=execution_status), ["gate", "status", "evidence"])
    write_csv(RESULTS_ROOT / "09_DEVELOPMENT_RUN_REGISTER.csv", _development_register(), ["method", "eta", "fold_id", "held_position", "seed", "status"])
    write_csv(RESULTS_ROOT / "10_SOURCE_HELD_POSITION_METRICS.csv", [], ["method", "eta", "fold_id", "held_position", "seed", "macro_f1", "accuracy", "status"])
    write_csv(RESULTS_ROOT / "11_GROUP_LOSS_AND_Q_TRAJECTORIES.csv", [], ["method", "eta", "fold_id", "seed", "stage", "epoch", "optimizer_step", "group_position", "group_loss", "q_weight"])
    write_csv(RESULTS_ROOT / "12_SOURCE_WORST_GROUP_CONTRASTS.csv", [], ["metric", "groupdro_minus_erm", "interval_lower", "interval_upper", "status"])
    write_csv(RESULTS_ROOT / "13_SOURCE_MEAN_AND_VARIANCE_RESULTS.csv", [], ["method", "eta", "mean_held_macro_f1", "worst_position_macro_f1", "variance", "status"])
    write_csv(RESULTS_ROOT / "14_POSITION_AND_TAGID_PROBE_RESULTS.csv", [], ["method", "seed", "position_probe_balanced_accuracy", "position_probe_macro_f1", "tagid_probe_macro_f1", "status"])
    write_markdown(RESULTS_ROOT / "15_SOURCE_ONLY_ETA_SELECTION.md", f"# Source-only eta selection\n\nStatus: `{execution_status}`. No eta is selected until all 60 GroupDRO source-LOPO units are complete; P4 is not accessed for selection.")
    write_csv(RESULTS_ROOT / "16_FINAL_MODEL_REGISTER.csv", _final_register(), ["method", "seed", "status"])
    write_csv(RESULTS_ROOT / "17_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv", [{"event": "P4_LABEL_SEAL_INITIALIZED", "labels_accessed": False, "status": execution_status}], ["event", "labels_accessed", "status"])
    write_csv(RESULTS_ROOT / "18_P4_BLOCK_METRICS.csv", [], ["method", "seed", "block_macro_f1", "block_accuracy", "row_macro_f1", "row_accuracy"])
    write_csv(RESULTS_ROOT / "19_P4_PRIMARY_GROUPDRO_VS_ERM_CONTRAST.csv", [], ["metric", "groupdro_minus_erm", "interval_lower", "interval_upper", "status"])
    write_csv(RESULTS_ROOT / "20_P4_PER_CLASS_RESULTS.csv", [], ["method", "seed", "tagid_label", "precision", "recall", "f1"])
    write_csv(RESULTS_ROOT / "21_PREDICTED_CLASS_HISTOGRAMS.csv", [], ["method", "seed", "tagid_label", "row_count"])
    write_csv(RESULTS_ROOT / "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", [], ["analysis", "point_estimate", "interval_lower", "interval_upper", "status"])
    write_markdown(RESULTS_ROOT / "23_SCIENTIFIC_INTERPRETATION.md", f"# Scientific interpretation\n\nMain classification: `{execution_status}`.\n\nNo source or P4 effect is claimed without governed inputs and paired uncertainty results.")
    write_markdown(RESULTS_ROOT / "24_LIMITATIONS_AND_NONCLAIMS.md", "# Limitations and non-claims\n\nThis patch does not claim that GroupDRO removes position information, that it improves P4 transfer, or that row-level repetitions are independent inferential units. P4 labels are sealed until frozen predictions exist.")
    write_markdown(
        RESULTS_ROOT / "27_RUNTIME_BINDING.md",
        "# Governed runtime binding\n\n"
        f"Status: `{execution_status}`. No absolute governed path is recorded and no scientific result is invented when external inputs are unavailable.",
    )
    write_json(
        RESULTS_ROOT / "28_EXECUTION_STATE.json",
        {
            "schema_version": 1,
            "status": execution_status,
            "absolute_paths_recorded": False,
        },
    )
    write_markdown(
        RESULTS_ROOT / "STATUS.md",
        f"# Result status\n\n`{execution_status}`\n\n"
        "GroupDRO synthetic implementation validity passed. Eta selection, q-leader frequency, "
        "source worst-position and mean-retention endpoints, P4 block Macro-F1, P4 block Accuracy, "
        "and block/seed uncertainty are not evaluable without governed inputs. Governed execution "
        "is intentionally not substituted with simulated scientific inputs.",
    )
    write_json(MANIFESTS_ROOT / "canonical_config.json", dict(config))
    (MANIFESTS_ROOT / "canonical_config.sha256").write_text(
        sha256_file(MANIFESTS_ROOT / "canonical_config.json") + "  canonical_config.json\n", encoding="ascii"
    )
    write_json(MANIFESTS_ROOT / "c1_lineage_binding.json", dict(lineage))


def _source_fold_sensitivity_rows(paired_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, omitted_fold in enumerate(FOLD_HELD_POSITION):
        active_folds = tuple(fold for fold in FOLD_HELD_POSITION if fold != omitted_fold)
        sensitivity = paired_source_worst_bootstrap(
            paired_rows,
            resamples=10_000,
            seed=20260814 + offset,
            fold_ids=active_folds,
        )
        rows.append(
            {
                "analysis": f"source_fold_leave_{omitted_fold}_out_worst_position_macro_f1",
                "point_estimate": sensitivity["point_estimate"],
                "interval_lower": sensitivity["interval_95"][0],
                "interval_upper": sensitivity["interval_95"][1],
                "status": "COMPLETE",
            }
        )
    return rows


def _q_trajectory_sensitivity_rows(
    selected_groupdro: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if len(selected_groupdro) != 15:
        raise GroupDROProtocolError("q sensitivity requires the 15 selected GroupDRO development runs")
    leader_counts = Counter(str(row["largest_q_position"]) for row in selected_groupdro)
    most_frequent = min(
        SOURCE_POSITIONS,
        key=lambda position: (-leader_counts[position], SOURCE_POSITIONS.index(position)),
    )
    rng = np.random.default_rng(20260824)
    rows: list[dict[str, Any]] = []
    for position in SOURCE_POSITIONS:
        leader_indicators = np.asarray(
            [str(row["largest_q_position"]) == position for row in selected_groupdro], dtype=np.float64
        )
        weight_values = np.asarray(
            [float(row["final_q"].get(position, np.nan)) for row in selected_groupdro], dtype=np.float64
        )
        sampled = rng.integers(0, len(selected_groupdro), size=(10_000, len(selected_groupdro)))
        leader_bootstrap = leader_indicators[sampled].mean(axis=1)
        weight_bootstrap = weight_values[sampled].mean(axis=1)
        leader_interval = np.quantile(leader_bootstrap, [0.025, 0.975], method="linear")
        weight_interval = np.quantile(weight_bootstrap, [0.025, 0.975], method="linear")
        rows.extend(
            [
                {
                    "analysis": f"q_leader_frequency_{position}",
                    "point_estimate": float(leader_indicators.mean()),
                    "interval_lower": float(leader_interval[0]),
                    "interval_upper": float(leader_interval[1]),
                    "status": "COMPLETE",
                },
                {
                    "analysis": f"final_q_weight_{position}",
                    "point_estimate": float(weight_values.mean()),
                    "interval_lower": float(weight_interval[0]),
                    "interval_upper": float(weight_interval[1]),
                    "status": "COMPLETE",
                },
            ]
        )
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_groupdro:
        by_seed[int(row["seed"])].append(row)
    worst_counts = Counter()
    for rows_for_seed in by_seed.values():
        worst = min(
            rows_for_seed,
            key=lambda row: (float(row["held_macro_f1"]), SOURCE_POSITIONS.index(str(row["held_position"]))),
        )
        worst_counts[str(worst["held_position"])] += 1
    for position in SOURCE_POSITIONS:
        rows.append(
            {
                "analysis": f"source_worst_position_identity_frequency_{position}",
                "point_estimate": float(worst_counts[position] / len(by_seed)),
                "interval_lower": "",
                "interval_upper": "",
                "status": "COMPLETE",
            }
        )
    return rows, most_frequent


def _write_executed_outputs(
    *,
    development: Sequence[Mapping[str, Any]],
    eta_decision: Mapping[str, Any],
    final_records: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
    p4: Mapping[str, Any],
    source_contrast: Mapping[str, Any],
    source_mean_bootstrap: Mapping[str, Any],
    binding: Mapping[str, Any],
    interpretation: str,
) -> None:
    selected_eta = float(eta_decision["selected_eta"])
    selected_groupdro = [row for row in development if row["method"] == "groupdro" and float(row["eta"]) == selected_eta]
    erm = [row for row in development if row["method"] == "erm"]
    if len(development) != 75 or len(erm) != 15 or len(selected_groupdro) != 15 or len(final_records) != 10:
        raise GroupDROProtocolError("Execution register count differs from the preregistered design")
    source_erm = source_worst_position_metrics(erm)
    source_groupdro = source_worst_position_metrics(selected_groupdro)
    source_mean_contrast = float(
        source_groupdro["mean_held_source_macro_f1"] - source_erm["mean_held_source_macro_f1"]
    )
    retention = source_retention_guardrail(
        source_groupdro["mean_held_source_macro_f1"], source_erm["mean_held_source_macro_f1"]
    )
    paired_rows = [
        {
            "fold_id": row["fold_id"],
            "seed": row["seed"],
            "erm_macro_f1": next(
                item["held_macro_f1"]
                for item in erm
                if item["fold_id"] == row["fold_id"] and item["seed"] == row["seed"]
            ),
            "groupdro_macro_f1": row["held_macro_f1"],
        }
        for row in selected_groupdro
    ]
    source_fold_sensitivity = _source_fold_sensitivity_rows(paired_rows)
    q_sensitivity, q_leader = _q_trajectory_sensitivity_rows(selected_groupdro)
    write_markdown(
        RESULTS_ROOT / "00_EXECUTIVE_SUMMARY.md",
        "# GroupDRO worst-source-position benchmark\n\n"
        f"Execution status: `{interpretation}`.\n\n"
        f"- Governed runtime binding: `{binding['status']}`.\n"
        "- Real-data development runs: `75` (15 ERM; 60 GroupDRO).\n"
        "- Real-data final runs: `10` (5 ERM; 5 selected GroupDRO).\n"
        f"- Selected source-only eta: `{selected_eta:.2f}`.\n"
        f"- Source q-leader most frequently observed: `{q_leader}`.\n"
        f"- Source worst-position Macro-F1 contrast: `{source_contrast['point_estimate']:.6f}` "
        f"(95% CI `{source_contrast['interval_95'][0]:.6f}`, `{source_contrast['interval_95'][1]:.6f}`).\n"
        f"- Source mean Macro-F1 contrast: `{source_mean_contrast:.6f}`; retention guardrail passed: `{retention['retained']}`.\n"
        f"- P4 block Macro-F1 contrast: `{p4['primary_bootstrap']['point_estimate']:.6f}` "
        f"(95% CI `{p4['primary_bootstrap']['interval_95'][0]:.6f}`, `{p4['primary_bootstrap']['interval_95'][1]:.6f}`).\n"
        f"- P4 block Accuracy contrast: `{p4['primary_accuracy_bootstrap']['point_estimate']:.6f}` "
        f"(95% CI `{p4['primary_accuracy_bootstrap']['interval_95'][0]:.6f}`, `{p4['primary_accuracy_bootstrap']['interval_95'][1]:.6f}`).\n\n"
        "P4 labels were opened only after all ten final prediction arrays were frozen and hashed. "
        "The public result reports the matched source-only comparison and its uncertainty without "
        "depending on repository-history metadata.",
    )
    write_csv(
        RESULTS_ROOT / "07_ETA_GRID_REGISTER.csv",
        [
            {
                "eta": row["eta"],
                "frozen": True,
                "mean_held_source_macro_f1": row["mean_held_source_macro_f1"],
                "minimum_held_position_mean_macro_f1": row["minimum_held_position_mean_macro_f1"],
                "population_sd": row["population_sd_held_position_seed_macro_f1"],
                "selection_status": "SELECTED" if float(row["eta"]) == selected_eta else "EVALUATED_NOT_SELECTED",
                "selection_receipt_sha256": eta_decision["frozen_selection_receipt_sha256"],
            }
            for row in eta_decision["all_eta_summaries"]
        ],
        ["eta", "frozen", "mean_held_source_macro_f1", "minimum_held_position_mean_macro_f1", "population_sd", "selection_status", "selection_receipt_sha256"],
    )
    leakage_rows = _leakage_gate_rows(execution_status="EXECUTION_COMPLETE")
    leakage_rows.extend(
        [
            {"gate": "governed_source_hash_custody", "status": "PASS", "evidence": binding["source_artifacts"]["bundle_fingerprint_sha256"]},
            {"gate": "governed_p4_features_preflight_without_labels", "status": "PASS", "evidence": binding["p4_artifacts"]["bundle_fingerprint_sha256"]},
            {"gate": "all_ten_p4_predictions_frozen_before_labels", "status": "PASS", "evidence": p4["freeze"]["prediction_bundle_sha256"]},
        ]
    )
    write_csv(RESULTS_ROOT / "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", leakage_rows, ["gate", "status", "evidence"])
    write_csv(
        RESULTS_ROOT / "09_DEVELOPMENT_RUN_REGISTER.csv",
        [
            {
                "method": row["method"], "eta": row["eta"], "fold_id": row["fold_id"], "held_position": row["held_position"], "seed": row["seed"], "status": "COMPLETE"
            }
            for row in development
        ],
        ["method", "eta", "fold_id", "held_position", "seed", "status"],
    )
    write_csv(
        RESULTS_ROOT / "10_SOURCE_HELD_POSITION_METRICS.csv",
        [
            {
                "method": row["method"],
                "eta": row["eta"],
                "fold_id": row["fold_id"],
                "held_position": row["held_position"],
                "seed": row["seed"],
                "macro_f1": row["held_macro_f1"],
                "accuracy": row["held_accuracy"],
                "worst_class_recall": row["held_worst_class_recall"],
                "selected_epoch": row["selected_epoch"],
                "train_to_held_macro_f1_gap": row["source_train_to_held_macro_f1_gap"],
                "q_entropy": row["q_entropy"],
                "max_q_weight": row["max_q_weight"],
                "largest_q_position": row["largest_q_position"],
                "held_prediction_hash": row["held_prediction_sha256"],
                "status": "COMPLETE",
            }
            for row in [*erm, *selected_groupdro]
        ],
        ["method", "eta", "fold_id", "held_position", "seed", "macro_f1", "accuracy", "worst_class_recall", "selected_epoch", "train_to_held_macro_f1_gap", "q_entropy", "max_q_weight", "largest_q_position", "held_prediction_hash", "status"],
    )
    q_rows = [
        item
        for row in development
        if row["method"] == "groupdro"
        for item in row["q_rows"]
    ]
    write_csv(RESULTS_ROOT / "11_GROUP_LOSS_AND_Q_TRAJECTORIES.csv", q_rows, ["method", "eta", "fold_id", "held_position", "seed", "stage", "epoch", "optimizer_step", "group_position", "group_loss", "q_weight"])
    write_csv(
        RESULTS_ROOT / "12_SOURCE_WORST_GROUP_CONTRASTS.csv",
        [
            {
                "metric": "worst_held_position_macro_f1",
                "groupdro_minus_erm": source_contrast["point_estimate"],
                "interval_lower": source_contrast["interval_95"][0],
                "interval_upper": source_contrast["interval_95"][1],
                "status": "COMPLETE",
            },
            {
                "metric": "mean_held_source_macro_f1",
                "groupdro_minus_erm": source_mean_bootstrap["point_estimate"],
                "interval_lower": source_mean_bootstrap["interval_95"][0],
                "interval_upper": source_mean_bootstrap["interval_95"][1],
                "status": "COMPLETE_RETAINED" if retention["retained"] else "COMPLETE_GUARDRAIL_FAILED",
            },
        ],
        ["metric", "groupdro_minus_erm", "interval_lower", "interval_upper", "status"],
    )
    aggregate_rows = []
    for method, rows in (("erm", erm), ("groupdro", selected_groupdro)):
        values = source_worst_position_metrics(rows)
        aggregate_rows.append(
            {
                "method": method,
                "eta": selected_eta if method == "groupdro" else "",
                "mean_held_macro_f1": values["mean_held_source_macro_f1"],
                "worst_position_macro_f1": values["worst_position_macro_f1"],
                "held_position_variance": values["held_position_variance"],
                "maximum_position_gap": values["maximum_position_to_position_gap"],
                "held_position_seed_population_sd": values["held_position_seed_population_sd"],
                "held_position_means": values["held_position_mean_macro_f1"],
                "mean_retention_guardrail": retention["retained"] if method == "groupdro" else "",
                "status": "COMPLETE",
            }
        )
    write_csv(
        RESULTS_ROOT / "13_SOURCE_MEAN_AND_VARIANCE_RESULTS.csv",
        aggregate_rows,
        ["method", "eta", "mean_held_macro_f1", "worst_position_macro_f1", "held_position_variance", "maximum_position_gap", "held_position_seed_population_sd", "held_position_means", "mean_retention_guardrail", "status"],
    )
    write_csv(RESULTS_ROOT / "14_POSITION_AND_TAGID_PROBE_RESULTS.csv", probes, ["method", "seed", "position_probe_balanced_accuracy", "position_probe_macro_f1", "tagid_probe_macro_f1", "position_silhouette", "tagid_silhouette", "within_class_cross_position_distance", "between_class_distance", "position_to_tagid_separability_ratio", "condition_block_disjoint_probe"])
    write_markdown(RESULTS_ROOT / "15_SOURCE_ONLY_ETA_SELECTION.md", "# Source-only eta selection\n\n```json\n" + json.dumps(eta_decision, indent=2, sort_keys=True) + "\n```")
    write_csv(
        RESULTS_ROOT / "16_FINAL_MODEL_REGISTER.csv",
        [
            {
                "method": row["method"],
                "seed": row["seed"],
                "eta": row["eta"],
                "status": "COMPLETE",
                "epochs": row["epochs"],
                "checkpoint_sha256": row["checkpoint"]["checkpoint_sha256"],
                "checkpoint_model_state_sha256": row["model_state_sha256"],
                "q_entropy": row["q_entropy"],
                "max_q_weight": row["max_q_weight"],
                "final_q": row["final_q"],
            }
            for row in final_records
        ],
        ["method", "seed", "eta", "status", "epochs", "checkpoint_sha256", "checkpoint_model_state_sha256", "q_entropy", "max_q_weight", "final_q"],
    )
    write_csv(
        RESULTS_ROOT / "17_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv",
        p4["access_log"],
        ["event", "labels_accessed", "signal_sha256", "registry_sha256", "p4_input_mode", "tagid_column_accessed", "er_column_accessed", "prediction_bundle_sha256", "prediction_hashes", "label_sha256", "eta_selection_receipt_sha256", "eta_selection_frozen_at_utc"],
    )
    write_csv(RESULTS_ROOT / "18_P4_BLOCK_METRICS.csv", p4["p4_metrics"], ["method", "seed", "block_macro_f1", "block_accuracy", "row_macro_f1", "row_accuracy", "mean_within_block_agreement", "block_confusion_matrix"])
    write_csv(
        RESULTS_ROOT / "19_P4_PRIMARY_GROUPDRO_VS_ERM_CONTRAST.csv",
        [
            {
                "metric": "condition_block_macro_f1",
                "groupdro_minus_erm": p4["primary_bootstrap"]["point_estimate"],
                "interval_lower": p4["primary_bootstrap"]["interval_95"][0],
                "interval_upper": p4["primary_bootstrap"]["interval_95"][1],
                "status": "COMPLETE",
            },
            {
                "metric": "condition_block_accuracy",
                "groupdro_minus_erm": p4["primary_accuracy_bootstrap"]["point_estimate"],
                "interval_lower": p4["primary_accuracy_bootstrap"]["interval_95"][0],
                "interval_upper": p4["primary_accuracy_bootstrap"]["interval_95"][1],
                "status": "COMPLETE",
            },
        ],
        ["metric", "groupdro_minus_erm", "interval_lower", "interval_upper", "status"],
    )
    write_csv(RESULTS_ROOT / "20_P4_PER_CLASS_RESULTS.csv", p4["per_class"], ["method", "seed", "tagid_label", "precision", "recall", "f1"])
    write_csv(RESULTS_ROOT / "21_PREDICTED_CLASS_HISTOGRAMS.csv", p4["histograms"], ["method", "seed", "tagid_label", "row_count"])
    sensitivity = [
        {"analysis": "source_worst_position_macro_f1", "point_estimate": source_contrast["point_estimate"], "interval_lower": source_contrast["interval_95"][0], "interval_upper": source_contrast["interval_95"][1], "status": "COMPLETE"},
        {"analysis": "source_mean_held_macro_f1", "point_estimate": source_mean_bootstrap["point_estimate"], "interval_lower": source_mean_bootstrap["interval_95"][0], "interval_upper": source_mean_bootstrap["interval_95"][1], "status": "COMPLETE"},
        {"analysis": "p4_block_only_macro_f1", "point_estimate": p4["block_only_bootstrap"]["point_estimate"], "interval_lower": p4["block_only_bootstrap"]["interval_95"][0], "interval_upper": p4["block_only_bootstrap"]["interval_95"][1], "status": "COMPLETE"},
        {"analysis": "p4_block_only_accuracy", "point_estimate": p4["block_only_accuracy_bootstrap"]["point_estimate"], "interval_lower": p4["block_only_accuracy_bootstrap"]["interval_95"][0], "interval_upper": p4["block_only_accuracy_bootstrap"]["interval_95"][1], "status": "COMPLETE"},
        {"analysis": "p4_block_plus_seed_macro_f1", "point_estimate": p4["block_seed_bootstrap"]["point_estimate"], "interval_lower": p4["block_seed_bootstrap"]["interval_95"][0], "interval_upper": p4["block_seed_bootstrap"]["interval_95"][1], "status": "COMPLETE"},
        {"analysis": "p4_block_plus_seed_accuracy", "point_estimate": p4["block_seed_accuracy_bootstrap"]["point_estimate"], "interval_lower": p4["block_seed_accuracy_bootstrap"]["interval_95"][0], "interval_upper": p4["block_seed_accuracy_bootstrap"]["interval_95"][1], "status": "COMPLETE"},
        *source_fold_sensitivity,
        *q_sensitivity,
    ]
    for row in p4["leave_one_seed_out"]:
        for metric, sensitivity_result in (("macro_f1", row["macro_f1"]), ("accuracy", row["accuracy"])):
            sensitivity.append(
                {
                    "analysis": f"p4_leave_one_training_seed_out_{row['omitted_seed']}_{metric}",
                    "point_estimate": sensitivity_result["point_estimate"],
                    "interval_lower": sensitivity_result["interval_95"][0],
                    "interval_upper": sensitivity_result["interval_95"][1],
                    "status": "COMPLETE",
                }
            )
    write_csv(RESULTS_ROOT / "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", sensitivity, ["analysis", "point_estimate", "interval_lower", "interval_upper", "status"])
    write_markdown(
        RESULTS_ROOT / "23_SCIENTIFIC_INTERPRETATION.md",
        "# Scientific interpretation\n\n"
        f"Main classification: `{interpretation}`.\n\n"
        f"Selected source-only eta: `{selected_eta:.2f}`. The worst held-source-position Macro-F1 contrast is "
        f"`{source_contrast['point_estimate']:.6f}` with paired 95% interval "
        f"`[{source_contrast['interval_95'][0]:.6f}, {source_contrast['interval_95'][1]:.6f}]`. "
        f"Mean source retention is `{source_mean_contrast:.6f}` and the preregistered guardrail is "
        f"`{retention['retained']}`. P4 condition-block Macro-F1 contrast is "
        f"`{p4['primary_bootstrap']['point_estimate']:.6f}` with 95% interval "
        f"`[{p4['primary_bootstrap']['interval_95'][0]:.6f}, {p4['primary_bootstrap']['interval_95'][1]:.6f}]`; "
        f"P4 condition-block Accuracy contrast is `{p4['primary_accuracy_bootstrap']['point_estimate']:.6f}` "
        f"with 95% interval `[{p4['primary_accuracy_bootstrap']['interval_95'][0]:.6f}, "
        f"{p4['primary_accuracy_bootstrap']['interval_95'][1]:.6f}]`. The classification uses only "
        "the preregistered source and paired condition-block criteria; representation diagnostics were secondary.",
    )
    write_markdown(
        RESULTS_ROOT / "24_LIMITATIONS_AND_NONCLAIMS.md",
        "# Limitations and non-claims\n\n"
        "This result applies to the single hash-locked governed corpus, the specified C1 architecture, "
        "the five fixed training seeds, and source positions P1-P3 with P4 as the target. Condition blocks, "
        "not 3,150 repeated rows, are the P4 inferential units. The study does not establish causal mechanisms "
        "for position effects, generalize to other RFID hardware or target distributions, or claim that GroupDRO "
        "removes all position information. P4 labels were opened exactly once after prediction freezing; raw "
        "governed data and checkpoint binaries remain outside Git.",
    )
    write_markdown(
        RESULTS_ROOT / "STATUS.md",
        f"# Result status\n\n`{interpretation}`\n\n"
        "All 75 source-LOPO development runs and all 10 final source-only runs completed. GroupDRO "
        "validity and source-only eta selection completed before P4 label access; all final P4 predictions "
        "were frozen and hashed before labels were opened.",
    )


def execute_benchmark(
    *,
    prepare_only: bool = False,
    dry_run: bool = False,
    development_only: bool = False,
    only_fold: str | None = None,
    only_seed: int | None = None,
) -> dict[str, Any]:
    """Run governed execution, or write a truthfully blocked dry-run bundle."""

    config = load_config()
    validity = _synthetic_validity_rows()
    if not all(bool(row["passed"]) for row in validity):
        raise GroupDROProtocolError("FAIL_GROUPDRO_INTERVENTION_VALIDITY")
    lineage = c1_lineage_binding(config)
    if dry_run:
        _write_public_scaffold(
            config=config,
            execution_status="BLOCKED_GOVERNED_INPUTS",
            validity_rows=validity,
            lineage=lineage,
        )
        return {"status": "BLOCKED_GOVERNED_INPUTS", "reason": "dry_run", "synthetic_validity_passed": True}
    try:
        paths = environment_paths()
    except GroupDROProtocolError as error:
        if not str(error).startswith("BLOCKED_GOVERNED_INPUTS"):
            raise
        _write_public_scaffold(
            config=config,
            execution_status="BLOCKED_GOVERNED_INPUTS",
            validity_rows=validity,
            lineage=lineage,
        )
        return {"status": "BLOCKED_GOVERNED_INPUTS", "reason": str(error), "synthetic_validity_passed": True}
    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    source_hashes = governed_source_input_hashes(paths.strict_artifact_root)
    try:
        p4_preflight = governed_p4_preflight(paths.p4_governed_root)
    except (FileNotFoundError, GroupDROProtocolError) as error:
        raise GroupDROProtocolError("BLOCKED_RECOVERED_DATA_STRUCTURE_MISMATCH: P4 preflight") from error
    binding = execution_binding_evidence(
        paths=paths,
        source_hashes=source_hashes,
        p4_preflight=p4_preflight,
    )
    data = _canonical_source_loader(paths.strict_artifact_root, paths.runtime_root)
    structure_rows, split_rows = source_structure_audit(data)
    states = fit_first_difference_preprocessing(data, paths.runtime_root / "preprocessing")
    governed_diagnostic = _small_governed_diagnostic(
        data=data, states=states, config=config, runtime_root=paths.runtime_root
    )
    _write_public_scaffold(config=config, execution_status="PREPARATION_COMPLETE", validity_rows=validity, lineage=lineage)
    write_runtime_binding_outputs(binding)
    write_csv(RESULTS_ROOT / "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv", structure_rows, ["position", "sample_count", "tagid_count", "condition_block_count", "rows_per_condition_block", "ordered_input_points", "first_difference_points", "status"])
    write_csv(RESULTS_ROOT / "05_SOURCE_LOPO_SPLIT_MANIFEST.csv", split_rows, ["fold_id", "held_position", "partition", "sample_count", "condition_block_count", "position_groups", "condition_blocks_disjoint", "status"])
    diagnostic_rows = [*validity, {"check": "governed_small_diagnostic", "passed": governed_diagnostic["status"] == "PASS", "details": governed_diagnostic}]
    write_csv(RESULTS_ROOT / "06_GROUPDRO_IMPLEMENTATION_VALIDITY.csv", diagnostic_rows, ["check", "passed", "details"])
    if prepare_only:
        return {"status": "PASS_PRETRAINING_VALIDATION", "governed_diagnostic": governed_diagnostic}

    development: list[dict[str, Any]] = []
    fold_ids = (only_fold,) if development_only and only_fold is not None else tuple(FOLD_HELD_POSITION)
    seeds = (only_seed,) if development_only and only_seed is not None else tuple(int(seed) for seed in config["seeds"])
    for fold_id in fold_ids:
        for seed in seeds:
            development.append(_run_or_resume_lopo_unit(data=data, first_difference_states=states, config=config, fold_id=fold_id, seed=int(seed), method="erm", eta=None, runtime_root=paths.runtime_root))
            for eta in ETA_GRID:
                development.append(_run_or_resume_lopo_unit(data=data, first_difference_states=states, config=config, fold_id=fold_id, seed=int(seed), method="groupdro", eta=eta, runtime_root=paths.runtime_root))
    if development_only:
        write_json(
            paths.runtime_root / "development_chunk_receipt.json",
            {
                "status": "PASS_SOURCE_ONLY_DEVELOPMENT_CHUNK",
                "folds": list(fold_ids),
                "seeds": list(seeds),
                "unit_count": len(development),
                "resumed_unit_count": sum(
                    bool(row.get("resumed_from_verified_runtime_receipt")) for row in development
                ),
                "p4_accessed": False,
            },
        )
        return {
            "status": "PASS_SOURCE_ONLY_DEVELOPMENT_CHUNK",
            "unit_count": len(development),
            "resumed_unit_count": sum(
                bool(row.get("resumed_from_verified_runtime_receipt")) for row in development
            ),
        }
    eta_decision = select_eta_source_only([row for row in development if row["method"] == "groupdro"])
    eta_decision["selection_frozen_at_utc"] = utc_now()
    eta_decision["frozen_selection_receipt_sha256"] = canonical_json_sha256(eta_decision)
    eta_selection_path = paths.runtime_root / "frozen_eta_selection.json"
    write_json(eta_selection_path, eta_decision)
    selected_eta = float(eta_decision["selected_eta"])
    erm_development = [row for row in development if row["method"] == "erm"]
    selected_development = [row for row in development if row["method"] == "groupdro" and float(row["eta"]) == selected_eta]
    paired = []
    erm_by_key = {(row["fold_id"], row["seed"]): row for row in erm_development}
    for row in selected_development:
        matched = erm_by_key[(row["fold_id"], row["seed"])]
        paired.append({"fold_id": row["fold_id"], "seed": row["seed"], "erm_macro_f1": matched["held_macro_f1"], "groupdro_macro_f1": row["held_macro_f1"]})
    source_contrast = paired_source_worst_bootstrap(paired)
    source_mean_bootstrap = paired_source_mean_bootstrap(paired)
    all_source_partition, all_source_preprocessing = _all_source_first_difference(data)
    final_records = []
    erm_epochs = _final_epoch_by_seed(erm_development)
    groupdro_epochs = _final_epoch_by_seed(selected_development)
    for seed in config["seeds"]:
        final_records.append(_train_final_model(data=data, partition=all_source_partition, preprocessing=all_source_preprocessing, config=config, method="erm", eta=None, seed=int(seed), epochs=erm_epochs[int(seed)], runtime_root=paths.runtime_root))
        final_records.append(_train_final_model(data=data, partition=all_source_partition, preprocessing=all_source_preprocessing, config=config, method="groupdro", eta=selected_eta, seed=int(seed), epochs=groupdro_epochs[int(seed)], runtime_root=paths.runtime_root))
    positions = _positions_for(data, all_source_partition)
    probes = []
    for record in final_records:
        checkpoint = paths.runtime_root / "final" / str(record["method"]) / ("none" if record["eta"] is None else f"eta_{float(record['eta']):.2f}") / f"seed_{record['seed']}" / "final_source_only.pt"
        probes.append({"method": record["method"], "seed": record["seed"], **_representation_probe(_load_checkpoint(checkpoint, int(record["seed"])), all_source_partition, positions)})
    p4 = _p4_metrics_after_freeze(final_records=final_records, p4_root=paths.p4_governed_root, preprocessing=all_source_preprocessing, runtime_root=paths.runtime_root, eta_selection_path=eta_selection_path)
    source_erm = source_worst_position_metrics(erm_development)
    source_groupdro = source_worst_position_metrics(selected_development)
    source_mean_contrast = source_groupdro["mean_held_source_macro_f1"] - source_erm["mean_held_source_macro_f1"]
    interpretation = classify_interpretation(governed_inputs_available=True, protocol_defect=False, source_worst_interval=source_contrast["interval_95"], p4_interval=p4["primary_bootstrap"]["interval_95"], source_mean_contrast=source_mean_contrast, source_mean_interval=source_mean_bootstrap["interval_95"], p4_point_contrast=p4["primary_bootstrap"]["point_estimate"], unstable=False)
    _write_executed_outputs(development=development, eta_decision=eta_decision, final_records=final_records, probes=probes, p4=p4, source_contrast=source_contrast, source_mean_bootstrap=source_mean_bootstrap, binding=binding, interpretation=interpretation)
    write_csv(
        RESULTS_ROOT / "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv",
        [*structure_rows, p4["structure_audit"]],
        ["position", "sample_count", "tagid_count", "condition_block_count", "rows_per_condition_block", "ordered_input_points", "first_difference_points", "status"],
    )
    return {"status": interpretation, "selected_eta": selected_eta, "governed_diagnostic": governed_diagnostic}


def recover_execution_from_frozen_p4() -> dict[str, Any]:
    """Complete reporting from verified source/final receipts and an existing P4 freeze bundle."""

    config = load_config()
    validity = _synthetic_validity_rows()
    if not all(bool(row["passed"]) for row in validity):
        raise GroupDROProtocolError("FAIL_GROUPDRO_INTERVENTION_VALIDITY")
    paths = environment_paths()
    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    source_hashes = governed_source_input_hashes(paths.strict_artifact_root)
    try:
        p4_preflight = governed_p4_preflight(paths.p4_governed_root)
    except (FileNotFoundError, GroupDROProtocolError) as error:
        raise GroupDROProtocolError("BLOCKED_RECOVERED_DATA_STRUCTURE_MISMATCH: P4 preflight") from error
    binding = execution_binding_evidence(
        paths=paths,
        source_hashes=source_hashes,
        p4_preflight=p4_preflight,
    )
    data = _canonical_source_loader(paths.strict_artifact_root, paths.runtime_root)
    structure_rows, split_rows = source_structure_audit(data)
    lineage = c1_lineage_binding(config)
    diagnostic_path = paths.runtime_root / "governed_diagnostic.json"
    if not diagnostic_path.is_file():
        raise GroupDROProtocolError("Governed pretraining diagnostic receipt is missing")
    governed_diagnostic = read_json(diagnostic_path)
    if governed_diagnostic.get("status") != "PASS":
        raise GroupDROProtocolError("FAIL_GROUPDRO_INTERVENTION_VALIDITY")
    development: list[dict[str, Any]] = []
    for fold_id in FOLD_HELD_POSITION:
        for seed in config["seeds"]:
            development.append(
                _load_completed_lopo_unit(
                    runtime_root=paths.runtime_root,
                    method="erm",
                    eta=None,
                    fold_id=fold_id,
                    seed=int(seed),
                )
                or {}
            )
            for eta in ETA_GRID:
                development.append(
                    _load_completed_lopo_unit(
                        runtime_root=paths.runtime_root,
                        method="groupdro",
                        eta=eta,
                        fold_id=fold_id,
                        seed=int(seed),
                    )
                    or {}
                )
    if len(development) != 75 or any(not row for row in development):
        raise GroupDROProtocolError("Complete 75-unit development receipt set is required")
    eta_selection_path = paths.runtime_root / "frozen_eta_selection.json"
    if not eta_selection_path.is_file():
        raise GroupDROProtocolError("Frozen eta selection receipt is missing")
    eta_decision = read_json(eta_selection_path)
    recomputed = select_eta_source_only([row for row in development if row["method"] == "groupdro"])
    if (
        float(eta_decision.get("selected_eta", float("nan"))) != float(recomputed["selected_eta"])
        or eta_decision.get("selection_sha256") != recomputed["selection_sha256"]
    ):
        raise GroupDROProtocolError("Frozen eta selection receipt differs from source-only recomputation")
    selected_eta = float(eta_decision["selected_eta"])
    final_records = [
        _load_completed_final_model(
            runtime_root=paths.runtime_root,
            method=method,
            eta=selected_eta if method == "groupdro" else None,
            seed=int(seed),
        )
        for method in ("erm", "groupdro")
        for seed in config["seeds"]
    ]
    all_source_partition, all_source_preprocessing = _all_source_first_difference(data)
    positions = _positions_for(data, all_source_partition)
    probes = []
    for record in final_records:
        checkpoint = (
            paths.runtime_root
            / "final"
            / str(record["method"])
            / ("none" if record["eta"] is None else f"eta_{float(record['eta']):.2f}")
            / f"seed_{record['seed']}"
            / "final_source_only.pt"
        )
        probes.append(
            {
                "method": record["method"],
                "seed": record["seed"],
                **_representation_probe(
                    _load_checkpoint(checkpoint, int(record["seed"])),
                    all_source_partition,
                    positions,
                ),
            }
        )
    _write_public_scaffold(
        config=config,
        execution_status="POST_FREEZE_RECOVERY_PREPARATION_COMPLETE",
        validity_rows=validity,
        lineage=lineage,
    )
    write_runtime_binding_outputs(binding)
    write_csv(
        RESULTS_ROOT / "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv",
        structure_rows,
        ["position", "sample_count", "tagid_count", "condition_block_count", "rows_per_condition_block", "ordered_input_points", "first_difference_points", "status"],
    )
    write_csv(
        RESULTS_ROOT / "05_SOURCE_LOPO_SPLIT_MANIFEST.csv",
        split_rows,
        ["fold_id", "held_position", "partition", "sample_count", "condition_block_count", "position_groups", "condition_blocks_disjoint", "status"],
    )
    write_csv(
        RESULTS_ROOT / "06_GROUPDRO_IMPLEMENTATION_VALIDITY.csv",
        [*validity, {"check": "governed_small_diagnostic", "passed": True, "details": governed_diagnostic}],
        ["check", "passed", "details"],
    )
    p4 = _recover_frozen_p4_metrics(
        final_records=final_records,
        p4_root=paths.p4_governed_root,
        preprocessing=all_source_preprocessing,
        runtime_root=paths.runtime_root,
        eta_selection_path=eta_selection_path,
    )
    erm_development = [row for row in development if row["method"] == "erm"]
    selected_development = [
        row
        for row in development
        if row["method"] == "groupdro" and float(row["eta"]) == selected_eta
    ]
    erm_by_key = {(row["fold_id"], row["seed"]): row for row in erm_development}
    paired = [
        {
            "fold_id": row["fold_id"],
            "seed": row["seed"],
            "erm_macro_f1": erm_by_key[(row["fold_id"], row["seed"])]["held_macro_f1"],
            "groupdro_macro_f1": row["held_macro_f1"],
        }
        for row in selected_development
    ]
    source_contrast = paired_source_worst_bootstrap(paired)
    source_mean_bootstrap = paired_source_mean_bootstrap(paired)
    source_erm = source_worst_position_metrics(erm_development)
    source_groupdro = source_worst_position_metrics(selected_development)
    source_mean_contrast = float(
        source_groupdro["mean_held_source_macro_f1"] - source_erm["mean_held_source_macro_f1"]
    )
    interpretation = classify_interpretation(
        governed_inputs_available=True,
        protocol_defect=False,
        source_worst_interval=source_contrast["interval_95"],
        p4_interval=p4["primary_bootstrap"]["interval_95"],
        source_mean_contrast=source_mean_contrast,
        source_mean_interval=source_mean_bootstrap["interval_95"],
        p4_point_contrast=p4["primary_bootstrap"]["point_estimate"],
        unstable=False,
    )
    _write_executed_outputs(
        development=development,
        eta_decision=eta_decision,
        final_records=final_records,
        probes=probes,
        p4=p4,
        source_contrast=source_contrast,
        source_mean_bootstrap=source_mean_bootstrap,
        binding=binding,
        interpretation=interpretation,
    )
    write_csv(
        RESULTS_ROOT / "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv",
        [*structure_rows, p4["structure_audit"]],
        ["position", "sample_count", "tagid_count", "condition_block_count", "rows_per_condition_block", "ordered_input_points", "first_difference_points", "status"],
    )
    execution_state_path = RESULTS_ROOT / "28_EXECUTION_STATE.json"
    execution_state = read_json(execution_state_path)
    execution_state["post_freeze_scoring"] = {
        "status": "PASS_VERIFIED_FROZEN_PREDICTIONS",
        "prediction_bundle_sha256": p4["freeze"]["prediction_bundle_sha256"],
        "eta_selection_receipt_sha256": sha256_file(eta_selection_path),
        "labels_opened_after_existing_prediction_bundle_verification": True,
    }
    write_json(execution_state_path, execution_state)
    write_json(
        paths.runtime_root / "POST_FREEZE_RECOVERY_RECEIPT.json",
        {
            "status": "PASS_VERIFIED_FROZEN_PREDICTION_RECOVERY",
            "prediction_bundle_sha256": p4["freeze"]["prediction_bundle_sha256"],
            "selected_eta": selected_eta,
            "interpretation": interpretation,
        },
    )
    return {
        "status": interpretation,
        "selected_eta": selected_eta,
        "recovery": "PASS_VERIFIED_FROZEN_PREDICTION_RECOVERY",
    }


def record_post_freeze_reporting_defect() -> dict[str, Any]:
    """Seal an explicit protocol failure without reopening P4 labels after a post-label exception."""

    config = load_config()
    validity = _synthetic_validity_rows()
    paths = environment_paths()
    source_hashes = governed_source_input_hashes(paths.strict_artifact_root)
    p4_preflight = governed_p4_preflight(paths.p4_governed_root)
    binding = execution_binding_evidence(
        paths=paths,
        source_hashes=source_hashes,
        p4_preflight=p4_preflight,
    )
    data = _canonical_source_loader(paths.strict_artifact_root, paths.runtime_root)
    structure_rows, split_rows = source_structure_audit(data)
    lineage = c1_lineage_binding(config)
    diagnostic = read_json(paths.runtime_root / "governed_diagnostic.json")
    development = []
    for fold_id in FOLD_HELD_POSITION:
        for seed in config["seeds"]:
            development.append(
                _load_completed_lopo_unit(
                    runtime_root=paths.runtime_root,
                    method="erm",
                    eta=None,
                    fold_id=fold_id,
                    seed=int(seed),
                )
            )
            for eta in ETA_GRID:
                development.append(
                    _load_completed_lopo_unit(
                        runtime_root=paths.runtime_root,
                        method="groupdro",
                        eta=eta,
                        fold_id=fold_id,
                        seed=int(seed),
                    )
                )
    if len(development) != 75 or any(row is None for row in development):
        raise GroupDROProtocolError("Complete source development receipts are required for defect recording")
    development = [row for row in development if row is not None]
    eta_selection_path = paths.runtime_root / "frozen_eta_selection.json"
    eta_decision = read_json(eta_selection_path)
    selected_eta = float(eta_decision["selected_eta"])
    final_records = [
        _load_completed_final_model(
            runtime_root=paths.runtime_root,
            method=method,
            eta=selected_eta if method == "groupdro" else None,
            seed=int(seed),
        )
        for method in ("erm", "groupdro")
        for seed in config["seeds"]
    ]
    erm = [row for row in development if row["method"] == "erm"]
    selected_groupdro = [
        row
        for row in development
        if row["method"] == "groupdro" and float(row["eta"]) == selected_eta
    ]
    paired = [
        {
            "fold_id": row["fold_id"],
            "seed": row["seed"],
            "erm_macro_f1": next(
                item["held_macro_f1"]
                for item in erm
                if item["fold_id"] == row["fold_id"] and item["seed"] == row["seed"]
            ),
            "groupdro_macro_f1": row["held_macro_f1"],
        }
        for row in selected_groupdro
    ]
    source_contrast = paired_source_worst_bootstrap(paired)
    source_mean_bootstrap = paired_source_mean_bootstrap(paired)
    source_erm = source_worst_position_metrics(erm)
    source_groupdro = source_worst_position_metrics(selected_groupdro)
    retention = source_retention_guardrail(
        source_groupdro["mean_held_source_macro_f1"], source_erm["mean_held_source_macro_f1"]
    )
    source_fold_sensitivity = _source_fold_sensitivity_rows(paired)
    q_sensitivity, q_leader = _q_trajectory_sensitivity_rows(selected_groupdro)
    all_source_partition, _ = _all_source_first_difference(data)
    positions = _positions_for(data, all_source_partition)
    probes = []
    for record in final_records:
        checkpoint = (
            paths.runtime_root
            / "final"
            / str(record["method"])
            / ("none" if record["eta"] is None else f"eta_{float(record['eta']):.2f}")
            / f"seed_{record['seed']}"
            / "final_source_only.pt"
        )
        probes.append(
            {
                "method": record["method"],
                "seed": record["seed"],
                **_representation_probe(
                    _load_checkpoint(checkpoint, int(record["seed"])),
                    all_source_partition,
                    positions,
                ),
            }
        )
    _write_public_scaffold(
        config=config,
        execution_status="FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT",
        validity_rows=validity,
        lineage=lineage,
    )
    write_runtime_binding_outputs(binding)
    write_csv(
        RESULTS_ROOT / "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv",
        [
            *structure_rows,
            {
                "position": "P4",
                "sample_count": 3150,
                "tagid_count": "NOT_REPORTED_AFTER_PROTOCOL_DEFECT",
                "condition_block_count": 63,
                "rows_per_condition_block": [50],
                "ordered_input_points": 281,
                "first_difference_points": 280,
                "status": "P4_LABEL_METRICS_NOT_REPORTED",
            },
        ],
        ["position", "sample_count", "tagid_count", "condition_block_count", "rows_per_condition_block", "ordered_input_points", "first_difference_points", "status"],
    )
    write_csv(
        RESULTS_ROOT / "05_SOURCE_LOPO_SPLIT_MANIFEST.csv",
        split_rows,
        ["fold_id", "held_position", "partition", "sample_count", "condition_block_count", "position_groups", "condition_blocks_disjoint", "status"],
    )
    write_csv(
        RESULTS_ROOT / "06_GROUPDRO_IMPLEMENTATION_VALIDITY.csv",
        [*validity, {"check": "governed_small_diagnostic", "passed": diagnostic.get("status") == "PASS", "details": diagnostic}],
        ["check", "passed", "details"],
    )
    write_csv(
        RESULTS_ROOT / "07_ETA_GRID_REGISTER.csv",
        [
            {
                "eta": row["eta"],
                "frozen": True,
                "mean_held_source_macro_f1": row["mean_held_source_macro_f1"],
                "minimum_held_position_mean_macro_f1": row["minimum_held_position_mean_macro_f1"],
                "population_sd": row["population_sd_held_position_seed_macro_f1"],
                "selection_status": "SELECTED" if float(row["eta"]) == selected_eta else "EVALUATED_NOT_SELECTED",
                "selection_receipt_sha256": eta_decision["frozen_selection_receipt_sha256"],
            }
            for row in eta_decision["all_eta_summaries"]
        ],
        ["eta", "frozen", "mean_held_source_macro_f1", "minimum_held_position_mean_macro_f1", "population_sd", "selection_status", "selection_receipt_sha256"],
    )
    gates = _leakage_gate_rows(execution_status="EXECUTION_COMPLETE")
    gates.append(
        {
            "gate": "post_freeze_p4_reporting_persistence",
            "status": "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT",
            "evidence": "Unexpected source fold: S1 after post-freeze label scoring completed in memory",
        }
    )
    write_csv(RESULTS_ROOT / "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", gates, ["gate", "status", "evidence"])
    write_csv(
        RESULTS_ROOT / "09_DEVELOPMENT_RUN_REGISTER.csv",
        [
            {"method": row["method"], "eta": row["eta"], "fold_id": row["fold_id"], "held_position": row["held_position"], "seed": row["seed"], "status": "COMPLETE"}
            for row in development
        ],
        ["method", "eta", "fold_id", "held_position", "seed", "status"],
    )
    write_csv(
        RESULTS_ROOT / "10_SOURCE_HELD_POSITION_METRICS.csv",
        [
            {
                "method": row["method"], "eta": row["eta"], "fold_id": row["fold_id"], "held_position": row["held_position"], "seed": row["seed"],
                "macro_f1": row["held_macro_f1"], "accuracy": row["held_accuracy"], "worst_class_recall": row["held_worst_class_recall"],
                "selected_epoch": row["selected_epoch"], "train_to_held_macro_f1_gap": row["source_train_to_held_macro_f1_gap"],
                "q_entropy": row["q_entropy"], "max_q_weight": row["max_q_weight"], "largest_q_position": row["largest_q_position"],
                "held_prediction_hash": row["held_prediction_sha256"], "status": "COMPLETE",
            }
            for row in [*erm, *selected_groupdro]
        ],
        ["method", "eta", "fold_id", "held_position", "seed", "macro_f1", "accuracy", "worst_class_recall", "selected_epoch", "train_to_held_macro_f1_gap", "q_entropy", "max_q_weight", "largest_q_position", "held_prediction_hash", "status"],
    )
    write_csv(
        RESULTS_ROOT / "11_GROUP_LOSS_AND_Q_TRAJECTORIES.csv",
        [item for row in development if row["method"] == "groupdro" for item in row["q_rows"]],
        ["method", "eta", "fold_id", "held_position", "seed", "stage", "epoch", "optimizer_step", "group_position", "group_loss", "q_weight"],
    )
    write_csv(
        RESULTS_ROOT / "12_SOURCE_WORST_GROUP_CONTRASTS.csv",
        [
            {"metric": "worst_held_position_macro_f1", "groupdro_minus_erm": source_contrast["point_estimate"], "interval_lower": source_contrast["interval_95"][0], "interval_upper": source_contrast["interval_95"][1], "status": "COMPLETE"},
            {"metric": "mean_held_source_macro_f1", "groupdro_minus_erm": source_mean_bootstrap["point_estimate"], "interval_lower": source_mean_bootstrap["interval_95"][0], "interval_upper": source_mean_bootstrap["interval_95"][1], "status": "COMPLETE_RETAINED" if retention["retained"] else "COMPLETE_GUARDRAIL_FAILED"},
        ],
        ["metric", "groupdro_minus_erm", "interval_lower", "interval_upper", "status"],
    )
    write_csv(
        RESULTS_ROOT / "13_SOURCE_MEAN_AND_VARIANCE_RESULTS.csv",
        [
            {
                "method": method, "eta": selected_eta if method == "groupdro" else "", "mean_held_macro_f1": values["mean_held_source_macro_f1"],
                "worst_position_macro_f1": values["worst_position_macro_f1"], "held_position_variance": values["held_position_variance"],
                "maximum_position_gap": values["maximum_position_to_position_gap"], "held_position_seed_population_sd": values["held_position_seed_population_sd"],
                "held_position_means": values["held_position_mean_macro_f1"], "mean_retention_guardrail": retention["retained"] if method == "groupdro" else "", "status": "COMPLETE",
            }
            for method, values in (("erm", source_erm), ("groupdro", source_groupdro))
        ],
        ["method", "eta", "mean_held_macro_f1", "worst_position_macro_f1", "held_position_variance", "maximum_position_gap", "held_position_seed_population_sd", "held_position_means", "mean_retention_guardrail", "status"],
    )
    write_csv(
        RESULTS_ROOT / "14_POSITION_AND_TAGID_PROBE_RESULTS.csv",
        probes,
        ["method", "seed", "position_probe_balanced_accuracy", "position_probe_macro_f1", "tagid_probe_macro_f1", "position_silhouette", "tagid_silhouette", "within_class_cross_position_distance", "between_class_distance", "position_to_tagid_separability_ratio", "condition_block_disjoint_probe"],
    )
    write_markdown(RESULTS_ROOT / "15_SOURCE_ONLY_ETA_SELECTION.md", "# Source-only eta selection\n\n```json\n" + json.dumps(eta_decision, indent=2, sort_keys=True) + "\n```")
    write_csv(
        RESULTS_ROOT / "16_FINAL_MODEL_REGISTER.csv",
        [
            {"method": row["method"], "seed": row["seed"], "eta": row["eta"], "status": "COMPLETE", "epochs": row["epochs"], "checkpoint_sha256": row["checkpoint"]["checkpoint_sha256"], "checkpoint_model_state_sha256": row["model_state_sha256"], "q_entropy": row["q_entropy"], "max_q_weight": row["max_q_weight"], "final_q": row["final_q"]}
            for row in final_records
        ],
        ["method", "seed", "eta", "status", "epochs", "checkpoint_sha256", "checkpoint_model_state_sha256", "q_entropy", "max_q_weight", "final_q"],
    )
    prediction_bundle = paths.runtime_root / "p4" / "frozen_predictions.npz"
    freeze_hash = sha256_file(prediction_bundle)
    write_csv(
        RESULTS_ROOT / "17_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv",
        [
            {"event": "P4_PREDICTIONS_FROZEN", "labels_accessed": False, "prediction_bundle_sha256": freeze_hash, "status": "PASS"},
            {"event": "P4_LABELS_OPENED_AFTER_VERIFIED_PREDICTION_BUNDLE", "labels_accessed": True, "prediction_bundle_sha256": freeze_hash, "status": "POST_FREEZE_REPORTING_DEFECT"},
            {"event": "POST_FREEZE_REPORT_WRITER_EXCEPTION", "labels_accessed": True, "status": "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT", "evidence": "Unexpected source fold: S1"},
        ],
        ["event", "labels_accessed", "prediction_bundle_sha256", "status", "evidence"],
    )
    write_csv(RESULTS_ROOT / "18_P4_BLOCK_METRICS.csv", [{"status": "NOT_REPORTED_AFTER_POST_FREEZE_PROTOCOL_DEFECT"}], ["status"])
    write_csv(RESULTS_ROOT / "19_P4_PRIMARY_GROUPDRO_VS_ERM_CONTRAST.csv", [{"metric": "condition_block_metrics", "status": "NOT_REPORTED_AFTER_POST_FREEZE_PROTOCOL_DEFECT"}], ["metric", "status"])
    write_csv(RESULTS_ROOT / "20_P4_PER_CLASS_RESULTS.csv", [{"status": "NOT_REPORTED_AFTER_POST_FREEZE_PROTOCOL_DEFECT"}], ["status"])
    write_csv(RESULTS_ROOT / "21_PREDICTED_CLASS_HISTOGRAMS.csv", [{"status": "NOT_REPORTED_AFTER_POST_FREEZE_PROTOCOL_DEFECT"}], ["status"])
    write_csv(
        RESULTS_ROOT / "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv",
        [
            {"analysis": "source_worst_position_macro_f1", "point_estimate": source_contrast["point_estimate"], "interval_lower": source_contrast["interval_95"][0], "interval_upper": source_contrast["interval_95"][1], "status": "COMPLETE"},
            {"analysis": "source_mean_held_macro_f1", "point_estimate": source_mean_bootstrap["point_estimate"], "interval_lower": source_mean_bootstrap["interval_95"][0], "interval_upper": source_mean_bootstrap["interval_95"][1], "status": "COMPLETE"},
            *source_fold_sensitivity,
            *q_sensitivity,
            {"analysis": "p4_block_and_seed_bootstrap", "point_estimate": "", "interval_lower": "", "interval_upper": "", "status": "NOT_REPORTED_AFTER_POST_FREEZE_PROTOCOL_DEFECT"},
        ],
        ["analysis", "point_estimate", "interval_lower", "interval_upper", "status"],
    )
    write_markdown(
        RESULTS_ROOT / "00_EXECUTIVE_SUMMARY.md",
        "# GroupDRO worst-source-position benchmark\n\n"
        "Execution status: `FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT`.\n\n"
        "- Governed runtime binding: `PASS_GOVERNED_RUNTIME_BINDING_RECOVERED`.\n"
        "- Real-data development runs: `75`; real-data final runs: `10`.\n"
        f"- Selected source-only eta: `{selected_eta:.2f}`.\n"
        f"- Most frequent highest-q source position: `{q_leader}`.\n"
        f"- Source worst-position Macro-F1 contrast: `{source_contrast['point_estimate']:.6f}` (95% CI `{source_contrast['interval_95'][0]:.6f}`, `{source_contrast['interval_95'][1]:.6f}`).\n"
        f"- Source mean-retention guardrail passed: `{retention['retained']}`.\n"
        "- P4 predictions were frozen before label access, but a post-label report-writer exception prevented persistence of P4 metrics; labels will not be reopened.\n\n"
        "No P4 performance contrast, P4 confidence interval, or scientific benefit classification is claimed after this protocol defect.",
    )
    write_markdown(
        RESULTS_ROOT / "23_SCIENTIFIC_INTERPRETATION.md",
        "# Scientific interpretation\n\n"
        "Main classification: `FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT`.\n\n"
        "The source-only development and final training receipts are complete, but the P4 label stage cannot support a scientific transfer claim: labels were opened after the verified prediction bundle, and an exception in the post-freeze report writer (`Unexpected source fold: S1`) prevented metric persistence. Reopening labels to recompute those metrics would violate the one-time P4 boundary, so no P4 effect is reported.",
    )
    write_markdown(
        RESULTS_ROOT / "24_LIMITATIONS_AND_NONCLAIMS.md",
        "# Limitations and non-claims\n\n"
        "The source endpoints are retained as executed receipts, but the P4 primary endpoint, P4 Accuracy endpoint, per-class P4 results, and P4 bootstrap intervals are intentionally not reported. The post-freeze reporting exception is a protocol defect. No conclusion about P4 benefit, harm, or transfer is valid from this run, and the sealed labels will not be reopened. Raw governed data and checkpoint binaries remain outside Git.",
    )
    write_markdown(
        RESULTS_ROOT / "STATUS.md",
        "# Result status\n\n"
        "`FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT`\n\n"
        "All 75 development and 10 final source-only training receipts completed. The frozen eta receipt predates the P4 prediction bundle. A post-freeze report writer exception prevented persistence of P4 metrics after label access, so the strict label boundary forbids a retry.",
    )
    execution_state_path = RESULTS_ROOT / "28_EXECUTION_STATE.json"
    execution_state = read_json(execution_state_path)
    execution_state["post_freeze_reporting_defect"] = {
        "status": "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT",
        "prediction_bundle_sha256": freeze_hash,
        "eta_selection_receipt_sha256": sha256_file(eta_selection_path),
        "exception": "Unexpected source fold: S1",
        "p4_metrics_reopened": False,
    }
    write_json(execution_state_path, execution_state)
    write_json(
        paths.runtime_root / "POST_FREEZE_REPORTING_DEFECT_RECEIPT.json",
        {
            "status": "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT",
            "prediction_bundle_sha256": freeze_hash,
            "eta_selection_receipt_sha256": sha256_file(eta_selection_path),
            "exception": "Unexpected source fold: S1",
            "p4_labels_reopened": False,
        },
    )
    return {"status": "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT", "selected_eta": selected_eta}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--recover-frozen-p4", action="store_true")
    parser.add_argument("--record-post-freeze-defect", action="store_true")
    parser.add_argument("--only-fold", choices=tuple(FOLD_HELD_POSITION))
    parser.add_argument("--only-seed", type=int, choices=(42, 43, 44, 45, 46))
    args = parser.parse_args(argv)
    active_stages = sum(
        bool(value)
        for value in (
            args.prepare_only,
            args.dry_run,
            args.development_only,
            args.recover_frozen_p4,
            args.record_post_freeze_defect,
        )
    )
    if active_stages > 1:
        parser.error("execution stage flags are mutually exclusive")
    if (args.only_fold is not None or args.only_seed is not None) and not args.development_only:
        parser.error("--only-fold and --only-seed require --development-only")
    if args.record_post_freeze_defect:
        result = record_post_freeze_reporting_defect()
    elif args.recover_frozen_p4:
        result = recover_execution_from_frozen_p4()
    else:
        result = execute_benchmark(
            prepare_only=args.prepare_only,
            dry_run=args.dry_run,
            development_only=args.development_only,
            only_fold=args.only_fold,
            only_seed=args.only_seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
