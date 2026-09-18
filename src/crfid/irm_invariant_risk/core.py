"""Pure IRMv1, batching, stage, inference, and label-seal utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from crfid.strict_runtime.neutral_metrics import classification_metrics


SOURCE_POSITIONS = ("P1", "P2", "P3")
ALL_POSITIONS = (*SOURCE_POSITIONS, "P4")
CLASS_ORDER = tuple(range(7))
SEEDS = (42, 43, 44, 45, 46)
LAMBDA_GRID = (1.0, 10.0, 100.0, 1000.0)
FOLD_HELD_POSITION = {"S1": "P1", "S2": "P2", "S3": "P3"}
SOURCE_LOPO_DEVELOPMENT = "SOURCE_LOPO_DEVELOPMENT"
FINAL_SOURCE_TRAINING = "FINAL_SOURCE_TRAINING"
FINAL_P4_EVALUATION = "FINAL_P4_EVALUATION"
P4_RAW_FILENAMES = ("A1_P4.csv", "A2_P4.csv", "A3_P4.csv")
P4_RAW_HEADER = (
    "A3", "A2", "A1", "P4", "P3", "P2", "P1", "ER", "TagID",
    *(str(index) for index in range(281)),
)
P4_RAW_FILE_HASHES = {
    "A1_P4.csv": "8f63f8bee3deb7747a88ac0fe1f47eabe8c0feedfe5a65cb503e25fdcf1377ee",
    "A2_P4.csv": "6b20fb815d6df40ff9ad4c0ecbf3bc7a82762ac96b8fd7df7170504715d75a00",
    "A3_P4.csv": "6c3bffcca0e81664398423c5c4828c3d7d4698a69e804fe69368e53bb8a9f6e3",
}


class IRMProtocolError(RuntimeError):
    """Raised when a frozen scientific or label-boundary rule is violated."""


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_rng(*parts: object) -> np.random.Generator:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed)


def validate_execution_record(record: Mapping[str, Any]) -> None:
    """Validate one of the three mutually exclusive execution-stage schemas."""

    stage = record.get("execution_stage")
    fold = record.get("development_fold")
    training = record.get("training_positions")
    held = record.get("held_source_position")
    target = record.get("target_position")
    if stage == SOURCE_LOPO_DEVELOPMENT:
        if fold not in FOLD_HELD_POSITION:
            raise IRMProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")
        expected_held = FOLD_HELD_POSITION[str(fold)]
        expected_training = "_".join(
            position for position in SOURCE_POSITIONS if position != expected_held
        )
        valid = training == expected_training and held == expected_held and target is None
    elif stage == FINAL_SOURCE_TRAINING:
        valid = fold is None and training == "P1_P2_P3" and held is None and target is None
    elif stage == FINAL_P4_EVALUATION:
        valid = fold is None and training == "P1_P2_P3" and held is None and target == "P4"
    else:
        valid = False
    if not valid:
        raise IRMProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")


def execution_record(stage: str, fold: str | None = None) -> dict[str, Any]:
    if stage == SOURCE_LOPO_DEVELOPMENT and fold in FOLD_HELD_POSITION:
        held = FOLD_HELD_POSITION[fold]
        record = {
            "execution_stage": stage,
            "development_fold": fold,
            "training_positions": "_".join(p for p in SOURCE_POSITIONS if p != held),
            "held_source_position": held,
            "target_position": None,
        }
    elif stage == FINAL_SOURCE_TRAINING:
        record = {
            "execution_stage": stage,
            "development_fold": None,
            "training_positions": "P1_P2_P3",
            "held_source_position": None,
            "target_position": None,
        }
    elif stage == FINAL_P4_EVALUATION:
        record = {
            "execution_stage": stage,
            "development_fold": None,
            "training_positions": "P1_P2_P3",
            "held_source_position": None,
            "target_position": "P4",
        }
    else:
        raise IRMProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")
    validate_execution_record(record)
    return record


def validate_source_fold_record(record: Mapping[str, Any]) -> None:
    validate_execution_record(record)
    if record["execution_stage"] != SOURCE_LOPO_DEVELOPMENT:
        raise IRMProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")


def canonical_position_order(values: Iterable[str]) -> tuple[str, ...]:
    observed = {str(value) for value in values}
    if not observed or observed.difference(SOURCE_POSITIONS):
        raise IRMProtocolError("Active environments must be source positions only")
    return tuple(position for position in SOURCE_POSITIONS if position in observed)


def validate_condition_block_disjoint(
    train_condition_ids: Iterable[str], validation_condition_ids: Iterable[str]
) -> None:
    if set(train_condition_ids).intersection(validation_condition_ids):
        raise IRMProtocolError("Source condition block crosses train/validation boundary")


def validate_lopo_position_isolation(
    training_positions: Iterable[str], held_positions: Iterable[str], held_position: str
) -> None:
    expected = set(SOURCE_POSITIONS).difference({held_position})
    if set(training_positions) != expected or set(held_positions) != {held_position}:
        raise IRMProtocolError("Held source position leaked into LOPO development")


def validate_frozen_configuration(config: Mapping[str, Any]) -> None:
    irm = config["irmv1"]
    if tuple(float(value) for value in irm["lambda_grid"]) != LAMBDA_GRID:
        raise IRMProtocolError("Frozen IRMv1 lambda grid changed")
    if tuple(config["source_positions"]) != SOURCE_POSITIONS:
        raise IRMProtocolError("Source environments changed")
    if tuple(int(value) for value in config["seeds"]) != SEEDS:
        raise IRMProtocolError("Canonical seeds changed")
    if float(irm["scale_initial_value"]) != 1.0 or not bool(irm["create_graph"]):
        raise IRMProtocolError("IRMv1 scale-gradient contract changed")
    if float(irm["anneal_fraction"]) != 0.25:
        raise IRMProtocolError("IRMv1 annealing fraction changed")
    expected = math.floor(0.25 * int(irm["canonical_total_optimizer_steps"]))
    if int(irm["anneal_step"]) != expected:
        raise IRMProtocolError("IRMv1 anneal boundary changed")
    if bool(irm["mixed_precision"]):
        raise IRMProtocolError("Mixed precision is prohibited on the IRM penalty path")


@dataclass(frozen=True)
class BatchPlan:
    batches: tuple[np.ndarray, ...]
    audits: tuple[dict[str, Any], ...]
    source_positions: tuple[str, ...]
    replacement_policy: str
    replacement_count: int

    @property
    def signature_sha256(self) -> str:
        digest = hashlib.sha256()
        for batch in self.batches:
            digest.update(np.asarray(batch, dtype="<i8").tobytes(order="C"))
        return digest.hexdigest()


def _class_balanced_order(
    indices: np.ndarray, labels: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    queues: dict[int, list[int]] = {}
    for label in sorted(np.unique(labels[indices]).tolist()):
        selected = np.asarray(indices[labels[indices] == label], dtype=np.int64)
        queues[int(label)] = rng.permutation(selected).tolist()
    cycle = rng.permutation(np.asarray(sorted(queues), dtype=np.int64)).tolist()
    ordered: list[int] = []
    while any(queues.values()):
        for label in cycle:
            if queues[int(label)]:
                ordered.append(queues[int(label)].pop())
    return np.asarray(ordered, dtype=np.int64)


def make_environment_balanced_batches(
    position_labels: Sequence[str] | np.ndarray,
    class_labels: Sequence[int] | np.ndarray,
    *,
    seed: int,
    epoch: int,
    stage: str,
    batch_size: int,
    environments: Sequence[str] | None = None,
) -> BatchPlan:
    positions = np.asarray(position_labels, dtype=str)
    labels = np.asarray(class_labels, dtype=np.int64)
    if positions.ndim != 1 or positions.shape != labels.shape:
        raise ValueError("Position and class labels must be aligned")
    groups = tuple(environments) if environments is not None else canonical_position_order(positions)
    if groups != canonical_position_order(groups) or batch_size < len(groups):
        raise IRMProtocolError("Invalid environment-balanced sampler declaration")
    ordered: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for group in groups:
        selected = np.flatnonzero(positions == group).astype(np.int64)
        if not len(selected):
            raise IRMProtocolError(f"Missing environment in batch plan: {group}")
        counts[group] = len(selected)
        ordered[group] = _class_balanced_order(
            selected, labels, rng=stable_rng("balanced", seed, epoch, stage, group)
        )
    target = max(counts.values())
    replacement_count = 0
    for group in groups:
        base = ordered[group]
        if len(base) < target:
            pieces = [base]
            needed = target - len(base)
            round_id = 1
            while needed:
                refreshed = _class_balanced_order(
                    np.flatnonzero(positions == group).astype(np.int64),
                    labels,
                    rng=stable_rng("replacement", seed, epoch, stage, group, round_id),
                )
                take = min(needed, len(refreshed))
                pieces.append(refreshed[:take])
                needed -= take
                round_id += 1
            ordered[group] = np.concatenate(pieces)
            replacement_count += target - counts[group]
    cursors = {group: 0 for group in groups}
    batches: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    batch_index = 0
    while any(cursors[group] < target for group in groups):
        remaining = {group: target - cursors[group] for group in groups}
        base_count = min(batch_size // len(groups), min(remaining.values()))
        if base_count <= 0:
            raise IRMProtocolError("Incomplete batch omitted an active environment")
        allocation = {group: base_count for group in groups}
        spare = batch_size - base_count * len(groups)
        rotation = groups[batch_index % len(groups):] + groups[:batch_index % len(groups)]
        for group in rotation:
            if spare and remaining[group] > allocation[group]:
                allocation[group] += 1
                spare -= 1
        pieces = []
        for group in groups:
            start = cursors[group]
            stop = start + allocation[group]
            pieces.append(ordered[group][start:stop])
            cursors[group] = stop
        selected = np.concatenate(pieces)
        selected = stable_rng("batch-order", seed, epoch, stage, batch_index).permutation(selected)
        observed_positions = positions[selected]
        observed_labels = labels[selected]
        environment_counts = {
            group: int(np.count_nonzero(observed_positions == group)) for group in groups
        }
        if min(environment_counts.values()) <= 0 or max(environment_counts.values()) - min(environment_counts.values()) > 1:
            raise IRMProtocolError("Environment-balanced batch composition failed")
        tag_counts = {
            group: {
                str(label): int(np.count_nonzero(observed_labels[observed_positions == group] == label))
                for label in CLASS_ORDER
            }
            for group in groups
        }
        batches.append(np.asarray(selected, dtype=np.int64))
        audits.append({
            "epoch": int(epoch),
            "stage": stage,
            "batch_index": int(batch_index),
            "sample_count": int(len(selected)),
            "environment_counts": environment_counts,
            "tagid_counts_per_environment": tag_counts,
            "incomplete_final_batch": len(selected) < batch_size,
            "replacement_policy": "repeat_only_after_all_original_position_rows_consumed",
            "replacement_active": replacement_count > 0,
            "sampler_seed": canonical_json_sha256([seed, epoch, stage, batch_index]),
        })
        batch_index += 1
    return BatchPlan(
        batches=tuple(batches),
        audits=tuple(audits),
        source_positions=groups,
        replacement_policy="repeat_only_after_all_original_position_rows_consumed",
        replacement_count=int(replacement_count),
    )


@dataclass(frozen=True)
class IRMTerms:
    objective: torch.Tensor
    mean_risk: torch.Tensor
    penalty: torch.Tensor
    environment_risks: dict[str, torch.Tensor]
    environment_penalties: dict[str, torch.Tensor]
    scale: torch.Tensor
    lambda_effective: float


def irmv1_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positions: Sequence[str] | np.ndarray,
    environments: Sequence[str],
    *,
    configured_lambda: float,
    optimizer_step: int,
    anneal_step: int,
    create_graph: bool = True,
    detach_logits: bool = False,
) -> IRMTerms:
    """Standard scalar-logit IRMv1 objective with a shared scale at exactly 1."""

    if configured_lambda not in LAMBDA_GRID:
        raise IRMProtocolError("IRM lambda is outside the frozen grid")
    values = logits.detach() if detach_logits else logits
    values = values.float()
    if values.dtype != torch.float32:
        raise IRMProtocolError("IRM logits are not float32")
    labels = labels.to(dtype=torch.int64, device=values.device)
    position_array = np.asarray(positions, dtype=str)
    supplied_groups = tuple(str(value) for value in environments)
    groups = canonical_position_order(supplied_groups)
    if len(groups) != len(supplied_groups):
        raise IRMProtocolError("IRM environments are not unique source positions")
    scale = torch.tensor(1.0, dtype=torch.float32, device=values.device, requires_grad=True)
    risks: dict[str, torch.Tensor] = {}
    penalties: dict[str, torch.Tensor] = {}
    for group in groups:
        mask = torch.from_numpy(position_array == group).to(values.device)
        if not bool(mask.any()):
            raise IRMProtocolError(f"IRM batch is missing {group}")
        risk = F.cross_entropy(scale * values[mask], labels[mask], reduction="mean")
        gradient = torch.autograd.grad(risk, scale, create_graph=create_graph)[0]
        risks[group] = risk
        penalties[group] = gradient.pow(2)
    mean_risk = torch.stack([risks[group] for group in groups]).mean()
    penalty = torch.stack([penalties[group] for group in groups]).mean()
    lambda_effective = float(configured_lambda if optimizer_step >= anneal_step else 0.0)
    objective = mean_risk if lambda_effective == 0.0 else (
        mean_risk + lambda_effective * penalty
    ) / (1.0 + lambda_effective)
    return IRMTerms(
        objective=objective,
        mean_risk=mean_risk,
        penalty=penalty,
        environment_risks=risks,
        environment_penalties=penalties,
        scale=scale,
        lambda_effective=lambda_effective,
    )


def matched_erm_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positions: Sequence[str] | np.ndarray,
    environments: Sequence[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    values = logits.float()
    position_array = np.asarray(positions, dtype=str)
    risks: dict[str, torch.Tensor] = {}
    for group in environments:
        mask = torch.from_numpy(position_array == group).to(values.device)
        if not bool(mask.any()):
            raise IRMProtocolError(f"ERM batch is missing {group}")
        risks[group] = F.cross_entropy(values[mask], labels[mask], reduction="mean")
    return torch.stack([risks[group] for group in environments]).mean(), risks


def assert_penalty_connectivity(
    penalty: torch.Tensor, encoder_parameter: torch.Tensor, classifier_parameter: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not penalty.requires_grad:
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY: create_graph is disabled")
    try:
        encoder_gradient, classifier_gradient = torch.autograd.grad(
            penalty,
            (encoder_parameter, classifier_parameter),
            retain_graph=True,
            allow_unused=True,
        )
    except RuntimeError as error:
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY: penalty is detached") from error
    if encoder_gradient is None or classifier_gradient is None:
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY: penalty is detached")
    if not bool(torch.isfinite(encoder_gradient).all()) or not bool(torch.isfinite(classifier_gradient).all()):
        raise IRMProtocolError("FAIL_IRM_INTERVENTION_VALIDITY: non-finite second-order gradient")
    return encoder_gradient, classifier_gradient


def select_lambda_source_only(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for value in LAMBDA_GRID:
        selected = [row for row in run_rows if float(row["lambda"]) == value]
        if len(selected) != 15:
            raise IRMProtocolError(f"Lambda {value:g} does not have 15 source units")
        position_means = {
            position: float(np.mean([
                float(row["held_macro_f1"]) for row in selected
                if row["held_position"] == position
            ]))
            for position in SOURCE_POSITIONS
        }
        units = np.asarray([float(row["held_macro_f1"]) for row in selected], dtype=np.float64)
        summaries.append({
            "lambda": value,
            "mean_held_macro_f1": float(units.mean()),
            "minimum_held_position_mean_macro_f1": float(min(position_means.values())),
            "population_sd_held_position_seed_macro_f1": float(units.std(ddof=0)),
            "mean_held_accuracy": float(np.mean([float(row["held_accuracy"]) for row in selected])),
            "held_position_means": position_means,
            "mean_validation_irm_penalty": float(np.mean([float(row["validation_irm_penalty"]) for row in selected])),
            "penalty_stability_population_sd": float(np.std([float(row["validation_irm_penalty"]) for row in selected], ddof=0)),
        })
    best_mean = max(row["mean_held_macro_f1"] for row in summaries)
    eligible = [row for row in summaries if best_mean - row["mean_held_macro_f1"] <= 0.01 + 1e-15]
    best_minimum = max(row["minimum_held_position_mean_macro_f1"] for row in eligible)
    finalists = [row for row in eligible if best_minimum - row["minimum_held_position_mean_macro_f1"] <= 0.005 + 1e-15]
    best_sd = min(row["population_sd_held_position_seed_macro_f1"] for row in finalists)
    tied = [row for row in finalists if row["population_sd_held_position_seed_macro_f1"] - best_sd <= 0.001 + 1e-15]
    selected = min(tied, key=lambda row: row["lambda"])
    return {
        "selected_lambda": float(selected["lambda"]),
        "summaries": summaries,
        "eligibility_set": [float(row["lambda"]) for row in eligible],
        "p4_accessed": False,
        "selector": "frozen_lexicographic_source_only",
    }


def source_retention_guardrail(irm_mean: float, erm_mean: float) -> dict[str, Any]:
    contrast = float(irm_mean - erm_mean)
    return {"contrast": contrast, "threshold": -0.02, "passed": contrast >= -0.02 - 1e-15}


def early_stop_allowed(
    *,
    method: str,
    epochs_without_improvement: int,
    patience: int,
    completed_optimizer_steps: int,
    anneal_step: int,
) -> bool:
    """Never let validation patience truncate IRM before its frozen intervention."""

    if epochs_without_improvement < patience:
        return False
    if method == "irm" and completed_optimizer_steps <= anneal_step:
        return False
    return True


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def _metric(labels: np.ndarray, predictions: np.ndarray, name: str) -> float:
    matrix = _confusion(labels, predictions)
    if name == "accuracy":
        return float(np.trace(matrix) / max(matrix.sum(), 1))
    tp = np.diag(matrix).astype(np.float64)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(7), where=matrix.sum(axis=0) > 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros(7), where=matrix.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=(precision + recall) > 0)
    return float(f1.mean())


def paired_source_bootstrap(
    paired_rows: Sequence[Mapping[str, Any]], *, resamples: int = 10_000, seed: int = 20260805
) -> dict[str, Any]:
    if len(paired_rows) != 15:
        raise IRMProtocolError("Source bootstrap requires 3 positions x 5 paired seeds")
    lookup = {(str(row["held_position"]), int(row["seed"])): row for row in paired_rows}
    if set(lookup) != {(p, s) for p in SOURCE_POSITIONS for s in SEEDS}:
        raise IRMProtocolError("Source bootstrap pairing is incomplete")

    def endpoints(keys: Sequence[tuple[str, int]]) -> tuple[float, float]:
        irm_by_position: dict[str, list[float]] = {p: [] for p in SOURCE_POSITIONS}
        erm_by_position: dict[str, list[float]] = {p: [] for p in SOURCE_POSITIONS}
        for position, training_seed in keys:
            row = lookup[(position, training_seed)]
            irm_by_position[position].append(float(row["irm_macro_f1"]))
            erm_by_position[position].append(float(row["erm_macro_f1"]))
        active = [position for position in SOURCE_POSITIONS if irm_by_position[position]]
        mean_contrast = float(np.mean([
            value for position in active for value in np.asarray(irm_by_position[position]) - np.asarray(erm_by_position[position])
        ]))
        worst_contrast = float(
            min(np.mean(irm_by_position[p]) for p in active)
            - min(np.mean(erm_by_position[p]) for p in active)
        )
        return mean_contrast, worst_contrast

    original_keys = [(p, s) for p in SOURCE_POSITIONS for s in SEEDS]
    point_mean, point_worst = endpoints(original_keys)
    rng = np.random.default_rng(seed)
    mean_values = np.empty(resamples, dtype=np.float64)
    worst_values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_positions = rng.choice(SOURCE_POSITIONS, size=3, replace=True)
        keys: list[tuple[str, int]] = []
        position_alias: dict[str, str] = {}
        # Preserve paired seed draws; repeated folds contribute with their original identity.
        for draw, position in enumerate(sampled_positions):
            alias = SOURCE_POSITIONS[draw]
            position_alias[alias] = str(position)
            for training_seed in rng.choice(SEEDS, size=5, replace=True):
                keys.append((str(position), int(training_seed)))
        contrasts = [
            float(lookup[key]["irm_macro_f1"]) - float(lookup[key]["erm_macro_f1"])
            for key in keys
        ]
        mean_values[index] = float(np.mean(contrasts))
        irm_means = []
        erm_means = []
        cursor = 0
        for _ in range(3):
            rows = keys[cursor:cursor + 5]
            cursor += 5
            irm_means.append(float(np.mean([float(lookup[key]["irm_macro_f1"]) for key in rows])))
            erm_means.append(float(np.mean([float(lookup[key]["erm_macro_f1"]) for key in rows])))
        worst_values[index] = min(irm_means) - min(erm_means)
    return {
        "resamples": int(resamples),
        "pairing_unit": "held_position_x_training_seed",
        "mean_contrast": {"point_estimate": point_mean, "interval_95": np.quantile(mean_values, [0.025, 0.975]).tolist()},
        "worst_position_contrast": {"point_estimate": point_worst, "interval_95": np.quantile(worst_values, [0.025, 0.975]).tolist()},
    }


def deterministic_block_majority_vote(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    condition_ids: Sequence[str] | np.ndarray,
    *,
    expected_block_size: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predicted.shape or len(condition_ids) != len(truth):
        raise ValueError("Block-vote inputs are not aligned")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition_id in enumerate(condition_ids):
        groups.setdefault(str(condition_id), []).append(index)
    block_truth: list[int] = []
    block_predicted: list[int] = []
    rows: list[dict[str, Any]] = []
    for block_index, condition_id in enumerate(sorted(groups, key=str.casefold)):
        selected = groups[condition_id]
        if len(selected) != expected_block_size or len(np.unique(truth[selected])) != 1:
            raise IRMProtocolError("P4 condition block custody failed")
        votes = np.bincount(predicted[selected], minlength=7)
        chosen = int(np.flatnonzero(votes == votes.max())[0])
        block_truth.append(int(truth[selected[0]]))
        block_predicted.append(chosen)
        rows.append({
            "condition_block_index": block_index,
            "condition_block_id": condition_id,
            "true_label": block_truth[-1],
            "predicted_label": chosen,
            "within_block_agreement": float(votes[chosen] / len(selected)),
            "vote_counts": votes.astype(int).tolist(),
        })
    metrics = classification_metrics(np.asarray(block_truth), np.asarray(block_predicted))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    precision = np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(7), where=matrix.sum(axis=0) > 0)
    metrics["per_class_precision"] = precision.tolist()
    return metrics, rows


def paired_tagid_stratified_block_bootstrap(
    labels: np.ndarray,
    erm_predictions: np.ndarray,
    irm_predictions: np.ndarray,
    *,
    metric_name: str = "macro_f1",
    resamples: int = 10_000,
    seed: int = 20260805,
    resample_seeds: bool = False,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    erm = np.asarray(erm_predictions, dtype=np.int64)
    irm = np.asarray(irm_predictions, dtype=np.int64)
    if erm.shape != irm.shape or erm.shape != (5, 63) or labels.shape != (63,):
        raise ValueError("P4 bootstrap requires five paired seeds and 63 blocks")
    if metric_name not in {"macro_f1", "accuracy"}:
        raise ValueError(metric_name)
    point_seed = [
        _metric(labels, irm[index], metric_name) - _metric(labels, erm[index], metric_name)
        for index in range(5)
    ]
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    strata = [np.flatnonzero(labels == label) for label in CLASS_ORDER]
    if any(len(stratum) != 9 for stratum in strata):
        raise IRMProtocolError("P4 TagID strata differ from 9 blocks per class")
    for replicate in range(resamples):
        selected = np.concatenate([rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata])
        seed_indices = rng.choice(np.arange(5), size=5, replace=True) if resample_seeds else np.arange(5)
        contrasts = [
            _metric(labels[selected], irm[index, selected], metric_name)
            - _metric(labels[selected], erm[index, selected], metric_name)
            for index in seed_indices
        ]
        values[replicate] = float(np.mean(contrasts))
    return {
        "metric": metric_name,
        "point_estimate": float(np.mean(point_seed)),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
        "seed_wise_contrasts": [float(value) for value in point_seed],
        "resamples": int(resamples),
        "resample_training_seeds": bool(resample_seeds),
        "inferential_unit": "TagID_stratified_condition_block",
        "row_level_pseudoreplication": False,
    }


def classify_invariance_diagnostics(
    penalty_interval: Sequence[float], dispersion_interval: Sequence[float], guardrail_passed: bool
) -> str:
    penalty_improved = float(penalty_interval[1]) < 0.0
    dispersion_improved = float(dispersion_interval[1]) < 0.0
    if penalty_improved and dispersion_improved and guardrail_passed:
        return "SOURCE_INVARIANCE_DIAGNOSTICS_IMPROVED"
    if penalty_improved and dispersion_improved and not guardrail_passed:
        return "SOURCE_INVARIANCE_DIAGNOSTICS_IMPROVED_WITH_TAGID_TRADEOFF"
    if not penalty_improved and not dispersion_improved:
        return "SOURCE_INVARIANCE_DIAGNOSTICS_NOT_IMPROVED"
    return "SOURCE_INVARIANCE_DIAGNOSTICS_PARTIALLY_IMPROVED"


def classify_main_result(
    *,
    source_mean_interval: Sequence[float],
    source_worst_interval: Sequence[float],
    p4_interval: Sequence[float],
    source_guardrail_passed: bool,
    diagnostic_classification: str,
    unstable: bool,
    intervention_valid: bool = True,
    protocol_valid: bool = True,
) -> str:
    if not intervention_valid:
        return "FAIL_IRM_INTERVENTION_VALIDITY"
    if not protocol_valid:
        return "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"
    source_gain = float(source_mean_interval[0]) > 0 or float(source_worst_interval[0]) > 0
    p4_gain = float(p4_interval[0]) > 0
    p4_harm = float(p4_interval[1]) < 0
    diagnostics_improved = diagnostic_classification in {
        "SOURCE_INVARIANCE_DIAGNOSTICS_IMPROVED",
        "SOURCE_INVARIANCE_DIAGNOSTICS_IMPROVED_WITH_TAGID_TRADEOFF",
    }
    if diagnostics_improved and not source_guardrail_passed:
        return "IRM_SOURCE_PERFORMANCE_TRADEOFF"
    if p4_harm:
        return "IRM_HARMS_P4_TRANSFER"
    if source_gain and p4_gain and source_guardrail_passed:
        return "IRM_SOURCE_ROBUSTNESS_AND_P4_BENEFIT_CONFIRMED"
    if source_gain and not p4_gain:
        return "IRM_SOURCE_ROBUSTNESS_WITHOUT_P4_BENEFIT"
    if p4_gain and not source_gain:
        return "IRM_P4_BENEFIT_WITHOUT_SOURCE_ROBUSTNESS_GAIN"
    if diagnostics_improved and source_guardrail_passed and not p4_gain:
        return "IRM_INVARIANCE_DIAGNOSTICS_WITHOUT_P4_BENEFIT"
    if unstable:
        return "IRM_MIXED_OR_UNSTABLE_RESULT"
    return "IRM_NO_RELIABLE_BENEFIT"


@dataclass(frozen=True)
class SealedP4Data:
    signals: np.ndarray
    condition_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]


class P4LabelSeal:
    """One-way transaction: signals, freeze, receipt, labels once, metrics once."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._predictions_frozen = False
        self._labels_opened = False
        self._persistence_attempted = False
        self.access_log: list[dict[str, Any]] = []

    @staticmethod
    def _raw_block_id(path: Path, row_index: int) -> str:
        return f"{path.name}:opaque_block_{row_index // 50:02d}"

    def _paths(self) -> tuple[Path, ...]:
        paths = tuple(self.root / name for name in P4_RAW_FILENAMES)
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("BLOCKED_GOVERNED_INPUTS: governed P4 CSVs missing")
        return paths

    def load_unlabelled(self) -> SealedP4Data:
        signals: list[np.ndarray] = []
        condition_ids: list[str] = []
        sample_ids: list[str] = []
        file_hashes: dict[str, str] = {}
        for path in self._paths():
            digest = sha256_file(path)
            if digest != P4_RAW_FILE_HASHES[path.name]:
                raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
            file_hashes[path.name] = digest
            surface = path.stem.split("_")[0]
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != P4_RAW_HEADER:
                    raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
                count = 0
                for row_index, row in enumerate(reader):
                    active_surface = [name for name in ("A1", "A2", "A3") if float(row[name]) == 1.0]
                    active_position = [name for name in ("P1", "P2", "P3", "P4") if float(row[name]) == 1.0]
                    if active_surface != [surface] or active_position != ["P4"]:
                        raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
                    values = np.fromiter((float(row[str(i)]) for i in range(281)), dtype=np.float64, count=281)
                    if not np.isfinite(values).all():
                        raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
                    signals.append(values)
                    condition_ids.append(self._raw_block_id(path, row_index))
                    sample_ids.append(f"{path.stem}:opaque_row_{row_index:04d}")
                    count += 1
            if count != 1050:
                raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
        array = np.ascontiguousarray(np.vstack(signals), dtype=np.float64)
        block_counts = Counter(condition_ids)
        if array.shape != (3150, 281) or len(block_counts) != 63 or set(block_counts.values()) != {50}:
            raise IRMProtocolError("BLOCKED_IRM_LINEAGE_OR_DATA_MISMATCH")
        self.access_log.append({
            "event": "P4_SIGNALS_AND_OPAQUE_METADATA_OPENED",
            "labels_accessed": False,
            "tagid_column_accessed": False,
            "er_column_accessed": False,
            "signal_sha256": array_sha256(array),
            "registry_sha256": canonical_json_sha256(file_hashes),
        })
        return SealedP4Data(array, tuple(condition_ids), tuple(sample_ids))

    def freeze_predictions(
        self,
        predictions: Mapping[str, np.ndarray],
        checkpoint_hashes: Mapping[str, str],
        output_path: Path,
        stage_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        validate_execution_record(stage_record)
        if stage_record["execution_stage"] != FINAL_P4_EVALUATION:
            raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        if self._labels_opened or not predictions or set(predictions) != set(checkpoint_hashes):
            raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY")
        values = {key: np.asarray(value, dtype=np.int64) for key, value in predictions.items()}
        if any(value.shape != (3150,) for value in values.values()):
            raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **values)
        temporary.replace(output_path)
        self._predictions_frozen = True
        record = {
            **dict(stage_record),
            "event": "P4_PREDICTIONS_FROZEN",
            "labels_accessed": False,
            "prediction_bundle_sha256": sha256_file(output_path),
            "prediction_hashes": {key: array_sha256(value) for key, value in values.items()},
            "checkpoint_hashes": dict(checkpoint_hashes),
            "checkpoint_prediction_bindings": {
                key: canonical_json_sha256({"checkpoint": checkpoint_hashes[key], "prediction": array_sha256(values[key])})
                for key in sorted(values)
            },
        }
        self.access_log.append(record)
        return record

    def open_labels(self, stage_record: Mapping[str, Any]) -> np.ndarray:
        validate_execution_record(stage_record)
        if stage_record["execution_stage"] != FINAL_P4_EVALUATION:
            raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        if not self._predictions_frozen or self._labels_opened:
            raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY")
        labels: list[int] = []
        block_keys: dict[str, set[tuple[int, int, str]]] = {}
        for path in self._paths():
            if sha256_file(path) != P4_RAW_FILE_HASHES[path.name]:
                raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY")
            surface = path.stem.split("_")[0]
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader):
                    tag_id = int(float(row["TagID"]))
                    er = int(float(row["ER"]))
                    if tag_id not in range(1, 8) or er not in {0, 1, 2}:
                        raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY")
                    label = tag_id - 1
                    labels.append(label)
                    block_keys.setdefault(self._raw_block_id(path, row_index), set()).add((label, er, surface))
        values = np.ascontiguousarray(labels, dtype=np.int64)
        if values.shape != (3150,) or len(block_keys) != 63 or any(len(keys) != 1 for keys in block_keys.values()):
            raise IRMProtocolError("FAIL_P4_LABEL_BOUNDARY")
        self._labels_opened = True
        self.access_log.append({
            **dict(stage_record),
            "event": "P4_LABELS_OPENED_AFTER_PREDICTION_FREEZE",
            "labels_accessed": True,
            "label_sha256": array_sha256(values),
        })
        return values

    def persist_metrics_once(
        self,
        path: Path,
        payload: Mapping[str, Any],
        stage_record: Mapping[str, Any],
        *,
        fail_before_replace: bool = False,
    ) -> dict[str, Any]:
        validate_execution_record(stage_record)
        if stage_record["execution_stage"] != FINAL_P4_EVALUATION or not self._labels_opened:
            raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        if self._persistence_attempted:
            raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE: silent post-label retry rejected")
        self._persistence_attempted = True
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if fail_before_replace:
            temporary.unlink(missing_ok=True)
            raise IRMProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        temporary.replace(path)
        record = {**dict(stage_record), "event": "P4_METRICS_ATOMICALLY_PERSISTED", "sha256": sha256_file(path)}
        self.access_log.append(record)
        return record
