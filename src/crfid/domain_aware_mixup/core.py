"""Pure domain-aware mixup, batching, stage, inference, and label-seal utilities."""

from __future__ import annotations

import csv
import hashlib
import json
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
ALPHA_GRID = (0.20, 0.50, 1.00)
WITHIN_POSITION_ALPHA = 0.50
BETA = 1.0
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


class MixupProtocolError(RuntimeError):
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
            raise MixupProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")
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
        raise MixupProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")


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
        raise MixupProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")
    validate_execution_record(record)
    return record


def validate_source_fold_record(record: Mapping[str, Any]) -> None:
    validate_execution_record(record)
    if record["execution_stage"] != SOURCE_LOPO_DEVELOPMENT:
        raise MixupProtocolError("FAIL_EXECUTION_STAGE_SCHEMA_DEFECT")


def canonical_position_order(values: Iterable[str]) -> tuple[str, ...]:
    observed = {str(value) for value in values}
    if not observed or observed.difference(SOURCE_POSITIONS):
        raise MixupProtocolError("Active environments must be source positions only")
    return tuple(position for position in SOURCE_POSITIONS if position in observed)


def validate_condition_block_disjoint(
    train_condition_ids: Iterable[str], validation_condition_ids: Iterable[str]
) -> None:
    if set(train_condition_ids).intersection(validation_condition_ids):
        raise MixupProtocolError("Source condition block crosses train/validation boundary")


def validate_lopo_position_isolation(
    training_positions: Iterable[str], held_positions: Iterable[str], held_position: str
) -> None:
    expected = set(SOURCE_POSITIONS).difference({held_position})
    if set(training_positions) != expected or set(held_positions) != {held_position}:
        raise MixupProtocolError("Held source position leaked into LOPO development")


def validate_frozen_configuration(config: Mapping[str, Any]) -> None:
    mixup = config["mixup"]
    if tuple(float(value) for value in mixup["alpha_grid"]) != ALPHA_GRID:
        raise MixupProtocolError("Frozen domain-aware mixup alpha grid changed")
    if tuple(config["source_positions"]) != SOURCE_POSITIONS:
        raise MixupProtocolError("Source environments changed")
    if tuple(int(value) for value in config["seeds"]) != SEEDS:
        raise MixupProtocolError("Canonical seeds changed")
    if float(mixup["within_position_alpha"]) != WITHIN_POSITION_ALPHA:
        raise MixupProtocolError("Frozen within-position alpha changed")
    if float(mixup["beta"]) != BETA:
        raise MixupProtocolError("Frozen mixup beta changed")
    if mixup["location"] != "penultimate_encoder_embedding":
        raise MixupProtocolError("Frozen mixup location changed")
    if mixup["pairing_fields"] != ["tag_id", "er", "surface"]:
        raise MixupProtocolError("Frozen condition-matching fields changed")
    if mixup["coefficient_stabilization"] != "max_m_1_minus_m":
        raise MixupProtocolError("Frozen symmetric coefficient rule changed")


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
        raise MixupProtocolError("Invalid environment-balanced sampler declaration")
    ordered: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for group in groups:
        selected = np.flatnonzero(positions == group).astype(np.int64)
        if not len(selected):
            raise MixupProtocolError(f"Missing environment in batch plan: {group}")
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
            raise MixupProtocolError("Incomplete batch omitted an active environment")
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
            raise MixupProtocolError("Environment-balanced batch composition failed")
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
class PairPlan:
    parent_indices: np.ndarray
    partner_indices: np.ndarray
    mode: str
    requested_count: int
    missing_count: int
    pair_counts: dict[str, int]

    @property
    def coverage(self) -> float:
        return float(len(self.partner_indices) / max(self.requested_count, 1))

    @property
    def signature_sha256(self) -> str:
        payload = np.column_stack((self.parent_indices, self.partner_indices)).astype("<i8")
        return hashlib.sha256(payload.tobytes(order="C")).hexdigest()


def make_condition_matched_pairs(
    labels: Sequence[int] | np.ndarray,
    er_values: Sequence[int] | np.ndarray,
    surfaces: Sequence[str] | np.ndarray,
    positions: Sequence[str] | np.ndarray,
    repeats: Sequence[int] | np.ndarray,
    partition_roles: Sequence[str] | np.ndarray,
    *,
    mode: str,
    seed: int,
    epoch: int,
    parent_indices: Sequence[int] | np.ndarray | None = None,
) -> PairPlan:
    """Build deterministic training-only same-condition partners for every parent."""

    if mode not in {"within_position", "cross_position"}:
        raise ValueError(mode)
    y = np.asarray(labels, dtype=np.int64)
    er = np.asarray(er_values, dtype=np.int64)
    surface = np.asarray(surfaces, dtype=str)
    position = np.asarray(positions, dtype=str)
    repeat = np.asarray(repeats, dtype=np.int64)
    roles = np.asarray(partition_roles, dtype=str)
    if not (y.shape == er.shape == surface.shape == position.shape == repeat.shape == roles.shape):
        raise ValueError("Pairing metadata must be aligned")
    parents = (
        np.flatnonzero(roles == "training").astype(np.int64)
        if parent_indices is None
        else np.asarray(parent_indices, dtype=np.int64)
    )
    if np.any(roles[parents] != "training"):
        raise MixupProtocolError("Pair parents must come from the training partition only")
    cells: dict[tuple[int, int, str, str], np.ndarray] = {}
    for tag in np.unique(y[roles == "training"]):
        for er_value in np.unique(er[(roles == "training") & (y == tag)]):
            for surface_value in np.unique(surface[(roles == "training") & (y == tag) & (er == er_value)]):
                base = (roles == "training") & (y == tag) & (er == er_value) & (surface == surface_value)
                for position_value in np.unique(position[base]):
                    cells[(int(tag), int(er_value), str(surface_value), str(position_value))] = np.flatnonzero(
                        base & (position == position_value)
                    ).astype(np.int64)
    queue_state: dict[tuple[object, ...], tuple[np.ndarray, int]] = {}
    parent_out: list[int] = []
    partner_out: list[int] = []
    counts: Counter[str] = Counter()
    for parent in parents.tolist():
        key = (int(y[parent]), int(er[parent]), str(surface[parent]))
        if mode == "cross_position":
            alternatives = [p for p in SOURCE_POSITIONS if p != position[parent] and (*key, p) in cells]
            if not alternatives:
                counts["missing"] += 1
                continue
            choice_key = ("position", *key, str(position[parent]))
            if choice_key not in queue_state:
                values = stable_rng("alternative", seed, epoch, *choice_key).permutation(np.asarray(alternatives, dtype=str))
                queue_state[choice_key] = (values, 0)
            values, cursor = queue_state[choice_key]
            partner_position = str(values[cursor % len(values)])
            queue_state[choice_key] = (values, cursor + 1)
            eligible = cells[(*key, partner_position)]
        else:
            partner_position = str(position[parent])
            eligible = cells.get((*key, partner_position), np.empty(0, dtype=np.int64))
            different_repeat = eligible[repeat[eligible] != repeat[parent]]
            if len(different_repeat):
                eligible = different_repeat
            else:
                eligible = eligible[eligible != parent]
            if not len(eligible):
                counts["missing"] += 1
                continue
        queue_key = ("rows", mode, *key, str(position[parent]), partner_position)
        if queue_key not in queue_state:
            values = stable_rng("partner", seed, epoch, *queue_key).permutation(eligible)
            queue_state[queue_key] = (values, 0)
        values, cursor = queue_state[queue_key]
        if cursor and cursor % len(values) == 0:
            values = stable_rng("partner-cycle", seed, epoch, *queue_key, cursor // len(values)).permutation(eligible)
        partner = int(values[cursor % len(values)])
        queue_state[queue_key] = (values, cursor + 1)
        if roles[partner] != "training" or y[partner] != y[parent] or er[partner] != er[parent] or surface[partner] != surface[parent]:
            raise MixupProtocolError("Condition-matched training-only pairing failed")
        if mode == "cross_position" and position[partner] == position[parent]:
            raise MixupProtocolError("Cross-position pair did not cross position")
        if mode == "within_position" and position[partner] != position[parent]:
            raise MixupProtocolError("Within-position pair crossed position")
        parent_out.append(parent)
        partner_out.append(partner)
        position_pair = "-".join(sorted((str(position[parent]), str(position[partner]))))
        counts[f"tag={y[parent]}|er={er[parent]}|surface={surface[parent]}|positions={position_pair}"] += 1
    plan = PairPlan(
        parent_indices=np.asarray(parent_out, dtype=np.int64),
        partner_indices=np.asarray(partner_out, dtype=np.int64),
        mode=mode,
        requested_count=int(len(parents)),
        missing_count=int(counts.pop("missing", 0)),
        pair_counts=dict(sorted(counts.items())),
    )
    if plan.coverage < 0.99 - 1e-15:
        raise MixupProtocolError("FAIL_DOMAIN_AWARE_PAIRING_COVERAGE")
    return plan


def sample_mix_coefficients(alpha: float, count: int, *, seed: int, epoch: int, batch_index: int) -> np.ndarray:
    if alpha not in ALPHA_GRID:
        raise MixupProtocolError("Mixup alpha is outside the frozen grid")
    values = stable_rng("beta", seed, epoch, batch_index, alpha).beta(alpha, alpha, size=count)
    return np.maximum(values, 1.0 - values).astype(np.float32)


@dataclass(frozen=True)
class MixupTerms:
    objective: torch.Tensor
    original_loss: torch.Tensor
    mixup_loss: torch.Tensor
    mixed_embeddings: torch.Tensor
    mixed_logits: torch.Tensor


def mixup_objective(
    original_logits: torch.Tensor,
    original_labels: torch.Tensor,
    parent_embeddings: torch.Tensor,
    partner_embeddings: torch.Tensor,
    mix_labels: torch.Tensor,
    coefficients: torch.Tensor,
    classifier: torch.nn.Module,
    *,
    beta: float = BETA,
) -> MixupTerms:
    if beta != BETA:
        raise MixupProtocolError("Frozen mixup beta changed")
    if parent_embeddings.shape != partner_embeddings.shape or parent_embeddings.ndim != 2:
        raise MixupProtocolError("Parent embedding dimensions differ")
    if len(coefficients) != len(parent_embeddings) or torch.any(coefficients < 0.5) or torch.any(coefficients > 1.0):
        raise MixupProtocolError("Symmetric mix coefficient is outside [0.5, 1.0]")
    if not torch.equal(original_labels[:0], mix_labels[:0]):
        raise MixupProtocolError("Label dtype or device mismatch")
    weights = coefficients.to(parent_embeddings.device, dtype=parent_embeddings.dtype).reshape(-1, 1)
    mixed = weights * parent_embeddings + (1.0 - weights) * partner_embeddings
    mixed_logits = classifier(mixed)
    original_loss = F.cross_entropy(original_logits.float(), original_labels, reduction="mean")
    mix_loss = F.cross_entropy(mixed_logits.float(), mix_labels, reduction="mean")
    objective = (original_loss + beta * mix_loss) / (1.0 + beta)
    return MixupTerms(objective, original_loss, mix_loss, mixed, mixed_logits)


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
            raise MixupProtocolError(f"ERM batch is missing {group}")
        risks[group] = F.cross_entropy(values[mask], labels[mask], reduction="mean")
    return F.cross_entropy(values, labels, reduction="mean"), risks


def assert_mixup_connectivity(
    mixup_loss: torch.Tensor,
    parent_embeddings: torch.Tensor,
    partner_embeddings: torch.Tensor,
    encoder_parameter: torch.Tensor,
    classifier_parameter: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    gradients = torch.autograd.grad(
        mixup_loss,
        (parent_embeddings, partner_embeddings, encoder_parameter, classifier_parameter),
        retain_graph=True,
        allow_unused=True,
    )
    if any(value is None for value in gradients) or any(not bool(torch.isfinite(value).all()) for value in gradients if value is not None):
        raise MixupProtocolError("FAIL_MIXUP_INTERVENTION_VALIDITY")
    return gradients


def select_alpha_source_only(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for value in ALPHA_GRID:
        selected = [
            row for row in run_rows
            if row["method"] == "cross_position_mixup" and float(row["alpha"]) == value
        ]
        if len(selected) != 15:
            raise MixupProtocolError(f"Alpha {value:g} does not have 15 source units")
        position_means = {
            position: float(np.mean([
                float(row["held_macro_f1"]) for row in selected
                if row["held_position"] == position
            ]))
            for position in SOURCE_POSITIONS
        }
        units = np.asarray([float(row["held_macro_f1"]) for row in selected], dtype=np.float64)
        summaries.append({
            "alpha": value,
            "mean_held_macro_f1": float(units.mean()),
            "minimum_held_position_mean_macro_f1": float(min(position_means.values())),
            "population_sd_held_position_seed_macro_f1": float(units.std(ddof=0)),
            "mean_held_accuracy": float(np.mean([float(row["held_accuracy"]) for row in selected])),
            "held_position_means": position_means,
            "mean_pairing_coverage": float(np.mean([float(row["pair_coverage"]) for row in selected])),
            "seed_stability_population_sd": float(units.std(ddof=0)),
        })
    best_mean = max(row["mean_held_macro_f1"] for row in summaries)
    eligible = [row for row in summaries if best_mean - row["mean_held_macro_f1"] <= 0.01 + 1e-15]
    best_minimum = max(row["minimum_held_position_mean_macro_f1"] for row in eligible)
    finalists = [row for row in eligible if best_minimum - row["minimum_held_position_mean_macro_f1"] <= 0.005 + 1e-15]
    best_sd = min(row["population_sd_held_position_seed_macro_f1"] for row in finalists)
    tied = [row for row in finalists if row["population_sd_held_position_seed_macro_f1"] - best_sd <= 0.001 + 1e-15]
    selected = min(tied, key=lambda row: row["alpha"])
    return {
        "selected_alpha": float(selected["alpha"]),
        "summaries": summaries,
        "eligibility_set": [float(row["alpha"]) for row in eligible],
        "p4_accessed": False,
        "selector": "frozen_lexicographic_source_only",
    }


def source_retention_guardrail(cross_position_mean: float, erm_mean: float) -> dict[str, Any]:
    contrast = float(cross_position_mean - erm_mean)
    return {"contrast": contrast, "threshold": -0.02, "passed": contrast >= -0.02 - 1e-15}


def early_stop_allowed(
    *,
    method: str,
    epochs_without_improvement: int,
    patience: int,
    completed_optimizer_steps: int = 0,
    anneal_step: int = 0,
) -> bool:
    """Use the canonical source-validation patience rule for every method."""

    del method, completed_optimizer_steps, anneal_step
    return epochs_without_improvement >= patience


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
        raise MixupProtocolError("Source bootstrap requires 3 positions x 5 paired seeds")
    lookup = {(str(row["held_position"]), int(row["seed"])): row for row in paired_rows}
    if set(lookup) != {(p, s) for p in SOURCE_POSITIONS for s in SEEDS}:
        raise MixupProtocolError("Source bootstrap pairing is incomplete")

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
            raise MixupProtocolError("P4 condition block custody failed")
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
        raise MixupProtocolError("P4 TagID strata differ from 9 blocks per class")
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


def classify_domain_bridging_diagnostics(
    *,
    cross_position_distance_interval: Sequence[float],
    position_decodability_interval: Sequence[float],
    tagid_retention_passed: bool,
    cross_vs_within_interval: Sequence[float],
) -> str:
    distance_improved = float(cross_position_distance_interval[1]) < 0.0
    position_improved = float(position_decodability_interval[1]) < 0.0
    domain_specific = float(cross_vs_within_interval[1]) < 0.0
    if not tagid_retention_passed:
        return "CROSS_POSITION_MIXUP_HARMS_TAGID_REPRESENTATION"
    if distance_improved and position_improved and domain_specific:
        return "CROSS_POSITION_GEOMETRY_IMPROVED_WITH_TAGID_RETENTION"
    if not domain_specific and (distance_improved or position_improved):
        return "GENERIC_MIXUP_REGULARISATION_ONLY"
    if distance_improved or position_improved or domain_specific:
        return "CROSS_POSITION_GEOMETRY_PARTIALLY_IMPROVED"
    return "CROSS_POSITION_GEOMETRY_NOT_IMPROVED"


def classify_main_result(
    *,
    source_mean_interval: Sequence[float],
    source_worst_interval: Sequence[float],
    p4_cross_vs_erm_interval: Sequence[float],
    p4_cross_vs_within_interval: Sequence[float],
    p4_within_vs_erm_interval: Sequence[float],
    source_guardrail_passed: bool,
    diagnostic_classification: str,
    unstable: bool,
    intervention_valid: bool = True,
    protocol_valid: bool = True,
) -> str:
    if not intervention_valid:
        return "FAIL_MIXUP_INTERVENTION_VALIDITY"
    if not protocol_valid:
        return "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"
    source_gain = float(source_mean_interval[0]) > 0 or float(source_worst_interval[0]) > 0
    p4_gain = float(p4_cross_vs_erm_interval[0]) > 0
    exceeds_control = float(p4_cross_vs_within_interval[0]) > 0
    within_gain = float(p4_within_vs_erm_interval[0]) > 0
    p4_harm = float(p4_cross_vs_erm_interval[1]) < 0
    if diagnostic_classification == "CROSS_POSITION_MIXUP_HARMS_TAGID_REPRESENTATION" or not source_guardrail_passed:
        return "DOMAIN_AWARE_MIXUP_TAGID_TRADEOFF"
    if p4_harm:
        return "DOMAIN_AWARE_MIXUP_HARMS_P4_TRANSFER"
    if unstable:
        return "DOMAIN_AWARE_MIXUP_MIXED_OR_UNSTABLE_RESULT"
    if source_gain and p4_gain and exceeds_control:
        return "DOMAIN_AWARE_MIXUP_SOURCE_AND_P4_BENEFIT_CONFIRMED"
    if p4_gain and exceeds_control and not source_gain:
        return "DOMAIN_AWARE_MIXUP_P4_BENEFIT_CONFIRMED"
    if source_gain and not p4_gain:
        return "DOMAIN_AWARE_MIXUP_SOURCE_BENEFIT_WITHOUT_P4_BENEFIT"
    if (p4_gain and within_gain and not exceeds_control) or diagnostic_classification == "GENERIC_MIXUP_REGULARISATION_ONLY":
        return "GENERIC_MIXUP_BENEFIT_NOT_DOMAIN_SPECIFIC"
    return "DOMAIN_AWARE_MIXUP_NO_RELIABLE_BENEFIT"


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
                raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
            file_hashes[path.name] = digest
            surface = path.stem.split("_")[0]
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != P4_RAW_HEADER:
                    raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
                count = 0
                for row_index, row in enumerate(reader):
                    active_surface = [name for name in ("A1", "A2", "A3") if float(row[name]) == 1.0]
                    active_position = [name for name in ("P1", "P2", "P3", "P4") if float(row[name]) == 1.0]
                    if active_surface != [surface] or active_position != ["P4"]:
                        raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
                    values = np.fromiter((float(row[str(i)]) for i in range(281)), dtype=np.float64, count=281)
                    if not np.isfinite(values).all():
                        raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
                    signals.append(values)
                    condition_ids.append(self._raw_block_id(path, row_index))
                    sample_ids.append(f"{path.stem}:opaque_row_{row_index:04d}")
                    count += 1
            if count != 1050:
                raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
        array = np.ascontiguousarray(np.vstack(signals), dtype=np.float64)
        block_counts = Counter(condition_ids)
        if array.shape != (3150, 281) or len(block_counts) != 63 or set(block_counts.values()) != {50}:
            raise MixupProtocolError("BLOCKED_MIXUP_LINEAGE_OR_DATA_MISMATCH")
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
            raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        if self._labels_opened or not predictions or set(predictions) != set(checkpoint_hashes):
            raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY")
        values = {key: np.asarray(value, dtype=np.int64) for key, value in predictions.items()}
        if any(value.shape != (3150,) for value in values.values()):
            raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
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
            raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        if not self._predictions_frozen or self._labels_opened:
            raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY")
        labels: list[int] = []
        block_keys: dict[str, set[tuple[int, int, str]]] = {}
        for path in self._paths():
            if sha256_file(path) != P4_RAW_FILE_HASHES[path.name]:
                raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY")
            surface = path.stem.split("_")[0]
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader):
                    tag_id = int(float(row["TagID"]))
                    er = int(float(row["ER"]))
                    if tag_id not in range(1, 8) or er not in {0, 1, 2}:
                        raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY")
                    label = tag_id - 1
                    labels.append(label)
                    block_keys.setdefault(self._raw_block_id(path, row_index), set()).add((label, er, surface))
        values = np.ascontiguousarray(labels, dtype=np.int64)
        if values.shape != (3150,) or len(block_keys) != 63 or any(len(keys) != 1 for keys in block_keys.values()):
            raise MixupProtocolError("FAIL_P4_LABEL_BOUNDARY")
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
            raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        if self._persistence_attempted:
            raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE: silent post-label retry rejected")
        self._persistence_attempted = True
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if fail_before_replace:
            temporary.unlink(missing_ok=True)
            raise MixupProtocolError("FAIL_P4_FINAL_STAGE_PERSISTENCE")
        temporary.replace(path)
        record = {**dict(stage_record), "event": "P4_METRICS_ATOMICALLY_PERSISTED", "sha256": sha256_file(path)}
        self.access_log.append(record)
        return record
