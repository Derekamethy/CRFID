"""Pure GroupDRO, batching, inference, and protocol utilities.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as functional

from crfid.strict_runtime.neutral_metrics import classification_metrics


SOURCE_POSITIONS = ("P1", "P2", "P3")
ALL_POSITIONS = (*SOURCE_POSITIONS, "P4")
CLASS_ORDER = tuple(range(7))
ETA_GRID = (0.01, 0.05, 0.10, 0.20)
FOLD_HELD_POSITION = {"S1": "P1", "S2": "P2", "S3": "P3"}
P4_RAW_FILENAMES = ("A1_P4.csv", "A2_P4.csv", "A3_P4.csv")
P4_RAW_HEADER = (
    "A3",
    "A2",
    "A1",
    "P4",
    "P3",
    "P2",
    "P1",
    "ER",
    "TagID",
    *(str(index) for index in range(281)),
)
P4_RAW_FILE_HASHES = {
    "A1_P4.csv": "8f63f8bee3deb7747a88ac0fe1f47eabe8c0feedfe5a65cb503e25fdcf1377ee",
    "A2_P4.csv": "6b20fb815d6df40ff9ad4c0ecbf3bc7a82762ac96b8fd7df7170504715d75a00",
    "A3_P4.csv": "6c3bffcca0e81664398423c5c4828c3d7d4698a69e804fe69368e53bb8a9f6e3",
}


class GroupDROProtocolError(RuntimeError):
    """Raised when a frozen scientific or label-boundary rule is violated."""


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def canonical_position_order(values: Iterable[str], *, source_only: bool = True) -> tuple[str, ...]:
    allowed = SOURCE_POSITIONS if source_only else ALL_POSITIONS
    observed = set(str(value) for value in values)
    unknown = observed.difference(allowed)
    if unknown:
        raise GroupDROProtocolError(f"Unknown position groups: {sorted(unknown)}")
    ordered = tuple(position for position in allowed if position in observed)
    if not ordered:
        raise GroupDROProtocolError("No declared source-position group is present")
    return ordered


def validate_condition_block_disjoint(
    train_condition_ids: Iterable[str], validation_condition_ids: Iterable[str]
) -> None:
    overlap = set(train_condition_ids).intersection(validation_condition_ids)
    if overlap:
        raise GroupDROProtocolError("Source condition block crosses train/validation boundary")


def validate_lopo_position_isolation(
    training_positions: Iterable[str], held_positions: Iterable[str], held_position: str
) -> None:
    training = set(training_positions)
    held = set(held_positions)
    expected_training = set(SOURCE_POSITIONS).difference({held_position})
    if training != expected_training or held != {held_position}:
        raise GroupDROProtocolError("Held source position leaked into LOPO development")


def stable_rng(*parts: object) -> np.random.Generator:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little", signed=False)
    return np.random.default_rng(seed)


def validate_frozen_configuration(config: Mapping[str, Any]) -> None:
    if tuple(float(value) for value in config["groupdro"]["eta_grid"]) != ETA_GRID:
        raise GroupDROProtocolError("Frozen GroupDRO eta grid changed")
    if tuple(config["source_positions"]) != SOURCE_POSITIONS:
        raise GroupDROProtocolError("Source-position group declaration changed")
    if tuple(int(value) for value in config["seeds"]) != (42, 43, 44, 45, 46):
        raise GroupDROProtocolError("Canonical seed declaration changed")
    if config["groupdro"]["initial_q"] != "uniform":
        raise GroupDROProtocolError("GroupDRO q initialization is not uniform")
    if config["groupdro"]["update_frequency"] != "optimizer_step":
        raise GroupDROProtocolError("GroupDRO q update frequency changed")
    if config["groupdro"]["group_variable"] != "source_position":
        raise GroupDROProtocolError("GroupDRO primary group variable changed")
    if config["groupdro"]["q_momentum"] is not None:
        raise GroupDROProtocolError("GroupDRO q momentum is prohibited")
    if config["groupdro"]["q_clipping"] is not None:
        raise GroupDROProtocolError("GroupDRO q clipping is prohibited")


@dataclass(frozen=True)
class BatchPlan:
    """A deterministic balanced ordering shared unchanged by ERM and GroupDRO."""

    batches: tuple[np.ndarray, ...]
    audits: tuple[dict[str, Any], ...]
    source_positions: tuple[str, ...]
    replacement_policy: str
    replacement_count: int

    @property
    def sample_count(self) -> int:
        return int(sum(len(batch) for batch in self.batches))

    @property
    def signature_sha256(self) -> str:
        digest = hashlib.sha256()
        for batch in self.batches:
            digest.update(np.asarray(batch, dtype="<i8").tobytes(order="C"))
        return digest.hexdigest()


def _class_balanced_order(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Interleave shuffled class queues without changing source samples."""

    queues: dict[int, list[int]] = {}
    for label in sorted(np.unique(labels[indices]).tolist()):
        selected = np.asarray(indices[labels[indices] == label], dtype=np.int64)
        queues[int(label)] = rng.permutation(selected).tolist()
    class_cycle = rng.permutation(np.asarray(sorted(queues), dtype=np.int64)).tolist()
    ordered: list[int] = []
    while any(queues.values()):
        for label in class_cycle:
            if queues[int(label)]:
                ordered.append(queues[int(label)].pop())
    return np.asarray(ordered, dtype=np.int64)


def make_group_balanced_batches(
    position_labels: Sequence[str] | np.ndarray,
    class_labels: Sequence[int] | np.ndarray,
    *,
    seed: int,
    epoch: int,
    stage: str,
    batch_size: int,
    groups: Sequence[str] | None = None,
) -> BatchPlan:
    """Create group-balanced and class-interleaved batches for one epoch.

    Positions with fewer rows are deterministically repeated only after each of
    their original rows is consumed. This policy is exposed in every batch
    audit and is applied identically to the two model families.
    """

    positions = np.asarray(position_labels, dtype=str)
    labels = np.asarray(class_labels, dtype=np.int64)
    if positions.shape != labels.shape or positions.ndim != 1:
        raise ValueError("Position and class labels must be aligned one-dimensional arrays")
    group_order = tuple(groups) if groups is not None else canonical_position_order(positions)
    if tuple(group_order) != canonical_position_order(group_order):
        raise GroupDROProtocolError("Group order must use canonical source-position order")
    if batch_size < len(group_order):
        raise ValueError("Batch size cannot represent every source position")

    ordered_by_group: dict[str, np.ndarray] = {}
    original_counts: dict[str, int] = {}
    for position in group_order:
        selected = np.flatnonzero(positions == position).astype(np.int64)
        if len(selected) == 0:
            raise GroupDROProtocolError(f"Missing position in valid batch plan: {position}")
        original_counts[position] = int(len(selected))
        ordered_by_group[position] = _class_balanced_order(
            selected,
            labels,
            rng=stable_rng("group-balanced", seed, epoch, stage, position),
        )

    target_per_group = max(original_counts.values())
    replacement_count = 0
    for position in group_order:
        order = ordered_by_group[position]
        if len(order) < target_per_group:
            repeats: list[np.ndarray] = [order]
            needed = target_per_group - len(order)
            repeat_round = 1
            while needed > 0:
                refreshed = _class_balanced_order(
                    np.flatnonzero(positions == position).astype(np.int64),
                    labels,
                    rng=stable_rng("group-balanced-replacement", seed, epoch, stage, position, repeat_round),
                )
                take = min(needed, len(refreshed))
                repeats.append(refreshed[:take])
                needed -= take
                repeat_round += 1
            ordered_by_group[position] = np.concatenate(repeats)
            replacement_count += target_per_group - original_counts[position]

    cursors = {position: 0 for position in group_order}
    batches: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    batch_index = 0
    while any(cursors[position] < target_per_group for position in group_order):
        remaining = {
            position: target_per_group - cursors[position] for position in group_order
        }
        base = min(batch_size // len(group_order), min(remaining.values()))
        if base <= 0:
            raise GroupDROProtocolError("Incomplete final batch would omit a source position")
        counts = {position: base for position in group_order}
        remaining_slots = batch_size - base * len(group_order)
        rotation = group_order[batch_index % len(group_order) :] + group_order[: batch_index % len(group_order)]
        for position in rotation:
            if remaining_slots <= 0:
                break
            if remaining[position] > counts[position]:
                counts[position] += 1
                remaining_slots -= 1
        selected_parts: list[np.ndarray] = []
        for position in group_order:
            start = cursors[position]
            stop = start + counts[position]
            selected_parts.append(ordered_by_group[position][start:stop])
            cursors[position] = stop
        selected = np.concatenate(selected_parts)
        shuffled = stable_rng("batch-order", seed, epoch, stage, batch_index).permutation(selected)
        observed_positions = positions[shuffled]
        observed_labels = labels[shuffled]
        position_counts = {
            position: int(np.count_nonzero(observed_positions == position))
            for position in group_order
        }
        if set(position_counts) != set(group_order) or min(position_counts.values()) <= 0:
            raise GroupDROProtocolError("A valid training batch omitted a source position")
        tag_counts = {
            position: {
                str(label): int(np.count_nonzero(observed_labels[observed_positions == position] == label))
                for label in CLASS_ORDER
            }
            for position in group_order
        }
        batches.append(np.asarray(shuffled, dtype=np.int64))
        audits.append(
            {
                "epoch": int(epoch),
                "stage": stage,
                "batch_index": int(batch_index),
                "sample_count": int(len(shuffled)),
                "position_counts": position_counts,
                "tagid_counts": tag_counts,
                "incomplete_final_batch": len(shuffled) < batch_size,
                "replacement_policy": "repeat_only_after_all_original_position_rows_consumed",
                "replacement_active": replacement_count > 0,
            }
        )
        batch_index += 1
    return BatchPlan(
        batches=tuple(batches),
        audits=tuple(audits),
        source_positions=group_order,
        replacement_policy="repeat_only_after_all_original_position_rows_consumed",
        replacement_count=int(replacement_count),
    )


def group_mean_losses(
    per_example_losses: torch.Tensor,
    batch_positions: Sequence[str] | np.ndarray,
    groups: Sequence[str],
) -> dict[str, torch.Tensor]:
    positions = np.asarray(batch_positions, dtype=str)
    if len(positions) != len(per_example_losses):
        raise ValueError("Per-example losses and position labels are misaligned")
    result: dict[str, torch.Tensor] = {}
    for group in groups:
        mask = torch.from_numpy(positions == group).to(per_example_losses.device)
        if not bool(mask.any()):
            raise GroupDROProtocolError(f"GroupDRO batch is missing {group}")
        result[group] = per_example_losses[mask].mean()
    return result


@dataclass
class GroupDROState:
    """Detached exponentiated-gradient weights over declared source positions."""

    eta: float
    groups: tuple[str, ...]
    epsilon: float = 1e-12
    q: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.eta not in ETA_GRID:
            raise GroupDROProtocolError(f"Eta is outside the frozen grid: {self.eta}")
        if self.groups != canonical_position_order(self.groups):
            raise GroupDROProtocolError("GroupDRO state has noncanonical group order")
        self.q = torch.full((len(self.groups),), 1.0 / len(self.groups), dtype=torch.float64)
        self.q.requires_grad_(False)

    def update(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = torch.stack([losses[group] for group in self.groups])
        detached = values.detach().to(dtype=torch.float64, device="cpu")
        if detached.requires_grad or not bool(torch.isfinite(detached).all()):
            raise GroupDROProtocolError("Group-loss update must use finite detached values")
        log_weights = torch.log(torch.clamp(self.q, min=self.epsilon)) + self.eta * detached
        stabilized = log_weights - torch.max(log_weights)
        weights = torch.exp(stabilized)
        normalizer = torch.clamp(weights.sum(), min=self.epsilon)
        self.q = (weights / normalizer).detach().cpu()
        self.q.requires_grad_(False)
        if not bool(torch.isfinite(self.q).all()) or bool((self.q < 0).any()):
            raise GroupDROProtocolError("GroupDRO q is not finite and non-negative")
        if not torch.isclose(self.q.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12):
            raise GroupDROProtocolError("GroupDRO q does not sum to one")
        return self.q.clone()

    def weighted_loss(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = torch.stack([losses[group] for group in self.groups])
        weights = self.q.to(device=values.device, dtype=values.dtype).detach()
        return torch.sum(weights * values)

    def entropy(self) -> float:
        values = self.q.detach().cpu().numpy()
        return float(-np.sum(values * np.log(np.clip(values, self.epsilon, None))))


def erm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    per_example = functional.cross_entropy(logits, labels, reduction="none")
    return per_example.mean(), per_example


def groupdro_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positions: Sequence[str] | np.ndarray,
    state: GroupDROState,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    per_example = functional.cross_entropy(logits, labels, reduction="none")
    losses = group_mean_losses(per_example, positions, state.groups)
    q = state.update(losses)
    return state.weighted_loss(losses), per_example, losses, q


def position_group_losses(
    logits: np.ndarray,
    labels: np.ndarray,
    positions: Sequence[str] | np.ndarray,
    groups: Sequence[str],
) -> dict[str, float]:
    tensor_logits = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    tensor_labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    per_example = functional.cross_entropy(tensor_logits, tensor_labels, reduction="none")
    return {
        group: float(loss.detach().cpu().item())
        for group, loss in group_mean_losses(per_example, positions, groups).items()
    }


def per_class_precision(confusion: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(confusion, dtype=np.float64)
    values = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    return [
        float(value)
        for value in np.divide(values, predicted, out=np.zeros_like(values), where=predicted > 0)
    ]


def deterministic_block_majority_vote(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    condition_ids: Sequence[str] | np.ndarray,
    *,
    expected_block_size: int = 50,
    class_order: Sequence[int] = CLASS_ORDER,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate row predictions once per condition by vote and fixed tie order."""

    true_values = np.asarray(labels, dtype=np.int64)
    predicted_values = np.asarray(predictions, dtype=np.int64)
    if true_values.shape != predicted_values.shape or len(condition_ids) != len(true_values):
        raise ValueError("Block-vote inputs are not aligned")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition_id in enumerate(condition_ids):
        groups.setdefault(str(condition_id), []).append(index)
    block_labels: list[int] = []
    block_predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    rank = {int(value): index for index, value in enumerate(class_order)}
    for block_index, condition_id in enumerate(sorted(groups, key=str.casefold)):
        selected = groups[condition_id]
        if len(selected) != expected_block_size:
            raise GroupDROProtocolError(f"Incomplete P4 condition block: {condition_id}")
        block_truth = np.unique(true_values[selected])
        if len(block_truth) != 1:
            raise GroupDROProtocolError(f"P4 block labels disagree: {condition_id}")
        votes = np.bincount(predicted_values[selected], minlength=len(class_order))
        maximal = np.flatnonzero(votes == votes.max()).tolist()
        chosen = min(maximal, key=lambda value: rank[int(value)])
        agreement = float(votes[chosen] / len(selected))
        block_labels.append(int(block_truth[0]))
        block_predictions.append(int(chosen))
        rows.append(
            {
                "condition_block_index": int(block_index),
                "sample_count": int(len(selected)),
                "true_label": int(block_truth[0]),
                "predicted_label": int(chosen),
                "within_block_agreement": agreement,
                "vote_counts": [int(value) for value in votes.tolist()],
            }
        )
    metrics = classification_metrics(
        np.asarray(block_labels, dtype=np.int64), np.asarray(block_predictions, dtype=np.int64)
    )
    metrics["per_class_precision"] = per_class_precision(metrics["confusion_matrix"])
    return metrics, rows


def source_eta_summary(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for eta in ETA_GRID:
        selected = [row for row in run_rows if float(row["eta"]) == eta]
        if len(selected) != 15:
            raise GroupDROProtocolError(f"Eta {eta:.2f} does not have 15 source LOPO units")
        means_by_position = {
            position: float(
                np.mean([float(row["held_macro_f1"]) for row in selected if row["held_position"] == position])
            )
            for position in SOURCE_POSITIONS
        }
        units = np.asarray([float(row["held_macro_f1"]) for row in selected], dtype=np.float64)
        accuracy = np.asarray([float(row["held_accuracy"]) for row in selected], dtype=np.float64)
        max_q = np.asarray([float(row.get("max_q_weight", np.nan)) for row in selected], dtype=np.float64)
        summaries.append(
            {
                "eta": eta,
                "mean_held_source_macro_f1": float(units.mean()),
                "minimum_held_position_mean_macro_f1": float(min(means_by_position.values())),
                "held_position_means": means_by_position,
                "population_sd_held_position_seed_macro_f1": float(units.std(ddof=0)),
                "mean_held_source_accuracy": float(accuracy.mean()),
                "source_worst_position_macro_f1": float(min(means_by_position.values())),
                "mean_max_q_weight": float(np.nanmean(max_q)),
                "q_weight_stability_population_sd": float(np.nanstd(max_q, ddof=0)),
            }
        )
    return summaries


def select_eta_source_only(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen five-step selector before any P4 labels are opened."""

    summaries = source_eta_summary(run_rows)
    best_mean = max(row["mean_held_source_macro_f1"] for row in summaries)
    eligible = [
        row
        for row in summaries
        if row["mean_held_source_macro_f1"] >= best_mean - 0.01 - 1e-12
    ]
    best_minimum = max(row["minimum_held_position_mean_macro_f1"] for row in eligible)
    minimum_tied = [
        row
        for row in eligible
        if best_minimum - row["minimum_held_position_mean_macro_f1"] <= 0.005 + 1e-12
    ]
    best_sd = min(row["population_sd_held_position_seed_macro_f1"] for row in minimum_tied)
    sd_tied = [
        row
        for row in minimum_tied
        if row["population_sd_held_position_seed_macro_f1"] <= best_sd + 0.001 + 1e-12
    ]
    selected = min(sd_tied, key=lambda row: float(row["eta"]))
    decision = {
        "selector": "frozen_lexicographic_source_only_v1",
        "selection_input_p4_accessed": False,
        "best_mean_held_source_macro_f1": best_mean,
        "eligibility_threshold": best_mean - 0.01,
        "eligible_etas": [row["eta"] for row in eligible],
        "minimum_position_tied_etas": [row["eta"] for row in minimum_tied],
        "sd_tied_etas": [row["eta"] for row in sd_tied],
        "selected_eta": float(selected["eta"]),
        "all_eta_summaries": summaries,
    }
    decision["selection_sha256"] = canonical_json_sha256(decision)
    return decision


def source_worst_position_metrics(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(run_rows) != 15:
        raise GroupDROProtocolError("Source endpoint requires 15 fold/seed runs")
    by_position = {
        position: [float(row["held_macro_f1"]) for row in run_rows if row["held_position"] == position]
        for position in SOURCE_POSITIONS
    }
    means = {position: float(np.mean(values)) for position, values in by_position.items()}
    all_units = np.asarray([value for values in by_position.values() for value in values], dtype=np.float64)
    return {
        "held_position_mean_macro_f1": means,
        "worst_position_macro_f1": float(min(means.values())),
        "mean_held_source_macro_f1": float(all_units.mean()),
        "held_position_variance": float(np.var(list(means.values()), ddof=0)),
        "maximum_position_to_position_gap": float(max(means.values()) - min(means.values())),
        "held_position_seed_population_sd": float(all_units.std(ddof=0)),
    }


def source_retention_guardrail(groupdro_mean_macro_f1: float, erm_mean_macro_f1: float) -> dict[str, Any]:
    contrast = float(groupdro_mean_macro_f1) - float(erm_mean_macro_f1)
    return {
        "groupdro_minus_erm_mean_macro_f1": contrast,
        "maximum_permitted_decline": -0.02,
        "retained": contrast >= -0.02,
    }


def percentile_interval(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("Cannot form an interval from empty or non-finite samples")
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def paired_source_worst_bootstrap(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int = 10_000,
    seed: int = 20260804,
    fold_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bootstrap a paired worst-position contrast while retaining fold/seed units."""

    active_folds = tuple(fold_ids) if fold_ids is not None else tuple(FOLD_HELD_POSITION)
    if not active_folds or set(active_folds).difference(FOLD_HELD_POSITION):
        raise GroupDROProtocolError("Source bootstrap has invalid active folds")
    by_fold: dict[str, list[Mapping[str, Any]]] = {fold: [] for fold in active_folds}
    for row in paired_rows:
        fold = str(row["fold_id"])
        if fold not in FOLD_HELD_POSITION:
            raise GroupDROProtocolError(f"Unexpected source fold: {fold}")
        if fold in by_fold:
            by_fold[fold].append(row)
    if any(len(rows) != 5 for rows in by_fold.values()):
        raise GroupDROProtocolError("Paired source bootstrap requires five seeds per source fold")
    rng = np.random.default_rng(seed)
    contrasts = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        erm_means: list[float] = []
        groupdro_means: list[float] = []
        for fold in active_folds:
            rows = by_fold[fold]
            selected = rng.integers(0, len(rows), size=len(rows))
            erm_means.append(float(np.mean([float(rows[index]["erm_macro_f1"]) for index in selected])))
            groupdro_means.append(
                float(np.mean([float(rows[index]["groupdro_macro_f1"]) for index in selected]))
            )
        contrasts[replicate] = min(groupdro_means) - min(erm_means)
    return {
        "method": "paired_held_position_seed_bootstrap",
        "active_folds": list(active_folds),
        "resamples": int(resamples),
        "seed": int(seed),
        "point_estimate": float(
            min(
                np.mean([float(row["groupdro_macro_f1"]) for row in by_fold[fold]])
                for fold in active_folds
            )
            - min(
                np.mean([float(row["erm_macro_f1"]) for row in by_fold[fold]])
                for fold in active_folds
            )
        ),
        "interval_95": percentile_interval(contrasts),
    }


def paired_source_mean_bootstrap(
    paired_rows: Sequence[Mapping[str, Any]], *, resamples: int = 10_000, seed: int = 20260804
) -> dict[str, Any]:
    """Bootstrap the paired mean held-source Macro-F1 contrast by fold and seed."""

    by_fold: dict[str, list[Mapping[str, Any]]] = {fold: [] for fold in FOLD_HELD_POSITION}
    for row in paired_rows:
        fold = str(row["fold_id"])
        if fold not in by_fold:
            raise GroupDROProtocolError(f"Unexpected source fold: {fold}")
        by_fold[fold].append(row)
    if any(len(rows) != 5 for rows in by_fold.values()):
        raise GroupDROProtocolError("Paired source bootstrap requires five seeds per source fold")
    rng = np.random.default_rng(seed)
    contrasts = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        erm_values: list[float] = []
        groupdro_values: list[float] = []
        for fold in FOLD_HELD_POSITION:
            rows = by_fold[fold]
            selected = rng.integers(0, len(rows), size=len(rows))
            erm_values.extend(float(rows[index]["erm_macro_f1"]) for index in selected)
            groupdro_values.extend(float(rows[index]["groupdro_macro_f1"]) for index in selected)
        contrasts[replicate] = float(np.mean(groupdro_values) - np.mean(erm_values))
    point_estimate = float(
        np.mean([float(row["groupdro_macro_f1"]) for rows in by_fold.values() for row in rows])
        - np.mean([float(row["erm_macro_f1"]) for rows in by_fold.values() for row in rows])
    )
    return {
        "method": "paired_held_position_seed_mean_bootstrap",
        "resamples": int(resamples),
        "seed": int(seed),
        "point_estimate": point_estimate,
        "interval_95": percentile_interval(contrasts),
    }


def paired_tagid_stratified_block_bootstrap(
    labels: Sequence[int] | np.ndarray,
    erm_predictions: np.ndarray,
    groupdro_predictions: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 20260804,
    resample_seeds: bool = False,
    metric_name: str = "macro_f1",
) -> dict[str, Any]:
    """Paired block bootstrap; rows are never treated as inferential units."""

    true_labels = np.asarray(labels, dtype=np.int64)
    erm = np.asarray(erm_predictions, dtype=np.int64)
    groupdro = np.asarray(groupdro_predictions, dtype=np.int64)
    if erm.shape != groupdro.shape or erm.ndim != 2 or erm.shape[1] != len(true_labels):
        raise ValueError("Paired P4 predictions must be [seed, condition_block]")
    if metric_name not in {"macro_f1", "accuracy"}:
        raise ValueError("P4 bootstrap metric must be macro_f1 or accuracy")
    strata = {label: np.flatnonzero(true_labels == label) for label in CLASS_ORDER}
    if any(len(indices) == 0 for indices in strata.values()):
        raise GroupDROProtocolError("P4 block bootstrap requires every TagID stratum")
    rng = np.random.default_rng(seed)
    selection_weights = np.zeros((resamples, len(true_labels)), dtype=np.float64)
    replicate_indices = np.arange(resamples)[:, None]
    for indices in strata.values():
        chosen = rng.integers(0, len(indices), size=(resamples, len(indices)))
        np.add.at(selection_weights, (replicate_indices, indices[chosen]), 1.0)
    true_one_hot = np.eye(len(CLASS_ORDER), dtype=np.float64)[true_labels]

    def bootstrap_metric(predictions: np.ndarray) -> np.ndarray:
        prediction_one_hot = np.eye(len(CLASS_ORDER), dtype=np.float64)[predictions]
        confusion = np.einsum(
            "rb,bi,bj->rij",
            selection_weights,
            true_one_hot,
            prediction_one_hot,
            optimize=True,
        )
        diagonal = np.diagonal(confusion, axis1=1, axis2=2)
        if metric_name == "accuracy":
            return diagonal.sum(axis=1) / np.maximum(confusion.sum(axis=(1, 2)), 1.0)
        row_totals = confusion.sum(axis=2)
        column_totals = confusion.sum(axis=1)
        precision = np.divide(
            diagonal,
            column_totals,
            out=np.zeros_like(diagonal),
            where=column_totals > 0.0,
        )
        recall = np.divide(
            diagonal,
            row_totals,
            out=np.zeros_like(diagonal),
            where=row_totals > 0.0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0.0,
        )
        return f1.mean(axis=1)

    contrasts_by_seed = np.stack(
        [
            bootstrap_metric(groupdro[seed_index]) - bootstrap_metric(erm[seed_index])
            for seed_index in range(erm.shape[0])
        ],
        axis=0,
    )
    if resample_seeds:
        seed_indices = rng.integers(0, erm.shape[0], size=(resamples, erm.shape[0]))
        contrasts = contrasts_by_seed[seed_indices, np.arange(resamples)[:, None]].mean(axis=1)
    else:
        contrasts = contrasts_by_seed.mean(axis=0)
    point_seed_contrasts = [
        float(
            classification_metrics(true_labels, groupdro[index])[metric_name]
            - classification_metrics(true_labels, erm[index])[metric_name]
        )
        for index in range(erm.shape[0])
    ]
    return {
        "method": "paired_tagid_stratified_condition_block_bootstrap",
        "metric": metric_name,
        "resamples": int(resamples),
        "seed": int(seed),
        "resample_training_seeds": bool(resample_seeds),
        "point_estimate": float(np.mean(point_seed_contrasts)),
        "interval_95": percentile_interval(contrasts),
        "seed_wise_contrasts": point_seed_contrasts,
    }


def classify_interpretation(
    *,
    governed_inputs_available: bool,
    protocol_defect: bool,
    source_worst_interval: Sequence[float] | None,
    p4_interval: Sequence[float] | None,
    source_mean_contrast: float | None,
    source_mean_interval: Sequence[float] | None,
    p4_point_contrast: float | None,
    unstable: bool,
) -> str:
    if protocol_defect:
        return "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"
    if not governed_inputs_available:
        return "BLOCKED_GOVERNED_INPUTS"
    if source_worst_interval is None or p4_interval is None or source_mean_contrast is None:
        raise GroupDROProtocolError("Interpretation requested before required paired intervals")
    source_gain = float(source_worst_interval[0]) > 0.0
    p4_gain = float(p4_interval[0]) > 0.0
    p4_harm = float(p4_interval[1]) < 0.0 or (p4_point_contrast is not None and p4_point_contrast < 0.0)
    mean_harm = float(source_mean_contrast) < -0.02 and (
        source_mean_interval is None or float(source_mean_interval[1]) < -0.02
    )
    if p4_harm:
        return "GROUPDRO_HARMS_P4_TRANSFER"
    if source_gain and mean_harm:
        return "GROUPDRO_MEAN_PERFORMANCE_TRADEOFF"
    if source_gain and p4_gain:
        return "GROUPDRO_SOURCE_ROBUSTNESS_AND_P4_BENEFIT_CONFIRMED"
    if source_gain:
        return "GROUPDRO_SOURCE_ROBUSTNESS_WITHOUT_P4_BENEFIT"
    if p4_gain:
        return "GROUPDRO_P4_BENEFIT_WITHOUT_SOURCE_WORST_GROUP_GAIN"
    if unstable:
        return "GROUPDRO_MIXED_OR_UNSTABLE_RESULT"
    return "GROUPDRO_NO_RELIABLE_BENEFIT"


@dataclass(frozen=True)
class SealedP4Data:
    signals: np.ndarray
    condition_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    metadata_rows: tuple[dict[str, str], ...]


class P4LabelSeal:
    """One-way seal that separates prediction generation from P4 label access."""

    forbidden_registry_fields = {"tagid", "tag_id", "label", "label_index", "class", "class_index"}

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._predictions_frozen = False
        self._labels_opened = False
        self.access_log: list[dict[str, Any]] = []

    def _raw_csv_paths(self) -> tuple[Path, ...]:
        paths = tuple(self.root / name for name in P4_RAW_FILENAMES)
        return paths if all(path.is_file() for path in paths) else ()

    def _raw_csv_mode(self) -> bool:
        return bool(self._raw_csv_paths())

    @staticmethod
    def _raw_block_id(path: Path, row_index: int) -> str:
        return f"{path.name}:opaque_block_{row_index // 50:02d}"

    def _load_raw_unlabelled(self) -> SealedP4Data:
        signals: list[np.ndarray] = []
        rows: list[dict[str, str]] = []
        file_hashes: dict[str, str] = {}
        for path in self._raw_csv_paths():
            digest = sha256_file(path)
            if digest != P4_RAW_FILE_HASHES[path.name]:
                raise GroupDROProtocolError("P4 raw CSV custody hash differs")
            file_hashes[path.name] = digest
            expected_surface = path.stem.split("_")[0]
            row_count = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != P4_RAW_HEADER:
                    raise GroupDROProtocolError("P4 raw CSV schema differs")
                for row_index, row in enumerate(reader):
                    active_surface = [
                        name for name in ("A1", "A2", "A3") if float(row[name]) == 1.0
                    ]
                    active_position = [
                        name for name in ("P1", "P2", "P3", "P4") if float(row[name]) == 1.0
                    ]
                    if active_surface != [expected_surface] or active_position != ["P4"]:
                        raise GroupDROProtocolError("P4 raw CSV domain metadata differs")
                    signal = np.fromiter(
                        (float(row[column]) for column in P4_RAW_HEADER[9:]),
                        dtype=np.float64,
                        count=281,
                    )
                    if signal.shape != (281,) or not np.isfinite(signal).all():
                        raise GroupDROProtocolError("P4 raw signal custody differs")
                    signals.append(signal)
                    rows.append(
                        {
                            "sample_id": f"{path.stem}:opaque_row_{row_index:04d}",
                            "raw_condition_id": self._raw_block_id(path, row_index),
                            "position": "P4",
                        }
                    )
                    row_count += 1
            if row_count != 1050:
                raise GroupDROProtocolError("P4 raw CSV row count differs")
        values = np.ascontiguousarray(np.vstack(signals), dtype=np.float64)
        condition_ids = tuple(row["raw_condition_id"] for row in rows)
        condition_counts = Counter(condition_ids)
        if values.shape != (3150, 281) or len(condition_counts) != 63 or set(condition_counts.values()) != {50}:
            raise GroupDROProtocolError("P4 raw condition-block custody differs")
        self.access_log.append(
            {
                "event": "P4_SIGNALS_AND_OPAQUE_METADATA_OPENED",
                "labels_accessed": False,
                "signal_sha256": array_sha256(values),
                "registry_sha256": canonical_json_sha256(file_hashes),
                "p4_input_mode": "GOVERNED_RAW_CSV",
                "tagid_column_accessed": False,
                "er_column_accessed": False,
            }
        )
        return SealedP4Data(
            signals=values,
            condition_ids=condition_ids,
            sample_ids=tuple(row["sample_id"] for row in rows),
            metadata_rows=tuple(rows),
        )

    def _open_raw_labels(self) -> np.ndarray:
        labels: list[int] = []
        block_keys: dict[str, set[tuple[int, int, str]]] = {}
        file_hashes: dict[str, str] = {}
        for path in self._raw_csv_paths():
            digest = sha256_file(path)
            if digest != P4_RAW_FILE_HASHES[path.name]:
                raise GroupDROProtocolError("P4 raw CSV custody hash differs after prediction freeze")
            file_hashes[path.name] = digest
            expected_surface = path.stem.split("_")[0]
            row_count = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != P4_RAW_HEADER:
                    raise GroupDROProtocolError("P4 raw CSV schema changed after prediction freeze")
                for row_index, row in enumerate(reader):
                    tag_id = int(float(row["TagID"]))
                    er = int(float(row["ER"]))
                    if tag_id not in range(1, 8) or er not in {0, 1, 2}:
                        raise GroupDROProtocolError("P4 raw label metadata differs")
                    label = tag_id - 1
                    labels.append(label)
                    block_id = self._raw_block_id(path, row_index)
                    block_keys.setdefault(block_id, set()).add((label, er, expected_surface))
                    row_count += 1
            if row_count != 1050:
                raise GroupDROProtocolError("P4 raw CSV row count changed after prediction freeze")
        values = np.asarray(labels, dtype=np.int64)
        if values.shape != (3150,) or set(values.tolist()) != set(CLASS_ORDER):
            raise GroupDROProtocolError("P4 raw label custody differs")
        if len(block_keys) != 63 or any(len(keys) != 1 for keys in block_keys.values()):
            raise GroupDROProtocolError("P4 opaque blocks do not map to TagID x ER x surface")
        if len({next(iter(keys)) for keys in block_keys.values()}) != 63:
            raise GroupDROProtocolError("P4 condition-block identity differs")
        self.access_log.append(
            {
                "event": "P4_LABELS_OPENED_AFTER_PREDICTION_FREEZE",
                "labels_accessed": True,
                "label_sha256": array_sha256(values),
                "registry_sha256": canonical_json_sha256(file_hashes),
                "p4_input_mode": "GOVERNED_RAW_CSV",
                "tagid_column_accessed": True,
                "er_column_accessed": True,
            }
        )
        return np.ascontiguousarray(values, dtype=np.int64)

    def load_unlabelled(self) -> SealedP4Data:
        if self._raw_csv_mode():
            return self._load_raw_unlabelled()
        signals_path = self.root / "p4_signals_float64.npy"
        registry_path = self.root / "p4_unlabelled_registry.csv"
        if not signals_path.is_file() or not registry_path.is_file():
            raise FileNotFoundError("P4 sealed signals or unlabelled registry is missing")
        with registry_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise GroupDROProtocolError("P4 unlabelled registry has no header")
            normalized = {name.casefold() for name in reader.fieldnames}
            if normalized.intersection(self.forbidden_registry_fields):
                raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY")
            required = {"sample_id", "raw_condition_id", "position"}
            if not required.issubset(set(reader.fieldnames)):
                raise GroupDROProtocolError("P4 unlabelled registry does not carry required opaque metadata")
            rows = tuple(dict(row) for row in reader)
        signals = np.load(signals_path, allow_pickle=False)
        if signals.shape != (3150, 281) or signals.dtype.str != "<f8" or len(rows) != len(signals):
            raise GroupDROProtocolError("P4 sealed input custody differs")
        if {row["position"] for row in rows} != {"P4"}:
            raise GroupDROProtocolError("P4 unlabelled registry contains a non-P4 row")
        condition_ids = tuple(row["raw_condition_id"] for row in rows)
        condition_counts = {condition_id: condition_ids.count(condition_id) for condition_id in set(condition_ids)}
        if len(condition_counts) != 63 or set(condition_counts.values()) != {50}:
            raise GroupDROProtocolError("P4 opaque condition-block structure differs")
        self.access_log.append(
            {
                "event": "P4_SIGNALS_AND_OPAQUE_METADATA_OPENED",
                "labels_accessed": False,
                "signal_sha256": array_sha256(signals),
                "registry_sha256": sha256_file(registry_path),
            }
        )
        return SealedP4Data(
            signals=np.ascontiguousarray(signals, dtype=np.float64),
            condition_ids=condition_ids,
            sample_ids=tuple(row["sample_id"] for row in rows),
            metadata_rows=rows,
        )

    def freeze_predictions(self, predictions: Mapping[str, np.ndarray], output_path: Path) -> dict[str, Any]:
        if self._labels_opened:
            raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY")
        if not predictions:
            raise ValueError("No P4 predictions were supplied for freezing")
        serializable = {key: np.asarray(value, dtype=np.int64) for key, value in predictions.items()}
        if any(value.shape != (3150,) for value in serializable.values()):
            raise GroupDROProtocolError("P4 prediction freeze expects 3,150 row predictions per model")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **serializable)
        temporary.replace(output_path)
        prediction_hashes = {key: array_sha256(value) for key, value in serializable.items()}
        self._predictions_frozen = True
        record = {
            "event": "P4_PREDICTIONS_FROZEN",
            "labels_accessed": False,
            "prediction_bundle_sha256": sha256_file(output_path),
            "prediction_hashes": prediction_hashes,
        }
        self.access_log.append(record)
        return record

    def recover_frozen_predictions(
        self,
        input_path: Path,
        *,
        expected_keys: Iterable[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Adopt an already-hashed prediction bundle after an interrupted post-freeze process."""

        if self._labels_opened:
            raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY")
        if not Path(input_path).is_file():
            raise FileNotFoundError(input_path)
        with np.load(input_path, allow_pickle=False) as payload:
            recovered = {
                key: np.ascontiguousarray(payload[key], dtype=np.int64) for key in payload.files
            }
        required = set(expected_keys)
        if set(recovered) != required or any(values.shape != (3150,) for values in recovered.values()):
            raise GroupDROProtocolError("P4 frozen prediction bundle identity differs")
        self._predictions_frozen = True
        record = {
            "event": "P4_FROZEN_PREDICTIONS_RECOVERED_FOR_POST_FREEZE_SCORING",
            "labels_accessed": False,
            "prediction_bundle_sha256": sha256_file(Path(input_path)),
            "prediction_hashes": {key: array_sha256(values) for key, values in recovered.items()},
        }
        self.access_log.append(record)
        return recovered, record

    def open_labels(self) -> np.ndarray:
        if not self._predictions_frozen:
            raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY")
        if self._labels_opened:
            raise GroupDROProtocolError("FAIL_P4_LABEL_BOUNDARY")
        if self._raw_csv_mode():
            labels = self._open_raw_labels()
            self._labels_opened = True
            return labels
        labels_path = self.root / "p4_labels_int64.npy"
        if not labels_path.is_file():
            raise FileNotFoundError(labels_path)
        labels = np.load(labels_path, allow_pickle=False)
        if labels.shape != (3150,) or labels.dtype.str != "<i8" or np.any((labels < 0) | (labels >= 7)):
            raise GroupDROProtocolError("P4 label custody differs")
        self._labels_opened = True
        self.access_log.append(
            {
                "event": "P4_LABELS_OPENED_AFTER_PREDICTION_FREEZE",
                "labels_accessed": True,
                "label_sha256": array_sha256(labels),
            }
        )
        return np.ascontiguousarray(labels, dtype=np.int64)
