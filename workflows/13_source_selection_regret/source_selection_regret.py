"""Source-only leave-one-position-out selection-regret analysis.

The workflow deliberately has no P4 loader.  It accepts only a small explicit
allowlist of source-side evidence and rejects target/P4-classified paths before
opening them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import random
import statistics
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


CANDIDATES = (
    "C0_NEUTRAL_ERM_1DCNN",
    "C1_FIRST_DIFFERENCE_ERM_1DCNN",
    "C2_SOURCE_EUCLIDEAN_NCM_READOUT",
    "C3_SOURCE_CORAL_ERM_1DCNN",
)
FOLDS = ("S1", "S2", "S3")
POSITIONS = ("P1", "P2", "P3")
SEEDS = (42, 43, 44, 45, 46)
FOLD_TO_POSITION = dict(zip(FOLDS, POSITIONS))
POSITION_TO_FOLD = dict(zip(POSITIONS, FOLDS))
CLASS_COUNT = 7
NUMERICAL_TOLERANCE = 1e-12
SECONDARY_TOLERANCE = 0.01

EVENTS = (
    {"event_id": "HOLD_P1", "allowed_positions": ("P2", "P3"), "held_position": "P1"},
    {"event_id": "HOLD_P2", "allowed_positions": ("P1", "P3"), "held_position": "P2"},
    {"event_id": "HOLD_P3", "allowed_positions": ("P1", "P2"), "held_position": "P3"},
)

ORDERED_SELECTOR = (
    ("worst_outer_held_position_macro_f1", "maximize"),
    ("mean_outer_held_position_macro_f1", "maximize"),
    ("worst_class_recall", "maximize"),
    ("zero_recall_class_frequency", "minimize"),
    ("condition_block_macro_f1", "maximize"),
    ("unique_signal_weighted_macro_f1", "maximize"),
    ("across_seed_standard_deviation", "minimize"),
    ("candidate_id", "ascending_deterministic"),
)

ARCHIVE_INPUTS = {
    "canonical_config": "configs/strict_dg/canonical.yaml",
    "candidate_registry": "outputs/strict_dg/source_selection/candidate_registry.json",
    "per_run_metrics": "outputs/strict_dg/source_selection/per_run_metrics.csv",
    "per_fold_metrics": "outputs/strict_dg/source_selection/per_fold_metrics.csv",
    "candidate_aggregate_metrics": "outputs/strict_dg/source_selection/candidate_aggregate_metrics.csv",
    "source_fold_manifest": "outputs/strict_dg/source_selection/source_fold_manifest.csv",
    "selection_decision": "outputs/strict_dg/source_selection/source_selection_decision.json",
    "selected_recipe": "outputs/strict_dg/source_selection/selected_recipe.json",
    "execution_source": "src/crfid/strict_runtime/phase3b_execution.py",
    "summary_source": "src/crfid/strict_runtime/phase3b_summary.py",
    "metric_source": "src/crfid/strict_runtime/neutral_metrics.py",
    "reproduction_script": "scripts/reproduce_strict_dg_source_selection.py",
    "unit_index": "Failure_Mechanism_Physical_Analysis_Final/evidence/source_selection/60_UNIT_INDEX.csv",
    "checkpoint_audit": "Failure_Mechanism_Physical_Analysis_Final/evidence/source_only_diagnostics/CHECKPOINT_INFERENCE_AUDIT.csv",
}

P4_DENY_RULES = (
    ("D01", "path token p4", ("p4",)),
    ("D02", "path token target", ("target",)),
    ("D03", "matched-target evidence", ("matched_target", "matched-target")),
    ("D04", "target evaluation", ("target_evaluation", "target-evaluation")),
    ("D05", "mixed final strict-DG result package", ("strict_dg_final_results",)),
    ("D06", "final P4 result", ("final_p4", "final-p4")),
    ("D07", "Few-Shot patch or evidence", ("few_shot", "few-shot")),
    ("D08", "OpenEMS evidence", ("openems",)),
    ("D09", "archive/ZIP input", (".zip",)),
)

EXPECTED_INPUT_HASHES = {
    "configs/strict_dg/canonical.yaml": "2338bbe1f7ec501546512d286b3d2b3d3067f9f8e458a547e2a237c15a1b835e",
    "src/crfid/strict_runtime/phase3b_execution.py": "ddc65b7fa4bb8287140cd2837f7a392df968f9dcabd0dfabc5df3b479cf9b010",
    "src/crfid/strict_runtime/phase3b_summary.py": "dd76cca10f32982d161cd04b1746cd74287d73440b4684ace7ada1dc23409bf3",
    "scripts/reproduce_strict_dg_source_selection.py": "ed3aca30a4189f02959f6079e52c79042521e30b843b7b0dde3227fe094e1b8e",
    "outputs/strict_dg/source_selection/candidate_registry.json": "c650b3ae65f64ac4bcc3ab5b912e04ec05167c595c536bfd5788808f8486250b",
    "outputs/strict_dg/source_selection/source_fold_manifest.csv": "41a275a6f499ee3d3105e09a420ceb92bb8272d879a89a6bb6623ae6f9bc82ca",
    "outputs/strict_dg/source_selection/per_run_metrics.csv": "1fcbedafb9813f96f13e47929b029e1e40d8d8f36815af5391ee271d4d63d35d",
    "outputs/strict_dg/source_selection/per_fold_metrics.csv": "f51dfed773536c6cc9538776a4e5641c650266d57533e9290def35082bdc0d24",
    "outputs/strict_dg/source_selection/candidate_aggregate_metrics.csv": "8c5dad77f87de80626551d1b5dd7caeda6bfd5f5139fbfae63482e5165f05bd5",
    "outputs/strict_dg/source_selection/selected_recipe.json": "4805f8040f3609dbf4a02553b15c64471f64e3d2a9bc7a6c1068a6342c29d1b6",
    "outputs/strict_dg/source_selection/source_selection_decision.json": "e5479dd552f32e81ae5a4a3e3b8295891e02fbe2a67b84095cc67d8e1a10fc45",
    "Failure_Mechanism_Physical_Analysis_Final/evidence/source_selection/60_UNIT_INDEX.csv": "ea7a4b323821bc875b7049b5a5c135797c68fe9d58444d2aa8fc7f2f5cb353c6",
    "Failure_Mechanism_Physical_Analysis_Final/evidence/source_only_diagnostics/CHECKPOINT_INFERENCE_AUDIT.csv": "559669266b6e3e706d1f954681690b9de722ada4d2b4ac810e84ec199186f762",
}


class ProtocolError(RuntimeError):
    """A deterministic protocol failure."""


class P4SealViolation(ProtocolError):
    """Raised before a P4-classified path can be opened."""


class HeldMetricSealError(ProtocolError):
    """Raised when held metrics are requested before selection is frozen."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256_bytes(data)


def classify_denied_path(path: str | Path) -> tuple[str, str] | None:
    normalized = str(path).replace("\\", "/").casefold()
    for rule_id, description, tokens in P4_DENY_RULES:
        if any(token in normalized for token in tokens):
            return rule_id, description
    return None


@dataclass
class AccessLedger:
    root: Path
    entries: list[dict] = field(default_factory=list)

    def _resolve(self, relative_path: str, purpose: str) -> Path:
        denied = classify_denied_path(relative_path)
        if denied:
            rule_id, description = denied
            self.entries.append(
                {
                    "record_type": "access",
                    "rule_id": rule_id,
                    "path_classification": "P4_OR_TARGET_SCIENTIFIC_EVIDENCE",
                    "path_or_pattern": f"archive::{relative_path.replace(chr(92), '/')}",
                    "action": "REJECTED_BEFORE_OPEN",
                    "purpose": purpose,
                    "sha256": "",
                    "notes": description,
                }
            )
            raise P4SealViolation(f"FAIL_P4_SEAL_VIOLATION: {relative_path}")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ProtocolError(f"Input escapes authority root: {relative_path}") from exc
        return candidate

    def read_bytes(self, relative_path: str, purpose: str) -> bytes:
        path = self._resolve(relative_path, purpose)
        data = path.read_bytes()
        self.entries.append(
            {
                "record_type": "access",
                "rule_id": "ALLOW_SOURCE_ONLY",
                "path_classification": "SOURCE_ONLY",
                "path_or_pattern": f"archive::{relative_path.replace(chr(92), '/')}",
                "action": "OPENED",
                "purpose": purpose,
                "sha256": sha256_bytes(data),
                "notes": "scientific input or lineage evidence",
            }
        )
        return data

    def read_text(self, relative_path: str, purpose: str) -> str:
        return self.read_bytes(relative_path, purpose).decode("utf-8")

    def hash_only(self, relative_path: str, purpose: str) -> str:
        path = self._resolve(relative_path, purpose)
        digest = sha256_file(path)
        self.entries.append(
            {
                "record_type": "access",
                "rule_id": "ALLOW_SOURCE_ONLY",
                "path_classification": "SOURCE_ONLY",
                "path_or_pattern": f"archive::{relative_path.replace(chr(92), '/')}",
                "action": "HASHED_NOT_PARSED",
                "purpose": purpose,
                "sha256": digest,
                "notes": "integrity or lineage identity only",
            }
        )
        return digest


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(list(values)))


def population_sd(values: Iterable[float]) -> float:
    supplied = list(values)
    return float(statistics.pstdev(supplied)) if len(supplied) > 1 else 0.0


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ProtocolError("Prediction and label shapes differ")
    dtype = np.float64 if weights is not None else np.int64
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=dtype)
    if weights is None:
        np.add.at(matrix, (labels, predictions), 1)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != labels.shape or np.any(supplied <= 0):
            raise ProtocolError("Invalid unique-signal weights")
        np.add.at(matrix, (labels, predictions), supplied)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    values = np.asarray(matrix)
    true_support = values.sum(axis=1).astype(np.float64)
    predicted_support = values.sum(axis=0).astype(np.float64)
    true_positive = np.diag(values).astype(np.float64)
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive),
        where=true_support > 0,
    )
    denominator = (
        2.0 * true_positive
        + (predicted_support - true_positive)
        + (true_support - true_positive)
    )
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = float(values.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(per_class_f1.mean()),
        "per_class_recall": [float(value) for value in recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "zero_recall_class_count": int(np.count_nonzero(recall == 0.0)),
        "worst_class_recall": float(recall.min()),
        "confusion_matrix": values.tolist(),
    }


def recompute_prediction_bundle(path: Path) -> dict:
    """Independently recompute released source metrics without returning IDs/arrays."""
    with np.load(path, allow_pickle=False) as bundle:
        labels = np.asarray(bundle["true_labels"], dtype=np.int64)
        logits = np.asarray(bundle["logits"], dtype=np.float32)
        condition_ids = np.asarray(bundle["condition_ids"], dtype=str)
        weights = np.asarray(bundle["unique_signal_weights"], dtype=np.float64)
        saved_predictions = np.asarray(bundle["predictions"], dtype=np.int64)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    if not np.array_equal(predictions, saved_predictions):
        raise ProtocolError(f"Saved predictions disagree with logits: {path.name}")
    sample = metrics_from_confusion(confusion_matrix(labels, predictions))
    unique = metrics_from_confusion(confusion_matrix(labels, predictions, weights))
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition_id in enumerate(condition_ids):
        groups.setdefault(str(condition_id), []).append(index)
    block_labels: list[int] = []
    block_predictions: list[int] = []
    for indices in groups.values():
        unique_labels = np.unique(labels[indices])
        if len(unique_labels) != 1:
            raise ProtocolError("Condition block contains multiple labels")
        block_labels.append(int(unique_labels[0]))
        mean_logits = logits[indices].astype(np.float64).mean(axis=0)
        block_predictions.append(int(np.argmax(mean_logits)))
    condition = metrics_from_confusion(
        confusion_matrix(
            np.asarray(block_labels, dtype=np.int64),
            np.asarray(block_predictions, dtype=np.int64),
        )
    )
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = float(
        np.mean(
            -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-300, None)), axis=1
            )
        )
    )
    counts = np.bincount(predictions, minlength=CLASS_COUNT)
    return {
        "sample": sample,
        "condition": condition,
        "unique_signal_weighted": unique,
        "dominant_predicted_class_fraction": float(counts.max() / len(predictions)),
        "mean_prediction_entropy": entropy,
        "sample_count": int(len(labels)),
        "condition_count": int(len(groups)),
    }


REPORTED_METRIC_MAP = {
    "sample_accuracy": ("sample", "accuracy"),
    "sample_macro_f1": ("sample", "macro_f1"),
    "sample_worst_class_recall": ("sample", "worst_class_recall"),
    "sample_zero_recall_class_count": ("sample", "zero_recall_class_count"),
    "condition_accuracy": ("condition", "accuracy"),
    "condition_macro_f1": ("condition", "macro_f1"),
    "unique_signal_weighted_accuracy": ("unique_signal_weighted", "accuracy"),
    "unique_signal_weighted_macro_f1": ("unique_signal_weighted", "macro_f1"),
}


def validate_metric_replay(published: dict[str, str], recomputed: dict) -> None:
    for column, (level, metric) in REPORTED_METRIC_MAP.items():
        left = float(published[column])
        right = float(recomputed[level][metric])
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=NUMERICAL_TOLERANCE):
            raise ProtocolError(f"Metric replay mismatch: {column}")
    for column in ("dominant_predicted_class_fraction", "mean_prediction_entropy"):
        if not math.isclose(
            float(published[column]),
            float(recomputed[column]),
            rel_tol=0.0,
            abs_tol=NUMERICAL_TOLERANCE,
        ):
            raise ProtocolError(f"Metric replay mismatch: {column}")


def validate_inventory_grid(rows: Sequence[dict]) -> None:
    expected = {(candidate, fold, seed) for candidate in CANDIDATES for fold in FOLDS for seed in SEEDS}
    observed: list[tuple[str, str, int]] = []
    required = set(REPORTED_METRIC_MAP) | {
        "outer_development_sample_accuracy",
        "outer_development_sample_macro_f1",
        "inner_selected_sample_macro_f1",
        "selected_inner_epoch",
    }
    for row in rows:
        key = (row["candidate_id"], row["fold_id"], int(row["seed"]))
        observed.append(key)
        if row.get("execution_status") not in {
            "REUSED_CANONICAL_CONTROL",
            "NEWLY_TRAINED_FROZEN_CANDIDATE",
            "REUSED_CANONICAL_ENCODER_NEW_SOURCE_ONLY_READOUT",
        }:
            raise ProtocolError(f"Incomplete run: {key}")
        if any(row.get(column, "") == "" for column in required):
            raise ProtocolError(f"Missing required metric: {key}")
    duplicates = [key for key, count in Counter(observed).items() if count != 1]
    if duplicates:
        raise ProtocolError(f"Duplicate evidence rows: {duplicates}")
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected:
        raise ProtocolError(f"Evidence grid mismatch; missing={missing}, unexpected={unexpected}")


def _row_index(rows: Sequence[dict]) -> dict[tuple[str, str, int], dict]:
    return {
        (row["candidate_id"], row["position"], int(row["seed"])): row
        for row in rows
    }


def selector_statistics(
    rows: Sequence[dict],
    allowed_positions: Sequence[str],
    seed_sample: Sequence[int] = SEEDS,
) -> list[dict]:
    index = _row_index(rows)
    statistics_rows: list[dict] = []
    for candidate in CANDIDATES:
        units = [index[(candidate, position, int(seed))] for position in allowed_positions for seed in seed_sample]
        position_means = [
            mean(index[(candidate, position, int(seed))]["sample_macro_f1"] for seed in seed_sample)
            for position in allowed_positions
        ]
        per_class_means = [
            mean(unit["per_class_recall"][class_index] for unit in units)
            for class_index in range(CLASS_COUNT)
        ]
        seed_means = [
            mean(index[(candidate, position, int(seed))]["sample_macro_f1"] for position in allowed_positions)
            for seed in seed_sample
        ]
        statistics_rows.append(
            {
                "candidate_id": candidate,
                "worst_outer_held_position_macro_f1": min(position_means),
                "mean_outer_held_position_macro_f1": mean(unit["sample_macro_f1"] for unit in units),
                "worst_class_recall": min(per_class_means),
                "zero_recall_class_frequency": sum(unit["sample_zero_recall_class_count"] for unit in units)
                / (len(units) * CLASS_COUNT),
                "condition_block_macro_f1": mean(unit["condition_macro_f1"] for unit in units),
                "unique_signal_weighted_macro_f1": mean(
                    unit["unique_signal_weighted_macro_f1"] for unit in units
                ),
                "across_seed_standard_deviation": population_sd(seed_means),
                "allowed_unit_count": len(units),
            }
        )
    return statistics_rows


def selector_key(row: dict) -> tuple:
    return (
        -float(row["worst_outer_held_position_macro_f1"]),
        -float(row["mean_outer_held_position_macro_f1"]),
        -float(row["worst_class_recall"]),
        float(row["zero_recall_class_frequency"]),
        -float(row["condition_block_macro_f1"]),
        -float(row["unique_signal_weighted_macro_f1"]),
        float(row["across_seed_standard_deviation"]),
        row["candidate_id"],
    )


def rank_selector(statistics_rows: Sequence[dict]) -> list[dict]:
    return [dict(row, rank=index) for index, row in enumerate(sorted(statistics_rows, key=selector_key), start=1)]


@dataclass
class EvidenceVault:
    rows: list[dict]
    row_loader: Callable[[dict, Sequence[str], str], list[dict]] | None = None
    frozen: dict[str, dict] = field(default_factory=dict)
    sequence_log: list[dict] = field(default_factory=list)

    def selection_rows(self, event: dict) -> list[dict]:
        allowed = set(event["allowed_positions"])
        self.sequence_log.append({"event_id": event["event_id"], "action": "SELECTION_ROWS_OPENED"})
        if self.row_loader is not None:
            return self.row_loader(event, tuple(sorted(allowed)), "selection")
        return [row for row in self.rows if row["position"] in allowed]

    def freeze(self, event: dict, ranking: Sequence[dict]) -> str:
        payload = {
            "event_id": event["event_id"],
            "allowed_positions": list(event["allowed_positions"]),
            "held_position": event["held_position"],
            "ordered_selector": list(ORDERED_SELECTOR),
            "ranking": list(ranking),
        }
        decision_hash = canonical_sha256(payload)
        self.frozen[event["event_id"]] = {
            "decision_sha256": decision_hash,
            "selected_candidate": ranking[0]["candidate_id"],
        }
        self.sequence_log.append({"event_id": event["event_id"], "action": "SELECTION_FROZEN", "decision_sha256": decision_hash})
        return decision_hash

    def held_rows(self, event: dict) -> list[dict]:
        if event["event_id"] not in self.frozen:
            raise HeldMetricSealError(f"Held metrics requested before freeze: {event['event_id']}")
        self.sequence_log.append({"event_id": event["event_id"], "action": "HELD_ROWS_OPENED_AFTER_FREEZE"})
        if self.row_loader is not None:
            return self.row_loader(event, (event["held_position"],), "held")
        return [row for row in self.rows if row["position"] == event["held_position"]]


def average_ranks(values: Sequence[float], *, descending: bool = True) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    mean_a = mean(ranks_a)
    mean_b = mean(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denominator = math.sqrt(
        sum((a - mean_a) ** 2 for a in ranks_a)
        * sum((b - mean_b) ** 2 for b in ranks_b)
    )
    return float(numerator / denominator) if denominator else 0.0


def kendall_tau_b(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    concordant = discordant = ties_a = ties_b = 0
    for left, right in itertools.combinations(range(len(values_a)), 2):
        delta_a = values_a[left] - values_a[right]
        delta_b = values_b[left] - values_b[right]
        if delta_a == 0 and delta_b == 0:
            continue
        if delta_a == 0:
            ties_a += 1
        elif delta_b == 0:
            ties_b += 1
        elif delta_a * delta_b > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    )
    return float((concordant - discordant) / denominator) if denominator else 0.0


def held_candidate_summaries(held_rows: Sequence[dict], metric: str) -> list[dict]:
    summaries = []
    for candidate in CANDIDATES:
        selected = [row for row in held_rows if row["candidate_id"] == candidate]
        summaries.append(
            {
                "candidate_id": candidate,
                "mean": mean(row[metric] for row in selected),
                "population_sd": population_sd(row[metric] for row in selected),
            }
        )
    return [dict(row, rank=index) for index, row in enumerate(sorted(summaries, key=lambda row: (-row["mean"], row["candidate_id"])), start=1)]


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ProtocolError("Cannot take quantile of empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze_events(
    rows: list[dict] | None = None, *, vault: EvidenceVault | None = None
) -> dict:
    vault = vault or EvidenceVault(rows or [])
    decisions: list[dict] = []
    held_table: list[dict] = []
    primary: list[dict] = []
    accuracy: list[dict] = []
    correlations: list[dict] = []
    event_state: dict[str, dict] = {}

    # Pass 1 freezes all three decisions.  No held accessor is called in this
    # pass, so even the first held evaluation occurs only after every decision
    # hash exists.
    selection_state: dict[str, dict] = {}
    for event in EVENTS:
        allowed_rows = vault.selection_rows(event)
        ranking = rank_selector(selector_statistics(allowed_rows, event["allowed_positions"]))
        decision_hash = vault.freeze(event, ranking)
        selected_candidate = ranking[0]["candidate_id"]
        score_margin = ranking[0]["worst_outer_held_position_macro_f1"] - ranking[1]["worst_outer_held_position_macro_f1"]
        score_relative_margin = score_margin / abs(
            ranking[0]["worst_outer_held_position_macro_f1"]
        )
        for row in ranking:
            decisions.append(
                {
                    "event_id": event["event_id"],
                    "allowed_positions": "+".join(event["allowed_positions"]),
                    "held_position": event["held_position"],
                    "rank": row["rank"],
                    "candidate_id": row["candidate_id"],
                    "selected": row["rank"] == 1,
                    "decision_sha256": decision_hash,
                    "score_margin_top1_minus_top2": score_margin,
                    "score_relative_margin": score_relative_margin,
                    **{key: row[key] for key, _ in ORDERED_SELECTOR if key != "candidate_id"},
                }
            )

        selection_state[event["event_id"]] = {
            "ranking": ranking,
            "decision_sha256": decision_hash,
            "selected_candidate": selected_candidate,
            "score_margin": score_margin,
            "score_relative_margin": score_relative_margin,
        }

    # Pass 2 opens each event's held view only after the complete selection
    # manifest is frozen.
    for event in EVENTS:
        state = selection_state[event["event_id"]]
        ranking = state["ranking"]
        decision_hash = state["decision_sha256"]
        selected_candidate = state["selected_candidate"]
        score_margin = state["score_margin"]
        score_relative_margin = state["score_relative_margin"]
        held_rows = vault.held_rows(event)
        held_f1 = held_candidate_summaries(held_rows, "sample_macro_f1")
        held_accuracy = held_candidate_summaries(held_rows, "sample_accuracy")
        oracle_candidate = held_f1[0]["candidate_id"]
        oracle_value = held_f1[0]["mean"]
        selected_value = next(row["mean"] for row in held_f1 if row["candidate_id"] == selected_candidate)
        event_regret = oracle_value - selected_value
        if event_regret < -NUMERICAL_TOLERANCE:
            raise ProtocolError("Negative event regret")
        event_regret = max(0.0, event_regret)
        held_margin = held_f1[0]["mean"] - held_f1[1]["mean"]
        oracle_rows = [
            item for item in held_rows if item["candidate_id"] == oracle_candidate
        ]
        oracle_confusion = sum(
            (np.asarray(item["confusion_matrix"], dtype=np.float64) for item in oracle_rows),
            start=np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.float64),
        )
        confusion_total = float(oracle_confusion.sum())
        for row in held_f1:
            candidate_rows = [item for item in held_rows if item["candidate_id"] == row["candidate_id"]]
            per_class = [mean(item["per_class_recall"][index] for item in candidate_rows) for index in range(CLASS_COUNT)]
            candidate_confusion = sum(
                (
                    np.asarray(item["confusion_matrix"], dtype=np.float64)
                    for item in candidate_rows
                ),
                start=np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.float64),
            )
            held_table.append(
                {
                    "event_id": event["event_id"],
                    "held_position": event["held_position"],
                    "candidate_id": row["candidate_id"],
                    "held_rank": row["rank"],
                    "held_macro_f1_mean": row["mean"],
                    "held_macro_f1_population_sd": row["population_sd"],
                    "held_accuracy_mean": next(item["mean"] for item in held_accuracy if item["candidate_id"] == row["candidate_id"]),
                    "held_worst_class_recall": min(per_class),
                    "held_per_class_recall_json": json.dumps(per_class),
                    "confusion_matrix_l1_fraction_vs_oracle": float(
                        np.abs(candidate_confusion - oracle_confusion).sum()
                        / confusion_total
                    ),
                    "oracle": row["rank"] == 1,
                }
            )
        held_index = _row_index(held_rows)
        position = event["held_position"]
        for seed in SEEDS:
            values = [(candidate, held_index[(candidate, position, seed)]["sample_macro_f1"]) for candidate in CANDIDATES]
            seed_oracle, seed_oracle_value = sorted(values, key=lambda item: (-item[1], item[0]))[0]
            seed_selected_value = held_index[(selected_candidate, position, seed)]["sample_macro_f1"]
            seed_regret = max(0.0, seed_oracle_value - seed_selected_value)
            primary.append(
                {
                    "row_type": "seed",
                    "event_id": event["event_id"],
                    "held_position": position,
                    "seed": seed,
                    "selected_candidate": selected_candidate,
                    "oracle_candidate": seed_oracle,
                    "selected_held_macro_f1": seed_selected_value,
                    "oracle_held_macro_f1": seed_oracle_value,
                    "regret": seed_regret,
                    "zero_regret": seed_regret <= NUMERICAL_TOLERANCE,
                    "within_secondary_tolerance_0_01": seed_regret <= SECONDARY_TOLERANCE,
                }
            )
            accuracy_values = [(candidate, held_index[(candidate, position, seed)]["sample_accuracy"]) for candidate in CANDIDATES]
            acc_oracle, acc_oracle_value = sorted(accuracy_values, key=lambda item: (-item[1], item[0]))[0]
            acc_selected = held_index[(selected_candidate, position, seed)]["sample_accuracy"]
            accuracy.append(
                {
                    "row_type": "seed",
                    "event_id": event["event_id"],
                    "held_position": position,
                    "seed": seed,
                    "selected_candidate": selected_candidate,
                    "oracle_candidate": acc_oracle,
                    "selected_held_accuracy": acc_selected,
                    "oracle_held_accuracy": acc_oracle_value,
                    "regret": max(0.0, acc_oracle_value - acc_selected),
                }
            )
        primary.append(
            {
                "row_type": "event",
                "event_id": event["event_id"],
                "held_position": position,
                "seed": "",
                "selected_candidate": selected_candidate,
                "oracle_candidate": oracle_candidate,
                "selected_held_macro_f1": selected_value,
                "oracle_held_macro_f1": oracle_value,
                "regret": event_regret,
                "zero_regret": event_regret <= NUMERICAL_TOLERANCE,
                "within_secondary_tolerance_0_01": event_regret <= SECONDARY_TOLERANCE,
            }
        )
        selected_acc_value = next(row["mean"] for row in held_accuracy if row["candidate_id"] == selected_candidate)
        acc_oracle = held_accuracy[0]
        accuracy.append(
            {
                "row_type": "event",
                "event_id": event["event_id"],
                "held_position": position,
                "seed": "",
                "selected_candidate": selected_candidate,
                "oracle_candidate": acc_oracle["candidate_id"],
                "selected_held_accuracy": selected_acc_value,
                "oracle_held_accuracy": acc_oracle["mean"],
                "regret": max(0.0, acc_oracle["mean"] - selected_acc_value),
            }
        )
        selector_values = [next(row["worst_outer_held_position_macro_f1"] for row in ranking if row["candidate_id"] == candidate) for candidate in CANDIDATES]
        held_values = [next(row["mean"] for row in held_f1 if row["candidate_id"] == candidate) for candidate in CANDIDATES]
        selector_top2 = {row["candidate_id"] for row in ranking[:2]}
        held_top2 = {row["candidate_id"] for row in held_f1[:2]}
        correlations.append(
            {
                "event_id": event["event_id"],
                "held_position": position,
                "spearman_rho": spearman(selector_values, held_values),
                "kendall_tau_b": kendall_tau_b(selector_values, held_values),
                "top1_recovered": selected_candidate == oracle_candidate,
                "selected_in_held_top2": selected_candidate in held_top2,
                "held_oracle_in_selector_top2": oracle_candidate in selector_top2,
                "selector_top1_top2_margin": score_margin,
                "selector_top1_top2_relative_margin": score_relative_margin,
                "held_top1_top2_margin": held_margin,
                "held_top1_top2_relative_margin": held_margin
                / abs(held_f1[0]["mean"]),
            }
        )
        event_state[event["event_id"]] = {
            "event": event,
            "ranking": ranking,
            "selected_candidate": selected_candidate,
            "held_f1": held_f1,
            "event_regret": event_regret,
            "decision_sha256": decision_hash,
        }
    event_primary = [row for row in primary if row["row_type"] == "event"]
    primary.append(
        {
            "row_type": "summary",
            "event_id": "ALL_EVENTS",
            "held_position": "P1+P2+P3",
            "seed": "",
            "selected_candidate": "",
            "oracle_candidate": "",
            "selected_held_macro_f1": mean(row["selected_held_macro_f1"] for row in event_primary),
            "oracle_held_macro_f1": mean(row["oracle_held_macro_f1"] for row in event_primary),
            "regret": mean(row["regret"] for row in event_primary),
            "median_regret": statistics.median(row["regret"] for row in event_primary),
            "maximum_regret": max(row["regret"] for row in event_primary),
            "zero_regret_event_count": sum(bool(row["zero_regret"]) for row in event_primary),
            "top1_recovery_count": sum(row["selected_candidate"] == row["oracle_candidate"] for row in event_primary),
            "within_secondary_tolerance_event_count": sum(bool(row["within_secondary_tolerance_0_01"]) for row in event_primary),
        }
    )
    return {
        "vault": vault,
        "decisions": decisions,
        "held_table": held_table,
        "primary": primary,
        "accuracy": accuracy,
        "correlations": correlations,
        "event_state": event_state,
    }


def seed_stability(rows: list[dict], event_state: dict[str, dict]) -> list[dict]:
    output: list[dict] = []
    for event in EVENTS:
        full = event_state[event["event_id"]]["selected_candidate"]
        single_winners = []
        for seed in SEEDS:
            winner = rank_selector(selector_statistics(rows, event["allowed_positions"], [seed]))[0]["candidate_id"]
            single_winners.append(winner)
            output.append({"analysis_type": "single_seed", "event_id": event["event_id"], "seed_or_omitted_seed": seed, "winner": winner, "candidate_id": winner, "count": 1, "frequency": 1.0, "matches_full_selection": winner == full})
        counts = Counter(single_winners)
        for candidate in CANDIDATES:
            output.append({"analysis_type": "single_seed_winner_frequency", "event_id": event["event_id"], "seed_or_omitted_seed": "", "winner": "", "candidate_id": candidate, "count": counts[candidate], "frequency": counts[candidate] / len(SEEDS), "matches_full_selection": candidate == full})
        for omitted in SEEDS:
            retained = [seed for seed in SEEDS if seed != omitted]
            winner = rank_selector(selector_statistics(rows, event["allowed_positions"], retained))[0]["candidate_id"]
            output.append({"analysis_type": "leave_one_seed_out", "event_id": event["event_id"], "seed_or_omitted_seed": omitted, "winner": winner, "candidate_id": winner, "count": 1, "frequency": 1.0, "matches_full_selection": winner == full})
    return output


def baseline_comparison(rows: list[dict], event_state: dict[str, dict]) -> list[dict]:
    index = _row_index(rows)
    output: list[dict] = []
    selectors = (
        "CANONICAL_HISTORICAL_SELECTOR",
        "ALWAYS_C0",
        "ALWAYS_C1",
        "TWO_POSITION_MEAN_MACRO_F1",
        "TWO_POSITION_WORST_CASE_MACRO_F1",
        "TWO_POSITION_MEAN_MINUS_VARIABILITY",
    )
    regrets: dict[str, list[float]] = {name: [] for name in selectors}
    for event in EVENTS:
        position = event["held_position"]
        held_means = {candidate: mean(index[(candidate, position, seed)]["sample_macro_f1"] for seed in SEEDS) for candidate in CANDIDATES}
        oracle_candidate = sorted(CANDIDATES, key=lambda candidate: (-held_means[candidate], candidate))[0]
        oracle_value = held_means[oracle_candidate]
        allowed_units = {
            candidate: [index[(candidate, allowed, seed)]["sample_macro_f1"] for allowed in event["allowed_positions"] for seed in SEEDS]
            for candidate in CANDIDATES
        }
        worst_scores = {
            candidate: min(mean(index[(candidate, allowed, seed)]["sample_macro_f1"] for seed in SEEDS) for allowed in event["allowed_positions"])
            for candidate in CANDIDATES
        }
        chosen = {
            "CANONICAL_HISTORICAL_SELECTOR": event_state[event["event_id"]]["selected_candidate"],
            "ALWAYS_C0": CANDIDATES[0],
            "ALWAYS_C1": CANDIDATES[1],
            "TWO_POSITION_MEAN_MACRO_F1": sorted(CANDIDATES, key=lambda candidate: (-mean(allowed_units[candidate]), candidate))[0],
            "TWO_POSITION_WORST_CASE_MACRO_F1": sorted(CANDIDATES, key=lambda candidate: (-worst_scores[candidate], candidate))[0],
            "TWO_POSITION_MEAN_MINUS_VARIABILITY": sorted(CANDIDATES, key=lambda candidate: (-(mean(allowed_units[candidate]) - population_sd(allowed_units[candidate])), candidate))[0],
        }
        for selector in selectors:
            selected = chosen[selector]
            regret = max(0.0, oracle_value - held_means[selected])
            regrets[selector].append(regret)
            output.append({"row_type": "event", "selector": selector, "event_id": event["event_id"], "selected_candidate": selected, "oracle_candidate": oracle_candidate, "selected_held_macro_f1": held_means[selected], "oracle_held_macro_f1": oracle_value, "regret": regret, "mean_regret": "", "median_regret": "", "maximum_regret": "", "lower_mean_regret_than_canonical": ""})
    canonical_mean = mean(regrets["CANONICAL_HISTORICAL_SELECTOR"])
    for selector in selectors:
        values = regrets[selector]
        output.append({"row_type": "summary", "selector": selector, "event_id": "ALL_EVENTS", "selected_candidate": "", "oracle_candidate": "", "selected_held_macro_f1": "", "oracle_held_macro_f1": "", "regret": "", "mean_regret": mean(values), "median_regret": statistics.median(values), "maximum_regret": max(values), "lower_mean_regret_than_canonical": mean(values) < canonical_mean - NUMERICAL_TOLERANCE})
    return output


def margin_sensitivity(event_state: dict[str, dict], rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for event in EVENTS:
        state = event_state[event["event_id"]]
        base = {row["candidate_id"]: row["worst_outer_held_position_macro_f1"] for row in state["ranking"]}
        ordered_nominal = sorted(
            CANDIDATES, key=lambda candidate: (-base[candidate], candidate)
        )
        nominal_margin = base[ordered_nominal[0]] - base[ordered_nominal[1]]
        loso_changed = any(
            rank_selector(
                selector_statistics(
                    rows,
                    event["allowed_positions"],
                    [seed for seed in SEEDS if seed != omitted],
                )
            )[0]["candidate_id"]
            != state["selected_candidate"]
            for omitted in SEEDS
        )
        for delta in (0.001, 0.0025, 0.005):
            counts: Counter[str] = Counter()
            for signs in itertools.product((-1.0, 1.0), repeat=len(CANDIDATES)):
                scores = {candidate: base[candidate] + delta * sign for candidate, sign in zip(CANDIDATES, signs)}
                winner = sorted(CANDIDATES, key=lambda candidate: (-scores[candidate], candidate))[0]
                counts[winner] += 1
            for candidate in CANDIDATES:
                output.append({"event_id": event["event_id"], "delta_macro_f1_equivalent": delta, "candidate_id": candidate, "winner_scenario_count": counts[candidate], "enumerated_scenario_count": 16, "scenario_fraction_not_probability": counts[candidate] / 16.0, "nominal_winner": state["selected_candidate"], "nominal_top_score": base[ordered_nominal[0]], "nominal_second_score": base[ordered_nominal[1]], "nominal_absolute_margin": nominal_margin, "nominal_relative_margin": nominal_margin / abs(base[ordered_nominal[0]]), "leave_one_seed_out_winner_changed": loso_changed, "nominal_winner_robust_all_scenarios": counts[state["selected_candidate"]] == 16, "interpretation": "deterministic symmetric enumeration; not probabilistic"})
    return output


def bootstrap_analysis(rows: list[dict], replicates: int = 10000, random_seed: int = 20260804) -> tuple[list[dict], dict]:
    index = _row_index(rows)
    rng = random.Random(random_seed)
    event_regrets: dict[str, list[float]] = {event["event_id"]: [] for event in EVENTS}
    mean_regrets: list[float] = []
    winner_counts: dict[str, Counter[str]] = {event["event_id"]: Counter() for event in EVENTS}
    sampled_units: list[tuple[int, ...]] = []
    for _ in range(replicates):
        sampled = tuple(rng.choice(SEEDS) for _ in SEEDS)
        if len(sampled_units) < 3:
            sampled_units.append(sampled)
        replicate_regrets = []
        for event in EVENTS:
            ranking = rank_selector(selector_statistics(rows, event["allowed_positions"], sampled))
            selected = ranking[0]["candidate_id"]
            winner_counts[event["event_id"]][selected] += 1
            held = {
                candidate: mean(index[(candidate, event["held_position"], seed)]["sample_macro_f1"] for seed in sampled)
                for candidate in CANDIDATES
            }
            oracle = sorted(CANDIDATES, key=lambda candidate: (-held[candidate], candidate))[0]
            regret = max(0.0, held[oracle] - held[selected])
            event_regrets[event["event_id"]].append(regret)
            replicate_regrets.append(regret)
        mean_regrets.append(mean(replicate_regrets))
    output: list[dict] = []
    quantiles = (0.0, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 1.0)
    for scope, values in [*event_regrets.items(), ("MEAN_THREE_EVENT_REGRET", mean_regrets)]:
        for probability in quantiles:
            output.append({"record_type": "regret_distribution_quantile", "scope": scope, "candidate_id": "", "quantile": probability, "value": quantile(values, probability), "count": replicates, "probability_or_frequency": "", "notes": "seed-paired bootstrap"})
        output.append({"record_type": "zero_regret_probability", "scope": scope, "candidate_id": "", "quantile": "", "value": "", "count": sum(value <= NUMERICAL_TOLERANCE for value in values), "probability_or_frequency": sum(value <= NUMERICAL_TOLERANCE for value in values) / replicates, "notes": "bootstrap probability"})
    for event in EVENTS:
        for candidate in CANDIDATES:
            count = winner_counts[event["event_id"]][candidate]
            output.append({"record_type": "selector_winner_frequency", "scope": event["event_id"], "candidate_id": candidate, "quantile": "", "value": "", "count": count, "probability_or_frequency": count / replicates, "notes": "candidate selection probability under paired seed bootstrap"})
    for candidate in CANDIDATES:
        count = sum(winner_counts[event["event_id"]][candidate] for event in EVENTS)
        output.append({"record_type": "candidate_selection_probability_over_events", "scope": "ALL_EVENTS", "candidate_id": candidate, "quantile": "", "value": "", "count": count, "probability_or_frequency": count / (replicates * len(EVENTS)), "notes": "pooled event-replicate selection fraction; three events are not treated as independent samples"})
    metadata = {
        "replicates": replicates,
        "random_seed": random_seed,
        "resampling_unit": "seed",
        "paired_across_candidates_and_positions": True,
        "row_pseudoreplication": False,
        "first_three_sampled_seed_tuples": sampled_units,
        "mean_regret_ci_95": [quantile(mean_regrets, 0.025), quantile(mean_regrets, 0.975)],
    }
    return output, metadata


def ranking_consistency(event_state: dict[str, dict]) -> list[dict]:
    output = []
    for event in EVENTS:
        state = event_state[event["event_id"]]
        for candidate in CANDIDATES:
            selector_rank = next(row["rank"] for row in state["ranking"] if row["candidate_id"] == candidate)
            held_rank = next(row["rank"] for row in state["held_f1"] if row["candidate_id"] == candidate)
            output.append({"event_id": event["event_id"], "held_position": event["held_position"], "candidate_id": candidate, "selector_rank": selector_rank, "held_rank": held_rank, "rank_difference_selector_minus_held": selector_rank - held_rank, "exact_rank_match": selector_rank == held_rank})
    return output


def classify_result(primary: Sequence[dict], correlations: Sequence[dict]) -> str:
    events = [row for row in primary if row["row_type"] == "event"]
    top1 = sum(row["selected_candidate"] == row["oracle_candidate"] for row in events)
    if all(float(row["regret"]) <= NUMERICAL_TOLERANCE for row in events):
        return "SOURCE_SELECTION_VALIDITY_SUPPORTED"
    median_spearman = statistics.median(float(row["spearman_rho"]) for row in correlations)
    if 1 <= top1 <= 2 and median_spearman > 0:
        return "SOURCE_SELECTION_VALIDITY_PARTIALLY_SUPPORTED"
    if top1 == 0 and median_spearman <= 0:
        return "SOURCE_SELECTION_RULE_NOT_PREDICTIVE"
    return "SOURCE_SELECTION_UNSTABLE_INCONCLUSIVE"


def format_float(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def validate_fold_mapping(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    observed: dict[str, set[str]] = {fold: set() for fold in FOLDS}
    for row in rows:
        if row["partition"] == "outer_held":
            observed[row["fold_id"]].add(row["position"])
    mapping = {fold: next(iter(positions)) if len(positions) == 1 else "INVALID" for fold, positions in observed.items()}
    if mapping != FOLD_TO_POSITION:
        raise ProtocolError(f"Fold-to-position mismatch: {mapping}")
    return mapping


def checkpoint_bindings(archive_root: Path, ledger: AccessLedger, unit_index: Sequence[dict], audit_rows: Sequence[dict]) -> tuple[dict, dict[tuple[str, str, int], dict]]:
    prediction_index: dict[tuple[str, str, int], dict] = {}
    for row in unit_index:
        key = (row["candidate"], row["fold"], int(row["seed"]))
        prediction_index[key] = row
    c0_outer = {
        (row["fold_id"], int(row["seed"])): row
        for row in audit_rows
        if row["stage"] == "outer_refit"
    }
    c1_hashes = []
    per_run: dict[tuple[str, str, int], dict] = {}
    for candidate in CANDIDATES:
        for fold in FOLDS:
            for seed in SEEDS:
                prediction = prediction_index[(candidate, fold, seed)]
                if candidate == CANDIDATES[0]:
                    source = c0_outer[(fold, seed)]
                    binding = {"status": "BOUND_PHASE2_OUTER_CHECKPOINT", "checkpoint_file_sha256": source["file_sha256_before"], "model_state_sha256": source["model_state_sha256"]}
                elif candidate == CANDIDATES[1]:
                    rel = f"outputs/strict_dg/source_selection/checkpoints/{candidate}/{fold}/seed_{seed}.pt"
                    digest = ledger.hash_only(rel, "C1 released checkpoint identity")
                    c1_hashes.append(f"{fold},{seed},{digest}")
                    binding = {"status": "BOUND_PHASE3B_OUTER_CHECKPOINT", "checkpoint_file_sha256": digest, "model_state_sha256": "NOT_EXPOSED_WITHOUT_DESERIALIZING_CHECKPOINT"}
                elif candidate == CANDIDATES[2]:
                    source = c0_outer[(fold, seed)]
                    binding = {"status": "BOUND_SHARED_C0_PHASE2_ENCODER_CHECKPOINT", "checkpoint_file_sha256": source["file_sha256_before"], "model_state_sha256": source["model_state_sha256"]}
                else:
                    binding = {"status": "CHECKPOINT_NOT_RETAINED_IN_COMPACT_RELEASE", "checkpoint_file_sha256": "UNAVAILABLE", "model_state_sha256": "UNAVAILABLE", "surviving_output_identity": prediction["sha256"]}
                binding["prediction_file_sha256"] = prediction["sha256"]
                per_run[(candidate, fold, seed)] = binding
    c0_lines = [f"{fold},{seed},{row['file_sha256_before']},{row['model_state_sha256']}" for (fold, seed), row in sorted(c0_outer.items())]
    candidate_sets = {
        CANDIDATES[0]: {"checkpoint_binding_status": "BOUND", "checkpoint_set_sha256": sha256_bytes(("\n".join(c0_lines) + "\n").encode("ascii")), "checkpoint_count": 15},
        CANDIDATES[1]: {"checkpoint_binding_status": "BOUND", "checkpoint_set_sha256": sha256_bytes(("\n".join(sorted(c1_hashes)) + "\n").encode("ascii")), "checkpoint_count": 15},
        CANDIDATES[2]: {"checkpoint_binding_status": "BOUND_SHARED_C0_ENCODER", "checkpoint_set_sha256": sha256_bytes(("\n".join(c0_lines) + "\n").encode("ascii")), "checkpoint_count": 15},
        CANDIDATES[3]: {"checkpoint_binding_status": "NOT_RETAINED_BUT_15_PREDICTION_OUTPUTS_HASH_BOUND", "checkpoint_set_sha256": "UNAVAILABLE", "checkpoint_count": 0},
    }
    return candidate_sets, per_run


def run_study(repo_root: Path, archive_root: Path, output_root: Path, *, bootstrap_replicates: int = 10000) -> dict:
    ledger = AccessLedger(archive_root)
    protocol = json.loads((repo_root / "configs/source_selection_regret/protocol.json").read_text(encoding="utf-8"))
    if tuple(protocol["candidate_ids"]) != CANDIDATES or tuple(protocol["seeds"]) != SEEDS:
        raise ProtocolError("Frozen protocol identity mismatch")
    for relative_path, expected in EXPECTED_INPUT_HASHES.items():
        observed = ledger.hash_only(relative_path, "frozen scientific input identity")
        if observed != expected:
            raise ProtocolError(f"Input hash mismatch: {relative_path}")
    config_text = ledger.read_text(ARCHIVE_INPUTS["canonical_config"], "historical selector recovery")
    for metric, operation in ORDERED_SELECTOR[:-1]:
        config_operation = "ascending" if operation == "ascending_deterministic" else operation
        if metric not in config_text or config_operation not in config_text:
            raise ProtocolError("Historical selector is not bound to canonical config")
    registry = json.loads(ledger.read_text(ARCHIVE_INPUTS["candidate_registry"], "candidate lineage binding"))
    if tuple(row["candidate_id"] for row in registry["candidates"]) != CANDIDATES:
        raise ProtocolError("Candidate registry identity mismatch")
    fold_rows = read_csv_text(ledger.read_text(ARCHIVE_INPUTS["source_fold_manifest"], "source fold-to-position validation"))
    validate_fold_mapping(fold_rows)
    unit_index = read_csv_text(ledger.read_text(ARCHIVE_INPUTS["unit_index"], "prediction evidence identity"))
    expected_grid = {
        (candidate, fold, seed)
        for candidate in CANDIDATES
        for fold in FOLDS
        for seed in SEEDS
    }
    observed_grid = [
        (row["candidate"], row["fold"], int(row["seed"])) for row in unit_index
    ]
    if len(observed_grid) != 60 or set(observed_grid) != expected_grid:
        raise ProtocolError("Prediction evidence index is not the exact 60-row grid")
    if any(count != 1 for count in Counter(observed_grid).values()):
        raise ProtocolError("Prediction evidence index contains duplicates")
    checkpoint_audit = read_csv_text(ledger.read_text(ARCHIVE_INPUTS["checkpoint_audit"], "C0/C2 checkpoint lineage"))
    checkpoint_sets, per_run_checkpoint = checkpoint_bindings(archive_root, ledger, unit_index, checkpoint_audit)
    index_rows = {(row["candidate"], row["fold"], int(row["seed"])): row for row in unit_index}
    held_row_map: dict[tuple[str, str, int], dict] = {}

    def load_event_rows(
        event: dict, requested_positions: Sequence[str], phase: str
    ) -> list[dict]:
        loaded: list[dict] = []
        for position in requested_positions:
            fold = POSITION_TO_FOLD[position]
            for candidate in CANDIDATES:
                for seed in SEEDS:
                    identity = index_rows[(candidate, fold, seed)]
                    rel = (
                        "outputs/strict_dg/source_selection/"
                        + identity["prediction_reference"]
                    )
                    path = ledger._resolve(
                        rel,
                        f"{event['event_id']} {phase} source metric recomputation",
                    )
                    observed_hash = sha256_file(path)
                    if observed_hash != identity["sha256"]:
                        raise ProtocolError(
                            f"Prediction hash mismatch: {candidate}/{fold}/{seed}"
                        )
                    recomputed = recompute_prediction_bundle(path)
                    action = (
                        "OPENED_FOR_SELECTION_BEFORE_GLOBAL_HELD_RELEASE"
                        if phase == "selection"
                        else "OPENED_HELD_AFTER_ALL_SELECTIONS_FROZEN"
                    )
                    ledger.entries.append(
                        {
                            "record_type": "access",
                            "rule_id": "ALLOW_SOURCE_ONLY",
                            "path_classification": "SOURCE_ONLY_PREDICTION_BUNDLE",
                            "path_or_pattern": f"archive::{rel}",
                            "action": action,
                            "purpose": f"{event['event_id']} independent Accuracy/Macro-F1 and selector-component replay",
                            "sha256": observed_hash,
                            "notes": "event-local access; arrays and private identifiers were not emitted",
                        }
                    )
                    row = {
                        "candidate_id": candidate,
                        "fold_id": fold,
                        "position": position,
                        "seed": seed,
                        "sample_accuracy": recomputed["sample"]["accuracy"],
                        "sample_macro_f1": recomputed["sample"]["macro_f1"],
                        "sample_worst_class_recall": recomputed["sample"]["worst_class_recall"],
                        "sample_zero_recall_class_count": recomputed["sample"]["zero_recall_class_count"],
                        "per_class_recall": recomputed["sample"]["per_class_recall"],
                        "confusion_matrix": recomputed["sample"]["confusion_matrix"],
                        "condition_accuracy": recomputed["condition"]["accuracy"],
                        "condition_macro_f1": recomputed["condition"]["macro_f1"],
                        "unique_signal_weighted_accuracy": recomputed[
                            "unique_signal_weighted"
                        ]["accuracy"],
                        "unique_signal_weighted_macro_f1": recomputed[
                            "unique_signal_weighted"
                        ]["macro_f1"],
                        "dominant_predicted_class_fraction": recomputed[
                            "dominant_predicted_class_fraction"
                        ],
                        "mean_prediction_entropy": recomputed[
                            "mean_prediction_entropy"
                        ],
                        "prediction_sha256": observed_hash,
                        **per_run_checkpoint[(candidate, fold, seed)],
                    }
                    loaded.append(row)
                    if phase == "held":
                        held_row_map[(candidate, position, seed)] = row
        return loaded

    vault = EvidenceVault([], row_loader=load_event_rows)
    event_results = analyze_events(vault=vault)
    if len(event_results["vault"].frozen) != 3:
        raise ProtocolError("All three decisions were not frozen")
    first_held = next(
        index
        for index, row in enumerate(event_results["vault"].sequence_log)
        if row["action"] == "HELD_ROWS_OPENED_AFTER_FREEZE"
    )
    if sum(
        row["action"] == "SELECTION_FROZEN"
        for row in event_results["vault"].sequence_log[:first_held]
    ) != 3:
        raise ProtocolError("Held release occurred before all selections were frozen")

    # The compact numeric table is opened only after all three selection
    # decisions and held evaluations are complete.  It is a replay target, not
    # a selector input.
    published_rows = read_csv_text(
        ledger.read_text(
            ARCHIVE_INPUTS["per_run_metrics"],
            "post-freeze published source metric replay target",
        )
    )
    validate_inventory_grid(published_rows)
    published_index = {
        (row["candidate_id"], row["fold_id"], int(row["seed"])): row
        for row in published_rows
    }
    scientific_rows = [
        held_row_map[(candidate, position, seed)]
        for candidate in CANDIDATES
        for position in POSITIONS
        for seed in SEEDS
    ]
    replay_count = 0
    for row in scientific_rows:
        published = published_index[
            (row["candidate_id"], row["fold_id"], int(row["seed"]))
        ]
        recomputed = {
            "sample": {
                "accuracy": row["sample_accuracy"],
                "macro_f1": row["sample_macro_f1"],
                "worst_class_recall": row["sample_worst_class_recall"],
                "zero_recall_class_count": row["sample_zero_recall_class_count"],
            },
            "condition": {
                "accuracy": row["condition_accuracy"],
                "macro_f1": row["condition_macro_f1"],
            },
            "unique_signal_weighted": {
                "accuracy": row["unique_signal_weighted_accuracy"],
                "macro_f1": row["unique_signal_weighted_macro_f1"],
            },
            "dominant_predicted_class_fraction": row[
                "dominant_predicted_class_fraction"
            ],
            "mean_prediction_entropy": row["mean_prediction_entropy"],
        }
        validate_metric_replay(published, recomputed)
        row.update(
            {
                "execution_status": published["execution_status"],
                "selected_inner_epoch": int(published["selected_inner_epoch"]),
                "inner_selected_sample_macro_f1": float(
                    published["inner_selected_sample_macro_f1"]
                ),
                "outer_development_sample_accuracy": float(
                    published["outer_development_sample_accuracy"]
                ),
                "outer_development_sample_macro_f1": float(
                    published["outer_development_sample_macro_f1"]
                ),
            }
        )
        replay_count += 1
    stability = seed_stability(scientific_rows, event_results["event_state"])
    baselines = baseline_comparison(scientific_rows, event_results["event_state"])
    sensitivity = margin_sensitivity(event_results["event_state"], scientific_rows)
    bootstrap_rows, bootstrap_metadata = bootstrap_analysis(scientific_rows, bootstrap_replicates, int(protocol["bootstrap"]["random_seed"]))
    consistency = ranking_consistency(event_results["event_state"])
    classification = classify_result(event_results["primary"], event_results["correlations"])
    synthetic_denied = "outputs/strict_dg/strict_dg_final_results/FINAL_P4_RESULTS.csv"
    try:
        ledger.read_bytes(synthetic_denied, "required P4 access rejection test")
    except P4SealViolation:
        pass
    else:
        raise ProtocolError("P4 access rejection test did not reject")
    preprocessing_artifacts = []
    for fold in FOLDS:
        for name in (
            "inner_selection_mean_float64.npy",
            "inner_selection_scale_float64.npy",
            "outer_refit_mean_float64.npy",
            "outer_refit_scale_float64.npy",
        ):
            relative_path = (
                f"outputs/strict_dg/source_selection/preprocessing_state/{fold}/{name}"
            )
            preprocessing_artifacts.append(
                {
                    "path": relative_path,
                    "sha256": ledger.hash_only(
                        relative_path, "released source preprocessing-state identity"
                    ),
                }
            )
    preprocessing_set_sha256 = canonical_sha256(preprocessing_artifacts)
    lineage = {
        "schema_version": 1,
        "candidate_registry_sha256": EXPECTED_INPUT_HASHES[ARCHIVE_INPUTS["candidate_registry"]],
        "canonical_config_sha256": EXPECTED_INPUT_HASHES[ARCHIVE_INPUTS["canonical_config"]],
        "source_implementation": [
            {"path": ARCHIVE_INPUTS["execution_source"], "sha256": EXPECTED_INPUT_HASHES[ARCHIVE_INPUTS["execution_source"]]},
            {"path": ARCHIVE_INPUTS["summary_source"], "sha256": EXPECTED_INPUT_HASHES[ARCHIVE_INPUTS["summary_source"]]},
            {"path": ARCHIVE_INPUTS["metric_source"], "sha256": ledger.hash_only(ARCHIVE_INPUTS["metric_source"], "metric implementation lineage")},
        ],
        "preprocessing": {
            "policy": "representation_then_featurewise_source_train_standardization; ddof=0; float64 fit; minimum scale 1e-12 replaced by 1.0; float32 model input; no target refit",
            "released_state_artifacts": preprocessing_artifacts,
            "released_state_set_sha256": preprocessing_set_sha256,
        },
        "fold_definition": {
            "path": ARCHIVE_INPUTS["source_fold_manifest"],
            "sha256": EXPECTED_INPUT_HASHES[ARCHIVE_INPUTS["source_fold_manifest"]],
            "mapping": FOLD_TO_POSITION,
        },
        "seed_definition": list(SEEDS),
        "selection_rule": {"status": protocol["selector_status"], "ordered_selector": protocol["ordered_selector"], "protocol_sha256": sha256_file(repo_root / "configs/source_selection_regret/protocol.json")},
        "candidates": [],
    }
    registry_by_id = {row["candidate_id"]: row for row in registry["candidates"]}
    prediction_set_by_candidate = {
        candidate: canonical_sha256(sorted(row["prediction_sha256"] for row in scientific_rows if row["candidate_id"] == candidate))
        for candidate in CANDIDATES
    }
    for candidate in CANDIDATES:
        definition = registry_by_id[candidate]
        lineage["candidates"].append({
            "candidate_id": candidate,
            "architecture_or_transformation": definition["architecture_or_transformation"],
            "configuration": definition["hyperparameters"],
            "preprocessing": {
                "policy_binding": "candidate registry hyperparameters plus canonical representation-then-ddof0-standardization policy",
                "released_state_set_sha256": preprocessing_set_sha256,
            },
            "checkpoint_lineage": checkpoint_sets[candidate],
            "folds": list(FOLDS),
            "positions": list(POSITIONS),
            "seeds": list(SEEDS),
            "metric_implementation": ARCHIVE_INPUTS["metric_source"],
            "selection_rule": protocol["selector_status"],
            "released_evidence": {"prediction_bundle_count": 15, "prediction_set_sha256": prediction_set_by_candidate[candidate], "per_run_metrics_sha256": EXPECTED_INPUT_HASHES[ARCHIVE_INPUTS["per_run_metrics"]]},
        })
    inventory_rows = []
    for row in scientific_rows:
        held_event = f"HOLD_{row['position']}"
        selection_events = "+".join(event["event_id"] for event in EVENTS if row["position"] in event["allowed_positions"])
        inventory_rows.append({
            "candidate_id": row["candidate_id"], "held_source_position": row["position"], "fold_id": row["fold_id"], "seed": row["seed"],
            "candidate_training_output": row["execution_status"], "source_validation_output": "inner_selected_and_outer_development_metrics_available", "held_source_position_output": "independently_recomputed_from_prediction_bundle", "diagnostic_only_artifact": "per-class recall and condition/unique-weighted components", "candidate_selection_input_events": selection_events, "final_selection_output_event": held_event,
            "accuracy": row["sample_accuracy"], "macro_f1": row["sample_macro_f1"], "validation_macro_f1": row["outer_development_sample_macro_f1"], "selected_inner_epoch": row["selected_inner_epoch"],
            "checkpoint_binding_status": row["status"], "checkpoint_file_sha256": row["checkpoint_file_sha256"], "model_state_sha256": row["model_state_sha256"], "prediction_sha256": row["prediction_sha256"], "metric_replay_match": True,
        })
    selector_hash = canonical_sha256(list(ORDERED_SELECTOR))
    event_manifest = []
    for sequence, event in enumerate(EVENTS, start=1):
        state = event_results["event_state"][event["event_id"]]
        event_manifest.append({"sequence": sequence, "event_id": event["event_id"], "allowed_positions": "+".join(event["allowed_positions"]), "held_position": event["held_position"], "selection_rows_opened_before_freeze": True, "selection_decision_frozen": True, "decision_sha256": state["decision_sha256"], "all_three_decisions_frozen_before_any_held_open": True, "held_rows_opened_after_freeze": True, "selector_sha256": selector_hash})
    return {
        "ledger": ledger,
        "protocol": protocol,
        "lineage": lineage,
        "inventory": inventory_rows,
        "event_manifest": event_manifest,
        "event_results": event_results,
        "stability": stability,
        "baselines": baselines,
        "sensitivity": sensitivity,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_metadata": bootstrap_metadata,
        "consistency": consistency,
        "classification": classification,
        "metric_replay_count": replay_count,
        "scientific_rows": scientific_rows,
    }


def emit_results(repo_root: Path, study: dict, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    event_results = study["event_results"]
    events = [row for row in event_results["primary"] if row["row_type"] == "event"]
    summary = next(row for row in event_results["primary"] if row["row_type"] == "summary")
    correlations = event_results["correlations"]
    baseline_summaries = [row for row in study["baselines"] if row["row_type"] == "summary"]
    canonical_baseline = next(row for row in baseline_summaries if row["selector"] == "CANONICAL_HISTORICAL_SELECTOR")
    lower_baselines = [row["selector"] for row in baseline_summaries if row["lower_mean_regret_than_canonical"]]
    selections = "; ".join(f"{row['event_id']} -> {row['selected_candidate']}" for row in events)
    oracles = "; ".join(f"{row['event_id']} -> {row['oracle_candidate']}" for row in events)
    executive = f"""# Source-internal selection-regret result

Classification: `{study['classification']}`

- Evidence complete: yes, 60/60 candidate × held-position × seed rows, each independently replayed from a hash-bound source prediction bundle.
- Reruns: none; execution case A used frozen source-only evidence.
- Selector: exact historical lexicographic selector restricted to the two allowed source positions.
- Selections: {selections}.
- Held-position oracles: {oracles}.
- Mean/median/maximum three-event regret: {float(summary['regret']):.12g} / {float(summary['median_regret']):.12g} / {float(summary['maximum_regret']):.12g}.
- Top-1 recovery: {int(summary['top1_recovery_count'])}/3; zero-regret events: {int(summary['zero_regret_event_count'])}/3.
- Seed stability: see `13_SEED_WINNER_STABILITY.csv`; the bootstrap used paired seed units only.
- Simpler baseline with lower mean regret: {', '.join(lower_baselines) if lower_baselines else 'none'} (canonical mean regret {float(canonical_baseline['mean_regret']):.12g}).
- P4 seal: intact; the workflow's synthetic P4 access was rejected before open.

Selection regret is conditional on the current candidate set and cannot prove regret relative to all possible models or representations.
"""
    write_text(output_root / "00_EXECUTIVE_SUMMARY.md", executive)
    input_binding = {
        "schema_version": 1,
        "status": "PASS_SOURCE_ONLY_INPUT_BINDING",
        "source_only_input_hashes": EXPECTED_INPUT_HASHES,
        "candidate_ids": [candidate["candidate_id"] for candidate in study["lineage"]["candidates"]],
        "prediction_bundle_count": len(study["scientific_rows"]),
        "p4_scientific_inputs_used_for_selection": False,
    }
    write_json(output_root / "01_INPUT_BINDING.json", input_binding)
    deny_rows = [{"record_type": "denylist", "rule_id": rule_id, "path_classification": "P4_OR_TARGET_SCIENTIFIC_EVIDENCE", "path_or_pattern": " | ".join(tokens), "action": "REJECT_BEFORE_OPEN", "purpose": "P4 scientific seal", "sha256": "", "notes": description} for rule_id, description, tokens in P4_DENY_RULES]
    deny_rows.extend(study["ledger"].entries)
    write_csv(output_root / "02_P4_ACCESS_DENYLIST_AND_LOG.csv", deny_rows)
    write_json(output_root / "03_CANDIDATE_LINEAGE_BINDING.json", study["lineage"])
    write_csv(output_root / "04_SOURCE_EVIDENCE_INVENTORY.csv", study["inventory"])
    c3 = next(row for row in study["lineage"]["candidates"] if row["candidate_id"] == CANDIDATES[3])
    decision = f"""# Execution decision

`CASE A — COMPLETE EXISTING EVIDENCE`

All 60 expected source candidate × held-position × seed prediction bundles and compact metric rows are present, hash-bound, mutually compatible, and independently replay to numerical tolerance. No training or rerun was performed. C0 and C2 bind to the 15 canonical Phase-2 outer checkpoints; C1 binds to 15 retained Phase-3B checkpoints. C3's nonwinning checkpoint files were not retained in the compact release, but its exact candidate registry/config/code lineage and all 15 hash-bound prediction outputs survive; this is recorded as a lineage limitation and does not require inferring any score.

Metric replay count: {study['metric_replay_count']}/60. C3 checkpoint status: {c3['checkpoint_lineage']['checkpoint_binding_status']}.
"""
    write_text(output_root / "05_EXECUTION_DECISION.md", decision)
    frozen = "# Frozen selection rule\n\nStatus: `EXACT_HISTORICAL_SELECTOR_RESTRICTED_TO_TWO_ALLOWED_SOURCE_POSITIONS`.\n\n" + "\n".join(f"{index}. `{metric}` — {operation}." for index, (metric, operation) in enumerate(ORDERED_SELECTOR, start=1)) + "\n\nNo scalar weights or normalization are used. Each event replaces the historical three-position aggregation domain with its two allowed positions and changes nothing else. All three decisions are frozen and SHA-256 hashed before any held accessor is enabled.\n"
    write_text(output_root / "06_FROZEN_SELECTION_RULE.md", frozen)
    write_csv(output_root / "07_SELECTION_EVENT_MANIFEST.csv", study["event_manifest"])
    write_csv(output_root / "08_SELECTION_DECISIONS.csv", event_results["decisions"])
    write_csv(output_root / "09_HELD_POSITION_ORACLE_RESULTS.csv", event_results["held_table"])
    write_csv(output_root / "10_PRIMARY_SELECTION_REGRET.csv", event_results["primary"])
    write_csv(output_root / "11_ACCURACY_SELECTION_REGRET.csv", event_results["accuracy"])
    write_csv(output_root / "12_RANK_CORRELATION_RESULTS.csv", correlations)
    write_csv(output_root / "13_SEED_WINNER_STABILITY.csv", study["stability"])
    write_csv(output_root / "14_SELECTOR_BASELINE_COMPARISON.csv", study["baselines"])
    write_csv(output_root / "15_SELECTION_MARGIN_SENSITIVITY.csv", study["sensitivity"])
    write_csv(output_root / "16_BOOTSTRAP_RESULTS.csv", study["bootstrap_rows"])
    write_csv(output_root / "17_CANDIDATE_RANKING_CONSISTENCY.csv", study["consistency"])
    interpretation = f"""# Scientific interpretation

The frozen source-only selector is classified as `{study['classification']}` under the preregistered rule. It recovered held-position top-1 in {int(summary['top1_recovery_count'])} of three events. Event-wise Spearman correlations were {', '.join(format(float(row['spearman_rho']), '.6g') for row in correlations)}; Kendall tau-b values were {', '.join(format(float(row['kendall_tau_b']), '.6g') for row in correlations)}.

These are exact results for three held-source events, not asymptotic population claims. Bootstrap probabilities describe only paired resampling of the five canonical seed identities. Margin perturbation scenario fractions are deterministic enumeration summaries and have no probabilistic interpretation.
"""
    write_text(output_root / "18_SCIENTIFIC_INTERPRETATION.md", interpretation)
    limitations = """# Limitations and nonclaims

- There are only three held-source-position selection events and five canonical seeds.
- No P4 evidence, external data, new candidate, new representation, or target-informed threshold was used.
- C3 nonwinner checkpoint files were not retained; its identity is instead bound to the frozen registry/config/code and 15 released prediction hashes. No C3 score was inferred.
- The bootstrap resamples paired seed units and cannot establish a broader sampling distribution.
- Symmetric margin perturbations are deterministic sensitivity checks, not probabilities.
- Accuracy and the fixed 0.01 threshold are secondary diagnostics only.

Selection regret is conditional on the current candidate set and cannot prove regret relative to all possible models or representations.
"""
    write_text(output_root / "19_LIMITATIONS_AND_NONCLAIMS.md", limitations)
    status = f"""# Result status

Scientific classification: `{study['classification']}`.

The complete frozen source-only 60-row candidate grid was replayed exactly within numerical tolerance. Across three held-source selection events, the paired bootstrap used {study['bootstrap_metadata']['replicates']} replicates; the 95% interval for mean regret was [{study['bootstrap_metadata']['mean_regret_ci_95'][0]:.12g}, {study['bootstrap_metadata']['mean_regret_ci_95'][1]:.12g}].

P4 remains outside this source-selection diagnostic; no target performance claim is made from these selection events.
"""
    write_text(output_root / "STATUS.md", status)
    write_json(repo_root / "manifests/source_selection_regret/SCIENTIFIC_INPUT_MANIFEST.json", {"schema_version": 1, "source_only_input_hashes": EXPECTED_INPUT_HASHES, "prediction_bundle_count": 60, "prediction_bundle_set_sha256": canonical_sha256(sorted(row["prediction_sha256"] for row in study["scientific_rows"])), "bootstrap": study["bootstrap_metadata"], "p4_inputs": []})
    write_text(repo_root / "docs/SOURCE_SELECTION_REGRET_RESULTS.md", executive)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    archive_root = (args.archive_root or repo_root.parent / "CRFID_research_code_v2").resolve()
    output_root = repo_root / "results/canonical_metrics/source_selection_regret"
    study = run_study(repo_root, archive_root, output_root, bootstrap_replicates=args.bootstrap_replicates)
    emit_results(repo_root, study, output_root)
    print(json.dumps({"classification": study["classification"], "metric_replay_count": study["metric_replay_count"], "output_root": output_root.as_posix(), "p4_sealed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
