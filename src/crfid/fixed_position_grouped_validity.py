"""Fixed-position, condition-grouped learning-validity benchmark.

This additive adapter reuses the canonical Strict-DG C1 model, initialization,
optimizer, deterministic training loop, checkpoint rule, and metrics without
changing the frozen implementation. Large training artifacts remain under the
ignored runtime root; only compact review artifacts are written to Git paths.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data.tyndall import map_raw_tag_label
from .strict_runtime.hashing import array_sha256, canonical_json_sha256, exact_signal_sha256, stable_id
from .strict_runtime.neutral_data import CanonicalPhase2Data, PartitionData
from .strict_runtime.neutral_metrics import confusion_matrix, metrics_from_confusion
from .strict_runtime.neutral_model import initialize_model, parameter_count
from .strict_runtime.phase3b_execution import (
    C1,
    _criterion,
    _erm_train_epoch,
    _optimizer,
    evaluate_logits,
    extract_logits_embeddings,
    load_checkpoint_model,
    run_trainable_candidate,
)
from .strict_runtime.scale_policy import CANONICAL_MODE, apply_scale_policy, fallback_report
from .strict_runtime.target_evaluation import _active_one_hot, _integral_value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fixed_position_grouped_validity" / "canonical.json"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "canonical_metrics" / "fixed_position_grouped_validity"
DEFAULT_MANIFESTS = PROJECT_ROOT / "manifests" / "fixed_position_grouped_validity"
PROTOCOL_PATH = DEFAULT_RESULTS / "02_PREREGISTERED_PROTOCOL.md"
EXPECTED_POSITIONS = ("P1", "P2", "P3", "P4")
METADATA_COLUMNS = ("A3", "A2", "A1", "P4", "P3", "P2", "P1", "ER", "TagID")
SIGNAL_COLUMNS = tuple(str(index) for index in range(281))
EXPECTED_HEADER = METADATA_COLUMNS + SIGNAL_COLUMNS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames are required for an empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FixedPositionData:
    position: str
    rows: list[dict[str, Any]]
    signals: np.ndarray
    labels: np.ndarray
    source_data_fingerprint: str
    source_files: tuple[dict[str, Any], ...]

    def assert_isolated(self) -> None:
        observed = {str(row["position"]) for row in self.rows}
        if observed != {self.position}:
            raise RuntimeError(f"Cross-position rows reached {self.position}: {sorted(observed)}")
        if self.signals.shape != (3150, 281) or self.labels.shape != (3150,):
            raise RuntimeError(f"Unexpected fixed-position arrays for {self.position}")


@dataclass(frozen=True)
class PositionDataAdapter:
    """Minimum interface required by canonical ``run_trainable_candidate``."""

    registry_rows: list[dict[str, Any]]


def load_governed_position(raw_root: Path, position: str) -> FixedPositionData:
    """Generalize the canonical P4 byte-stream parser to one declared position."""

    if position not in EXPECTED_POSITIONS:
        raise ValueError(f"Unknown position: {position}")
    raw_root = Path(raw_root).resolve()
    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    condition_counts: Counter[str] = Counter()
    source_files: list[dict[str, Any]] = []
    for surface in ("A1", "A2", "A3"):
        path = raw_root / f"{surface}_{position}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw_bytes = path.read_bytes()
        file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        source_files.append(
            {
                "file_name": path.name,
                "size_bytes": len(raw_bytes),
                "sha256": file_sha256,
                "filesystem_read_count": 1,
            }
        )
        reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig"), newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != EXPECTED_HEADER:
            raise RuntimeError(f"Unexpected governed schema: {path.name}")
        file_id = stable_id("src", f"{path.name}|{file_sha256}")
        file_rows = 0
        for source_row_index, raw_row in enumerate(reader):
            observed_surface = _active_one_hot(raw_row, ("A1", "A2", "A3"), "surface")
            observed_position = _active_one_hot(raw_row, ("P1", "P2", "P3", "P4"), "position")
            if observed_surface != surface or observed_position != position:
                raise RuntimeError(f"File/domain mismatch: {path.name}/{source_row_index}")
            tag_id = _integral_value(raw_row["TagID"], "TagID")
            er = _integral_value(raw_row["ER"], "ER")
            label_index = map_raw_tag_label(tag_id)
            if er not in {0, 1, 2}:
                raise RuntimeError(f"Unexpected ER in {path.name}: {er}")
            signal = np.fromiter(
                (float(raw_row[column]) for column in SIGNAL_COLUMNS),
                dtype="<f8",
                count=281,
            )
            if signal.shape != (281,) or not np.isfinite(signal).all():
                raise RuntimeError(f"Invalid signal: {path.name}/{source_row_index}")
            signal_hash = exact_signal_sha256(signal)
            condition_key = (
                f"tag={tag_id}|er={er}|surface={surface}|position={position}"
            )
            condition_id = stable_id("cond", condition_key)
            repeat_index = condition_counts[condition_id]
            condition_counts[condition_id] += 1
            sample_key = (
                f"{path.name}|row={source_row_index}|{condition_key}|signal={signal_hash}"
            )
            rows.append(
                {
                    "registry_row": len(rows),
                    "sample_id": stable_id("sample", sample_key, length=32),
                    "raw_condition_id": condition_id,
                    "repeat_group_id": condition_id,
                    "repeat_id": f"{condition_id}_r{repeat_index:02d}",
                    "repeat_index": repeat_index,
                    "tag_id": tag_id,
                    "label_index": label_index,
                    "er": er,
                    "surface": surface,
                    "position": position,
                    "source_file_id": file_id,
                    "source_file_name": path.name,
                    "source_row_index": source_row_index,
                    "source_csv_line_number": source_row_index + 2,
                    "exact_signal_sha256": signal_hash,
                }
            )
            signals.append(signal)
            file_rows += 1
        if file_rows != 1050:
            raise RuntimeError(f"Unexpected row count in {path.name}: {file_rows}")

    signal_array = np.ascontiguousarray(np.stack(signals), dtype="<f8")
    labels = np.asarray([row["label_index"] for row in rows], dtype="<i8")
    multiplicities = Counter(row["exact_signal_sha256"] for row in rows)
    for row in rows:
        count = multiplicities[row["exact_signal_sha256"]]
        row["exact_signal_multiplicity"] = count
        row["unique_signal_weight"] = float(1.0 / count)
        row["signal_point_count"] = 281
        row["signal_dtype"] = "<f8"
    if (
        signal_array.shape != (3150, 281)
        or labels.dtype.str != "<i8"
        or len(condition_counts) != 63
        or set(condition_counts.values()) != {50}
        or len({row["sample_id"] for row in rows}) != 3150
        or len({row["repeat_id"] for row in rows}) != 3150
    ):
        raise RuntimeError(f"Governed fixed-position population differs: {position}")
    fingerprint = canonical_json_sha256(
        {
            "position": position,
            "source_files": source_files,
            "signals_array_sha256": array_sha256(signal_array),
            "labels_array_sha256": array_sha256(labels),
            "condition_ids": sorted(condition_counts),
        }
    )
    result = FixedPositionData(
        position=position,
        rows=rows,
        signals=signal_array,
        labels=labels,
        source_data_fingerprint=fingerprint,
        source_files=tuple(source_files),
    )
    result.assert_isolated()
    return result


def _canonical_source_loader(strict_root: Path, runtime_root: Path) -> CanonicalPhase2Data:
    config = {
        "input": {
            "canonical_signal_path": "source_inputs/source_signals_float64.npy",
            "canonical_label_path": "source_inputs/source_labels_int64.npy",
            "registry_path": "source_inputs/CANONICAL_SOURCE_REGISTRY.csv",
            "split_path": "source_inputs/SOURCE_ONLY_LOPO_SPLITS.csv",
            "custody_shape": [9450, 281],
            "custody_dtype": "<f8",
        },
        "folds": {"S1": {}, "S2": {}, "S3": {}},
    }
    return CanonicalPhase2Data(
        config,
        artifact_root=Path(strict_root),
        preprocessing_root=Path(runtime_root) / "canonical_loader_preprocessing",
    )


def audit_canonical_compatibility(
    data_by_position: dict[str, FixedPositionData], strict_root: Path, runtime_root: Path
) -> dict[str, dict[str, Any]]:
    """Prove raw adapter identity against canonical source and P4 registries."""

    strict_root = Path(strict_root).resolve()
    canonical = _canonical_source_loader(strict_root, runtime_root)
    registry_path = strict_root / "source_inputs" / "CANONICAL_SOURCE_REGISTRY.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        source_registry = list(csv.DictReader(handle))
    if len(source_registry) != 9450:
        raise RuntimeError("Canonical source registry row count differs")

    results: dict[str, dict[str, Any]] = {}
    for position in ("P1", "P2", "P3"):
        raw = data_by_position[position]
        raw_index = {row["sample_id"]: index for index, row in enumerate(raw.rows)}
        selected = [row for row in source_registry if row["position"] == position]
        if len(selected) != 3150 or set(raw_index) != {row["sample_id"] for row in selected}:
            raise RuntimeError(f"Raw and canonical source sample identities differ: {position}")
        canonical_indices = np.asarray([int(row["registry_row"]) for row in selected], dtype=np.int64)
        raw_indices = np.asarray([raw_index[row["sample_id"]] for row in selected], dtype=np.int64)
        signals_equal = np.array_equal(canonical.signals[canonical_indices], raw.signals[raw_indices])
        labels_equal = np.array_equal(canonical.labels[canonical_indices], raw.labels[raw_indices])
        metadata_equal = all(
            int(row["tag_id"]) == raw.rows[int(raw_index[row["sample_id"]])]["tag_id"]
            and int(row["label_index"]) == raw.rows[int(raw_index[row["sample_id"]])]["label_index"]
            and int(row["er"]) == raw.rows[int(raw_index[row["sample_id"]])]["er"]
            and row["surface"] == raw.rows[int(raw_index[row["sample_id"]])]["surface"]
            and row["raw_condition_id"]
            == raw.rows[int(raw_index[row["sample_id"]])]["raw_condition_id"]
            for row in selected
        )
        if not signals_equal or not labels_equal or not metadata_equal:
            raise RuntimeError(f"Canonical source compatibility failed: {position}")
        results[position] = {
            "canonical_loader": "CanonicalPhase2Data",
            "sample_identity_equal": True,
            "signals_bitwise_equal": True,
            "labels_bitwise_equal": True,
            "condition_metadata_equal": True,
        }

    p4_bundle = strict_root / "final_p4" / "predictions" / "seed_42.npz"
    if not p4_bundle.is_file():
        raise FileNotFoundError(p4_bundle)
    raw_p4 = data_by_position["P4"]
    with np.load(p4_bundle, allow_pickle=False) as bundle:
        sample_ids = np.asarray(bundle["sample_ids"], dtype=str)
        condition_ids = np.asarray(bundle["condition_ids"], dtype=str)
        labels = np.asarray(bundle["true_labels"], dtype=np.int64)
    registry_equal = (
        np.array_equal(sample_ids, np.asarray([row["sample_id"] for row in raw_p4.rows], dtype=str))
        and np.array_equal(
            condition_ids,
            np.asarray([row["raw_condition_id"] for row in raw_p4.rows], dtype=str),
        )
        and np.array_equal(labels, raw_p4.labels)
    )
    if not registry_equal:
        raise RuntimeError("P4 adapter registry differs from canonical final P4 registry")
    results["P4"] = {
        "canonical_loader": "target_evaluation.load_p4_once parser-compatible adapter",
        "sample_identity_equal": True,
        "signals_bitwise_equal": "NOT_AVAILABLE_IN_COMMITTED_CANONICAL_BUNDLE",
        "labels_bitwise_equal": True,
        "condition_metadata_equal": True,
    }
    return results


def structure_audit_rows(
    data_by_position: dict[str, FixedPositionData],
    compatibility: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in EXPECTED_POSITIONS:
        data = data_by_position[position]
        group_counts = Counter(row["raw_condition_id"] for row in data.rows)
        tag_ids = sorted({int(row["tag_id"]) for row in data.rows})
        ers = sorted({int(row["er"]) for row in data.rows})
        surfaces = sorted({str(row["surface"]) for row in data.rows})
        label_mapping_ok = all(
            int(row["label_index"]) == map_raw_tag_label(int(row["tag_id"]))
            for row in data.rows
        )
        passed = (
            tag_ids == list(range(1, 8))
            and ers == [0, 1, 2]
            and surfaces == ["A1", "A2", "A3"]
            and len(group_counts) == 63
            and set(group_counts.values()) == {50}
            and len(data.rows) == 3150
            and label_mapping_ok
            and data.signals.shape == (3150, 281)
        )
        if not passed:
            raise RuntimeError(f"DATA_STRUCTURE_FAILURE: {position}")
        rows.append(
            {
                "position": position,
                "tag_id_count": len(tag_ids),
                "tag_ids": ";".join(map(str, tag_ids)),
                "er_level_count": len(ers),
                "er_levels": ";".join(map(str, ers)),
                "surface_count": len(surfaces),
                "surfaces": ";".join(surfaces),
                "condition_block_count": len(group_counts),
                "minimum_rows_per_block": min(group_counts.values()),
                "maximum_rows_per_block": max(group_counts.values()),
                "row_count": len(data.rows),
                "signal_point_count": data.signals.shape[1],
                "first_difference_point_count": data.signals.shape[1] - 1,
                "label_mapping_matches_canonical": label_mapping_ok,
                "canonical_registry_compatibility": all(
                    value is True or isinstance(value, str)
                    for key, value in compatibility[position].items()
                    if key != "canonical_loader"
                ),
                "source_file_names": ";".join(row["file_name"] for row in data.source_files),
                "source_file_sha256s": ";".join(row["sha256"] for row in data.source_files),
                "source_data_fingerprint": data.source_data_fingerprint,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return rows


def _stable_rng(protocol_id: str, position: str, fold: int, seed: int, tag_id: int) -> np.random.Generator:
    material = f"{protocol_id}|{position}|fold={fold}|seed={seed}|tag={tag_id}".encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)
    return np.random.default_rng(value)


def generate_split_manifest(
    data_by_position: dict[str, FixedPositionData], config: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    surfaces = {value: index for index, value in enumerate(config["data"]["surface_order"])}
    for position in config["positions"]:
        data = data_by_position[position]
        blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for condition_id in sorted({row["raw_condition_id"] for row in data.rows}):
            member = next(row for row in data.rows if row["raw_condition_id"] == condition_id)
            blocks[int(member["tag_id"])].append(
                {
                    "condition_id": condition_id,
                    "tag_id": int(member["tag_id"]),
                    "er": int(member["er"]),
                    "surface": str(member["surface"]),
                    "row_count": sum(
                        row["raw_condition_id"] == condition_id for row in data.rows
                    ),
                }
            )
        for fold in range(3):
            for seed in config["seeds"]:
                for tag_id in config["data"]["tag_ids"]:
                    ordered = sorted(blocks[int(tag_id)], key=lambda row: (row["er"], row["surface"]))
                    test = [
                        row
                        for row in ordered
                        if (row["er"] + surfaces[row["surface"]]) % 3 == fold
                    ]
                    remaining = [row for row in ordered if row not in test]
                    if len(test) != 3 or len(remaining) != 6:
                        raise RuntimeError("Outer balanced fold construction failed")
                    permutation = _stable_rng(
                        config["protocol_id"], position, fold, int(seed), int(tag_id)
                    ).permutation(len(remaining))
                    validation_ids = {
                        remaining[int(index)]["condition_id"] for index in permutation[:2]
                    }
                    test_ids = {row["condition_id"] for row in test}
                    for block in ordered:
                        if block["condition_id"] in test_ids:
                            assignment = "test"
                        elif block["condition_id"] in validation_ids:
                            assignment = "validation"
                        else:
                            assignment = "train"
                        records.append(
                            {
                                "position": position,
                                "fold": fold + 1,
                                "seed": int(seed),
                                "TagID": block["tag_id"],
                                "ER": block["er"],
                                "surface": block["surface"],
                                "condition_block_identifier": block["condition_id"],
                                "row_count": block["row_count"],
                                "split_assignment": assignment,
                                "source_data_fingerprint": data.source_data_fingerprint,
                            }
                        )
    return records


def _run_manifest_rows(
    records: list[dict[str, Any]], position: str, fold: int, seed: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if row["position"] == position and int(row["fold"]) == fold and int(row["seed"]) == seed
    ]


def split_validation_rows(
    records: list[dict[str, Any]],
    repeated_records: list[dict[str, Any]],
    config: dict[str, Any],
    compatibility: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: dict[str, list[str]] = {str(index): [] for index in range(1, 11)}
    for position in config["positions"]:
        for fold in range(1, 4):
            seed_runs = {
                int(seed): _run_manifest_rows(records, position, fold, int(seed))
                for seed in config["seeds"]
            }
            for seed, run in seed_runs.items():
                by_condition = defaultdict(set)
                for row in run:
                    by_condition[row["condition_block_identifier"]].add(row["split_assignment"])
                if len(by_condition) != 63 or any(len(value) != 1 for value in by_condition.values()):
                    errors["1"].append(f"{position}/F{fold}/S{seed}")
                if any(int(row["row_count"]) != 50 for row in run):
                    errors["2"].append(f"{position}/F{fold}/S{seed}")
                test_tags = {int(row["TagID"]) for row in run if row["split_assignment"] == "test"}
                if test_tags != set(range(1, 8)):
                    errors["3"].append(f"{position}/F{fold}/S{seed}")
                expected = {"train": (28, 1400, 4), "validation": (14, 700, 2), "test": (21, 1050, 3)}
                for split, (blocks, rows, per_tag) in expected.items():
                    selected = [row for row in run if row["split_assignment"] == split]
                    tag_counts = Counter(int(row["TagID"]) for row in selected)
                    if len(selected) != blocks or sum(int(row["row_count"]) for row in selected) != rows or set(tag_counts.values()) != {per_tag}:
                        errors["4"].append(f"{position}/F{fold}/S{seed}/{split}")
                if {row["position"] for row in run} != {position}:
                    errors["7"].append(f"{position}/F{fold}/S{seed}")
                if any(row["position"] != position for row in run):
                    errors["8"].append(f"{position}/F{fold}/S{seed}")

            first_test = {
                row["condition_block_identifier"]
                for row in seed_runs[int(config["seeds"][0])]
                if row["split_assignment"] == "test"
            }
            validation_sets = []
            for seed in config["seeds"]:
                run = seed_runs[int(seed)]
                test = {
                    row["condition_block_identifier"]
                    for row in run
                    if row["split_assignment"] == "test"
                }
                if test != first_test:
                    errors["10"].append(f"test_changed/{position}/F{fold}/S{seed}")
                validation_sets.append(
                    frozenset(
                        row["condition_block_identifier"]
                        for row in run
                        if row["split_assignment"] == "validation"
                    )
                )
            if len(set(validation_sets)) == 1:
                errors["10"].append(f"validation_unchanged/{position}/F{fold}")

    if any(not all(value is True or isinstance(value, str) for key, value in compatibility[position].items() if key != "canonical_loader") for position in config["positions"]):
        errors["5"].append("canonical_compatibility")
    if int(config["preprocessing"]["input_feature_count"]) != 281 or int(config["preprocessing"]["output_feature_count"]) != 280:
        errors["6"].append("carrier_dimensions")
    if canonical_json_sha256(records) != canonical_json_sha256(repeated_records):
        errors["9"].append("repeat_manifest_digest")

    descriptions = {
        "1": "no condition block appears in more than one split",
        "2": "all 50 repetitions of each condition block remain together",
        "3": "all seven TagIDs appear in every test fold",
        "4": "class, block, and row counts are exact",
        "5": "label mappings and registries match canonical Strict-DG",
        "6": "ordered carrier and preprocessing dimensions match canonical Strict-DG",
        "7": "P1, P2, P3, and P4 run manifests are isolated",
        "8": "no target-assisted or cross-position row is supplied",
        "9": "split manifests are deterministic for a fixed seed",
        "10": "seed changes affect only validation/train membership and training randomness",
    }
    output = []
    for gate in map(str, range(1, 11)):
        output.append(
            {
                "gate": int(gate),
                "validation": descriptions[gate],
                "status": "PASS" if not errors[gate] else "FAIL",
                "checked_run_count": 60,
                "details": "all checks passed" if not errors[gate] else ";".join(errors[gate]),
            }
        )
    if any(row["status"] != "PASS" for row in output):
        raise RuntimeError("PRETRAINING_SPLIT_VALIDATION_FAILURE")
    return output


def _indices_from_manifest(
    data: FixedPositionData, run_manifest: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    assignment = {
        str(row["condition_block_identifier"]): str(row["split_assignment"])
        for row in run_manifest
    }
    result: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(data.rows):
        result[assignment[row["raw_condition_id"]]].append(index)
    arrays = {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}
    if {key: len(value) for key, value in arrays.items()} != {
        "train": 1400,
        "validation": 700,
        "test": 1050,
    }:
        raise RuntimeError("Run row counts differ")
    return arrays


def _partition(data: FixedPositionData, indices: np.ndarray, values: np.ndarray) -> PartitionData:
    rows = [data.rows[int(index)] for index in indices]
    return PartitionData(
        registry_rows=np.asarray(indices, dtype=np.int64),
        inputs=np.ascontiguousarray(values, dtype=np.float32),
        labels=np.ascontiguousarray(data.labels[indices], dtype=np.int64),
        sample_ids=[row["sample_id"] for row in rows],
        condition_ids=[row["raw_condition_id"] for row in rows],
        exact_signal_hashes=[row["exact_signal_sha256"] for row in rows],
        unique_signal_weights=np.asarray(
            [row["unique_signal_weight"] for row in rows], dtype=np.float64
        ),
    )


def build_partitions(
    data: FixedPositionData, run_manifest: list[dict[str, Any]]
) -> tuple[dict[str, PartitionData], dict[str, Any]]:
    data.assert_isolated()
    indices = _indices_from_manifest(data, run_manifest)
    differenced = np.ascontiguousarray(np.diff(data.signals, axis=1), dtype=np.float64)

    def fit(fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        fit_values = differenced[fit_indices]
        mean = np.ascontiguousarray(fit_values.mean(axis=0, dtype=np.float64))
        raw_scale = np.ascontiguousarray(fit_values.std(axis=0, ddof=0, dtype=np.float64))
        scale = apply_scale_policy(raw_scale, mode=CANONICAL_MODE)
        return mean, scale, fallback_report(raw_scale)

    inner_mean, inner_scale, inner_fallback = fit(indices["train"])
    outer_development = np.sort(np.concatenate((indices["train"], indices["validation"])))
    outer_mean, outer_scale, outer_fallback = fit(outer_development)

    def transform(selected: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(((differenced[selected] - mean) / scale).astype(np.float32))

    partitions = {
        "inner_train": _partition(
            data, indices["train"], transform(indices["train"], inner_mean, inner_scale)
        ),
        "inner_validation": _partition(
            data,
            indices["validation"],
            transform(indices["validation"], inner_mean, inner_scale),
        ),
        "outer_development": _partition(
            data,
            outer_development,
            transform(outer_development, outer_mean, outer_scale),
        ),
        "outer_held": _partition(
            data, indices["test"], transform(indices["test"], outer_mean, outer_scale)
        ),
    }
    state = {
        "inner": {
            "fit_count": len(indices["train"]),
            "mean_sha256": array_sha256(inner_mean),
            "scale_sha256": array_sha256(inner_scale),
            "fallback": inner_fallback,
        },
        "outer": {
            "fit_count": len(outer_development),
            "mean_sha256": array_sha256(outer_mean),
            "scale_sha256": array_sha256(outer_scale),
            "fallback": outer_fallback,
        },
        "input_dimension": 281,
        "output_dimension": 280,
        "mode": CANONICAL_MODE,
    }
    return partitions, state


def canonical_base_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer": dict(config["optimizer"]),
        "loss": {"weight": None, "label_smoothing": 0.0},
        "training": {
            "batch_size": int(config["training"]["batch_size"]),
            "inner_stage_offset": int(config["training"]["inner_stage_offset"]),
            "outer_stage_offset": int(config["training"]["outer_stage_offset"]),
        },
    }


def run_learning_validity(
    data: FixedPositionData,
    manifest_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config["learning_validity"]["diagnostic_seed"])
    fold = int(config["learning_validity"]["diagnostic_fold"]) + 1
    run_manifest = _run_manifest_rows(manifest_records, data.position, fold, seed)
    partitions, _ = build_partitions(data, run_manifest)
    train = partitions["inner_train"]
    rows_per_class = int(config["learning_validity"]["rows_per_class"])
    selected: list[int] = []
    for class_index in range(7):
        candidates = [
            index for index, label in enumerate(train.labels) if int(label) == class_index
        ]
        candidates.sort(key=lambda index: train.sample_ids[index])
        selected.extend(candidates[:rows_per_class])
    selected_array = np.asarray(selected, dtype=np.int64)
    subset = PartitionData(
        registry_rows=train.registry_rows[selected_array],
        inputs=np.ascontiguousarray(train.inputs[selected_array], dtype=np.float32),
        labels=np.ascontiguousarray(train.labels[selected_array], dtype=np.int64),
        sample_ids=[train.sample_ids[index] for index in selected],
        condition_ids=[train.condition_ids[index] for index in selected],
        exact_signal_hashes=[train.exact_signal_hashes[index] for index in selected],
        unique_signal_weights=np.ascontiguousarray(
            train.unique_signal_weights[selected_array], dtype=np.float64
        ),
    )
    model = initialize_model(seed)
    if parameter_count(model) != int(config["model"]["parameter_count"]):
        raise RuntimeError("Canonical model parameter count differs")
    base_config = canonical_base_config(config)
    optimizer = _optimizer(model, base_config)
    criterion = _criterion(base_config)
    initial_logits, _ = extract_logits_embeddings(model, subset)
    initial = evaluate_logits(subset, initial_logits)["sample"]
    final = initial
    final_loss = float("nan")
    steps = 0
    maximum = int(config["learning_validity"]["maximum_full_batch_steps"])
    for step in range(1, maximum + 1):
        losses, _ = _erm_train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            partition=subset,
            fold_id=f"{data.position}_LEARNING_VALIDITY",
            seed=seed,
            epoch=step,
            stage="inner_selection",
            base_config=base_config,
            initial_hash="diagnostic_initial_state_recorded_by_canonical_initializer",
            candidate_id=C1,
        )
        logits, _ = extract_logits_embeddings(model, subset)
        final = evaluate_logits(subset, logits)["sample"]
        final_loss = float(losses["total_loss"])
        steps = step
        if (
            float(final["accuracy"]) >= float(config["learning_validity"]["pass_accuracy"])
            and float(final["macro_f1"]) >= float(config["learning_validity"]["pass_macro_f1"])
        ):
            break
    passed = (
        float(final["accuracy"]) >= float(config["learning_validity"]["pass_accuracy"])
        and float(final["macro_f1"]) >= float(config["learning_validity"]["pass_macro_f1"])
    )
    return {
        "position": data.position,
        "diagnostic_fold": fold,
        "diagnostic_seed": seed,
        "training_subset_rows": len(subset.labels),
        "rows_per_class": rows_per_class,
        "maximum_full_batch_steps": maximum,
        "steps_executed": steps,
        "initial_accuracy": float(initial["accuracy"]),
        "initial_macro_f1": float(initial["macro_f1"]),
        "final_accuracy": float(final["accuracy"]),
        "final_macro_f1": float(final["macro_f1"]),
        "final_loss": final_loss,
        "pass_accuracy_threshold": float(config["learning_validity"]["pass_accuracy"]),
        "pass_macro_f1_threshold": float(config["learning_validity"]["pass_macro_f1"]),
        "status": "PASS" if passed else "LEARNING_VALIDITY_FAILURE",
        "affects_primary_hyperparameters": False,
    }


def _per_class_rows(
    position: str, fold: int, seed: int, level: str, metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for class_index in range(7):
        output.append(
            {
                "position": position,
                "fold": fold,
                "seed": seed,
                "evaluation_level": level,
                "class_index": class_index,
                "TagID": class_index + 1,
                "precision": float(metrics["per_class_precision"][class_index]),
                "recall": float(metrics["per_class_recall"][class_index]),
                "f1": float(metrics["per_class_f1"][class_index]),
                "support": int(np.asarray(metrics["confusion_matrix"])[class_index].sum()),
            }
        )
    return output


def _confusion_rows(
    position: str, fold: int, seed: int, level: str, matrix: Iterable[Iterable[int]]
) -> list[dict[str, Any]]:
    output = []
    values = np.asarray(matrix, dtype=np.int64)
    for true_class in range(7):
        row: dict[str, Any] = {
            "position": position,
            "fold": fold,
            "seed": seed,
            "evaluation_level": level,
            "true_class_index": true_class,
            "true_TagID": true_class + 1,
        }
        row.update({f"predicted_{index}": int(values[true_class, index]) for index in range(7)})
        output.append(row)
    return output


def _block_confusions(
    data: FixedPositionData,
    partition: PartitionData,
    evaluation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    predictions = np.asarray(evaluation["predictions"], dtype=np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, condition_id in enumerate(partition.condition_ids):
        groups[str(condition_id)].append(index)
    block_predictions = {
        str(row["raw_condition_id"]): int(row["predicted_label"])
        for row in evaluation["condition"]["rows"]
    }
    metadata = {row["raw_condition_id"]: row for row in data.rows}
    result = {}
    for condition_id, indices in groups.items():
        labels = partition.labels[np.asarray(indices, dtype=np.int64)]
        predicted = predictions[np.asarray(indices, dtype=np.int64)]
        label = int(labels[0])
        result[condition_id] = {
            "tag_id": int(metadata[condition_id]["tag_id"]),
            "label": label,
            "row_confusion": confusion_matrix(labels, predicted),
            "block_confusion": confusion_matrix(
                np.asarray([label], dtype=np.int64),
                np.asarray([block_predictions[condition_id]], dtype=np.int64),
            ),
        }
    return result


def execute_primary_runs(
    data_by_position: dict[str, FixedPositionData],
    manifest_records: list[dict[str, Any]],
    config: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    base_config = canonical_base_config(config)
    candidate_sha256 = canonical_json_sha256(config)
    run_register: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    histograms: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    confusions: list[dict[str, Any]] = []
    block_store: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    consistency_failures: list[str] = []
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    for position in config["positions"]:
        data = data_by_position[position]
        data.assert_isolated()
        for fold in range(1, 4):
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                run_id = f"{position}_F{fold}_S{seed}"
                started_at = utc_now()
                started = time.perf_counter()
                run_manifest = _run_manifest_rows(manifest_records, position, fold, seed)
                partitions, preprocessing = build_partitions(data, run_manifest)
                if any(
                    {data.rows[int(index)]["position"] for index in partition.registry_rows}
                    != {position}
                    for partition in partitions.values()
                ):
                    raise RuntimeError(f"Cross-position partition reached primary run: {run_id}")
                split_sha256 = canonical_json_sha256(run_manifest)
                inner_preprocessing_sha256 = canonical_json_sha256(preprocessing["inner"])
                outer_preprocessing_sha256 = canonical_json_sha256(preprocessing["outer"])
                initial_model = initialize_model(seed)
                initial_logits, _ = extract_logits_embeddings(
                    initial_model, partitions["outer_held"]
                )
                initial_evaluation = evaluate_logits(partitions["outer_held"], initial_logits)
                run_directory = Path(runtime_root) / "primary" / position / f"fold_{fold}" / f"seed_{seed}"
                run_directory.mkdir(parents=True, exist_ok=True)
                result = run_trainable_candidate(
                    candidate_id=C1,
                    candidate_config_sha256=candidate_sha256,
                    fold_id=f"{position}_F{fold}",
                    seed=seed,
                    data=PositionDataAdapter(data.rows),  # type: ignore[arg-type]
                    partitions=partitions,
                    base_config=base_config,
                    split_sha256=split_sha256,
                    inner_preprocessing_sha256=inner_preprocessing_sha256,
                    outer_preprocessing_sha256=outer_preprocessing_sha256,
                    run_directory=run_directory,
                )
                training = result["development_evaluation"]["sample"]
                validation = result["inner_evaluation"]["sample"]
                held = result["held_evaluation"]
                held_row = held["sample"]
                held_block = held["condition"]["metrics"]
                histogram = np.bincount(
                    np.asarray(held["predictions"], dtype=np.int64), minlength=7
                )
                row_majority = max(
                    np.bincount(partitions["outer_held"].labels, minlength=7)
                ) / len(partitions["outer_held"].labels)
                block_true = np.asarray(
                    [row["true_label"] for row in held["condition"]["rows"]], dtype=np.int64
                )
                block_majority = max(np.bincount(block_true, minlength=7)) / len(block_true)
                duration = time.perf_counter() - started
                completed_at = utc_now()
                run_register.append(
                    {
                        "run_id": run_id,
                        "position": position,
                        "fold": fold,
                        "seed": seed,
                        "method_id": C1,
                        "status": "PASS",
                        "started_at_utc": started_at,
                        "completed_at_utc": completed_at,
                        "duration_seconds": duration,
                        "protocol_sha256": protocol_sha256,
                        "split_sha256": split_sha256,
                        "source_data_fingerprint": data.source_data_fingerprint,
                        "selected_epoch": int(result["selected_epoch"]),
                        "checkpoint_verified_before_held": bool(
                            result["checkpoint_verified_before_outer_held"]
                        ),
                        "held_evaluation_count": int(result["outer_held_evaluation_count"]),
                        "model_updates_after_held": int(
                            result["model_updates_after_outer_held_evaluation"]
                        ),
                    }
                )
                run_metrics.append(
                    {
                        "run_id": run_id,
                        "position": position,
                        "fold": fold,
                        "seed": seed,
                        "selected_epoch": int(result["selected_epoch"]),
                        "initial_held_accuracy": float(initial_evaluation["sample"]["accuracy"]),
                        "initial_held_macro_f1": float(initial_evaluation["sample"]["macro_f1"]),
                        "training_accuracy": float(training["accuracy"]),
                        "training_macro_f1": float(training["macro_f1"]),
                        "validation_accuracy": float(validation["accuracy"]),
                        "validation_macro_f1": float(validation["macro_f1"]),
                        "held_row_accuracy": float(held_row["accuracy"]),
                        "held_row_macro_f1": float(held_row["macro_f1"]),
                        "held_block_accuracy": float(held_block["accuracy"]),
                        "held_block_macro_f1": float(held_block["macro_f1"]),
                        "train_to_held_accuracy_gap": float(
                            training["accuracy"] - held_row["accuracy"]
                        ),
                        "uniform_seven_class_chance": 1.0 / 7.0,
                        "empirical_row_majority_baseline": float(row_majority),
                        "condition_block_majority_baseline": float(block_majority),
                        "dominant_predicted_class_fraction": float(
                            held["dominant_predicted_class_fraction"]
                        ),
                        "mean_prediction_entropy": float(held["mean_prediction_entropy"]),
                    }
                )
                per_class.extend(_per_class_rows(position, fold, seed, "row", held_row))
                per_class.extend(_per_class_rows(position, fold, seed, "condition_block", held_block))
                for class_index, count in enumerate(histogram):
                    histograms.append(
                        {
                            "position": position,
                            "fold": fold,
                            "seed": seed,
                            "predicted_class_index": class_index,
                            "predicted_TagID": class_index + 1,
                            "row_count": int(count),
                            "row_fraction": float(count / histogram.sum()),
                        }
                    )
                confusions.extend(
                    _confusion_rows(position, fold, seed, "row", held_row["confusion_matrix"])
                )
                confusions.extend(
                    _confusion_rows(
                        position, fold, seed, "condition_block", held_block["confusion_matrix"]
                    )
                )
                for row in result["inner_history"]:
                    curves.append(
                        {
                            "position": position,
                            "fold": fold,
                            "seed": seed,
                            "stage": "inner_selection",
                            **row,
                        }
                    )
                for row in result["outer_history"]:
                    curves.append(
                        {
                            "position": position,
                            "fold": fold,
                            "seed": seed,
                            "stage": "outer_refit",
                            **row,
                            "inner_validation_accuracy": "",
                            "inner_validation_macro_f1": "",
                            "improved": "",
                            "epochs_without_improvement": "",
                        }
                    )
                block_store[position][seed].update(
                    _block_confusions(data, partitions["outer_held"], held)
                )
                row_matrix = np.asarray(held_row["confusion_matrix"], dtype=np.int64)
                block_matrix = np.asarray(held_block["confusion_matrix"], dtype=np.int64)
                if (
                    int(row_matrix.sum()) != 1050
                    or int(block_matrix.sum()) != 21
                    or int(histogram.sum()) != 1050
                    or not np.array_equal(row_matrix.sum(axis=0), histogram)
                ):
                    consistency_failures.append(run_id)
                print(
                    json.dumps(
                        {
                            "event": "fixed_position_run_complete",
                            "run_id": run_id,
                            "selected_epoch": int(result["selected_epoch"]),
                            "held_row_accuracy": float(held_row["accuracy"]),
                            "held_block_accuracy": float(held_block["accuracy"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    for position in config["positions"]:
        for seed in config["seeds"]:
            if len(block_store[position][int(seed)]) != 63:
                consistency_failures.append(f"incomplete_block_store/{position}/S{seed}")
    if consistency_failures:
        raise RuntimeError(
            "CONFUSION_CONSISTENCY_FAILURE: " + ";".join(consistency_failures)
        )
    return {
        "run_register": run_register,
        "run_metrics": run_metrics,
        "per_class": per_class,
        "histograms": histograms,
        "curves": curves,
        "confusions": confusions,
        "block_store": block_store,
        "consistency_failures": consistency_failures,
    }


def _bootstrap_intervals(
    position: str,
    store: dict[int, dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, tuple[float, float, float]]:
    seeds = [int(value) for value in config["seeds"]]
    first = store[seeds[0]]
    by_tag: dict[int, list[str]] = defaultdict(list)
    for condition_id, record in first.items():
        by_tag[int(record["tag_id"])].append(condition_id)
    if sorted((tag, len(ids)) for tag, ids in by_tag.items()) != [
        (tag, 9) for tag in range(1, 8)
    ]:
        raise RuntimeError(f"Bootstrap block structure differs: {position}")
    rng = np.random.default_rng(int(config["confidence_intervals"]["seed"]))
    resamples = int(config["confidence_intervals"]["resamples"])
    values = {
        "held_row_accuracy": np.empty(resamples, dtype=np.float64),
        "held_row_macro_f1": np.empty(resamples, dtype=np.float64),
        "held_block_accuracy": np.empty(resamples, dtype=np.float64),
        "held_block_macro_f1": np.empty(resamples, dtype=np.float64),
    }
    ordered_by_tag = {tag: sorted(ids) for tag, ids in by_tag.items()}
    for replicate in range(resamples):
        selected: list[str] = []
        for tag in range(1, 8):
            ids = ordered_by_tag[tag]
            selected.extend(ids[int(index)] for index in rng.integers(0, 9, size=9))
        row_accuracy = []
        row_macro = []
        block_accuracy = []
        block_macro = []
        for seed in seeds:
            row_matrix = np.zeros((7, 7), dtype=np.int64)
            block_matrix = np.zeros((7, 7), dtype=np.int64)
            for condition_id in selected:
                row_matrix += store[seed][condition_id]["row_confusion"]
                block_matrix += store[seed][condition_id]["block_confusion"]
            row_metrics = metrics_from_confusion(row_matrix)
            block_metrics = metrics_from_confusion(block_matrix)
            row_accuracy.append(float(row_metrics["accuracy"]))
            row_macro.append(float(row_metrics["macro_f1"]))
            block_accuracy.append(float(block_metrics["accuracy"]))
            block_macro.append(float(block_metrics["macro_f1"]))
        values["held_row_accuracy"][replicate] = statistics.fmean(row_accuracy)
        values["held_row_macro_f1"][replicate] = statistics.fmean(row_macro)
        values["held_block_accuracy"][replicate] = statistics.fmean(block_accuracy)
        values["held_block_macro_f1"][replicate] = statistics.fmean(block_macro)
    point_values: dict[str, list[float]] = defaultdict(list)
    all_condition_ids = sorted(first)
    for seed in seeds:
        row_matrix = np.zeros((7, 7), dtype=np.int64)
        block_matrix = np.zeros((7, 7), dtype=np.int64)
        for condition_id in all_condition_ids:
            row_matrix += store[seed][condition_id]["row_confusion"]
            block_matrix += store[seed][condition_id]["block_confusion"]
        row_metrics = metrics_from_confusion(row_matrix)
        block_metrics = metrics_from_confusion(block_matrix)
        point_values["held_row_accuracy"].append(float(row_metrics["accuracy"]))
        point_values["held_row_macro_f1"].append(float(row_metrics["macro_f1"]))
        point_values["held_block_accuracy"].append(float(block_metrics["accuracy"]))
        point_values["held_block_macro_f1"].append(float(block_metrics["macro_f1"]))
    return {
        metric: (
            statistics.fmean(point_values[metric]),
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        )
        for metric, samples in values.items()
    }


def aggregate_positions(
    run_metrics: list[dict[str, Any]],
    block_store: dict[str, dict[int, dict[str, dict[str, Any]]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = (
        "initial_held_accuracy",
        "initial_held_macro_f1",
        "training_accuracy",
        "training_macro_f1",
        "validation_accuracy",
        "validation_macro_f1",
        "held_row_accuracy",
        "held_row_macro_f1",
        "held_block_accuracy",
        "held_block_macro_f1",
        "train_to_held_accuracy_gap",
        "uniform_seven_class_chance",
        "empirical_row_majority_baseline",
        "condition_block_majority_baseline",
    )
    output: list[dict[str, Any]] = []
    for position in config["positions"]:
        selected = [row for row in run_metrics if row["position"] == position]
        if len(selected) != 15:
            raise RuntimeError(f"Aggregate run count differs: {position}")
        intervals = _bootstrap_intervals(position, block_store[position], config)
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            ci = intervals.get(metric)
            output.append(
                {
                    "position": position,
                    "metric": metric,
                    "run_count": len(values),
                    "mean": statistics.fmean(values),
                    "population_standard_deviation": statistics.pstdev(values),
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "confidence_interval_point_estimate": "" if ci is None else ci[0],
                    "confidence_interval_95_low": "" if ci is None else ci[1],
                    "confidence_interval_95_high": "" if ci is None else ci[2],
                    "confidence_interval_estimand": (
                        "not_applicable"
                        if ci is None
                        else "mean across five seed metrics after pooling all 63 held condition blocks per seed"
                    ),
                    "confidence_interval_method": (
                        "not_applicable"
                        if ci is None
                        else "10000-resample TagID-stratified condition-block cluster percentile bootstrap; paired across five seeds"
                    ),
                }
            )
    return output


def reaggregate_from_runtime(
    *,
    config_path: Path = DEFAULT_CONFIG,
    raw_root: Path,
    strict_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    """Rebuild block-bootstrap aggregates from saved primary checkpoints only."""

    config = read_json(config_path)
    data_by_position, manifest_records, _ = prepare_inputs_and_manifests(
        config=config,
        raw_root=raw_root,
        strict_root=strict_root,
        runtime_root=runtime_root,
    )
    with (DEFAULT_RESULTS / "06_PER_RUN_METRICS.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        run_metrics = list(csv.DictReader(handle))
    metric_lookup = {row["run_id"]: row for row in run_metrics}
    block_store: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for position in config["positions"]:
        data = data_by_position[position]
        for fold in range(1, 4):
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                run_id = f"{position}_F{fold}_S{seed}"
                run_manifest = _run_manifest_rows(manifest_records, position, fold, seed)
                partitions, _ = build_partitions(data, run_manifest)
                checkpoint = (
                    Path(runtime_root)
                    / "primary"
                    / position
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                    / "outer_refit_checkpoint.pt"
                )
                model, _ = load_checkpoint_model(checkpoint, seed)
                logits, _ = extract_logits_embeddings(model, partitions["outer_held"])
                held = evaluate_logits(partitions["outer_held"], logits)
                reported = metric_lookup[run_id]
                if not (
                    float(held["sample"]["accuracy"]) == float(reported["held_row_accuracy"])
                    and float(held["sample"]["macro_f1"])
                    == float(reported["held_row_macro_f1"])
                    and float(held["condition"]["metrics"]["accuracy"])
                    == float(reported["held_block_accuracy"])
                    and float(held["condition"]["metrics"]["macro_f1"])
                    == float(reported["held_block_macro_f1"])
                ):
                    raise RuntimeError(f"Saved checkpoint metrics differ: {run_id}")
                block_store[position][seed].update(
                    _block_confusions(data, partitions["outer_held"], held)
                )
    aggregates = aggregate_positions(run_metrics, block_store, config)
    write_csv(DEFAULT_RESULTS / "07_POSITION_AGGREGATES.csv", aggregates)
    return {
        "status": "PASS_AGGREGATES_REBUILT_WITHOUT_RETRAINING",
        "checkpoint_count": 60,
        "aggregate_rows": len(aggregates),
    }


def prepare_inputs_and_manifests(
    *,
    config: dict[str, Any],
    raw_root: Path,
    strict_root: Path,
    runtime_root: Path,
    results_root: Path = DEFAULT_RESULTS,
    manifests_root: Path = DEFAULT_MANIFESTS,
) -> tuple[dict[str, FixedPositionData], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("Preregistered protocol must exist before preparation")
    data_by_position = {
        position: load_governed_position(raw_root, position) for position in config["positions"]
    }
    compatibility = audit_canonical_compatibility(data_by_position, strict_root, runtime_root)
    audit_rows = structure_audit_rows(data_by_position, compatibility)
    write_csv(results_root / "03_DATA_AND_GROUP_STRUCTURE_AUDIT.csv", audit_rows)
    manifest_records = generate_split_manifest(data_by_position, config)
    repeated_records = generate_split_manifest(data_by_position, config)
    validations = split_validation_rows(
        manifest_records, repeated_records, config, compatibility
    )
    manifest_path = manifests_root / "exact_group_split_manifest.csv"
    write_csv(manifest_path, manifest_records)
    digest = sha256_file(manifest_path)
    (manifests_root / "exact_group_split_manifest.sha256").write_text(
        f"{digest}  exact_group_split_manifest.csv\n", encoding="ascii"
    )
    write_csv(results_root / "04_SPLIT_VALIDATION_RESULTS.csv", validations)
    return data_by_position, manifest_records, compatibility


def execute_benchmark(
    *,
    config_path: Path = DEFAULT_CONFIG,
    raw_root: Path,
    strict_root: Path,
    runtime_root: Path,
    prepare_only: bool = False,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config["protocol_id"] != "FIXED_POSITION_CONDITION_GROUPED_VALIDITY_V1":
        raise RuntimeError("Unexpected protocol identifier")
    runtime_root = Path(runtime_root).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    if not str(runtime_root).casefold().startswith(str(PROJECT_ROOT.resolve()).casefold()):
        raise RuntimeError(
            "The canonical checkpoint recorder requires CRFID_FIXED_POSITION_RUN_ROOT "
            "inside this clone; use the ignored outputs/ directory"
        )
    data_by_position, manifest_records, compatibility = prepare_inputs_and_manifests(
        config=config,
        raw_root=raw_root,
        strict_root=strict_root,
        runtime_root=runtime_root,
    )
    if prepare_only:
        return {
            "status": "PASS_PRETRAINING_VALIDATION",
            "manifest_rows": len(manifest_records),
            "compatibility": compatibility,
        }

    learning_results = [
        run_learning_validity(data_by_position[position], manifest_records, config)
        for position in config["positions"]
    ]
    write_csv(DEFAULT_RESULTS / "11_LEARNING_VALIDITY_RESULTS.csv", learning_results)
    execution = execute_primary_runs(
        data_by_position, manifest_records, config, runtime_root
    )
    aggregates = aggregate_positions(
        execution["run_metrics"], execution["block_store"], config
    )
    write_csv(DEFAULT_RESULTS / "05_RUN_REGISTER.csv", execution["run_register"])
    write_csv(DEFAULT_RESULTS / "06_PER_RUN_METRICS.csv", execution["run_metrics"])
    write_csv(DEFAULT_RESULTS / "07_POSITION_AGGREGATES.csv", aggregates)
    write_csv(DEFAULT_RESULTS / "08_PER_CLASS_RESULTS.csv", execution["per_class"])
    write_csv(DEFAULT_RESULTS / "09_PREDICTED_CLASS_HISTOGRAMS.csv", execution["histograms"])
    write_csv(DEFAULT_RESULTS / "learning_curves.csv", execution["curves"])
    write_csv(DEFAULT_RESULTS / "confusion_matrices.csv", execution["confusions"])
    report = (
        "# Confusion consistency report\n\n"
        "All 60 primary runs passed the compact confusion-matrix consistency checks. "
        "Each row-level matrix sums to 1,050, each condition-block matrix sums to 21, "
        "each predicted-class histogram sums to 1,050, and every histogram equals the "
        "corresponding confusion-matrix column sums.\n\n"
        "The committed matrices contain counts only; logits and full predictions remain "
        "under the ignored runtime boundary and are not committed.\n"
    )
    (DEFAULT_RESULTS / "10_CONFUSION_CONSISTENCY_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    return {
        "status": "PASS_FULL_BENCHMARK_EXECUTED",
        "run_count": len(execution["run_metrics"]),
        "learning_validity": learning_results,
        "aggregate_rows": len(aggregates),
    }


def environment_paths() -> tuple[Path, Path, Path]:
    required = {
        "CRFID_FIXED_POSITION_DATA_ROOT": os.environ.get("CRFID_FIXED_POSITION_DATA_ROOT", "").strip(),
        "CRFID_STRICT_DG_ARTIFACT_ROOT": os.environ.get("CRFID_STRICT_DG_ARTIFACT_ROOT", "").strip(),
        "CRFID_FIXED_POSITION_RUN_ROOT": os.environ.get("CRFID_FIXED_POSITION_RUN_ROOT", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("BLOCKED_GOVERNED_INPUTS: missing " + ", ".join(missing))
    return tuple(Path(required[name]).expanduser() for name in required)  # type: ignore[return-value]
