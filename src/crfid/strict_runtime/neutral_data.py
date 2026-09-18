"""Read-only source-domain artifact loader with train-partition-only preprocessing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .governance import write_json
from .hashing import array_sha256, canonical_json_sha256, hash_lines, sha256_file
from .scale_policy import MODE_REPLACE_WITH_ONE, apply_scale_policy


@dataclass(frozen=True)
class PartitionData:
    registry_rows: np.ndarray
    inputs: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    condition_ids: list[str]
    exact_signal_hashes: list[str]
    unique_signal_weights: np.ndarray


class CanonicalPhase2Data:
    """Validated P1-P3 arrays, registry, splits, and local preprocessing states."""

    def __init__(self, config: dict, artifact_root: Path, preprocessing_root: Path) -> None:
        self.config = config
        self.artifact_root = artifact_root.resolve()
        self.preprocessing_root = preprocessing_root.resolve()
        input_config = config["input"]
        signal_path = self.artifact_root / input_config["canonical_signal_path"]
        label_path = self.artifact_root / input_config["canonical_label_path"]
        registry_path = self.artifact_root / input_config["registry_path"]
        split_path = self.artifact_root / input_config["split_path"]
        for path in (signal_path, label_path, registry_path, split_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            if "p4" in str(path).casefold():
                raise RuntimeError(f"Target-domain path blocked during source selection: {path}")
        self.input_paths = (signal_path, label_path, registry_path, split_path)
        self.signals = np.load(signal_path, allow_pickle=False)
        self.labels = np.load(label_path, allow_pickle=False)
        if tuple(self.signals.shape) != tuple(input_config["custody_shape"]):
            raise RuntimeError(f"Canonical signal shape changed: {self.signals.shape}")
        if self.signals.dtype.str != input_config["custody_dtype"]:
            raise RuntimeError(f"Canonical signal dtype changed: {self.signals.dtype.str}")
        if self.labels.shape != (len(self.signals),) or self.labels.dtype.str != "<i8":
            raise RuntimeError("Canonical label custody changed")
        self.registry_rows = self._load_registry(registry_path)
        self.partitions = self._load_splits(split_path)
        self.split_file_sha256 = sha256_file(split_path)
        self.split_definition_sha256 = canonical_json_sha256(
            {
                fold: {name: hash_lines(map(str, values)) for name, values in parts.items()}
                for fold, parts in self.partitions.items()
            }
        )
        self._states: dict[tuple[str, str], dict] = {}
        self._fit_raw_preprocessing()

    def _load_registry(self, path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(self.signals):
            raise RuntimeError("Registry and signal sample counts differ")
        converted = []
        for expected, row in enumerate(rows):
            if int(row["registry_row"]) != expected:
                raise RuntimeError("Registry row order changed")
            converted.append(
                {
                    "registry_row": expected,
                    "sample_id": row["sample_id"],
                    "raw_condition_id": row["raw_condition_id"],
                    "tag_id": int(row["tag_id"]),
                    "label_index": int(row["label_index"]),
                    "position": row["position"],
                    "exact_signal_sha256": row["exact_signal_sha256"],
                    "unique_signal_weight": float(row["unique_signal_weight"]),
                }
            )
        observed = np.asarray([row["label_index"] for row in converted], dtype=np.int64)
        if not np.array_equal(observed, self.labels):
            raise RuntimeError("Registry labels differ from label array")
        return converted

    def _load_splits(self, path: Path) -> dict[str, dict[str, np.ndarray]]:
        names = ("inner_train", "inner_validation", "outer_held")
        assignments = {fold: {name: [] for name in names} for fold in self.config["folds"]}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                fold, part = row["fold_id"], row["partition"]
                if fold not in assignments or part not in assignments[fold]:
                    raise RuntimeError(f"Unexpected split assignment: {fold}/{part}")
                index = int(row["registry_row"])
                if row["sample_id"] != self.registry_rows[index]["sample_id"]:
                    raise RuntimeError("Split sample ID mismatch")
                assignments[fold][part].append(index)
        expected = set(range(len(self.signals)))
        result = {}
        for fold, parts in assignments.items():
            converted = {name: np.asarray(sorted(values), dtype=np.int64) for name, values in parts.items()}
            concatenated = np.concatenate(list(converted.values()))
            if set(concatenated.tolist()) != expected or len(concatenated) != len(expected):
                raise RuntimeError(f"Fold {fold} is not a complete disjoint partition")
            result[fold] = converted
        return result

    def _indices(self, fold_id: str, stage: str, name: str) -> np.ndarray:
        if stage == "inner_selection":
            return self.partitions[fold_id][name]
        if stage == "outer_refit" and name == "outer_development":
            return np.sort(np.concatenate((self.partitions[fold_id]["inner_train"], self.partitions[fold_id]["inner_validation"])))
        if stage == "outer_refit" and name == "outer_held":
            return self.partitions[fold_id]["outer_held"]
        raise ValueError(f"Invalid partition: {fold_id}/{stage}/{name}")

    def _fit_raw_preprocessing(self) -> None:
        for fold in self.config["folds"]:
            for stage, fit_name in (("inner_selection", "inner_train"), ("outer_refit", "outer_development")):
                indices = self._indices(fold, stage, fit_name)
                fit = np.ascontiguousarray(self.signals[indices], dtype=np.float64)
                mean = np.ascontiguousarray(fit.mean(axis=0, dtype=np.float64))
                raw_scale = np.ascontiguousarray(fit.std(axis=0, ddof=0, dtype=np.float64))
                # Historical raw-representation behaviour, preserved bit-for-bit
                # via an explicit mode. See crfid.strict_runtime.scale_policy.
                scale = apply_scale_policy(raw_scale, mode=MODE_REPLACE_WITH_ONE)
                directory = self.preprocessing_root / "raw" / fold
                directory.mkdir(parents=True, exist_ok=True)
                mean_path = directory / f"{stage}_mean_float64.npy"
                scale_path = directory / f"{stage}_scale_float64.npy"
                np.save(mean_path, mean, allow_pickle=False)
                np.save(scale_path, scale, allow_pickle=False)
                semantic = {
                    "schema_version": 1,
                    "fold_id": fold,
                    "stage": stage,
                    "fit_partition": fit_name,
                    "fit_sample_count": int(len(indices)),
                    "fit_registry_rows_sha256": hash_lines(map(str, indices)),
                    "ddof": 0,
                    "minimum_scale": 1e-12,
                    "scale_replacement": 1.0,
                    "mean_array_sha256": array_sha256(mean),
                    "scale_array_sha256": array_sha256(scale),
                    "mean_file": str(mean_path),
                    "scale_file": str(scale_path),
                    "p4_used": False,
                }
                state = {**semantic, "state_sha256": canonical_json_sha256(semantic)}
                write_json(directory / f"{stage}_state.json", state)
                self._states[(fold, stage)] = state

    def preprocessing_state(self, fold_id: str, stage: str) -> dict:
        return dict(self._states[(fold_id, stage)])

    def transformed_partition(self, fold_id: str, stage: str, partition_name: str) -> PartitionData:
        indices = self._indices(fold_id, stage, partition_name)
        state = self._states[(fold_id, stage)]
        mean = np.load(state["mean_file"], allow_pickle=False)
        scale = np.load(state["scale_file"], allow_pickle=False)
        values = np.ascontiguousarray(((self.signals[indices] - mean) / scale).astype(np.float32))
        rows = [self.registry_rows[int(index)] for index in indices]
        return PartitionData(
            registry_rows=indices.copy(),
            inputs=values,
            labels=np.ascontiguousarray(self.labels[indices], dtype=np.int64),
            sample_ids=[row["sample_id"] for row in rows],
            condition_ids=[row["raw_condition_id"] for row in rows],
            exact_signal_hashes=[row["exact_signal_sha256"] for row in rows],
            unique_signal_weights=np.asarray([row["unique_signal_weight"] for row in rows], dtype=np.float64),
        )
