"""Condition-block-disjoint representation-versus-signal diagnostic primitives.

This module is additive.  It binds the canonical data schema, C1 frozen
encoder, reference-peak extractor, metrics, folds, probe and paired inference
without changing any historical implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import types
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .analysis.reference_peaks import strongest_window_peaks
from .evaluation.metrics import classification_metrics


CLASS_COUNT = 7
REPRESENTATIONS = (
    "RAW_SIGNAL_281",
    "FIRST_DIFFERENCE_280",
    "C1_ENCODER_EMBEDDING",
    "PEAK_DESCRIPTOR",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("status_before_execution") != "PREREGISTERED":
        raise RuntimeError("Protocol is not preregistered")
    if tuple(config.get("representations", ())) != REPRESENTATIONS:
        raise RuntimeError("Representation family binding changed")
    if int(config["bootstrap"]["replicates"]) < 10_000:
        raise RuntimeError("Bootstrap replicate count is below 10,000")
    return config


@dataclass(frozen=True)
class PositionData:
    position: str
    signals: np.ndarray
    labels: np.ndarray
    tag_ids: np.ndarray
    er: np.ndarray
    surfaces: np.ndarray
    repeat_index: np.ndarray
    block_tokens: np.ndarray
    source_files: tuple[dict[str, Any], ...]

    def block_keys(self) -> list[tuple[int, int, str]]:
        return sorted(
            {
                (int(tag), int(er), str(surface))
                for tag, er, surface in zip(self.tag_ids, self.er, self.surfaces)
            }
        )


def _block_token(position: str, tag_id: int, er: int, surface: str) -> str:
    digest = hashlib.sha256(f"{position}|{tag_id}|{er}|{surface}".encode("ascii")).hexdigest()
    return f"BLOCK-{digest[:16]}"


def resolve_governed_data_root(
    repository_root: str | Path, config: Mapping[str, Any]
) -> tuple[Path, str]:
    repository_root = Path(repository_root).resolve()
    governed = config["governed_data"]
    candidates: list[tuple[Path, str]] = []
    environment_name = str(governed["environment_variable"])
    if os.environ.get(environment_name):
        candidates.append((Path(os.environ[environment_name]).expanduser(), f"ENV:{environment_name}"))
    for index, relative in enumerate(governed["repository_relative_candidates"]):
        candidates.append((repository_root / relative, f"REPOSITORY_RELATIVE_CANDIDATE_{index}"))
    expected = [
        f"A{surface}_P{position}.csv"
        for position in range(1, 5)
        for surface in range(1, 4)
    ]
    for candidate, token in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and all((resolved / name).is_file() for name in expected):
            return resolved, token
    raise FileNotFoundError("BLOCKED_GOVERNED_INPUTS")


def load_governed_positions(
    repository_root: str | Path, config: Mapping[str, Any]
) -> tuple[dict[str, PositionData], dict[str, Any]]:
    root, root_token = resolve_governed_data_root(repository_root, config)
    expected_meta = ["A3", "A2", "A1", "P4", "P3", "P2", "P1", "ER", "TagID"]
    expected_signal = [str(index) for index in range(int(config["raw_dimension"]))]
    result: dict[str, PositionData] = {}
    custody_files: list[dict[str, Any]] = []
    for position in config["positions"]:
        signals: list[list[float]] = []
        labels: list[int] = []
        tag_ids: list[int] = []
        er_values: list[int] = []
        surfaces: list[str] = []
        repeat_indices: list[int] = []
        block_tokens: list[str] = []
        source_files: list[dict[str, Any]] = []
        counters: Counter[tuple[int, int, str]] = Counter()
        position_index = int(str(position)[1:])
        for surface_index, surface in enumerate(config["surfaces"], start=1):
            path = root / f"A{surface_index}_{position}.csv"
            file_record = {
                "path_token": f"{root_token}/{path.name}",
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            source_files.append(file_record)
            custody_files.append(file_record)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or reader.fieldnames[:9] != expected_meta:
                    raise RuntimeError(f"BLOCKED_DATA_OR_LINEAGE_MISMATCH: metadata schema {path.name}")
                if reader.fieldnames[9:] != expected_signal:
                    raise RuntimeError(f"BLOCKED_DATA_OR_LINEAGE_MISMATCH: signal schema {path.name}")
                for row in reader:
                    tag_id = int(float(row["TagID"]))
                    er = int(float(row["ER"]))
                    if tag_id not in config["tag_ids"] or er not in config["er_levels"]:
                        raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: label value")
                    surface_sum = sum(int(float(row[f"A{i}"])) for i in range(1, 4))
                    position_sum = sum(int(float(row[f"P{i}"])) for i in range(1, 5))
                    if (
                        surface_sum != 1
                        or int(float(row[f"A{surface_index}"])) != 1
                        or position_sum != 1
                        or int(float(row[f"P{position_index}"])) != 1
                    ):
                        raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: one-hot metadata")
                    vector = [float(row[str(index)]) for index in range(int(config["raw_dimension"]))]
                    if not np.isfinite(vector).all():
                        raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: nonfinite signal")
                    key = (tag_id, er, surface)
                    repeat_index = counters[key]
                    counters[key] += 1
                    signals.append(vector)
                    labels.append(tag_id - 1)
                    tag_ids.append(tag_id)
                    er_values.append(er)
                    surfaces.append(surface)
                    repeat_indices.append(repeat_index)
                    block_tokens.append(_block_token(str(position), tag_id, er, surface))
        data = PositionData(
            position=str(position),
            signals=np.ascontiguousarray(signals, dtype=np.float64),
            labels=np.ascontiguousarray(labels, dtype=np.int64),
            tag_ids=np.ascontiguousarray(tag_ids, dtype=np.int64),
            er=np.ascontiguousarray(er_values, dtype=np.int64),
            surfaces=np.asarray(surfaces, dtype="<U2"),
            repeat_index=np.ascontiguousarray(repeat_indices, dtype=np.int64),
            block_tokens=np.asarray(block_tokens, dtype="<U22"),
            source_files=tuple(source_files),
        )
        expected_rows = 3150
        expected_blocks = 63
        expected_repeats = int(config["repeats_per_block"])
        if data.signals.shape != (expected_rows, int(config["raw_dimension"])):
            raise RuntimeError(f"BLOCKED_DATA_OR_LINEAGE_MISMATCH: {position} signal shape")
        if len(data.block_keys()) != expected_blocks or sorted(set(counters.values())) != [expected_repeats]:
            raise RuntimeError(f"BLOCKED_DATA_OR_LINEAGE_MISMATCH: {position} blocks")
        if sorted(set(data.labels.tolist())) != list(config["class_order"]):
            raise RuntimeError(f"BLOCKED_DATA_OR_LINEAGE_MISMATCH: {position} labels")
        if np.diff(data.signals, axis=1).shape[1] != int(config["first_difference_dimension"]):
            raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: first-difference dimension")
        result[str(position)] = data
    return result, {"root_token": root_token, "files": custody_files}


def cell_latin_fold(er: int, surface: str, config: Mapping[str, Any]) -> int:
    for fold, cells in config["folds"]["test_cells"].items():
        if [int(er), str(surface)] in cells:
            return int(fold)
    raise ValueError(f"Cell is absent from frozen Latin square: {(er, surface)}")


def split_role(er: int, surface: str, test_fold: int, config: Mapping[str, Any]) -> str:
    cell_fold = cell_latin_fold(er, surface, config)
    if cell_fold == test_fold:
        return "test"
    if cell_fold == (test_fold + 1) % 3:
        return "validation"
    if cell_fold == (test_fold + 2) % 3:
        return "train"
    raise AssertionError("Unreachable Latin-square assignment")


def split_indices(
    data: PositionData, test_fold: int, config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    roles = np.asarray(
        [split_role(int(er), str(surface), test_fold, config) for er, surface in zip(data.er, data.surfaces)]
    )
    result = {role: np.flatnonzero(roles == role) for role in ("train", "validation", "test")}
    if any(len(indices) != 1050 for indices in result.values()):
        raise RuntimeError("Frozen fold does not contain 1,050 rows per split")
    block_sets = {role: set(data.block_tokens[indices].tolist()) for role, indices in result.items()}
    if any(len(blocks) != 21 for blocks in block_sets.values()):
        raise RuntimeError("Frozen fold does not contain 21 blocks per split")
    if block_sets["train"] & block_sets["validation"] or block_sets["train"] & block_sets["test"] or block_sets["validation"] & block_sets["test"]:
        raise RuntimeError("Condition-block leakage detected")
    return result


def synchronized_split_manifest(
    positions: Mapping[str, PositionData], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in config["positions"]:
        data = positions[str(position)]
        for test_fold in range(3):
            for tag_id, er, surface in data.block_keys():
                mask = (data.tag_ids == tag_id) & (data.er == er) & (data.surfaces == surface)
                tokens = sorted(set(data.block_tokens[mask].tolist()))
                if len(tokens) != 1 or int(mask.sum()) != int(config["repeats_per_block"]):
                    raise RuntimeError("Block manifest construction failed")
                rows.append(
                    {
                        "position": position,
                        "fold": test_fold,
                        "tag_id": tag_id,
                        "er": er,
                        "surface": surface,
                        "block_token": tokens[0],
                        "split": split_role(er, surface, test_fold, config),
                        "repeat_count": int(mask.sum()),
                    }
                )
    return rows


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    fit_row_count: int
    fit_partition: str = "train"

    @classmethod
    def fit(cls, values: np.ndarray, *, partition: str = "train") -> "FeatureStandardizer":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
            raise ValueError("Standardizer needs a finite nonempty matrix")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(np.ascontiguousarray(mean), np.ascontiguousarray(scale), len(matrix), partition)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1:] != self.mean.shape:
            raise ValueError("Standardizer feature dimension changed")
        return np.ascontiguousarray((matrix - self.mean) / self.scale, dtype=np.float64)

    @property
    def state_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "mean_sha256": array_sha256(self.mean),
                "scale_sha256": array_sha256(self.scale),
                "fit_row_count": self.fit_row_count,
                "fit_partition": self.fit_partition,
            }
        )


def raw_representation(data: PositionData) -> np.ndarray:
    values = np.asarray(data.signals, dtype=np.float64)
    if values.shape != (3150, 281):
        raise RuntimeError("RAW_SIGNAL_281 dimension changed")
    return np.ascontiguousarray(values)


def first_difference_representation(data: PositionData) -> np.ndarray:
    values = np.ascontiguousarray(np.diff(data.signals, axis=1), dtype=np.float64)
    if values.shape != (3150, 280):
        raise RuntimeError("FIRST_DIFFERENCE_280 dimension changed")
    return values


def peak_descriptor_matrix(
    signals: np.ndarray, peak_config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 281:
        raise ValueError("Peak descriptor requires [N, 281] raw signals")
    axis = np.arange(281, dtype=np.float64)
    windows = OrderedDict(
        (name, (float(bounds[0]), float(bounds[1])))
        for name, bounds in peak_config["ordered_index_windows_inclusive"].items()
    )
    descriptors = np.empty((len(values), int(peak_config["dimension"])), dtype=np.float64)
    missing = np.empty((len(values), len(windows)), dtype=np.bool_)
    for row_index, signal in enumerate(values):
        found = strongest_window_peaks(signal, axis, windows, mode=str(peak_config["mode"]))
        cursor = 0
        for window_index, name in enumerate(windows):
            record = found[name]
            is_missing = not bool(record["interior_local_extremum"])
            descriptors[row_index, cursor : cursor + 3] = (
                float(record["index"]),
                float(record["value"]),
                float(is_missing),
            )
            missing[row_index, window_index] = is_missing
            cursor += 3
    if descriptors.shape[1] != int(peak_config["dimension"]) or not np.isfinite(descriptors).all():
        raise RuntimeError("Peak descriptor dimensionality or finiteness changed")
    return descriptors, missing


def synthetic_peak_controls(peak_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    axis = np.arange(281, dtype=np.float64)
    windows = OrderedDict(
        (name, (float(bounds[0]), float(bounds[1])))
        for name, bounds in peak_config["ordered_index_windows_inclusive"].items()
    )
    centers = [30, 95, 155, 220]

    def dips(shift: int = 0, width: float = 1.5, amplitude: float = 5.0) -> np.ndarray:
        signal = np.zeros(281, dtype=np.float64)
        for center in centers:
            signal -= amplitude * np.exp(-0.5 * ((axis - (center + shift)) / width) ** 2)
        return signal

    rows: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: object, expected: object) -> None:
        rows.append(
            {
                "record_type": "synthetic_control",
                "position": "SYNTHETIC",
                "block_token": name,
                "check": name,
                "observed": json.dumps(observed, separators=(",", ":")),
                "expected": json.dumps(expected, separators=(",", ":")),
                "pass": bool(passed),
            }
        )

    clean = strongest_window_peaks(dips(), axis, windows, mode="min")
    clean_indices = [int(clean[name]["index"]) for name in windows]
    record("clean_known_peak", clean_indices == centers, clean_indices, centers)

    two = dips()
    two -= 7.0 * np.exp(-0.5 * ((axis - 50) / 1.2) ** 2)
    two_found = strongest_window_peaks(two, axis, windows, mode="min")
    record("two_known_separated_peaks", int(two_found["W0"]["index"]) == 50, int(two_found["W0"]["index"]), 50)

    shifted = strongest_window_peaks(dips(shift=3), axis, windows, mode="min")
    shifted_indices = [int(shifted[name]["index"]) for name in windows]
    record("shifted_peaks", shifted_indices == [center + 3 for center in centers], shifted_indices, [center + 3 for center in centers])

    broad = strongest_window_peaks(dips(width=6.0), axis, windows, mode="min")
    broad_indices = [int(broad[name]["index"]) for name in windows]
    record("broadened_peaks", broad_indices == centers, broad_indices, centers)

    low = strongest_window_peaks(dips(amplitude=1e-4), axis, windows, mode="min")
    low_flags = [bool(low[name]["interior_local_extremum"]) for name in windows]
    record("low_amplitude_peaks", all(low_flags), low_flags, [True] * 4)

    absent = strongest_window_peaks(np.zeros(281), axis, windows, mode="min")
    absent_flags = [not bool(absent[name]["interior_local_extremum"]) for name in windows]
    record("absent_peaks", all(absent_flags), absent_flags, [True] * 4)

    tied = np.zeros(281, dtype=np.float64)
    tied[25] = 2.0
    tied[45] = 2.0
    tied_found = strongest_window_peaks(tied, axis, {"W0": windows["W0"]}, mode="max")
    record("tied_local_maxima", int(tied_found["W0"]["index"]) == 25, int(tied_found["W0"]["index"]), 25)
    return rows


def validate_peak_extractor(
    positions: Mapping[str, PositionData], peak_config: Mapping[str, Any]
) -> tuple[bool, dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    descriptors: dict[str, np.ndarray] = {}
    missing_arrays: dict[str, np.ndarray] = {}
    deterministic = True
    block_intermediate: dict[str, list[dict[str, Any]]] = {}
    for position, data in positions.items():
        matrix, missing = peak_descriptor_matrix(data.signals, peak_config)
        second, second_missing = peak_descriptor_matrix(data.signals[:1], peak_config)
        deterministic = deterministic and np.array_equal(matrix[:1], second) and np.array_equal(missing[:1], second_missing)
        descriptors[position] = matrix
        missing_arrays[position] = missing
        rows: list[dict[str, Any]] = []
        for tag_id, er, surface in data.block_keys():
            mask = (data.tag_ids == tag_id) & (data.er == er) & (data.surfaces == surface)
            selected = matrix[mask]
            selected_missing = missing[mask]
            rows.append(
                {
                    "record_type": "condition_block",
                    "position": position,
                    "block_token": str(data.block_tokens[np.flatnonzero(mask)[0]]),
                    "check": "repeat_stability",
                    "tag_id": tag_id,
                    "er": er,
                    "surface": surface,
                    "repeat_count": int(mask.sum()),
                    "within_peak_index_variance": float(np.mean(np.var(selected[:, 0::3], axis=0, ddof=0))),
                    "within_amplitude_variance": float(np.mean(np.var(selected[:, 1::3], axis=0, ddof=0))),
                    "missing_peak_frequency": float(selected_missing.mean()),
                    "mean_indices": selected[:, 0::3].mean(axis=0),
                    "mean_amplitudes": selected[:, 1::3].mean(axis=0),
                }
            )
        block_intermediate[position] = rows

    output_rows: list[dict[str, Any]] = []
    index_ratios: list[float] = []
    amplitude_ratios: list[float] = []
    for position, rows in block_intermediate.items():
        between_index = float(np.mean(np.var(np.stack([row["mean_indices"] for row in rows]), axis=0, ddof=0)))
        between_amplitude = float(np.mean(np.var(np.stack([row["mean_amplitudes"] for row in rows]), axis=0, ddof=0)))
        for row in rows:
            index_ratio = float(row["within_peak_index_variance"] / between_index) if between_index > 0 else float("inf")
            amplitude_ratio = float(row["within_amplitude_variance"] / between_amplitude) if between_amplitude > 0 else float("inf")
            index_ratios.append(index_ratio)
            amplitude_ratios.append(amplitude_ratio)
            output_rows.append(
                {
                    **{key: value for key, value in row.items() if key not in {"mean_indices", "mean_amplitudes"}},
                    "between_condition_peak_index_variance": between_index,
                    "between_condition_amplitude_variance": between_amplitude,
                    "within_to_between_index_variance_ratio": index_ratio,
                    "within_to_between_amplitude_variance_ratio": amplitude_ratio,
                    "observed": "",
                    "expected": "",
                    "pass": "",
                }
            )
    synthetic = synthetic_peak_controls(peak_config)
    output_rows.extend(synthetic)
    thresholds = peak_config["validity_thresholds"]
    global_missing = float(np.concatenate([item.reshape(-1) for item in missing_arrays.values()]).mean())
    median_index_ratio = float(np.median(index_ratios))
    median_amplitude_ratio = float(np.median(amplitude_ratios))
    synthetic_pass = all(bool(row["pass"]) for row in synthetic)
    summary = {
        "deterministic": bool(deterministic),
        "descriptor_dimension": int(next(iter(descriptors.values())).shape[1]),
        "global_missing_frequency": global_missing,
        "median_within_to_between_index_variance_ratio": median_index_ratio,
        "median_within_to_between_amplitude_variance_ratio": median_amplitude_ratio,
        "synthetic_controls_pass": synthetic_pass,
        "uses_labels": False,
        "uses_test_informed_windows": False,
    }
    passed = bool(
        deterministic
        and summary["descriptor_dimension"] == int(peak_config["dimension"])
        and global_missing <= float(thresholds["maximum_global_missing_frequency"])
        and median_index_ratio <= float(thresholds["maximum_median_within_to_between_index_variance_ratio"])
        and median_amplitude_ratio <= float(thresholds["maximum_median_within_to_between_amplitude_variance_ratio"])
        and synthetic_pass
    )
    summary["status"] = "PASS" if passed else "PEAK_DESCRIPTOR_NOT_VALIDATED"
    return passed, descriptors, output_rows, summary


def resolve_forensic_archive(repository_root: str | Path, config: Mapping[str, Any]) -> Path:
    path = (Path(repository_root).resolve() / config["forensic_archive"]["repository_relative_root"]).resolve()
    if not path.is_dir():
        raise FileNotFoundError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: forensic archive missing")
    return path


def load_c1_binding(
    repository_root: str | Path, config: Mapping[str, Any]
) -> tuple[Any, dict[int, tuple[Any, dict[str, Any]]], np.ndarray, np.ndarray, dict[str, Any]]:
    archive = resolve_forensic_archive(repository_root, config)
    frozen_root = archive / "outputs" / "few_shot" / "frozen_branch"
    module_path = frozen_root / "src" / "few_shot" / "frozen_encoder.py"
    module = types.ModuleType("crfid_frozen_c1_binding")
    module.__file__ = str(module_path)
    module.__package__ = ""
    source = module_path.read_text(encoding="utf-8")
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    recipe = frozen_root / "01_frozen_source_import" / "frozen_last_code_snapshot" / "10_final_recipe_freeze"
    mean_path = recipe / "FINAL_SOURCE_PREPROCESSING_MEAN_FLOAT64.npy"
    scale_path = recipe / "FINAL_SOURCE_PREPROCESSING_SCALE_FLOAT64.npy"
    if sha256_file(mean_path) != config["source_preprocessing"]["mean_file_sha256"]:
        raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: source mean hash")
    if sha256_file(scale_path) != config["source_preprocessing"]["scale_file_sha256"]:
        raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: source scale hash")
    mean = np.load(mean_path, allow_pickle=False)
    scale = np.load(scale_path, allow_pickle=False)
    models: dict[int, tuple[Any, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for seed in config["checkpoint_seeds"]:
        checkpoint = (
            frozen_root
            / "01_frozen_source_import"
            / "frozen_last_code_snapshot"
            / "11_final_p4_evaluation"
            / "runs"
            / f"seed_{seed}"
            / "FINAL_CHECKPOINT.pt"
        )
        model, payload = module.load_frozen_checkpoint(
            checkpoint,
            expected_file_sha256=config["checkpoint_file_sha256"][str(seed)],
            expected_seed=int(seed),
        )
        state_hash = module.model_state_sha256(model)
        if state_hash != config["checkpoint_state_sha256"][str(seed)]:
            raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: model-state hash")
        models[int(seed)] = (model, payload)
        records.append(
            {
                "seed": int(seed),
                "checkpoint_path_token": f"FORENSIC_ARCHIVE/frozen_c1/seed_{seed}/FINAL_CHECKPOINT.pt",
                "checkpoint_file_sha256": config["checkpoint_file_sha256"][str(seed)],
                "model_state_sha256": state_hash,
                "method_id": payload["method_id"],
                "p4_used": bool(payload["p4_used"]),
                "frozen": not model.training and all(not parameter.requires_grad for parameter in model.parameters()),
            }
        )
    architecture = module.architecture_specification(module.FrozenC1CNN1D())
    if architecture["embedding_module_path"] != "network.12" or architecture["embedding_dimension"] != 256:
        raise RuntimeError("BLOCKED_DATA_OR_LINEAGE_MISMATCH: embedding binding")
    binding = {
        "method_id": module.METHOD_ID,
        "architecture": architecture,
        "checkpoints": records,
        "source_preprocessing": {
            "mean_path_token": "FORENSIC_ARCHIVE/frozen_c1/FINAL_SOURCE_PREPROCESSING_MEAN_FLOAT64.npy",
            "mean_file_sha256": sha256_file(mean_path),
            "mean_shape": list(mean.shape),
            "scale_path_token": "FORENSIC_ARCHIVE/frozen_c1/FINAL_SOURCE_PREPROCESSING_SCALE_FLOAT64.npy",
            "scale_file_sha256": sha256_file(scale_path),
            "scale_shape": list(scale.shape),
            "scale_min": float(scale.min()),
            "role": "checkpoint-bound encoder input preprocessing; probe scaling remains training-only",
        },
    }
    return module, models, mean, scale, binding


def extract_c1_embeddings(
    module: Any,
    model: Any,
    raw_signals: np.ndarray,
    source_mean: np.ndarray,
    source_scale: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for start in range(0, len(raw_signals), batch_size):
        stop = min(start + batch_size, len(raw_signals))
        inputs = module.first_difference_source_standardize(raw_signals[start:stop], source_mean, source_scale)
        embeddings = module.extract_penultimate_embeddings(model, inputs)
        pieces.append(embeddings.detach().cpu().numpy())
    result = np.ascontiguousarray(np.concatenate(pieces, axis=0), dtype=np.float64)
    if result.shape != (len(raw_signals), 256):
        raise RuntimeError("Frozen C1 embedding shape changed")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen C1 encoder state changed")
    return result


class TestLabelSeal:
    """Instrumented one-way boundary: predictions must freeze before labels open."""

    def __init__(self, labels: np.ndarray) -> None:
        self.__labels = np.ascontiguousarray(labels, dtype=np.int64).copy()
        self._prediction_hash: str | None = None
        self._opened = False
        self._events = ["01_SEAL_CREATED"]

    def freeze_predictions(self, predictions: np.ndarray) -> str:
        if self._opened or self._prediction_hash is not None:
            raise RuntimeError("FAIL_TEST_LABEL_BOUNDARY")
        values = np.ascontiguousarray(predictions, dtype=np.int64)
        if values.shape != self.__labels.shape:
            raise ValueError("Prediction vector is not aligned with sealed labels")
        self._prediction_hash = array_sha256(values)
        self._events.append("02_PREDICTIONS_FROZEN_AND_HASHED")
        return self._prediction_hash

    def open_labels(self) -> np.ndarray:
        if self._prediction_hash is None or self._opened:
            raise RuntimeError("FAIL_TEST_LABEL_BOUNDARY")
        self._opened = True
        self._events.append("03_TEST_LABELS_OPENED_ONCE")
        return self.__labels.copy()

    @property
    def access_log(self) -> tuple[str, ...]:
        return tuple(self._events)

    @property
    def access_log_sha256(self) -> str:
        return canonical_json_sha256(self._events)


def deterministic_majority_vote(predictions: Sequence[int], *, class_count: int = CLASS_COUNT) -> int:
    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values >= class_count)):
        raise ValueError("Majority vote requires valid class predictions")
    counts = np.bincount(values, minlength=class_count)
    return int(np.flatnonzero(counts == counts.max())[0])


def block_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
    block_tokens: Sequence[str],
    *,
    expected_repeats: int = 50,
) -> list[dict[str, Any]]:
    actual = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.shape != predicted.shape or len(block_tokens) != len(actual):
        raise ValueError("Block prediction inputs are not aligned")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, token in enumerate(block_tokens):
        groups.setdefault(str(token), []).append(index)
    rows: list[dict[str, Any]] = []
    for token in sorted(groups):
        indices = groups[token]
        labels = np.unique(actual[indices])
        if len(indices) != expected_repeats or len(labels) != 1:
            raise RuntimeError("Condition block is not an indivisible single-label 50-row group")
        vote = deterministic_majority_vote(predicted[indices])
        counts = np.bincount(predicted[indices], minlength=CLASS_COUNT)
        rows.append(
            {
                "block_token": token,
                "truth": int(labels[0]),
                "prediction": vote,
                "agreement": float(counts.max() / len(indices)),
                "repeat_count": len(indices),
            }
        )
    return rows


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return classification_metrics(
        np.asarray(truth, dtype=np.int64),
        np.asarray(prediction, dtype=np.int64),
        class_count=CLASS_COUNT,
    )


def fit_fixed_softmax_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    probe_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    standardizer = FeatureStandardizer.fit(train_x, partition="train")
    x_train = torch.from_numpy(standardizer.transform(train_x).astype(np.float32, copy=False))
    x_validation = torch.from_numpy(standardizer.transform(validation_x).astype(np.float32, copy=False))
    x_test = torch.from_numpy(standardizer.transform(test_x).astype(np.float32, copy=False))
    y_train = torch.from_numpy(np.ascontiguousarray(train_y, dtype=np.int64))
    torch.manual_seed(int(seed))
    layer = torch.nn.Linear(x_train.shape[1], CLASS_COUNT)
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_metric = -float("inf")
    best_epoch = 0
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    maximum_epochs = int(probe_config["maximum_epochs"])
    patience = int(probe_config["early_stopping_patience"])
    minimum_delta = float(probe_config["early_stopping_min_delta"])
    for epoch in range(1, maximum_epochs + 1):
        layer.train()
        optimizer.zero_grad(set_to_none=True)
        logits = layer(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        layer.eval()
        with torch.inference_mode():
            validation_prediction = torch.softmax(layer(x_validation), dim=1).argmax(dim=1).cpu().numpy()
        validation_metric = metrics(validation_y, validation_prediction)["macro_f1"]
        if validation_metric > best_metric + minimum_delta:
            best_metric = float(validation_metric)
            best_epoch = epoch
            best_loss = float(loss.detach().cpu())
            best_state = {name: value.detach().cpu().clone() for name, value in layer.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Probe checkpoint selection failed")
    layer.load_state_dict(best_state, strict=True)
    layer.eval()
    with torch.inference_mode():
        train_prediction = torch.softmax(layer(x_train), dim=1).argmax(dim=1).cpu().numpy()
        validation_prediction = torch.softmax(layer(x_validation), dim=1).argmax(dim=1).cpu().numpy()
        test_prediction = torch.softmax(layer(x_test), dim=1).argmax(dim=1).cpu().numpy()
    return {
        "test_prediction": np.ascontiguousarray(test_prediction, dtype=np.int64),
        "train_metrics": metrics(np.asarray(train_y, dtype=np.int64), train_prediction),
        "validation_metrics": metrics(validation_y, validation_prediction),
        "selected_epoch": best_epoch,
        "epochs_executed": epoch,
        "selected_training_loss": best_loss,
        "scaler_state_sha256": standardizer.state_sha256,
        "scaler_fit_row_count": standardizer.fit_row_count,
        "scaler_fit_partition": standardizer.fit_partition,
    }


def fit_learning_validity_control(
    features: np.ndarray,
    labels: np.ndarray,
    subset_indices: np.ndarray,
    *,
    seed: int,
    probe_config: Mapping[str, Any],
    validity_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    selected_x = np.asarray(features[subset_indices], dtype=np.float64)
    selected_y = np.asarray(labels[subset_indices], dtype=np.int64)
    if sorted(selected_y.tolist()) != list(range(CLASS_COUNT)):
        raise RuntimeError("Learning-validity subset must contain one row per class")
    standardizer = FeatureStandardizer.fit(selected_x, partition="training_only_positive_control")
    inputs = torch.from_numpy(standardizer.transform(selected_x).astype(np.float32, copy=False))
    targets = torch.from_numpy(np.ascontiguousarray(selected_y))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    layer = torch.nn.Linear(inputs.shape[1], CLASS_COUNT)
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=float(probe_config["learning_rate"]),
        weight_decay=float(probe_config["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    maximum_steps = int(validity_config["maximum_steps"])
    fit_metrics: dict[str, Any] = {}
    final_loss = float("inf")
    for step in range(1, maximum_steps + 1):
        layer.train()
        optimizer.zero_grad(set_to_none=True)
        logits = layer(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        layer.eval()
        with torch.inference_mode():
            prediction = torch.softmax(layer(inputs), dim=1).argmax(dim=1).cpu().numpy()
        fit_metrics = metrics(selected_y, prediction)
        final_loss = float(loss.detach().cpu())
        if (
            fit_metrics["accuracy"] >= float(validity_config["fit_accuracy_threshold"])
            and fit_metrics["macro_f1"] >= float(validity_config["fit_macro_f1_threshold"])
        ):
            break
    passed = bool(
        fit_metrics["accuracy"] >= float(validity_config["fit_accuracy_threshold"])
        and fit_metrics["macro_f1"] >= float(validity_config["fit_macro_f1_threshold"])
    )
    return {
        "train_accuracy": fit_metrics["accuracy"],
        "train_macro_f1": fit_metrics["macro_f1"],
        "final_loss": final_loss,
        "optimization_steps": step,
        "pass": passed,
        "status": "PASS" if passed else "REPRESENTATION_PROBE_VALIDITY_FAILURE",
        "subset_size": len(subset_indices),
        "subset_hash": array_sha256(np.ascontiguousarray(subset_indices, dtype=np.int64)),
        "scaler_state_sha256": standardizer.state_sha256,
    }


def learning_validity_subset(
    data: PositionData, config: Mapping[str, Any]
) -> np.ndarray:
    fold = int(config["learning_validity"]["fold"])
    train = split_indices(data, fold, config)["train"]
    chosen: list[int] = []
    for class_index in config["class_order"]:
        candidates = train[data.labels[train] == int(class_index)]
        ordering = sorted(
            candidates.tolist(),
            key=lambda index: (
                str(data.block_tokens[index]),
                int(data.repeat_index[index]),
                index,
            ),
        )
        chosen.append(ordering[0])
    return np.ascontiguousarray(chosen, dtype=np.int64)


def cosine_ncm_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
) -> tuple[np.ndarray, str]:
    standardizer = FeatureStandardizer.fit(train_x, partition="train")
    train = standardizer.transform(train_x)
    evaluation = standardizer.transform(evaluation_x)

    def l2(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return values / norms

    normalized_train = l2(train)
    prototypes = np.stack([normalized_train[np.asarray(train_y) == class_index].mean(axis=0) for class_index in range(CLASS_COUNT)])
    prototypes = l2(prototypes)
    similarities = l2(evaluation) @ prototypes.T
    prediction = np.argmax(similarities, axis=1).astype(np.int64)
    return prediction, standardizer.state_sha256


def percentile_interval(values: Sequence[float], confidence_level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    return tuple(float(item) for item in np.quantile(np.asarray(values, dtype=np.float64), [alpha, 1.0 - alpha]))


def stratified_block_resample_indices(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(truth, dtype=np.int64)
    pieces = []
    for class_index in range(CLASS_COUNT):
        indices = np.flatnonzero(labels == class_index)
        if len(indices) == 0:
            raise ValueError("Every TagID must be present in a stratified bootstrap")
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(pieces)


def paired_block_bootstrap(
    truth: np.ndarray,
    predictions: Mapping[str, Mapping[int, np.ndarray]],
    contrasts: Sequence[tuple[str, str, str]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """Primary paired TagID-stratified bootstrap, averaging matched seeds."""

    rng = np.random.default_rng(seed)
    names = list(predictions)
    seed_order = sorted(next(iter(predictions.values())))
    distributions = {
        name: {
            "macro_f1": np.empty(replicates, dtype=np.float64),
            "accuracy": np.empty(replicates, dtype=np.float64),
        }
        for name in names
    }
    contrast_distributions = {name: np.empty(replicates, dtype=np.float64) for name, _, _ in contrasts}
    for replicate in range(replicates):
        sampled = stratified_block_resample_indices(truth, rng)
        replicate_f1: dict[str, float] = {}
        for name in names:
            seed_metrics = [metrics(truth[sampled], predictions[name][probe_seed][sampled]) for probe_seed in seed_order]
            distributions[name]["macro_f1"][replicate] = np.mean([item["macro_f1"] for item in seed_metrics])
            distributions[name]["accuracy"][replicate] = np.mean([item["accuracy"] for item in seed_metrics])
            replicate_f1[name] = distributions[name]["macro_f1"][replicate]
        for contrast_name, left, right in contrasts:
            contrast_distributions[contrast_name][replicate] = replicate_f1[left] - replicate_f1[right]
    rows: list[dict[str, Any]] = []
    for name in names:
        for metric_name in ("macro_f1", "accuracy"):
            low, high = percentile_interval(distributions[name][metric_name], confidence_level)
            rows.append(
                {
                    "scheme": "tagid_stratified_block_only",
                    "target_type": "representation",
                    "target": name,
                    "metric": f"block_{metric_name}",
                    "replicates": replicates,
                    "estimate": float(np.mean(distributions[name][metric_name])),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    for contrast_name, values in contrast_distributions.items():
        low, high = percentile_interval(values, confidence_level)
        rows.append(
            {
                "scheme": "tagid_stratified_block_only",
                "target_type": "representation_contrast",
                "target": contrast_name,
                "metric": "block_macro_f1_difference",
                "replicates": replicates,
                "estimate": float(np.mean(values)),
                "ci_low": low,
                "ci_high": high,
            }
        )
    distributions["__contrasts__"] = {name: values for name, values in contrast_distributions.items()}
    return rows, distributions


def seed_resampling_sensitivity(
    truth: np.ndarray,
    predictions: Mapping[str, Mapping[int, np.ndarray]],
    contrasts: Sequence[tuple[str, str, str]],
    fold_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    seed_order = np.asarray(sorted(next(iter(predictions.values()))), dtype=np.int64)
    simpler = [name for name in predictions if name != "C1_ENCODER_EMBEDDING"]
    rows: list[dict[str, Any]] = []
    schemes = ("block_plus_probe_seed", "block_plus_checkpoint_seed", "both_seed_dimensions")
    for scheme_index, scheme in enumerate(schemes):
        rng = np.random.default_rng(seed + 1000 * (scheme_index + 1))
        values = {name: np.empty(replicates, dtype=np.float64) for name, _, _ in contrasts}
        for replicate in range(replicates):
            sampled = stratified_block_resample_indices(truth, rng)
            probe_draw = rng.choice(seed_order, size=len(seed_order), replace=True)
            checkpoint_draw = rng.choice(seed_order, size=len(seed_order), replace=True)
            rep_value: dict[str, float] = {}
            for name in predictions:
                if scheme == "block_plus_probe_seed":
                    chosen = probe_draw
                elif scheme == "block_plus_checkpoint_seed":
                    chosen = checkpoint_draw if name == "C1_ENCODER_EMBEDDING" else seed_order
                else:
                    chosen = checkpoint_draw if name == "C1_ENCODER_EMBEDDING" else probe_draw
                rep_value[name] = float(
                    np.mean([metrics(truth[sampled], predictions[name][int(item)][sampled])["macro_f1"] for item in chosen])
                )
            for contrast_name, left, right in contrasts:
                values[contrast_name][replicate] = rep_value[left] - rep_value[right]
        for contrast_name, distribution in values.items():
            low, high = percentile_interval(distribution, confidence_level)
            rows.append(
                {
                    "scheme": scheme,
                    "target_type": "representation_contrast",
                    "target": contrast_name,
                    "metric": "block_macro_f1_difference",
                    "replicates": replicates,
                    "estimate": float(np.mean(distribution)),
                    "ci_low": low,
                    "ci_high": high,
                    "note": "C1 probe and checkpoint seeds are preregistered as aligned and cannot be independently identified",
                }
            )

    rng = np.random.default_rng(seed + 4000)
    fold_values = {name: np.empty(replicates, dtype=np.float64) for name, _, _ in contrasts}
    unique_folds = np.asarray(sorted(set(np.asarray(fold_ids, dtype=np.int64).tolist())), dtype=np.int64)
    for replicate in range(replicates):
        sampled_folds = rng.choice(unique_folds, size=len(unique_folds), replace=True)
        rep_value: dict[str, float] = {}
        for name in predictions:
            per_seed = []
            for probe_seed in seed_order:
                fold_metrics = []
                for fold in sampled_folds:
                    mask = fold_ids == fold
                    fold_metrics.append(metrics(truth[mask], predictions[name][int(probe_seed)][mask])["macro_f1"])
                per_seed.append(float(np.mean(fold_metrics)))
            rep_value[name] = float(np.mean(per_seed))
        for contrast_name, left, right in contrasts:
            fold_values[contrast_name][replicate] = rep_value[left] - rep_value[right]
    for contrast_name, distribution in fold_values.items():
        low, high = percentile_interval(distribution, confidence_level)
        rows.append(
            {
                "scheme": "fold_level_aggregate",
                "target_type": "representation_contrast",
                "target": contrast_name,
                "metric": "mean_fold_block_macro_f1_difference",
                "replicates": replicates,
                "estimate": float(np.mean(distribution)),
                "ci_low": low,
                "ci_high": high,
                "note": "three synchronized folds resampled as aggregate units",
            }
        )
    return rows


def angle_associated_contrast(position_accuracy: Mapping[str, float]) -> float:
    required = {"P1", "P2", "P3", "P4"}
    if set(position_accuracy) != required:
        raise ValueError("Angle contrast requires P1-P4")
    return 0.5 * (
        (float(position_accuracy["P2"]) - float(position_accuracy["P1"]))
        + (float(position_accuracy["P4"]) - float(position_accuracy["P3"]))
    )


def paired_angle_bootstrap(
    truths: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    positions = ("P1", "P2", "P3", "P4")
    reference_truth = truths["P1"]
    if any(not np.array_equal(truths[position], reference_truth) for position in positions[1:]):
        raise RuntimeError("Position pairing changed before angle bootstrap")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for rep_index, (representation, by_position) in enumerate(predictions.items()):
        seed_order = sorted(by_position["P1"])
        distribution = np.empty(replicates, dtype=np.float64)
        local_rng = np.random.default_rng(seed + rep_index * 10000)
        for replicate in range(replicates):
            sampled = stratified_block_resample_indices(reference_truth, local_rng)
            accuracies = {
                position: float(
                    np.mean(
                        [
                            metrics(truths[position][sampled], by_position[position][probe_seed][sampled])["accuracy"]
                            for probe_seed in seed_order
                        ]
                    )
                )
                for position in positions
            }
            distribution[replicate] = angle_associated_contrast(accuracies)
        low, high = percentile_interval(distribution, confidence_level)
        rows.append(
            {
                "representation": representation,
                "estimate": float(np.mean(distribution)),
                "ci_low": low,
                "ci_high": high,
                "replicates": replicates,
                "sampling": "TagID-stratified paired condition blocks with position pairing retained",
            }
        )
    return rows


def confusion_hash(truth: np.ndarray, prediction: np.ndarray) -> str:
    return canonical_json_sha256(metrics(truth, prediction)["confusion_matrix"])


def finite_or_blank(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def sanitize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{str(key): finite_or_blank(value) for key, value in row.items()} for row in rows]
