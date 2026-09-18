"""Governed source loading, canonical preprocessing, and matched batching."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from protocol import FOLDS, HELD_POSITION, POSITIONS


EXPECTED_INPUT_HASHES = {
    "CANONICAL_SOURCE_REGISTRY.csv": "a1cbb062b4e2a57d62a6662acbe27cb151703ee07fa0ea3ef447bca4f96ad7e7",
    "source_labels_int64.npy": "a0f74a067fd56c81df7bea54d023e634cf91252cb9b2f75ab18b4ecf7b8525a3",
    "SOURCE_ONLY_LOPO_SPLITS.csv": "41a275a6f499ee3d3105e09a420ceb92bb8272d879a89a6bb6623ae6f9bc82ca",
    "source_signals_float64.npy": "b8fa8d1699d3b5d0da9d7c45a3e709aff2802b6653afb10771d57b401f740aa8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceData:
    signals: np.ndarray
    labels: np.ndarray
    rows: tuple[dict, ...]
    splits: dict[str, dict[str, np.ndarray]]
    input_hashes: dict[str, str]

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([row["position"] for row in self.rows], dtype=str)

    @property
    def condition_ids(self) -> np.ndarray:
        return np.asarray([row["raw_condition_id"] for row in self.rows], dtype=str)


@dataclass(frozen=True)
class PreprocessingState:
    mean: np.ndarray
    scale: np.ndarray
    fit_rows: np.ndarray

    @property
    def semantic_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (self.mean, self.scale, self.fit_rows.astype("<i8", copy=False)):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def load_source_data(root: Path, *, enforce_hashes: bool = True) -> SourceData:
    root = Path(root).resolve()
    paths = {name: root / name for name in EXPECTED_INPUT_HASHES}
    if not all(path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"BLOCKED_GOVERNED_INPUTS: missing {missing}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if enforce_hashes and hashes != EXPECTED_INPUT_HASHES:
        raise RuntimeError("BLOCKED_DANN_LINEAGE_OR_DATA_MISMATCH: governed hashes differ")
    signals = np.load(paths["source_signals_float64.npy"], allow_pickle=False)
    labels = np.load(paths["source_labels_int64.npy"], allow_pickle=False)
    with paths["CANONICAL_SOURCE_REGISTRY.csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = tuple(csv.DictReader(handle))
    if signals.shape != (9450, 281) or signals.dtype.str != "<f8":
        raise RuntimeError("BLOCKED_DANN_LINEAGE_OR_DATA_MISMATCH: source signal custody")
    if labels.shape != (9450,) or labels.dtype.str != "<i8":
        raise RuntimeError("BLOCKED_DANN_LINEAGE_OR_DATA_MISMATCH: source label custody")
    if len(rows) != 9450:
        raise RuntimeError("BLOCKED_DANN_LINEAGE_OR_DATA_MISMATCH: registry length")
    converted: list[dict] = []
    for expected, row in enumerate(rows):
        if int(row["registry_row"]) != expected:
            raise RuntimeError("Registry order changed")
        converted.append(
            {
                "registry_row": expected,
                "sample_id": row["sample_id"],
                "raw_condition_id": row["raw_condition_id"],
                "tag_id": int(row["tag_id"]),
                "label_index": int(row["label_index"]),
                "er": int(row["er"]),
                "surface": row["surface"],
                "position": row["position"],
            }
        )
    rows = tuple(converted)
    if not np.array_equal(labels, np.asarray([row["label_index"] for row in rows])):
        raise RuntimeError("Registry and array labels differ")
    _validate_structure(rows)
    assignments = {
        fold: {name: [] for name in ("inner_train", "inner_validation", "outer_held")}
        for fold in FOLDS
    }
    with paths["SOURCE_ONLY_LOPO_SPLITS.csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            fold = row["fold_id"]
            part = row["partition"]
            index = int(row["registry_row"])
            if fold not in assignments or part not in assignments[fold]:
                raise RuntimeError("Unexpected LOPO split key")
            if row["sample_id"] != rows[index]["sample_id"]:
                raise RuntimeError("LOPO sample identity mismatch")
            assignments[fold][part].append(index)
    splits = {
        fold: {
            part: np.asarray(sorted(values), dtype=np.int64)
            for part, values in parts.items()
        }
        for fold, parts in assignments.items()
    }
    _validate_splits(rows, splits)
    return SourceData(signals, labels, rows, splits, hashes)


def _validate_structure(rows: tuple[dict, ...]) -> None:
    if set(row["tag_id"] for row in rows) != set(range(1, 8)):
        raise RuntimeError("Unexpected TagID mapping")
    if set(row["label_index"] for row in rows) != set(range(7)):
        raise RuntimeError("Unexpected label-index mapping")
    if set(row["er"] for row in rows) != {0, 1, 2}:
        raise RuntimeError("Unexpected ER levels")
    if set(row["surface"] for row in rows) != {"A1", "A2", "A3"}:
        raise RuntimeError("Unexpected surface levels")
    if set(row["position"] for row in rows) != set(POSITIONS):
        raise RuntimeError("Unexpected source-position mapping")
    by_position = Counter(row["position"] for row in rows)
    by_block = Counter(row["raw_condition_id"] for row in rows)
    if set(by_position.values()) != {3150} or len(by_block) != 189 or set(by_block.values()) != {50}:
        raise RuntimeError("BLOCKED_DANN_LINEAGE_OR_DATA_MISMATCH: block structure")


def _validate_splits(rows: tuple[dict, ...], splits: dict[str, dict[str, np.ndarray]]) -> None:
    expected_counts = {"inner_train": 4200, "inner_validation": 2100, "outer_held": 3150}
    all_rows = set(range(9450))
    for fold, parts in splits.items():
        concatenated = np.concatenate(tuple(parts.values()))
        if len(concatenated) != 9450 or set(concatenated.tolist()) != all_rows:
            raise RuntimeError(f"Fold {fold} is not a complete disjoint partition")
        for part, expected in expected_counts.items():
            if len(parts[part]) != expected:
                raise RuntimeError(f"Unexpected {fold}/{part} count")
        assert_held_position_isolated(rows, parts, HELD_POSITION[fold])
        assert_no_block_leakage(rows, parts)


def assert_no_block_leakage(rows: tuple[dict, ...], parts: dict[str, np.ndarray]) -> None:
    block_parts: dict[str, set[str]] = defaultdict(set)
    for part, indices in parts.items():
        for index in indices:
            block_parts[rows[int(index)]["raw_condition_id"]].add(part)
    if any(len(values) != 1 for values in block_parts.values()):
        raise RuntimeError("Condition block crosses a train/validation boundary")


def assert_held_position_isolated(
    rows: tuple[dict, ...], parts: dict[str, np.ndarray], held_position: str
) -> None:
    held_positions = {rows[int(index)]["position"] for index in parts["outer_held"]}
    development_positions = {
        rows[int(index)]["position"]
        for index in np.concatenate((parts["inner_train"], parts["inner_validation"]))
    }
    if held_positions != {held_position} or held_position in development_positions:
        raise RuntimeError("Held source position leaked into training/validation")


def fit_first_difference(signals: np.ndarray, fit_indices: np.ndarray) -> PreprocessingState:
    differenced = np.ascontiguousarray(np.diff(signals, axis=1), dtype=np.float64)
    if differenced.shape[1] != 280:
        raise RuntimeError("Canonical first-difference dimensionality changed")
    fit = differenced[np.asarray(fit_indices, dtype=np.int64)]
    mean = np.ascontiguousarray(fit.mean(axis=0, dtype=np.float64))
    scale = np.ascontiguousarray(fit.std(axis=0, ddof=0, dtype=np.float64))
    scale = np.maximum(scale, 1e-12)
    return PreprocessingState(mean, scale, np.asarray(fit_indices, dtype=np.int64).copy())


def transform(signals: np.ndarray, indices: np.ndarray, state: PreprocessingState) -> np.ndarray:
    values = np.diff(signals[np.asarray(indices, dtype=np.int64)], axis=1)
    return np.ascontiguousarray(((values - state.mean) / state.scale).astype(np.float32))


def domain_labels(positions: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    order = tuple(sorted(np.unique(np.asarray(positions, dtype=str)).tolist()))
    if len(order) not in {2, 3} or not set(order).issubset(POSITIONS):
        raise RuntimeError(f"Invalid source-domain set: {order}")
    mapping = {position: index for index, position in enumerate(order)}
    return np.asarray([mapping[str(value)] for value in positions], dtype=np.int64), order


def balanced_epoch_indices(
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Interleave shuffled position/TagID cells; every row appears exactly once."""

    positions = np.asarray(positions, dtype=str)
    labels = np.asarray(labels, dtype=np.int64)
    if positions.shape != labels.shape:
        raise ValueError("Position and TagID arrays are not aligned")
    cells = sorted({(str(position), int(label)) for position, label in zip(positions, labels)})
    expected_cells = len(set(positions)) * len(set(labels))
    if len(cells) != expected_cells:
        raise RuntimeError("Balanced sampler requires every position/TagID cell")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_000 + int(epoch))
    queues: dict[tuple[str, int], list[int]] = {}
    for cell in cells:
        selected = np.flatnonzero((positions == cell[0]) & (labels == cell[1]))
        permutation = torch.randperm(len(selected), generator=generator).numpy()
        queues[cell] = selected[permutation].tolist()
    output: list[int] = []
    offset = (int(epoch) - 1) % len(cells)
    ordered = cells[offset:] + cells[:offset]
    while any(queues[cell] for cell in cells):
        for cell in ordered:
            if queues[cell]:
                output.append(queues[cell].pop())
        ordered = ordered[1:] + ordered[:1]
    result = np.asarray(output, dtype=np.int64)
    if len(result) != len(labels) or len(np.unique(result)) != len(labels):
        raise RuntimeError("Balanced sampler lost or duplicated rows")
    return result


def batch_composition(
    order: np.ndarray, positions: np.ndarray, labels: np.ndarray, batch_size: int = 256
) -> list[dict]:
    records = []
    for batch_index, start in enumerate(range(0, len(order), int(batch_size))):
        selected = order[start : start + int(batch_size)]
        cells = Counter(
            f"{positions[int(index)]}:tag{int(labels[int(index)])}"
            for index in selected
        )
        records.append(
            {
                "batch_index": batch_index,
                "sample_count": len(selected),
                "position_counts": dict(Counter(positions[selected].tolist())),
                "position_tag_counts": dict(sorted(cells.items())),
                "max_minus_min_cell_count": max(cells.values()) - min(cells.values()),
            }
        )
    return records


def final_probe_masks(data: SourceData) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 6/3-per-TagID signature split, synchronized across P1-P3."""

    all_signatures = sorted(
        {
            (row["tag_id"], row["er"], row["surface"])
            for row in data.rows
        }
    )
    train_signatures: set[tuple[int, int, str]] = set()
    validation_signatures: set[tuple[int, int, str]] = set()
    for tag_id in range(1, 8):
        local = [signature for signature in all_signatures if signature[0] == tag_id]
        ranked = sorted(
            local,
            key=lambda signature: hashlib.sha256(
                f"dann-final-probe-v1|{signature[0]}|{signature[1]}|{signature[2]}".encode("ascii")
            ).hexdigest(),
        )
        train_signatures.update(ranked[:6])
        validation_signatures.update(ranked[6:])
    if (
        len(train_signatures) != 42
        or len(validation_signatures) != 21
        or train_signatures & validation_signatures
        or train_signatures | validation_signatures != set(all_signatures)
    ):
        raise RuntimeError("Frozen synchronized final-probe signature split changed")
    train = []
    validation = []
    for index, row in enumerate(data.rows):
        signature = (row["tag_id"], row["er"], row["surface"])
        (train if signature in train_signatures else validation).append(index)
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)
