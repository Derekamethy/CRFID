"""Governed P4 parsing with support/query views and sealed query labels."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    ALL_CELLS,
    CLASS_ORDER,
    EXPECTED_BLOCKS,
    EXPECTED_ROWS,
    P4_FILE_HASHES,
    QUERY_FOLDS,
    RAW_SIGNAL_LENGTH,
    ROWS_PER_BLOCK,
    cell_id,
)
from .protocol import ProtocolViolation, QueryLabelSeal, validate_outer_folds
from .selection import deterministic_row_offset


STRUCTURE_MISMATCH = "BLOCKED_P4_STRUCTURE_MISMATCH"
METADATA_COLUMNS = ("A3", "A2", "A1", "P4", "P3", "P2", "P1", "ER", "TagID")
SIGNAL_COLUMNS = tuple(str(index) for index in range(RAW_SIGNAL_LENGTH))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_block_id(class_index: int, er: int, surface: int) -> str:
    digest = hashlib.sha256(
        f"P4_FACTOR_AWARE_BLOCK_V1|{class_index}|{er}|{surface}".encode("ascii")
    ).hexdigest()
    return f"block_{digest[:20]}"


def _integral(value: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(STRUCTURE_MISMATCH)
    return int(numeric)


@dataclass(frozen=True)
class QueryView:
    indices: np.ndarray
    signals: np.ndarray
    block_ids: tuple[str, ...]
    factor_cells: tuple[tuple[int, int], ...]
    label_seal: QueryLabelSeal


@dataclass(frozen=True)
class SupportSet:
    indices: np.ndarray
    signals: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    selected_cells: tuple[tuple[int, int], ...]
    manifest_rows: tuple[dict[str, object], ...]


class GovernedP4:
    """In-memory P4 data whose query labels are available only through a seal."""

    def __init__(
        self,
        *,
        signals: np.ndarray,
        labels: np.ndarray,
        er: np.ndarray,
        surface: np.ndarray,
        source_row: np.ndarray,
        source_hashes: dict[str, str],
    ) -> None:
        self.signals = np.ascontiguousarray(signals, dtype=np.float64)
        self._labels = np.ascontiguousarray(labels, dtype=np.int64)
        self.er = np.ascontiguousarray(er, dtype=np.int64)
        self.surface = np.ascontiguousarray(surface, dtype=np.int64)
        self.source_row = np.ascontiguousarray(source_row, dtype=np.int64)
        self.source_hashes = dict(source_hashes)
        self._groups: dict[tuple[int, int, int], np.ndarray] = {}
        self._block_ids: list[str] = []
        grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, (label, er_value, surface_value) in enumerate(
            zip(self._labels, self.er, self.surface, strict=True)
        ):
            key = (int(label), int(er_value), int(surface_value))
            grouped[key].append(index)
            self._block_ids.append(_opaque_block_id(*key))
        self._groups = {
            key: np.asarray(sorted(indices), dtype=np.int64)
            for key, indices in grouped.items()
        }
        self._validate_structure()

    def _validate_structure(self) -> None:
        if self.signals.shape != (EXPECTED_ROWS, RAW_SIGNAL_LENGTH):
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        if any(array.shape != (EXPECTED_ROWS,) for array in (self._labels, self.er, self.surface)):
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        expected = {
            (class_index, er, surface)
            for class_index in CLASS_ORDER
            for er, surface in ALL_CELLS
        }
        if set(self._groups) != expected or len(self._groups) != EXPECTED_BLOCKS:
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        if {len(indices) for indices in self._groups.values()} != {ROWS_PER_BLOCK}:
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        if not np.isfinite(self.signals).all():
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        validate_outer_folds()

    def structure_rows(self) -> list[dict[str, object]]:
        rows = []
        for (class_index, er, surface), indices in sorted(self._groups.items()):
            rows.append(
                {
                    "class_index": class_index,
                    "factor_cell": cell_id((er, surface)),
                    "condition_block_id": _opaque_block_id(class_index, er, surface),
                    "row_count": len(indices),
                    "raw_signal_values": RAW_SIGNAL_LENGTH,
                    "first_difference_values": RAW_SIGNAL_LENGTH - 1,
                    "position": "P4",
                }
            )
        return rows

    def query_view(self, fold: int) -> QueryView:
        if fold not in QUERY_FOLDS:
            raise ValueError("unknown outer fold")
        query_cells = set(QUERY_FOLDS[fold])
        mask = np.asarray(
            [
                (int(er), int(surface)) in query_cells
                for er, surface in zip(self.er, self.surface, strict=True)
            ],
            dtype=bool,
        )
        indices = np.flatnonzero(mask).astype(np.int64)
        if len(indices) != 21 * ROWS_PER_BLOCK:
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        block_ids = tuple(self._block_ids[index] for index in indices)
        block_counts = Counter(block_ids)
        if len(block_counts) != 21 or set(block_counts.values()) != {ROWS_PER_BLOCK}:
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        return QueryView(
            indices=indices,
            signals=self.signals[indices],
            block_ids=block_ids,
            factor_cells=tuple(
                (int(self.er[index]), int(self.surface[index])) for index in indices
            ),
            label_seal=QueryLabelSeal(self._labels[indices]),
        )

    def support_set(
        self,
        *,
        fold: int,
        selected_cells: tuple[tuple[int, int], ...],
        support_seed: int,
    ) -> SupportSet:
        if set(selected_cells) & set(QUERY_FOLDS[fold]):
            raise ProtocolViolation("support/query condition-block overlap")
        chosen: list[int] = []
        sample_ids: list[str] = []
        manifest_rows: list[dict[str, object]] = []
        for class_index in CLASS_ORDER:
            for cell in selected_cells:
                group = self._groups.get((class_index, cell[0], cell[1]))
                if group is None or len(group) != ROWS_PER_BLOCK:
                    raise ProtocolViolation(STRUCTURE_MISMATCH)
                offset = deterministic_row_offset(
                    support_seed=support_seed,
                    fold=fold,
                    class_index=class_index,
                    cell=cell,
                    row_count=len(group),
                )
                index = int(group[offset])
                chosen.append(index)
                sample_ids.append(
                    f"SUP_F{fold}_C{class_index}_{cell_id(cell)}_R{offset:02d}"
                )
                manifest_rows.append(
                    {
                        "fold": fold,
                        "support_seed": support_seed,
                        "class_index": class_index,
                        "factor_cell": cell_id(cell),
                        "support_row_ordinal": offset,
                        "labelled_rows_from_block": 1,
                    }
                )
        indices = np.asarray(chosen, dtype=np.int64)
        labels = self._labels[indices]
        expected_per_class = len(selected_cells)
        if len(indices) != 7 * expected_per_class or len(set(indices.tolist())) != len(indices):
            raise ProtocolViolation("invalid labelled support-row budget")
        if not np.all(np.bincount(labels, minlength=7) == expected_per_class):
            raise ProtocolViolation("support labels are not class-balanced")
        block_keys = {
            (int(self._labels[index]), int(self.er[index]), int(self.surface[index]))
            for index in indices
        }
        if len(block_keys) != len(indices):
            raise ProtocolViolation("more than one labelled row came from a support block")
        return SupportSet(
            indices=indices,
            signals=self.signals[indices],
            labels=labels.copy(),
            sample_ids=tuple(sample_ids),
            selected_cells=selected_cells,
            manifest_rows=tuple(manifest_rows),
        )


def load_governed_p4(data_directory: str | Path) -> GovernedP4:
    root = Path(data_directory)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    er_values: list[int] = []
    surfaces: list[int] = []
    source_rows: list[int] = []
    observed_hashes: dict[str, str] = {}
    expected_header = METADATA_COLUMNS + SIGNAL_COLUMNS

    for surface_index, filename in enumerate(("A1_P4.csv", "A2_P4.csv", "A3_P4.csv")):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError("BLOCKED_GOVERNED_P4_INPUTS")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != P4_FILE_HASHES[filename]:
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        observed_hashes[filename] = digest
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != expected_header:
            raise ProtocolViolation(STRUCTURE_MISMATCH)
        file_row_count = 0
        for source_row_index, row in enumerate(reader):
            active_surface = [name for name in ("A1", "A2", "A3") if float(row[name]) == 1.0]
            active_position = [name for name in ("P1", "P2", "P3", "P4") if float(row[name]) == 1.0]
            if active_surface != [f"A{surface_index + 1}"] or active_position != ["P4"]:
                raise ProtocolViolation(STRUCTURE_MISMATCH)
            tag_id = _integral(row["TagID"])
            er = _integral(row["ER"])
            if tag_id not in range(1, 8) or er not in range(3):
                raise ProtocolViolation(STRUCTURE_MISMATCH)
            signal = np.fromiter(
                (float(row[column]) for column in SIGNAL_COLUMNS),
                dtype=np.float64,
                count=RAW_SIGNAL_LENGTH,
            )
            if signal.shape != (RAW_SIGNAL_LENGTH,) or not np.isfinite(signal).all():
                raise ProtocolViolation(STRUCTURE_MISMATCH)
            signals.append(signal)
            labels.append(tag_id - 1)
            er_values.append(er)
            surfaces.append(surface_index)
            source_rows.append(source_row_index)
            file_row_count += 1
        if file_row_count != 1050:
            raise ProtocolViolation(STRUCTURE_MISMATCH)

    return GovernedP4(
        signals=np.stack(signals),
        labels=np.asarray(labels),
        er=np.asarray(er_values),
        surface=np.asarray(surfaces),
        source_row=np.asarray(source_rows),
        source_hashes=observed_hashes,
    )
