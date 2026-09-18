"""Shared helpers for the composition_warp sub-module.

Two responsibilities:

* :func:`load_source` returns the canonical Strict-DG P1-P3 arrays and registry
  read-only, together with the conditional 281-point frequency axis. Nothing
  here writes outside ``06_failure_mechanism/composition_warp``.
* :class:`Canvas` is a dependency-free PNG rasterizer. The locked CRFID
  environment does not provide matplotlib and installing packages is
  prohibited, so the two required figures are rendered with ``zlib`` and
  ``numpy`` only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[1]
SOURCE_INPUTS_ENV = "CRFID_COMPOSITION_SOURCE_INPUTS"
FROZEN_STRICT_DG_ENV = "CRFID_COMPOSITION_FROZEN_STRICT_DG"
FREQUENCY_AXIS = (
    PROJECT_ROOT
    / "results"
    / "canonical_metrics"
    / "reference_peak"
    / "conditional_frequency_axis_281_5_to_8.csv"
)

POSITIONS = ("P1", "P2", "P3")
SURFACES = ("A1", "A2", "A3")
ENCODING_STATES = (0, 1, 2)
TAG_IDS = (1, 2, 3, 4, 5, 6, 7)
POINT_COUNT = 281
REPEAT_COUNT = 50


def _required_external_dir(env_name: str, purpose: str) -> Path:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(
            f"{purpose} is external to the public repository; set {env_name} to its directory"
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{env_name} does not point to a directory: {path}")
    return path


def source_inputs_root() -> Path:
    return _required_external_dir(SOURCE_INPUTS_ENV, "Strict-DG source inputs")


def frozen_strict_dg_root() -> Path:
    return _required_external_dir(FROZEN_STRICT_DG_ENV, "Frozen Strict-DG release")


#: Verified resonator windows in the project's conditional 5-8 GHz mapping.
#: The public summary is retained under
#: ``results/canonical_metrics/reference_peak/FREQUENCY_VALUE_MAP_REPORT.md``.
#: The task specification labels the two outer
#: windows the other way round (its "L1 = 5.84" / "L3 = 7.17"); the repository's
#: own naming is authoritative here and the discrepancy is recorded in the
#: sub-experiment report.
REFERENCE_WINDOWS = {
    "Lc": (5.100, 5.745),
    "L3": (5.800, 6.350),
    "L2": (6.450, 6.950),
    "L1": (7.050, 7.700),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class SourceData:
    """Read-only canonical P1-P3 custody."""

    signals: np.ndarray
    labels: np.ndarray
    rows: list[dict]
    tag_id: np.ndarray
    er: np.ndarray
    surface: np.ndarray
    position: np.ndarray
    repeat_index: np.ndarray
    condition_id: np.ndarray
    signals_sha256: str
    registry_sha256: str


def load_source() -> SourceData:
    """Load the canonical Strict-DG source arrays and registry, read-only."""

    source_inputs = source_inputs_root()
    signal_path = source_inputs / "source_signals_float64.npy"
    label_path = source_inputs / "source_labels_int64.npy"
    registry_path = source_inputs / "CANONICAL_SOURCE_REGISTRY.csv"

    signals = np.load(signal_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    if signals.shape != (9450, POINT_COUNT) or signals.dtype.str != "<f8":
        raise RuntimeError(f"Canonical source signal custody changed: {signals.shape}")
    if labels.shape != (9450,) or labels.dtype.str != "<i8":
        raise RuntimeError("Canonical source label custody changed")

    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(signals):
        raise RuntimeError("Registry and signal counts differ")
    for expected, row in enumerate(rows):
        if int(row["registry_row"]) != expected:
            raise RuntimeError("Registry row order changed")
        if int(row["label_index"]) != int(labels[expected]):
            raise RuntimeError("Registry label disagrees with the canonical label array")

    data = SourceData(
        signals=signals,
        labels=labels,
        rows=rows,
        tag_id=np.asarray([int(row["tag_id"]) for row in rows], dtype=np.int64),
        er=np.asarray([int(row["er"]) for row in rows], dtype=np.int64),
        surface=np.asarray([row["surface"] for row in rows], dtype="<U2"),
        position=np.asarray([row["position"] for row in rows], dtype="<U2"),
        repeat_index=np.asarray([int(row["repeat_index"]) for row in rows], dtype=np.int64),
        condition_id=np.asarray([row["raw_condition_id"] for row in rows], dtype=str),
        signals_sha256=sha256_file(signal_path),
        registry_sha256=sha256_file(registry_path),
    )
    if sorted(set(data.position.tolist())) != list(POSITIONS):
        raise RuntimeError("Source positions are not exactly P1/P2/P3")
    if "P4" in set(data.position.tolist()):
        raise RuntimeError("Target domain present in the source custody")
    return data


def load_frequency_axis() -> np.ndarray:
    """Conditional 281-point 5-8 GHz axis. Physical provenance is unresolved."""

    with FREQUENCY_AXIS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    axis = np.asarray([float(row["frequency_ghz"]) for row in rows], dtype=np.float64)
    if axis.shape != (POINT_COUNT,) or not np.all(np.diff(axis) > 0):
        raise RuntimeError("Conditional frequency axis is not a strictly increasing 281-vector")
    return axis


def condition_means(data: SourceData) -> tuple[dict[tuple[int, int, str, str], np.ndarray], list]:
    """Mean spectrum of every (tag, er, surface, position) block of 50 repeats."""

    means: dict[tuple[int, int, str, str], np.ndarray] = {}
    keys = []
    for tag in TAG_IDS:
        for er in ENCODING_STATES:
            for surface in SURFACES:
                for position in POSITIONS:
                    mask = (
                        (data.tag_id == tag)
                        & (data.er == er)
                        & (data.surface == surface)
                        & (data.position == position)
                    )
                    selected = np.flatnonzero(mask)
                    if selected.size != REPEAT_COUNT:
                        raise RuntimeError(
                            f"Condition block is not {REPEAT_COUNT} repeats: "
                            f"{(tag, er, surface, position)} -> {selected.size}"
                        )
                    key = (tag, er, surface, position)
                    means[key] = data.signals[selected].mean(axis=0)
                    keys.append(key)
    return means, keys


# --------------------------------------------------------------------------
# Dependency-free PNG rendering
# --------------------------------------------------------------------------

# 5x7 bitmap font. Each glyph is seven rows; the low five bits of each row are
# the pixels, most significant of the five on the left.
_FONT: dict[str, tuple[int, ...]] = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    "!": (0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x04),
    "%": (0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "*": (0x00, 0x0A, 0x04, 0x1F, 0x04, 0x0A, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    ",": (0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "[": (0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E),
    "]": (0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E),
    "_": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F),
    "a": (0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F),
    "b": (0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E),
    "c": (0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E),
    "d": (0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F),
    "e": (0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E),
    "f": (0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08),
    "g": (0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "h": (0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11),
    "i": (0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E),
    "j": (0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C),
    "k": (0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12),
    "l": (0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "m": (0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15),
    "n": (0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11),
    "o": (0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E),
    "p": (0x00, 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10),
    "q": (0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x01),
    "r": (0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10),
    "s": (0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E),
    "t": (0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06),
    "u": (0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D),
    "v": (0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "w": (0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A),
    "x": (0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11),
    "y": (0x00, 0x11, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "z": (0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F),
}
_GLYPH_WIDTH = 5
_GLYPH_HEIGHT = 7


def text_width(value: str, scale: int = 1) -> int:
    return len(value) * (_GLYPH_WIDTH + 1) * scale


class Canvas:
    """Minimal RGB raster canvas that serializes to PNG via ``zlib``."""

    def __init__(self, width: int, height: int, background: tuple[int, int, int] = (255, 255, 255)):
        self.width = int(width)
        self.height = int(height)
        self.pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.pixels[:, :, :] = np.asarray(background, dtype=np.uint8)

    def _set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y, x] = color

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        left, right = sorted((int(x0), int(x1)))
        top, bottom = sorted((int(y0), int(y1)))
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, self.width - 1)
        bottom = min(bottom, self.height - 1)
        if left <= right and top <= bottom:
            self.pixels[top : bottom + 1, left : right + 1] = color

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        offsets = range(-(thickness // 2), thickness // 2 + 1)
        for step in range(steps + 1):
            fraction = step / steps if steps else 0.0
            x = int(round(x0 + (x1 - x0) * fraction))
            y = int(round(y0 + (y1 - y0) * fraction))
            for dx in offsets:
                for dy in offsets:
                    self._set(x + dx, y + dy, color)

    def polyline(self, xs, ys, color: tuple[int, int, int], thickness: int = 1) -> None:
        for index in range(len(xs) - 1):
            self.line(xs[index], ys[index], xs[index + 1], ys[index + 1], color, thickness)

    def marker(self, x: float, y: float, color: tuple[int, int, int], size: int = 3, shape: str = "o") -> None:
        cx, cy = int(round(x)), int(round(y))
        if shape == "o":
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    if dx * dx + dy * dy <= size * size:
                        self._set(cx + dx, cy + dy, color)
        elif shape == "s":
            self.rect(cx - size, cy - size, cx + size, cy + size, color)
        elif shape == "^":
            for dy in range(-size, size + 1):
                span = size - abs(dy)
                for dx in range(-span, span + 1):
                    self._set(cx + dx, cy + dy, color)
        elif shape == "x":
            for offset in range(-size, size + 1):
                self._set(cx + offset, cy + offset, color)
                self._set(cx + offset, cy - offset, color)
        else:
            raise ValueError(f"Unknown marker shape: {shape}")

    def text(
        self,
        x: int,
        y: int,
        value: str,
        color: tuple[int, int, int] = (20, 20, 20),
        scale: int = 1,
    ) -> None:
        cursor = int(x)
        for character in value:
            glyph = _FONT.get(character)
            if glyph is None:
                glyph = _FONT["?"] if "?" in _FONT else _FONT[" "]
            for row_index, row in enumerate(glyph):
                for column in range(_GLYPH_WIDTH):
                    if row & (1 << (_GLYPH_WIDTH - 1 - column)):
                        for sx in range(scale):
                            for sy in range(scale):
                                self._set(
                                    cursor + column * scale + sx,
                                    int(y) + row_index * scale + sy,
                                    color,
                                )
            cursor += (_GLYPH_WIDTH + 1) * scale

    def text_right(self, x: int, y: int, value: str, color=(20, 20, 20), scale: int = 1) -> None:
        self.text(int(x) - text_width(value, scale), y, value, color, scale)

    def text_center(self, x: int, y: int, value: str, color=(20, 20, 20), scale: int = 1) -> None:
        self.text(int(x) - text_width(value, scale) // 2, y, value, color, scale)

    def save_png(self, path: Path, compression: int = 9) -> None:
        raw = bytearray()
        for row in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[row].tobytes())

        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
            )

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        blob = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), compression))
            + chunk(b"IEND", b"")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
