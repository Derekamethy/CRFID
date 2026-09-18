"""Tyndall/Paper task schema and explicit label mapping."""

from __future__ import annotations

from pathlib import Path

from .schemas import DatasetSchema
from ..exceptions import DataValidationError


CLASS_ORDER = tuple(range(7))
DOMAIN_ORDER = ("P1", "P2", "P3", "P4")
SOURCE_DOMAINS = DOMAIN_ORDER[:3]
TARGET_DOMAIN = "P4"
SCHEMA = DatasetSchema("tyndall", CLASS_ORDER, 281, DOMAIN_ORDER)


def map_raw_tag_label(raw_label: int) -> int:
    """Map raw tag labels 1..7 to canonical indices 0..6."""

    if raw_label not in range(1, 8):
        raise DataValidationError("Raw Tyndall tag label must be in 1..7")
    return raw_label - 1


def parse_measurement_name(path: str | Path) -> tuple[str, str]:
    """Return surface and position from names such as A1_P2.csv."""

    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) != 2 or parts[1] not in DOMAIN_ORDER:
        raise DataValidationError(f"Unexpected Tyndall measurement name: {stem}")
    return parts[0], parts[1]
