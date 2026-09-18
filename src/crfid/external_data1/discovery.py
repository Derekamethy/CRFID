"""Strict nine-file discovery for a user-supplied Data1 root."""

from __future__ import annotations

import re
from pathlib import Path

from ..exceptions import DataValidationError
from ..governance.integrity import sha256_file
from .schema import ALL_SETS, EXPECTED_FILE_COUNT, RawFile


_SET_PATTERN = re.compile(r"^set_([1-9])\.csv$")


def parse_set_id(path: str | Path) -> str:
    """Return the canonical set identifier from an exact raw filename."""

    name = Path(path).name
    match = _SET_PATTERN.fullmatch(name)
    if match is None:
        raise DataValidationError(f"Unsupported Data1 raw filename: {name}")
    return f"set_{int(match.group(1))}"


def discover_raw_files(data_root: str | Path) -> tuple[RawFile, ...]:
    """Discover exactly set_1.csv through set_9.csv, with no substitutions."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    csv_files = sorted(root.glob("*.csv"), key=lambda path: path.name)
    if len(csv_files) != EXPECTED_FILE_COUNT:
        raise DataValidationError(
            f"Expected exactly {EXPECTED_FILE_COUNT} raw CSV files, found {len(csv_files)}"
        )
    by_set = {parse_set_id(path): path for path in csv_files}
    if tuple(sorted(by_set, key=lambda item: int(item.split("_")[1]))) != ALL_SETS:
        raise DataValidationError("Data1 raw root must contain exactly set_1.csv through set_9.csv")
    return tuple(
        RawFile(
            set_id=set_id,
            path=by_set[set_id],
            size_bytes=by_set[set_id].stat().st_size,
            sha256=sha256_file(by_set[set_id]),
        )
        for set_id in ALL_SETS
    )

