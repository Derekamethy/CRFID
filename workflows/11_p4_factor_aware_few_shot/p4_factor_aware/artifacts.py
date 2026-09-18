"""Deterministic compact artifact writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return value.name
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="")


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: list[str] | None = None) -> None:
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)


def compact_float(value: float) -> str:
    return f"{float(value):.10f}"
