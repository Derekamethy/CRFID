"""Public artifact serialization and integrity helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..governance.integrity import sha256_file


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(destination)


def write_csv(
    path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(destination)


def write_prediction_bundle(
    path: str | Path,
    *,
    relative_path: str,
    sample_ids: tuple[str, ...],
    measurement_sets: tuple[str, ...],
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predicted = np.ascontiguousarray(predictions, dtype=np.int64)
    probabilities = np.ascontiguousarray(scores, dtype=np.float64)
    np.savez(
        destination,
        sample_ids=np.asarray(sample_ids),
        measurement_sets=np.asarray(measurement_sets),
        predictions=predicted,
        scores=probabilities,
    )
    return {
        "relative_path": relative_path,
        "sample_count": int(predicted.size),
        "file_sha256": sha256_file(destination),
        "prediction_array_sha256": array_sha256(predicted),
        "score_array_sha256": array_sha256(probabilities),
    }


def public_manifest(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    rows = []
    for path in sorted((item for item in base.rglob("*") if item.is_file())):
        rows.append(
            {
                "relative_path": path.relative_to(base).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows
