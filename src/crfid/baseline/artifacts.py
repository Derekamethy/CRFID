"""Small artifact writers used by the Pre-DG workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..governance.pre_dg_integrity import sha256_file


def write_json(path: str | Path, payload: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(destination)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(destination)


def file_manifest(root: str | Path, *, omit_directories: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if relative.parts and relative.parts[0] in omit_directories:
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows
