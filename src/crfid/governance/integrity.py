"""File and manifest integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_manifest(root: str | Path, paths: list[Path]) -> dict[str, str]:
    base = Path(root).resolve()
    result: dict[str, str] = {}
    for supplied in paths:
        path = supplied.resolve()
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError("Manifest path escapes its declared root") from exc
        result[relative] = sha256_file(path)
    return dict(sorted(result.items()))
