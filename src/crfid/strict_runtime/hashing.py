"""Deterministic hashes and identifiers for canonical artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, semantic_key: str, length: int = 24) -> str:
    digest = hashlib.sha256(semantic_key.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(value) for value in canonical.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def exact_signal_sha256(signal: np.ndarray) -> str:
    canonical = np.ascontiguousarray(signal, dtype="<f8")
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def array_descriptor(array: np.ndarray, path: str | Path) -> dict:
    target = Path(path)
    return {
        "path": str(target.resolve()),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "c_contiguous": bool(array.flags.c_contiguous),
        "array_sha256": array_sha256(array),
        "file_sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }
