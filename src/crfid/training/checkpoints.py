"""Explicit checkpoint save/load with optional hash verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import DependencyUnavailableError
from ..governance.integrity import sha256_file


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise DependencyUnavailableError("PyTorch is required for checkpoints") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return sha256_file(destination)


def load_checkpoint(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise DependencyUnavailableError("PyTorch is required for checkpoints") from exc
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError("Checkpoint hash mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    return payload
