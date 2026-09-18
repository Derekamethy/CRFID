"""Reject non-authoritative data roles from the Pre-DG workflow."""

from __future__ import annotations

from pathlib import Path


_ALLOWED_DATASETS = {"paper3_tyndall", "paper4_depolarizing"}
_REJECTED_PATH_PARTS = {
    "strict_dg",
    "target_assisted",
    "external_data1",
    "few_shot",
    "openems",
}


def validate_pre_dg_dataset_access(
    dataset_id: str, processed_root: Path, manifest_path: Path
) -> None:
    if dataset_id not in _ALLOWED_DATASETS:
        raise PermissionError(f"Dataset is not authorized for Pre-DG: {dataset_id}")
    for path in (processed_root, manifest_path):
        lowered = {part.lower() for part in path.parts}
        rejected = sorted(lowered & _REJECTED_PATH_PARTS)
        if rejected:
            raise PermissionError(f"Rejected non-Pre-DG artifact role: {rejected[0]}")
    if not processed_root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("Explicit Pre-DG custody path is unavailable")


def require_explicit_location(value: str, field: str) -> Path:
    if not value or "${" in value:
        raise ValueError(f"Explicit Pre-DG location is required: {field}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Pre-DG location must be explicit and absolute: {field}")
    return path.resolve()
