"""Hash-gated loader for exact frozen Pre-DG processed inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..governance.pre_dg_access import validate_pre_dg_dataset_access
from ..governance.pre_dg_integrity import require_file_sha256
from .pre_dg_schema import PreDGDatasetSchema


@dataclass(frozen=True)
class LoadedPreDGDataset:
    schema: PreDGDatasetSchema
    inputs: np.ndarray
    metadata: pd.DataFrame
    manifest: pd.DataFrame
    processed_root: Path
    manifest_path: Path


def load_authoritative_processed_dataset(
    schema: PreDGDatasetSchema,
    processed_root: str | Path,
    manifest_path: str | Path,
) -> LoadedPreDGDataset:
    root = Path(processed_root).resolve()
    manifest = Path(manifest_path).resolve()
    validate_pre_dg_dataset_access(schema.dataset_id, root, manifest)
    x_path = root / "X.npy"
    metadata_path = root / "metadata.csv"
    require_file_sha256(x_path, schema.x_sha256)
    require_file_sha256(metadata_path, schema.metadata_sha256)
    require_file_sha256(manifest, schema.manifest_sha256)
    inputs = np.load(x_path, allow_pickle=False)
    metadata_frame = pd.read_csv(metadata_path)
    manifest_frame = pd.read_csv(manifest)
    if list(inputs.shape) != [
        schema.row_count,
        schema.input_channels,
        schema.input_length,
    ]:
        raise ValueError(f"{schema.dataset_id} processed shape changed: {inputs.shape}")
    if inputs.dtype != np.float32 or not np.isfinite(inputs).all():
        raise ValueError(f"{schema.dataset_id} processed dtype or finiteness changed")
    if len(metadata_frame) != schema.row_count or len(manifest_frame) != schema.row_count:
        raise ValueError(f"{schema.dataset_id} metadata/manifest row count changed")
    for frame_name, frame in (("metadata", metadata_frame), ("manifest", manifest_frame)):
        if "sample_id" not in frame or not frame["sample_id"].is_unique:
            raise ValueError(f"{schema.dataset_id} {frame_name} sample identity is invalid")
    if metadata_frame["sample_id"].astype(str).tolist() != manifest_frame["sample_id"].astype(str).tolist():
        raise ValueError(f"{schema.dataset_id} processed/manifest sample order changed")
    observed_classes = tuple(
        sorted(metadata_frame[schema.label_column].dropna().astype(int).unique().tolist())
    )
    if observed_classes != schema.class_order:
        raise ValueError(
            f"{schema.dataset_id} class order changed: {observed_classes} != {schema.class_order}"
        )
    required = {"sample_id", schema.label_column, schema.group_column, *schema.condition_columns}
    missing = sorted(required.difference(metadata_frame.columns))
    if missing:
        raise ValueError(f"{schema.dataset_id} metadata fields missing: {missing}")
    return LoadedPreDGDataset(
        schema=schema,
        inputs=inputs,
        metadata=metadata_frame,
        manifest=manifest_frame,
        processed_root=root,
        manifest_path=manifest,
    )
