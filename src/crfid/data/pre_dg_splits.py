"""Authoritative split reconstruction and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..governance.pre_dg_integrity import require_file_sha256
from .pre_dg_schema import PreDGSplitSchema


@dataclass(frozen=True)
class ReconstructedSplit:
    schema: PreDGSplitSchema
    frame: pd.DataFrame
    indices: dict[str, np.ndarray]
    sample_ids: dict[str, tuple[str, ...]]
    integrity: dict[str, object]


def reconstruct_authoritative_split(
    split_path: str | Path,
    split_metadata_path: str | Path,
    schema: PreDGSplitSchema,
    metadata: pd.DataFrame,
    group_column: str,
) -> ReconstructedSplit:
    path = Path(split_path)
    metadata_path = Path(split_metadata_path)
    require_file_sha256(path, schema.file_sha256)
    require_file_sha256(metadata_path, schema.metadata_sha256)
    frame = pd.read_csv(path)
    required = {"sample_id", "dataset_id", "split", group_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Split {schema.split_name} is missing columns: {missing}")
    if len(frame) != schema.row_count or not frame["sample_id"].is_unique:
        raise ValueError(f"Split {schema.split_name} row/sample identity mismatch")
    if set(frame["dataset_id"].astype(str)) != {schema.dataset_id}:
        raise ValueError(f"Split {schema.split_name} dataset identity mismatch")
    observed_counts = frame["split"].value_counts().to_dict()
    expected_counts = {
        "train": schema.train_count,
        "val": schema.validation_count,
        "test": schema.test_count,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Split {schema.split_name} counts changed: {observed_counts} != {expected_counts}"
        )
    metadata_ids = metadata["sample_id"].astype(str)
    split_ids = frame["sample_id"].astype(str)
    if set(metadata_ids) != set(split_ids):
        raise ValueError(f"Split {schema.split_name} does not cover the dataset exactly")
    sample_to_index = {sample_id: index for index, sample_id in enumerate(metadata_ids)}
    indices: dict[str, np.ndarray] = {}
    sample_ids: dict[str, tuple[str, ...]] = {}
    group_sets: dict[str, set[str]] = {}
    for public_name, source_name in (("train", "train"), ("validation", "val"), ("test", "test")):
        part = frame.loc[frame["split"].eq(source_name)].copy()
        ids = tuple(part["sample_id"].astype(str))
        indices[public_name] = np.asarray([sample_to_index[value] for value in ids], dtype=np.int64)
        sample_ids[public_name] = ids
        group_sets[public_name] = set(part[group_column].astype(str))
    overlaps = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Raw-condition leakage in {schema.split_name}: {overlaps}")
    integrity = {
        "split_name": schema.split_name,
        "row_count": len(frame),
        "counts": {
            "train": len(indices["train"]),
            "validation": len(indices["validation"]),
            "test": len(indices["test"]),
        },
        "raw_condition_group_overlap": overlaps,
        "sample_order_preserved_within_partitions": True,
        "passed": True,
    }
    return ReconstructedSplit(schema, frame, indices, sample_ids, integrity)
