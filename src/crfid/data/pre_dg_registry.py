"""Parsing and validation for the machine-readable Pre-DG authority registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pre_dg_schema import PreDGDatasetSchema, PreDGSplitSchema


RECOGNIZED_DATASETS = ("paper3_tyndall", "paper4_depolarizing")


def parse_dataset_registry(payload: Mapping[str, Any]) -> dict[str, PreDGDatasetSchema]:
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or tuple(datasets) != RECOGNIZED_DATASETS:
        raise ValueError("Pre-DG dataset registry or order is not authoritative")
    parsed: dict[str, PreDGDatasetSchema] = {}
    for dataset_id, item in datasets.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"Dataset registry entry is not a mapping: {dataset_id}")
        class_order = tuple(int(value) for value in item["class_order"])
        if len(class_order) != int(item["class_count"]) or len(set(class_order)) != len(class_order):
            raise ValueError(f"Invalid class order for {dataset_id}")
        parsed[dataset_id] = PreDGDatasetSchema(
            dataset_id=dataset_id,
            row_count=int(item["row_count"]),
            input_channels=int(item["input_channels"]),
            input_length=int(item["input_length"]),
            class_order=class_order,
            class_count=int(item["class_count"]),
            label_column=str(item["label_column"]),
            condition_columns=tuple(str(value) for value in item["condition_columns"]),
            group_column=str(item["group_column"]),
            x_sha256=str(item["x_sha256"]),
            metadata_sha256=str(item["metadata_sha256"]),
            manifest_sha256=str(item["manifest_sha256"]),
            parameter_count=int(item["parameter_count"]),
        )
    return parsed


def parse_split_registry(payload: Mapping[str, Any]) -> tuple[PreDGSplitSchema, ...]:
    raw = payload.get("splits")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Pre-DG split registry is empty")
    parsed = tuple(
        PreDGSplitSchema(
            dataset_id=str(item["dataset_id"]),
            split_name=str(item["split_name"]),
            file_name=str(item["file_name"]),
            file_sha256=str(item["file_sha256"]),
            metadata_sha256=str(item["metadata_sha256"]),
            train_count=int(item["counts"]["train"]),
            validation_count=int(item["counts"]["validation"]),
            test_count=int(item["counts"]["test"]),
        )
        for item in raw
    )
    names = [item.split_name for item in parsed]
    if len(parsed) != 18 or len(set(names)) != len(names):
        raise ValueError("Expected 18 unique authoritative Pre-DG splits")
    if any(item.dataset_id not in RECOGNIZED_DATASETS for item in parsed):
        raise ValueError("Split registry contains an unrecognized dataset")
    return parsed
