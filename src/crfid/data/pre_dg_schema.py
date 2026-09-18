"""Typed schema contracts for the frozen Pre-DG datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreDGDatasetSchema:
    dataset_id: str
    row_count: int
    input_channels: int
    input_length: int
    class_order: tuple[int, ...]
    class_count: int
    label_column: str
    condition_columns: tuple[str, ...]
    group_column: str
    x_sha256: str
    metadata_sha256: str
    manifest_sha256: str
    parameter_count: int


@dataclass(frozen=True)
class PreDGSplitSchema:
    dataset_id: str
    split_name: str
    file_name: str
    file_sha256: str
    metadata_sha256: str
    train_count: int
    validation_count: int
    test_count: int

    @property
    def row_count(self) -> int:
        return self.train_count + self.validation_count + self.test_count
