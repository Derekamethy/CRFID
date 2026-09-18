"""Typed data records shared by all workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..exceptions import DataValidationError


@dataclass(frozen=True)
class DatasetSchema:
    dataset_id: str
    class_order: tuple[int, ...]
    signal_length: int
    domains: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id or self.signal_length <= 1:
            raise DataValidationError("Dataset schema has an invalid identifier or length")
        if tuple(sorted(set(self.class_order))) != self.class_order:
            raise DataValidationError("Class order must be sorted and unique")
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise DataValidationError("Domains must be non-empty and unique")


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    dataset_id: str
    class_index: int
    domain_id: str
    condition_id: str
    surface_id: str | None = None
    material_id: str | None = None
    repeat_id: str | None = None


@dataclass(frozen=True)
class SignalRecord:
    metadata: SampleMetadata
    signal: np.ndarray
    frequency_axis: np.ndarray | None = None

    def validate(self, schema: DatasetSchema) -> None:
        signal = np.asarray(self.signal)
        if signal.shape != (schema.signal_length,) or not np.isfinite(signal).all():
            raise DataValidationError("Signal shape or finite-value validation failed")
        if self.metadata.dataset_id != schema.dataset_id:
            raise DataValidationError("Record dataset does not match schema")
        if self.metadata.class_index not in schema.class_order:
            raise DataValidationError("Class label is outside the declared class order")
        if self.metadata.domain_id not in schema.domains:
            raise DataValidationError("Domain is outside the declared schema")
        if self.frequency_axis is not None:
            axis = np.asarray(self.frequency_axis, dtype=np.float64)
            if axis.shape != signal.shape or not np.all(np.diff(axis) > 0):
                raise DataValidationError("Frequency axis must be aligned and increasing")


def validate_records(records: Sequence[SignalRecord], schema: DatasetSchema) -> None:
    """Validate records and unique sample identities."""

    if not records:
        raise DataValidationError("At least one record is required")
    for record in records:
        record.validate(schema)
    identities = [record.metadata.sample_id for record in records]
    if len(set(identities)) != len(identities):
        raise DataValidationError("Sample identities must be unique")
