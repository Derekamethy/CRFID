"""Authoritative raw parsing with test labels sealed from callers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..data.data1 import load_csv
from ..exceptions import DataValidationError
from .discovery import discover_raw_files
from .identity import raw_bundle_aggregate, validate_expected_hashes, validate_loaded_identity
from .schema import LoadedData1, LoadedSet, Partition, TEST_SETS, TRAIN_SETS


def exact_signal_sha256(signal: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(signal, dtype="<f8"))
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _sample_id(
    file_sha256: str, row_index: int, label: int, signal_sha256: str
) -> str:
    return (
        f"DATA1::{file_sha256}::ROW::{row_index}::SIGNAL::{signal_sha256}"
    )


def _load_set(raw_file: object) -> LoadedSet:
    inputs, labels = load_csv(raw_file.path)
    signal_hashes = tuple(exact_signal_sha256(row) for row in inputs)
    sample_ids = tuple(
        _sample_id(raw_file.sha256, index, int(labels[index]), signal_hashes[index])
        for index in range(len(labels))
    )
    if len(set(sample_ids)) != len(sample_ids):
        raise DataValidationError(f"Repeated sample identifiers in {raw_file.set_id}")
    return LoadedSet(
        set_id=raw_file.set_id,
        raw_file=raw_file,
        inputs=inputs,
        labels=labels,
        sample_ids=sample_ids,
        signal_sha256=signal_hashes,
    )


def _partition(
    role: str, loaded: dict[str, LoadedSet], set_ids: tuple[str, ...], *, labels: bool
) -> Partition:
    inputs = np.ascontiguousarray(np.concatenate([loaded[item].inputs for item in set_ids]))
    label_array = (
        np.ascontiguousarray(np.concatenate([loaded[item].labels for item in set_ids]))
        if labels
        else None
    )
    sample_ids = tuple(
        sample for item in set_ids for sample in loaded[item].sample_ids
    )
    signal_hashes = tuple(
        value for item in set_ids for value in loaded[item].signal_sha256
    )
    groups = tuple(
        item for item in set_ids for _ in range(len(loaded[item].labels))
    )
    return Partition(role, inputs, label_array, sample_ids, groups, signal_hashes)


def load_authoritative_data1(
    data_root: str | Path,
    *,
    expected_file_sha256: dict[str, str],
    expected_aggregate_sha256: str,
) -> LoadedData1:
    """Load the exact manifest-authorized bundle and seal held-out labels."""

    raw_files = discover_raw_files(data_root)
    validate_expected_hashes(raw_files, expected_file_sha256)
    aggregate = raw_bundle_aggregate(raw_files)
    if aggregate != expected_aggregate_sha256:
        raise DataValidationError("Raw Data1 aggregate SHA-256 changed")
    loaded_sets = tuple(_load_set(item) for item in raw_files)
    validate_loaded_identity(loaded_sets)
    by_set = {item.set_id: item for item in loaded_sets}
    train = _partition("train", by_set, TRAIN_SETS, labels=True)
    test_with_labels = _partition("test", by_set, TEST_SETS, labels=True)
    test_features = Partition(
        "test",
        test_with_labels.inputs,
        None,
        test_with_labels.sample_ids,
        test_with_labels.measurement_sets,
        test_with_labels.signal_sha256,
    )
    if set(train.sample_ids).intersection(test_features.sample_ids):
        raise DataValidationError("Train/test sample identity overlap")
    if set(train.signal_sha256).intersection(test_features.signal_sha256):
        raise DataValidationError("Train/test exact-signal overlap")
    assert test_with_labels.labels is not None
    return LoadedData1(
        sets=loaded_sets,
        train=train,
        test_features=test_features,
        sealed_test_labels=np.ascontiguousarray(test_with_labels.labels),
        raw_aggregate_sha256=aggregate,
    )
