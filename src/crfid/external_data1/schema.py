"""Frozen Data1 schema and typed runtime records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATASET_ID = "external_data1_four_class"
SCIENTIFIC_CLAIM_TYPE = "INDEPENDENT_EXTERNAL_DATASET_PIPELINE_VALIDATION"
PUBLIC_BRANCH_NAME = "External Data1 Pipeline Validation"
CLASS_ORDER = (0, 1, 2, 3)
SIGNAL_LENGTH = 1601
EXPECTED_FILE_COUNT = 9
EXPECTED_SAMPLE_COUNT = 12_250
EXPECTED_CLASS_COUNTS = (3150, 3150, 3150, 2800)
ALL_SETS = tuple(f"set_{index}" for index in range(1, 10))
TRAIN_SETS = ("set_3",)
TEST_SETS = ("set_4", "set_5", "set_6", "set_7", "set_8", "set_9")
EXCLUDED_SETS = ("set_1", "set_2")
EXPECTED_SET_COUNTS = {
    "set_1": 2400,
    "set_2": 2400,
    "set_3": 5600,
    "set_4": 300,
    "set_5": 300,
    "set_6": 300,
    "set_7": 150,
    "set_8": 400,
    "set_9": 400,
}
EXPECTED_SET_CLASS_COUNTS = {
    "set_1": (600, 600, 600, 600),
    "set_2": (600, 600, 600, 600),
    "set_3": (1400, 1400, 1400, 1400),
    "set_4": (100, 100, 100, 0),
    "set_5": (100, 100, 100, 0),
    "set_6": (100, 100, 100, 0),
    "set_7": (50, 50, 50, 0),
    "set_8": (100, 100, 100, 100),
    "set_9": (100, 100, 100, 100),
}


@dataclass(frozen=True)
class RawFile:
    set_id: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LoadedSet:
    set_id: str
    raw_file: RawFile
    inputs: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    signal_sha256: tuple[str, ...]


@dataclass(frozen=True)
class Partition:
    role: str
    inputs: np.ndarray
    labels: np.ndarray | None
    sample_ids: tuple[str, ...]
    measurement_sets: tuple[str, ...]
    signal_sha256: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return int(self.inputs.shape[0])


@dataclass(frozen=True)
class LoadedData1:
    sets: tuple[LoadedSet, ...]
    train: Partition
    test_features: Partition
    sealed_test_labels: np.ndarray
    raw_aggregate_sha256: str

