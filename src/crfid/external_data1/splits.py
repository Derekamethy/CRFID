"""Frozen documented set-3 to sets-4-through-9 split."""

from __future__ import annotations

from collections import Counter

from ..exceptions import ProtocolViolation
from .schema import EXCLUDED_SETS, LoadedData1, TEST_SETS, TRAIN_SETS


SPLIT_ID = "documented_original_set3_to_sets4_9"
EXCLUSION_REASON = (
    "Sets 1, 2, and 3 are documented for different applications; "
    "the next six sets test models trained with the third set."
)


def validate_frozen_split(data: LoadedData1) -> dict[str, object]:
    if tuple(sorted(set(data.train.measurement_sets))) != TRAIN_SETS:
        raise ProtocolViolation("Only set_3 may be used for training")
    if tuple(sorted(set(data.test_features.measurement_sets))) != TEST_SETS:
        raise ProtocolViolation("Only sets 4 through 9 may be used for testing")
    if data.train.sample_count != 5600 or data.test_features.sample_count != 1850:
        raise ProtocolViolation("Frozen train/test sample counts changed")
    if set(data.train.sample_ids).intersection(data.test_features.sample_ids):
        raise ProtocolViolation("Frozen split is not sample-disjoint")
    train_counts = Counter(int(value) for value in data.train.labels)
    test_counts = Counter(int(value) for value in data.sealed_test_labels)
    return {
        "split_id": SPLIT_ID,
        "train_sets": list(TRAIN_SETS),
        "test_sets": list(TEST_SETS),
        "excluded_sets": list(EXCLUDED_SETS),
        "excluded_set_reason": EXCLUSION_REASON,
        "train_sample_count": data.train.sample_count,
        "test_sample_count": data.test_features.sample_count,
        "train_class_counts": [train_counts[index] for index in range(4)],
        "test_class_counts": [test_counts[index] for index in range(4)],
        "train_test_disjoint": True,
        "validation_sets": [],
    }


def split_manifest_rows(data: LoadedData1) -> list[dict[str, object]]:
    role_by_set = {
        "set_1": "documented_other_application_not_in_primary_split",
        "set_2": "documented_other_application_not_in_primary_split",
        "set_3": "train",
        **{item: "test" for item in TEST_SETS},
    }
    rows: list[dict[str, object]] = []
    for loaded_set in data.sets:
        for index, (sample_id, label, signal_hash) in enumerate(
            zip(
                loaded_set.sample_ids,
                loaded_set.labels,
                loaded_set.signal_sha256,
                strict=True,
            )
        ):
            rows.append(
                {
                    "split_id": SPLIT_ID,
                    "split_role": role_by_set[loaded_set.set_id],
                    "measurement_set_id": loaded_set.set_id,
                    "row_index": index,
                    "raw_label": int(label),
                    "sample_id": sample_id,
                    "exact_signal_sha256": signal_hash,
                    "raw_file_sha256": loaded_set.raw_file.sha256,
                }
            )
    return rows

