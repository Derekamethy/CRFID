"""Raw-bundle identity and schema verification."""

from __future__ import annotations

import hashlib
from collections import Counter

from ..exceptions import DataValidationError
from .schema import (
    CLASS_ORDER,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_FILE_COUNT,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SET_CLASS_COUNTS,
    EXPECTED_SET_COUNTS,
    LoadedSet,
    RawFile,
)


def raw_bundle_aggregate(files: tuple[RawFile, ...]) -> str:
    """Apply the frozen filename/size/raw-hash aggregate algorithm."""

    lines = [
        f"{item.path.name}\t{item.size_bytes}\t{item.sha256}"
        for item in sorted(files, key=lambda value: value.path.name)
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def validate_expected_hashes(
    files: tuple[RawFile, ...], expected_sha256: dict[str, str]
) -> None:
    observed = {item.path.name: item.sha256 for item in files}
    if observed != expected_sha256:
        missing = sorted(set(expected_sha256).difference(observed))
        changed = sorted(
            name for name in set(expected_sha256).intersection(observed)
            if observed[name] != expected_sha256[name]
        )
        raise DataValidationError(
            f"Raw Data1 authority mismatch; missing={missing}, changed={changed}"
        )


def validate_loaded_identity(sets: tuple[LoadedSet, ...]) -> dict[str, object]:
    if len(sets) != EXPECTED_FILE_COUNT:
        raise DataValidationError("Loaded Data1 file count changed")
    total_counts: Counter[int] = Counter()
    for item in sets:
        observed = tuple(
            int((item.labels == label).sum()) for label in CLASS_ORDER
        )
        if len(item.labels) != EXPECTED_SET_COUNTS[item.set_id]:
            raise DataValidationError(f"Row count changed for {item.set_id}")
        if observed != EXPECTED_SET_CLASS_COUNTS[item.set_id]:
            raise DataValidationError(f"Class counts changed for {item.set_id}")
        total_counts.update(int(value) for value in item.labels)
    counts = tuple(total_counts[label] for label in CLASS_ORDER)
    sample_count = sum(len(item.labels) for item in sets)
    if sample_count != EXPECTED_SAMPLE_COUNT or counts != EXPECTED_CLASS_COUNTS:
        raise DataValidationError("Full Data1 identity changed")
    return {
        "file_count": len(sets),
        "sample_count": sample_count,
        "class_order": list(CLASS_ORDER),
        "class_counts": list(counts),
        "signal_length": int(sets[0].inputs.shape[1]),
        "exact_signal_duplicate_count": (
            sum(len(item.signal_sha256) for item in sets)
            - len({value for item in sets for value in item.signal_sha256})
        ),
    }

