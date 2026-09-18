"""Historical target-use partition construction and overlap accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..exceptions import SplitIsolationError


HISTORICAL_ROLE = "selection_and_final_evaluation"


def historical_full_p4_partition(sample_ids: Iterable[str]) -> list[dict[str, str]]:
    identifiers = [str(value) for value in sample_ids]
    if len(identifiers) != len(set(identifiers)):
        raise SplitIsolationError("Duplicate target sample identity")
    return [
        {
            "sample_id": identity,
            "partition": "p4_full",
            "assigned_role": HISTORICAL_ROLE,
            "features_accessed": "true",
            "labels_accessed": "true",
            "purpose": "readout_selection_and_final_scoring",
        }
        for identity in identifiers
    ]


def overlap_report(partitions: Mapping[str, Iterable[str]], *, overlap_allowed: bool) -> dict[str, object]:
    sets = {name: set(values) for name, values in partitions.items()}
    overlaps: dict[str, list[str]] = {}
    names = sorted(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(sets[left] & sets[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    if overlaps and not overlap_allowed:
        raise SplitIsolationError("Target partitions overlap but the contract forbids reuse")
    return {
        "overlap_allowed": overlap_allowed,
        "overlap_detected": bool(overlaps),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "scientific_limitation": bool(overlaps),
    }
