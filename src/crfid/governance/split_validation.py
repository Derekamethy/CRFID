"""Protected group and identity isolation checks."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from ..exceptions import SplitIsolationError


def assert_disjoint_groups(partitions: Mapping[str, Collection[str]]) -> None:
    names = list(partitions)
    for index, left in enumerate(names):
        left_groups = set(partitions[left])
        for right in names[index + 1 :]:
            overlap = left_groups.intersection(partitions[right])
            if overlap:
                preview = sorted(overlap)[:3]
                raise SplitIsolationError(f"Protected groups overlap between {left} and {right}: {preview}")


def assert_aligned_lengths(*values: Sequence[object]) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise SplitIsolationError("Split arrays are not aligned")
