"""Condition-disjoint support/query validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import SplitIsolationError
from ..governance.split_validation import assert_disjoint_groups


@dataclass(frozen=True)
class SupportQuerySplit:
    support_indices: np.ndarray
    query_indices: np.ndarray
    support_conditions: tuple[str, ...]
    query_conditions: tuple[str, ...]

    def validate(self, sample_count: int) -> None:
        support = np.asarray(self.support_indices, dtype=np.int64)
        query = np.asarray(self.query_indices, dtype=np.int64)
        if support.ndim != 1 or query.ndim != 1 or support.size == 0 or query.size == 0:
            raise SplitIsolationError("Support and query must be non-empty index vectors")
        if np.any(support < 0) or np.any(query < 0) or np.any(support >= sample_count) or np.any(query >= sample_count):
            raise SplitIsolationError("Support/query index outside sample range")
        if set(support.tolist()).intersection(query.tolist()):
            raise SplitIsolationError("Support and query sample identities overlap")
        assert_disjoint_groups({"support": self.support_conditions, "query": self.query_conditions})
