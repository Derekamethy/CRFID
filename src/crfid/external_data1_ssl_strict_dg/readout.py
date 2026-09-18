"""Readouts: the primary learned classifier head and a secondary source-only NCM.

The primary endpoint of this branch is the learned classifier head, matching the
canonical Strict-DG selected candidate. The nearest-class-mean readout is a
*preregistered secondary diagnostic* built from source-domain class prototypes only.
It never sees a P4 prototype, a P4 support sample or a P4 label, and it does not
replace the primary endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PRIMARY_READOUT = "LEARNED_CLASSIFIER_HEAD"
SECONDARY_READOUT = "SOURCE_EUCLIDEAN_NEAREST_CLASS_MEAN"
CLASS_ORDER = tuple(range(7))


@dataclass(frozen=True)
class NearestClassMean:
    prototypes: np.ndarray  # [7, 256] float64
    fitted_sample_count: int
    fitted_on: str

    @classmethod
    def fit(cls, embeddings: np.ndarray, labels: np.ndarray, *, fitted_on: str) -> "NearestClassMean":
        values = np.asarray(embeddings, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or targets.shape != (values.shape[0],):
            raise ValueError("NCM fitting requires aligned [N,D] embeddings and [N] labels")
        if set(np.unique(targets).tolist()) != set(CLASS_ORDER):
            raise ValueError("NCM prototypes require every source class to be present")
        prototypes = np.stack(
            [values[targets == index].mean(axis=0) for index in CLASS_ORDER], axis=0
        )
        return cls(np.ascontiguousarray(prototypes), int(values.shape[0]), fitted_on)

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Negative squared Euclidean distance to each source class prototype."""

        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.prototypes.shape[1]:
            raise ValueError("NCM scoring requires matching embedding dimensions")
        squared = (
            (values**2).sum(axis=1, keepdims=True)
            - 2.0 * values @ self.prototypes.T
            + (self.prototypes**2).sum(axis=1)[None, :]
        )
        return -squared

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.scores(embeddings).argmax(axis=1).astype(np.int64)

    def as_record(self) -> dict[str, Any]:
        return {
            "readout": SECONDARY_READOUT,
            "distance": "squared_euclidean",
            "prototype_source": self.fitted_on,
            "fitted_sample_count": self.fitted_sample_count,
            "target_prototypes_used": False,
            "target_support_samples_used": False,
        }
