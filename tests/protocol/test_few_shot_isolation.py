from __future__ import annotations

import unittest

import numpy as np

from crfid.adaptation.prototypes import build_target_prototypes
from crfid.adaptation.support_query import SupportQuerySplit
from crfid.exceptions import DomainAccessViolation, SplitIsolationError
from crfid.protocols.common import AccessRequest, Purpose, Resource
from crfid.protocols.few_shot import FewShotProtocol


class FewShotIsolationTests(unittest.TestCase):
    def test_condition_overlap_rejected(self) -> None:
        split = SupportQuerySplit(np.asarray([0]), np.asarray([1]), ("condition-a",), ("condition-a",))
        with self.assertRaises(SplitIsolationError):
            split.validate(2)

    def test_query_labels_sealed_during_adaptation(self) -> None:
        protocol = FewShotProtocol()
        with self.assertRaises(DomainAccessViolation):
            protocol.authorize(AccessRequest("P4", Resource.LABELS, Purpose.ADAPTATION, "query"))
        protocol.close_adaptation()
        protocol.authorize(AccessRequest("P4", Resource.LABELS, Purpose.EVALUATION, "query"))

    def test_target_prototypes_require_exact_shots(self) -> None:
        embeddings = np.zeros((7, 256), dtype=np.float32)
        labels = np.arange(7, dtype=np.int64)
        prototypes = build_target_prototypes(embeddings, labels, [f"s{i}" for i in range(7)], shot_count=1)
        self.assertEqual(prototypes.shape, (7, 256))
        self.assertEqual(prototypes.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
