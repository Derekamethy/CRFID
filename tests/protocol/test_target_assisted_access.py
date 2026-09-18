from __future__ import annotations

import unittest

from crfid.exceptions import ProtocolViolation, SplitIsolationError, TargetAccessViolation
from crfid.governance.claims import RETROSPECTIVE_CLAIM, validate_target_assisted_claim
from crfid.governance.target_access import TargetAccessContract
from crfid.protocols.common import AccessRequest, Purpose, Resource
from crfid.protocols.target_assisted import TargetAssistedProtocol
from crfid.adaptation.target_partition import overlap_report


def contract(**changes: object) -> TargetAccessContract:
    values = {
        "allow_features": True,
        "allow_labels": True,
        "allowed_adaptation_partition": "p4_full",
        "allowed_selection_partition": "p4_full",
        "allowed_evaluation_partition": "p4_full",
        "metrics_may_influence_selection": True,
        "source_parameters_may_update": False,
        "layers_may_update": [],
        "source_preprocessing_may_refit": False,
        "target_preprocessing_may_fit": False,
        "transductive_query_features_allowed": True,
        "query_labels_allowed": True,
        "final_claim_type": RETROSPECTIVE_CLAIM,
    }
    values.update(changes)
    return TargetAccessContract.from_mapping(values)


class TargetAssistedAccessTests(unittest.TestCase):
    def test_declared_target_labels_allowed(self) -> None:
        TargetAssistedProtocol(contract()).authorize(
            AccessRequest("P4", Resource.LABELS, Purpose.EVALUATION, "p4_full")
        )

    def test_labels_rejected_when_disabled(self) -> None:
        with self.assertRaises(TargetAccessViolation):
            TargetAssistedProtocol(contract(allow_labels=False)).authorize(
                AccessRequest("P4", Resource.LABELS, Purpose.EVALUATION, "p4_full")
            )

    def test_outcome_selection_rejected_when_disabled(self) -> None:
        with self.assertRaises(TargetAccessViolation):
            TargetAssistedProtocol(contract(metrics_may_influence_selection=False)).authorize(
                AccessRequest("P4", Resource.OUTCOMES, Purpose.SELECTION, "p4_full")
            )

    def test_wrong_partition_rejected(self) -> None:
        with self.assertRaises(TargetAccessViolation):
            TargetAssistedProtocol(contract()).authorize(
                AccessRequest("P4", Resource.FEATURES, Purpose.ADAPTATION, "invented_support")
            )

    def test_source_update_rejected(self) -> None:
        with self.assertRaises(TargetAccessViolation):
            contract().authorize_parameter_update(("classifier.weight",))

    def test_overlap_detected_and_rejected_when_forbidden(self) -> None:
        with self.assertRaises(SplitIsolationError):
            overlap_report({"selection": ["a"], "evaluation": ["a"]}, overlap_allowed=False)

    def test_stronger_public_claim_rejected(self) -> None:
        with self.assertRaises(ProtocolViolation):
            validate_target_assisted_claim("INDEPENDENT_TARGET_EVALUATION")


if __name__ == "__main__":
    unittest.main()
