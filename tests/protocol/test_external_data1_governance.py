from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crfid.exceptions import DomainAccessViolation, ProtocolViolation
from crfid.external_data1.artifacts import write_prediction_bundle
from crfid.external_data1.schema import DATASET_ID, SCIENTIFIC_CLAIM_TYPE
from crfid.governance.external_claims import (
    PRIMARY_CLAIM,
    reject_direct_external_transfer_claim,
    validate_external_claim,
)
from crfid.governance.external_data_access import authorize_external_data
from crfid.governance.external_integrity import validate_external_mode
from crfid.governance.external_label_access import ExternalLabelVault
from crfid.protocols.external_data1 import validate_protocol


ROOT = Path(__file__).resolve().parents[2]


class ExternalData1GovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs" / "external_data1" / "ev2r_protocol.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_exact_protocol_is_accepted(self) -> None:
        validate_protocol(self.protocol)

    def test_altered_protocol_is_rejected(self) -> None:
        altered = json.loads(json.dumps(self.protocol))
        altered["cnn_seeds"][-1] = 99
        with self.assertRaises(ProtocolViolation):
            validate_protocol(altered)

    def test_altered_class_order_is_rejected(self) -> None:
        altered = json.loads(json.dumps(self.protocol))
        altered["class_order"] = [1, 0, 2, 3]
        with self.assertRaises(ProtocolViolation):
            validate_protocol(altered)

    def test_altered_tie_rule_is_rejected(self) -> None:
        altered = json.loads(json.dumps(self.protocol))
        altered["majority_tie_rule"]["selected_class"] = 1
        with self.assertRaises(ProtocolViolation):
            validate_protocol(altered)

    def test_test_labels_are_sealed_until_predictions_frozen(self) -> None:
        ids = ("a", "b")
        vault = ExternalLabelVault(np.asarray([0, 1]), ids, protocol_sha256="x")
        vault.freeze_candidate_registry({"methods": ["m"]})
        vault.register_frozen_model(
            "run", method="m", model_sha256="state", preprocessing_sha256=None
        )
        with self.assertRaises(DomainAccessViolation):
            vault.labels_for_scoring("run")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            write_prediction_bundle(
                path,
                relative_path="prediction.npz",
                sample_ids=ids,
                measurement_sets=("set_4", "set_4"),
                predictions=np.asarray([0, 1]),
                scores=np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
            )
            vault.freeze_predictions("run", path)
            self.assertEqual(vault.labels_for_scoring("run").tolist(), [0, 1])

    def test_paper4_and_p4_inputs_are_rejected(self) -> None:
        with self.assertRaises(DomainAccessViolation):
            authorize_external_data(DATASET_ID, "inputs/Paper4/data.csv")
        with self.assertRaises(DomainAccessViolation):
            authorize_external_data(DATASET_ID, "inputs/P4/data.csv")

    def test_excluded_modes_are_rejected(self) -> None:
        for mode in ("target_assisted", "few_shot", "data1_ssl"):
            with self.subTest(mode=mode), self.assertRaises(ProtocolViolation):
                validate_external_mode(mode)

    def test_claim_boundary(self) -> None:
        validate_external_claim(
            SCIENTIFIC_CLAIM_TYPE,
            PRIMARY_CLAIM,
            {"DATA1_LEARNABILITY_CLEAR"},
        )
        with self.assertRaises(ProtocolViolation):
            reject_direct_external_transfer_claim(
                "The seven-class model generalized to Data1."
            )


if __name__ == "__main__":
    unittest.main()

