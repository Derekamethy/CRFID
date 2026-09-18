from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from crfid.adaptation.adaptation_results import aggregate_metrics
from crfid.adaptation.prototypes import blend_prototypes, normalized_source_prototypes
from crfid.adaptation.readout_adaptation import load_readout_state, save_readout_state
from crfid.adaptation.target_assisted import execute_target_informed_ncm
from crfid.adaptation.target_partition import historical_full_p4_partition
from crfid.exceptions import AdaptationSpecificationError, SourceReleaseIntegrityError
from crfid.governance.adaptation_integrity import canonical_json_sha256, validate_source_release, verify_specification


ROOT = Path(__file__).resolve().parents[2]


class TargetAssistedCompleteTests(unittest.TestCase):
    def test_source_release_import_and_hash_validation(self) -> None:
        release = ROOT / "outputs" / "strict_dg" / "frozen_release"
        if not (release / "FROZEN_RELEASE_MANIFEST.json").is_file():
            self.skipTest("governed Strict-DG frozen release is external")
        result = validate_source_release(
            release,
            "4805f8040f3609dbf4a02553b15c64471f64e3d2a9bc7a6c1068a6342c29d1b6",
        )
        self.assertEqual(result["verification_status"], "PASS")
        self.assertEqual(result["imported_seeds"], [42, 43, 44, 45, 46])

    def test_modified_recipe_identity_rejected(self) -> None:
        with self.assertRaises(SourceReleaseIntegrityError):
            validate_source_release(ROOT / "outputs" / "strict_dg" / "frozen_release", "0" * 64)

    def test_normalized_prototype_class_order(self) -> None:
        embeddings = np.asarray([[0.0, 2.0], [3.0, 0.0]])
        result = normalized_source_prototypes(embeddings, np.asarray([1, 0]), (0, 1))
        np.testing.assert_array_equal(result, np.asarray([[1.0, 0.0], [0.0, 1.0]]))

    def test_cosine_prediction_and_lowest_index_tie(self) -> None:
        result = execute_target_informed_ncm(
            np.asarray([[1.0, 0.0], [-1.0, 0.0]]), np.asarray([0, 1]), np.asarray([[0.0, 1.0]]), class_order=(0, 1)
        )
        self.assertEqual(result.predictions.tolist(), [0])
        self.assertEqual(result.distances.tolist(), [[0.0, 0.0]])

    def test_prototype_blending(self) -> None:
        source = np.asarray([[1.0, 0.0]])
        target = np.asarray([[0.0, 1.0]])
        np.testing.assert_allclose(blend_prototypes(source, target, target_weight=0.5), [[2 ** -0.5, 2 ** -0.5]])

    def test_readout_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            save_readout_state(path, np.eye(2), (0, 1))
            prototypes, order = load_readout_state(path)
        np.testing.assert_array_equal(prototypes, np.eye(2))
        self.assertEqual(order, (0, 1))

    def test_metric_aggregation(self) -> None:
        result = aggregate_metrics([{"accuracy": 0.5, "macro_f1": 0.25}, {"accuracy": 1.0, "macro_f1": 0.75}])
        self.assertEqual(result["accuracy_mean"], 0.75)
        self.assertEqual(result["macro_f1_mean"], 0.5)

    def test_historical_partition_marks_reuse(self) -> None:
        rows = historical_full_p4_partition(["a", "b"])
        self.assertTrue(all(row["assigned_role"] == "selection_and_final_evaluation" for row in rows))

    def test_altered_specification_rejected(self) -> None:
        original = {"method": "source_cosine_ncm"}
        digest = canonical_json_sha256(original)
        verify_specification(original, digest)
        with self.assertRaises(AdaptationSpecificationError):
            verify_specification({"method": "changed"}, digest)


if __name__ == "__main__":
    unittest.main()
