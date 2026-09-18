from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows/11_p4_factor_aware_few_shot"
sys.path.insert(0, str(WORKFLOW))

from p4_factor_aware.constants import HEAD_LAMBDAS, HISTORICAL_BINDING_HASHES
from p4_factor_aware.models import (
    build_cosine_prototypes,
    cosine_similarity_predict,
    load_frozen_model,
    preprocess,
)
from p4_factor_aware.protocol import ProtocolViolation


class ModelTests(unittest.TestCase):
    def test_19_cosine_similarity_prototype_implementation(self) -> None:
        support = torch.zeros((7, 256), dtype=torch.float32)
        for class_index in range(7):
            support[class_index, class_index] = 1.0
        labels = np.arange(7)
        prototypes = build_cosine_prototypes(support, labels, 1)
        prediction, similarities = cosine_similarity_predict(support, prototypes)
        np.testing.assert_array_equal(prediction, labels)
        self.assertEqual(similarities.shape, (7, 7))

    def test_20_cosine_exact_tie_uses_frozen_class_order(self) -> None:
        query = torch.zeros((1, 256), dtype=torch.float32)
        query[0, 0] = 1.0
        prototypes = torch.zeros((7, 256), dtype=torch.float64)
        prototypes[:, 0] = 1.0
        prediction, _ = cosine_similarity_predict(query, prototypes)
        self.assertEqual(int(prediction[0]), 0)

    def test_21_unequal_label_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_cosine_prototypes(torch.ones((20, 256), dtype=torch.float32), np.repeat(np.arange(7), 3)[:20], 3)

    def test_22_first_difference_dimension_matches_c1(self) -> None:
        raw = np.arange(2 * 281, dtype=np.float64).reshape(2, 281)
        transformed = preprocess(raw, np.zeros(280), np.ones(280))
        self.assertEqual(tuple(transformed.shape), (2, 1, 280))

    def test_23_frozen_checkpoint_hash_gate_rejects_unknown_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FINAL_CHECKPOINT.pt"
            path.write_bytes(b"not a governed checkpoint")
            with self.assertRaises(ProtocolViolation):
                load_frozen_model(path, 42)

    def test_24_historical_linear_head_hyperparameter_binding(self) -> None:
        self.assertEqual(HEAD_LAMBDAS, {1: 100.0, 3: 100.0, 5: 0.1})

    def test_25_historical_result_preservation_bindings_exist(self) -> None:
        result_paths = [path for path in HISTORICAL_BINDING_HASHES if "RESULT" in path or "COMPARISON" in path]
        self.assertGreaterEqual(len(result_paths), 4)
        self.assertTrue(all(len(HISTORICAL_BINDING_HASHES[path]) == 64 for path in result_paths))


if __name__ == "__main__":
    unittest.main()
