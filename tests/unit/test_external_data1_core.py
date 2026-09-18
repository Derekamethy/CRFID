from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crfid.data.data1 import load_csv
from crfid.exceptions import DataValidationError, ProtocolViolation
from crfid.external_data1.aggregation import (
    metric_bundle,
    prediction_mismatch_count,
    summarize_values,
)
from crfid.external_data1.artifacts import array_sha256
from crfid.external_data1.baselines import majority_tie_decision, predict_majority
from crfid.external_data1.discovery import discover_raw_files, parse_set_id
from crfid.external_data1.pca_logreg import fit_and_save, predict_from_saved_state
from crfid.external_data1.preprocessing import fit_standardization, represent, transform_with_state


class ExternalData1CoreTests(unittest.TestCase):
    def test_set_id_parser(self) -> None:
        self.assertEqual(parse_set_id("set_9.csv"), "set_9")
        with self.assertRaises(DataValidationError):
            parse_set_id("set_10.csv")

    def test_nine_file_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(1, 10):
                (root / f"set_{index}.csv").write_text("x\n", encoding="utf-8")
            discovered = discover_raw_files(root)
            self.assertEqual([item.set_id for item in discovered], [f"set_{i}" for i in range(1, 10)])

    def test_schema_parser_accepts_1601_values_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "set_3.csv"
            header = ",".join(str(index) for index in range(1602))
            row = ",".join(["1.0"] * 1601 + ["2"])
            path.write_text(f"{header}\n{row}\n", encoding="utf-8")
            inputs, labels = load_csv(path)
            self.assertEqual(inputs.shape, (1, 1601))
            self.assertEqual(labels.tolist(), [2])

    def test_schema_parser_rejects_wrong_signal_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "set_3.csv"
            path.write_text("a,b\n1,0\n", encoding="utf-8")
            with self.assertRaises(DataValidationError):
                load_csv(path)

    def test_first_difference_output_length(self) -> None:
        represented = represent(np.zeros((3, 1601)), "first_difference")
        self.assertEqual(represented.shape, (3, 1600))

    def test_majority_four_way_tie(self) -> None:
        labels = np.repeat(np.arange(4), 1400)
        decision = majority_tie_decision(labels)
        self.assertEqual(decision["selected_class"], 0)
        prediction, _ = predict_majority(5)
        self.assertEqual(prediction.tolist(), [0] * 5)

    def test_test_preprocessing_refit_is_rejected(self) -> None:
        _, state = fit_standardization(np.arange(16 * 1601).reshape(16, 1601), "raw")
        with self.assertRaises(ProtocolViolation):
            transform_with_state(np.zeros((2, 1601)), state, allow_refit=True)

    def test_pca_logistic_serialization_and_reload(self) -> None:
        generator = np.random.default_rng(4)
        labels = np.repeat(np.arange(4), 20)
        inputs = generator.normal(size=(80, 24)) + labels[:, None] * 2.0
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.npz"
            metadata = fit_and_save(inputs, labels, state)
            prediction, scores = predict_from_saved_state(state, inputs[:8])
            self.assertGreater(metadata.retained_components, 0)
            self.assertEqual(prediction.shape, (8,))
            self.assertEqual(scores.shape, (8, 4))

    def test_metric_and_aggregation(self) -> None:
        metrics = metric_bundle(np.asarray([0, 1, 2, 3]), np.asarray([0, 1, 0, 3]))
        self.assertEqual(metrics["confusion_matrix"][2][0], 1)
        summary = summarize_values([1.0, 2.0, 3.0])
        self.assertAlmostEqual(summary["population_standard_deviation"], np.std([1, 2, 3]))

    def test_artifact_array_hash_is_deterministic(self) -> None:
        values = np.arange(12, dtype=np.int64)
        self.assertEqual(array_sha256(values), array_sha256(values.copy()))

    def test_mismatch_detection(self) -> None:
        self.assertEqual(
            prediction_mismatch_count(np.asarray([0, 1, 2]), np.asarray([0, 0, 2])),
            1,
        )


if __name__ == "__main__":
    unittest.main()

