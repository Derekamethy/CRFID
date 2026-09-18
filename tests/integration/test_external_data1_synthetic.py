from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crfid.external_data1.aggregation import metric_bundle
from crfid.external_data1.baselines import predict_majority
from crfid.external_data1.cnn_portability import (
    configure_determinism,
    infer_cnn,
    prepare_representation,
    save_and_reload_checkpoint,
    train_cnn,
)
from crfid.external_data1.pca_logreg import fit_and_save, predict_from_saved_state
from crfid.models.cnn1d import NeutralCNN1D, parameter_count


class ExternalData1SyntheticIntegrationTests(unittest.TestCase):
    def test_complete_majority_flow(self) -> None:
        prediction, _ = predict_majority(4)
        metrics = metric_bundle(np.asarray([0, 1, 2, 3]), prediction)
        self.assertEqual(metrics["accuracy"], 0.25)

    def test_complete_pca_logistic_flow(self) -> None:
        generator = np.random.default_rng(9)
        labels = np.repeat(np.arange(4), 25)
        inputs = generator.normal(size=(100, 32)) + labels[:, None] * 3
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "estimator.npz"
            fit_and_save(inputs, labels, state)
            prediction, _ = predict_from_saved_state(state, inputs)
            self.assertGreater(metric_bundle(labels, prediction)["accuracy"], 0.9)

    def test_cnn_four_class_output(self) -> None:
        import torch

        os.environ["PYTHONHASHSEED"] = "0"
        configure_determinism()
        model = NeutralCNN1D(class_count=4)
        output = model(torch.zeros((2, 1, 1601), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(parameter_count(model), 142084)

    def test_raw_cnn_train_checkpoint_reload_and_infer(self) -> None:
        os.environ["PYTHONHASHSEED"] = "0"
        generator = np.random.default_rng(3)
        labels = np.repeat(np.arange(4), 4).astype(np.int64)
        inputs = generator.normal(size=(16, 1601))
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "preprocessing.npz"
            train, state = prepare_representation(inputs, "raw", state_path)
            model, _ = train_cnn(train, labels, seed=42, epochs=1)
            checkpoint = Path(directory) / "checkpoint.pt"
            loaded, metadata = save_and_reload_checkpoint(
                checkpoint,
                model,
                run_id="synthetic",
                representation="raw",
                seed=42,
                fixed_epoch=1,
                preprocessing_sha256="synthetic",
            )
            prediction, scores = infer_cnn(loaded, train)
            self.assertTrue(metadata["reload_verified"])
            self.assertEqual(prediction.shape, (16,))
            self.assertEqual(scores.shape, (16, 4))

    def test_first_difference_cnn_flow(self) -> None:
        os.environ["PYTHONHASHSEED"] = "0"
        generator = np.random.default_rng(5)
        labels = np.repeat(np.arange(4), 2).astype(np.int64)
        inputs = generator.normal(size=(8, 1601))
        with tempfile.TemporaryDirectory() as directory:
            train, state = prepare_representation(
                inputs, "first_difference", Path(directory) / "preprocessing.npz"
            )
            self.assertEqual(train.shape, (8, 1600))
            model, _ = train_cnn(train, labels, seed=43, epochs=1)
            prediction, _ = infer_cnn(model, train)
            self.assertEqual(prediction.shape, (8,))


if __name__ == "__main__":
    unittest.main()
