"""Required lightweight tests for the representation-versus-signal diagnostic."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import numpy as np

from crfid import representation_signal_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = diagnostic.load_protocol(
    ROOT / "configs" / "representation_signal_diagnostic" / "protocol.json"
)


class RepresentationSignalDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.positions, cls.custody = diagnostic.load_governed_positions(ROOT, CONFIG)
            (
                cls.c1_module,
                cls.c1_models,
                cls.source_mean,
                cls.source_scale,
                cls.c1_binding,
            ) = diagnostic.load_c1_binding(ROOT, CONFIG)
        except FileNotFoundError as exc:
            raise unittest.SkipTest(
                "Requires governed Tyndall measurement data and frozen C1 artifacts, "
                "which are intentionally not distributed in the public repository."
            ) from exc

    def test_01_four_position_data_structure(self) -> None:
        self.assertEqual(set(self.positions), {"P1", "P2", "P3", "P4"})
        for data in self.positions.values():
            self.assertEqual(data.signals.shape, (3150, 281))
            self.assertEqual(len(data.block_keys()), 63)
            counts = np.unique(
                [np.sum(data.block_tokens == token) for token in set(data.block_tokens.tolist())]
            )
            np.testing.assert_array_equal(counts, [50])

    def test_02_synchronized_latin_square_folds(self) -> None:
        expected = {
            0: {(0, "A1"), (1, "A2"), (2, "A3")},
            1: {(0, "A2"), (1, "A3"), (2, "A1")},
            2: {(0, "A3"), (1, "A1"), (2, "A2")},
        }
        observed = {fold: set() for fold in range(3)}
        for er in CONFIG["er_levels"]:
            for surface in CONFIG["surfaces"]:
                observed[diagnostic.cell_latin_fold(er, surface, CONFIG)].add((er, surface))
        self.assertEqual(observed, expected)
        manifests = diagnostic.synchronized_split_manifest(self.positions, CONFIG)
        p1 = [(row["fold"], row["tag_id"], row["er"], row["surface"], row["split"]) for row in manifests if row["position"] == "P1"]
        p4 = [(row["fold"], row["tag_id"], row["er"], row["surface"], row["split"]) for row in manifests if row["position"] == "P4"]
        self.assertEqual(p1, p4)

    def test_03_condition_block_disjointness(self) -> None:
        for data in self.positions.values():
            for fold in range(3):
                splits = diagnostic.split_indices(data, fold, CONFIG)
                blocks = {name: set(data.block_tokens[index].tolist()) for name, index in splits.items()}
                self.assertFalse(blocks["train"] & blocks["validation"])
                self.assertFalse(blocks["train"] & blocks["test"])
                self.assertFalse(blocks["validation"] & blocks["test"])

    def test_04_exact_out_of_fold_block_coverage(self) -> None:
        for data in self.positions.values():
            counts = {token: 0 for token in set(data.block_tokens.tolist())}
            for fold in range(3):
                test = diagnostic.split_indices(data, fold, CONFIG)["test"]
                for token in set(data.block_tokens[test].tolist()):
                    counts[token] += 1
            self.assertEqual(set(counts.values()), {1})

    def test_05_raw_signal_dimensionality(self) -> None:
        self.assertEqual(diagnostic.raw_representation(self.positions["P1"]).shape, (3150, 281))

    def test_06_first_difference_dimensionality(self) -> None:
        self.assertEqual(
            diagnostic.first_difference_representation(self.positions["P1"]).shape,
            (3150, 280),
        )

    def test_07_frozen_c1_embedding_extraction(self) -> None:
        model, _ = self.c1_models[42]
        embeddings = diagnostic.extract_c1_embeddings(
            self.c1_module,
            model,
            self.positions["P1"].signals[:2],
            self.source_mean,
            self.source_scale,
        )
        self.assertEqual(embeddings.shape, (2, 256))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_08_checkpoint_hash_binding(self) -> None:
        records = self.c1_binding["checkpoints"]
        self.assertEqual([row["seed"] for row in records], [42, 43, 44, 45, 46])
        for row in records:
            seed = str(row["seed"])
            self.assertEqual(row["checkpoint_file_sha256"], CONFIG["checkpoint_file_sha256"][seed])
            self.assertEqual(row["model_state_sha256"], CONFIG["checkpoint_state_sha256"][seed])

    def test_09_training_only_scaling(self) -> None:
        train = np.asarray([[0.0, 1.0], [2.0, 3.0]])
        held = np.asarray([[100.0, 100.0]])
        scaler = diagnostic.FeatureStandardizer.fit(train, partition="train")
        np.testing.assert_allclose(scaler.mean, [1.0, 2.0])
        self.assertFalse(np.allclose(scaler.mean, np.vstack([train, held]).mean(axis=0)))
        self.assertEqual(scaler.fit_partition, "train")

    def test_10_validated_peak_extraction_gate(self) -> None:
        seal_path = ROOT / "manifests" / "representation_signal_diagnostic" / "PRE_PROBE_MANIFEST_SEAL.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        self.assertFalse(seal["peak_descriptor_validated"])
        self.assertEqual(seal["peak_descriptor_status"], "PEAK_DESCRIPTOR_NOT_VALIDATED")
        descriptors, missing = diagnostic.peak_descriptor_matrix(
            self.positions["P1"].signals[:50], CONFIG["peak_descriptor"]
        )
        self.assertEqual(descriptors.shape, (50, 12))
        self.assertEqual(missing.shape, (50, 4))

    def test_11_known_synthetic_peak_recovery(self) -> None:
        controls = diagnostic.synthetic_peak_controls(CONFIG["peak_descriptor"])
        self.assertEqual({row["check"] for row in controls}, {
            "clean_known_peak", "two_known_separated_peaks", "shifted_peaks",
            "broadened_peaks", "low_amplitude_peaks", "absent_peaks", "tied_local_maxima",
        })
        self.assertTrue(all(row["pass"] for row in controls))

    def test_12_within_block_peak_stability_calculation(self) -> None:
        data = self.positions["P1"]
        tag_id, er, surface = data.block_keys()[0]
        mask = (data.tag_ids == tag_id) & (data.er == er) & (data.surfaces == surface)
        descriptors, missing = diagnostic.peak_descriptor_matrix(
            data.signals[mask], CONFIG["peak_descriptor"]
        )
        self.assertEqual(len(descriptors), 50)
        self.assertTrue(np.isfinite(np.var(descriptors[:, 0::3], axis=0)).all())
        self.assertGreaterEqual(float(missing.mean()), 0.0)

    def test_13_common_probe_hyperparameter_binding(self) -> None:
        probe = CONFIG["linear_probe"]
        self.assertEqual(probe["family"], "FIXED_SOFTMAX_LINEAR_PROBE")
        self.assertEqual(probe["layers"], 1)
        self.assertEqual(CONFIG["probe_seeds"], CONFIG["checkpoint_seeds"])
        self.assertEqual(probe["validation_metric"], "row_macro_f1")

    def test_14_training_only_positive_control(self) -> None:
        features = np.eye(7, dtype=np.float64)
        labels = np.arange(7, dtype=np.int64)
        outcome = diagnostic.fit_learning_validity_control(
            features,
            labels,
            np.arange(7, dtype=np.int64),
            seed=42,
            probe_config=CONFIG["linear_probe"],
            validity_config=CONFIG["learning_validity"],
        )
        self.assertTrue(outcome["pass"])
        self.assertEqual(outcome["train_accuracy"], 1.0)

    def test_15_test_label_sealing(self) -> None:
        seal = diagnostic.TestLabelSeal(np.asarray([0, 1, 2]))
        with self.assertRaisesRegex(RuntimeError, "FAIL_TEST_LABEL_BOUNDARY"):
            seal.open_labels()
        seal.freeze_predictions(np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(seal.open_labels(), [0, 1, 2])
        self.assertEqual(
            seal.access_log,
            ("01_SEAL_CREATED", "02_PREDICTIONS_FROZEN_AND_HASHED", "03_TEST_LABELS_OPENED_ONCE"),
        )

    def test_16_deterministic_majority_vote(self) -> None:
        self.assertEqual(diagnostic.deterministic_majority_vote([2, 2, 1]), 2)

    def test_17_deterministic_tie_breaking(self) -> None:
        self.assertEqual(diagnostic.deterministic_majority_vote([4, 4, 2, 2]), 2)

    def test_18_block_macro_f1(self) -> None:
        truth = np.arange(7, dtype=np.int64)
        self.assertEqual(diagnostic.metrics(truth, truth)["macro_f1"], 1.0)

    def test_19_paired_representation_contrasts(self) -> None:
        truth = np.repeat(np.arange(7, dtype=np.int64), 9)
        perfect = {seed: truth.copy() for seed in CONFIG["probe_seeds"]}
        constant = {seed: np.zeros_like(truth) for seed in CONFIG["probe_seeds"]}
        rows, _ = diagnostic.paired_block_bootstrap(
            truth,
            {"LEFT": perfect, "RIGHT": constant},
            [("LEFT_MINUS_RIGHT", "LEFT", "RIGHT")],
            replicates=50,
            seed=1,
        )
        contrast = next(row for row in rows if row["target"] == "LEFT_MINUS_RIGHT")
        self.assertGreater(contrast["ci_low"], 0.0)

    def test_20_angle_associated_contrast(self) -> None:
        observed = diagnostic.angle_associated_contrast(
            {"P1": 0.8, "P2": 0.5, "P3": 0.6, "P4": 0.4}
        )
        self.assertAlmostEqual(observed, -0.25)

    def test_21_tagid_stratified_paired_bootstrap(self) -> None:
        truth = np.repeat(np.arange(7, dtype=np.int64), 9)
        first = diagnostic.stratified_block_resample_indices(truth, np.random.default_rng(5))
        second = diagnostic.stratified_block_resample_indices(truth, np.random.default_rng(5))
        np.testing.assert_array_equal(first, second)
        self.assertEqual([int(np.sum(truth[first] == item)) for item in range(7)], [9] * 7)

    def test_22_seed_resampling_sensitivity(self) -> None:
        truth = np.repeat(np.arange(7, dtype=np.int64), 9)
        folds = np.tile(np.arange(3, dtype=np.int64), 21)
        predictions = {
            name: {seed: truth.copy() for seed in CONFIG["probe_seeds"]}
            for name in ("RAW_SIGNAL_281", "FIRST_DIFFERENCE_280", "C1_ENCODER_EMBEDDING")
        }
        rows = diagnostic.seed_resampling_sensitivity(
            truth,
            predictions,
            [("RAW_MINUS_C1", "RAW_SIGNAL_281", "C1_ENCODER_EMBEDDING")],
            folds,
            replicates=25,
            seed=7,
        )
        self.assertEqual(
            {row["scheme"] for row in rows},
            {"block_plus_probe_seed", "block_plus_checkpoint_seed", "both_seed_dimensions", "fold_level_aggregate"},
        )

    def test_23_no_row_level_pseudoreplication(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "50-row group"):
            diagnostic.block_predictions(
                np.zeros(49, dtype=np.int64),
                np.zeros(49, dtype=np.int64),
                ["one-block"] * 49,
                expected_repeats=50,
            )
        self.assertEqual(CONFIG["bootstrap"]["sampling_unit"], "condition block within TagID")

if __name__ == "__main__":
    unittest.main()
