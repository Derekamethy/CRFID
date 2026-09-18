from __future__ import annotations

import unittest

from crfid.models.cnn1d import BASELINE_SPEC, STRICT_SPEC


class ModelContractTests(unittest.TestCase):
    def test_strict_spec(self) -> None:
        self.assertEqual((STRICT_SPEC.input_channels, STRICT_SPEC.input_length, STRICT_SPEC.class_count), (1, 280, 7))

    def test_baseline_spec(self) -> None:
        self.assertEqual((BASELINE_SPEC.input_channels, BASELINE_SPEC.input_length, BASELINE_SPEC.class_count), (2, 512, 7))


if __name__ == "__main__":
    unittest.main()
