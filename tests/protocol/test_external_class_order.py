from __future__ import annotations

import unittest

import numpy as np

from crfid.data.data1 import CLASS_ORDER, deterministic_majority_label


class ExternalClassTests(unittest.TestCase):
    def test_fixed_class_order(self) -> None:
        self.assertEqual(CLASS_ORDER, (0, 1, 2, 3))

    def test_majority_tie_uses_lowest_class(self) -> None:
        self.assertEqual(deterministic_majority_label(np.asarray([0, 1, 0, 1])), 0)


if __name__ == "__main__":
    unittest.main()
