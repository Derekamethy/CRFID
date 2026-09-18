from __future__ import annotations

import unittest

import numpy as np

from crfid.models.cnn1d import NumpyNeutralCNN1D


class SyntheticForwardTests(unittest.TestCase):
    def test_dummy_tensor_forward_shape_and_finiteness(self) -> None:
        model = NumpyNeutralCNN1D(seed=42)
        output = model.forward(np.zeros((2, 1, 32), dtype=np.float32))
        self.assertEqual(output.shape, (2, 7))
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
