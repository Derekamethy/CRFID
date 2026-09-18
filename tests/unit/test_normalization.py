from __future__ import annotations

import unittest

import numpy as np

from crfid.preprocessing.normalization import Standardizer


class NormalizationTests(unittest.TestCase):
    def test_population_standardization(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 6.0]])
        result = Standardizer.fit(values).transform(values, dtype=np.float64)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-12)

    def test_constant_feature_uses_unit_scale(self) -> None:
        state = Standardizer.fit(np.ones((3, 2)))
        np.testing.assert_array_equal(state.scale, np.ones(2))


if __name__ == "__main__":
    unittest.main()
