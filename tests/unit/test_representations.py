from __future__ import annotations

import unittest

import numpy as np

from crfid.preprocessing.representations import first_difference, two_channel_raw_gradient


class RepresentationTests(unittest.TestCase):
    def test_first_difference_has_no_padding(self) -> None:
        result = first_difference(np.asarray([[1.0, 4.0, 9.0]]))
        np.testing.assert_array_equal(result, [[3.0, 5.0]])

    def test_two_channel_shape(self) -> None:
        result = two_channel_raw_gradient(np.arange(12, dtype=np.float32).reshape(3, 4))
        self.assertEqual(result.shape, (3, 2, 4))


if __name__ == "__main__":
    unittest.main()
