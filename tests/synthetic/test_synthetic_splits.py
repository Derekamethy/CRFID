from __future__ import annotations

import unittest

from crfid.exceptions import SplitIsolationError
from crfid.governance.split_validation import assert_disjoint_groups


class SyntheticSplitTests(unittest.TestCase):
    def test_disjoint_groups_pass(self) -> None:
        assert_disjoint_groups({"train": {"a", "b"}, "test": {"c"}})

    def test_overlap_fails(self) -> None:
        with self.assertRaises(SplitIsolationError):
            assert_disjoint_groups({"support": {"a"}, "query": {"a"}})


if __name__ == "__main__":
    unittest.main()
