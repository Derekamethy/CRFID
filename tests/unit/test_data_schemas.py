from __future__ import annotations

import unittest

import numpy as np

from crfid.data.schemas import SampleMetadata, SignalRecord
from crfid.data.tyndall import SCHEMA, map_raw_tag_label
from crfid.exceptions import DataValidationError


class DataSchemaTests(unittest.TestCase):
    def test_valid_record(self) -> None:
        metadata = SampleMetadata("s1", "tyndall", 0, "P1", "c1")
        SignalRecord(metadata, np.zeros(281)).validate(SCHEMA)

    def test_invalid_label_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            map_raw_tag_label(0)


if __name__ == "__main__":
    unittest.main()
