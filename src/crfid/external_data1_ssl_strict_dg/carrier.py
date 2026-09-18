"""Signal carrier shared by Paper4 and Data1.

Carrier decision (see ``04_DATA_COMPATIBILITY_AND_CARRIER_DECISION.md``)

Neither dataset has a resolved physical frequency axis in current v2 evidence
(Tyndall: ``PHYSICAL_FREQUENCY_AXIS_PROVENANCE_UNRESOLVED``; Data1:
``AXIS_IDENTITY_UNRESOLVED``). A shared physical 5-8 GHz grid therefore cannot be
asserted, and normalised-index resampling would silently claim that position
``i/L`` means the same thing in both datasets. This branch instead uses an
**ordered-position variable-length carrier**:

* representation: first difference without padding (Paper4 281 -> 280, Data1 1601 -> 1600),
  identical to the canonical Strict-DG representation;
* normalisation: featurewise population standardisation fitted per dataset on its own
  declared training partition, so raw amplitude scale cannot act as a dataset tag;
* self-supervised stage: both datasets are randomly cropped to the *same* window
  length, so sequence length cannot act as a trivial dataset identifier and no padding
  exists anywhere in the branch;
* supervised stage and final evaluation: Paper4 is used at its full 280 positions with
  no crop, byte-for-byte the canonical Strict-DG input.

The backbone is length-agnostic (``AdaptiveAvgPool1d(1)``), so this needs no
architecture change relative to the canonical anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MINIMUM_SCALE = 1e-12
MINIMUM_SCALE_REPLACEMENT = 1.0
SSL_WINDOW_LENGTH = 256


def first_difference(signals: np.ndarray) -> np.ndarray:
    """``x[i+1] - x[i]`` without padding, matching the canonical representation."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("First difference requires finite [N,L] signals with L>=2")
    return np.ascontiguousarray(np.diff(values, axis=1))


@dataclass(frozen=True)
class FeatureStandardizer:
    """Featurewise population standardisation, ddof=0, float64 fit."""

    mean: np.ndarray
    scale: np.ndarray
    zero_scale_replacement_count: int

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureStandardizer":
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] == 0 or not np.isfinite(data).all():
            raise ValueError("Standardizer fit requires finite non-empty [N,F] data")
        mean = data.mean(axis=0, dtype=np.float64)
        raw_scale = data.std(axis=0, ddof=0, dtype=np.float64)
        degenerate = raw_scale < MINIMUM_SCALE
        scale = np.where(degenerate, MINIMUM_SCALE_REPLACEMENT, raw_scale)
        return cls(
            np.ascontiguousarray(mean),
            np.ascontiguousarray(scale),
            int(degenerate.sum()),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.mean.shape[0]:
            raise ValueError("Transform features do not match the fitted state")
        out = (data - self.mean) / self.scale
        if not np.isfinite(out).all():
            raise ValueError("Non-finite standardized value")
        return np.ascontiguousarray(out, dtype=np.float32)


def to_model_input(values: np.ndarray) -> np.ndarray:
    """Add the single channel axis expected by the backbone."""

    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("Model inputs are built from [N,L] arrays")
    return np.ascontiguousarray(data[:, None, :])


def prepare_supervised(
    raw_signals: np.ndarray, standardizer: FeatureStandardizer
) -> np.ndarray:
    """Canonical supervised pipeline: first difference -> standardize -> [N,1,L]."""

    return to_model_input(standardizer.transform(first_difference(raw_signals)))


def fit_supervised_standardizer(raw_signals: np.ndarray) -> FeatureStandardizer:
    return FeatureStandardizer.fit(first_difference(raw_signals))


def carrier_declaration() -> dict[str, object]:
    return {
        "carrier": "ORDERED_POSITION_VARIABLE_LENGTH",
        "physical_frequency_grid_asserted": False,
        "normalized_index_resampling_used": False,
        "padding_used": False,
        "representation": "first_difference_without_padding",
        "normalization": "featurewise_population_standardization_per_dataset",
        "ddof": 0,
        "minimum_scale": MINIMUM_SCALE,
        "minimum_scale_replacement": MINIMUM_SCALE_REPLACEMENT,
        "ssl_common_window_length": SSL_WINDOW_LENGTH,
        "supervised_paper4_length": 280,
        "ssl_data1_length": 1600,
        "length_is_a_trivial_dataset_identifier_during_ssl": False,
        "backbone_length_agnostic": True,
    }
