"""Single canonical definition of the Strict-DG degenerate-scale rule.

The external audit found the low-variance fallback implemented two different
ways across four call sites while the configuration declared both a
``minimum_scale`` of 1e-12 and a ``minimum_scale_replacement`` of 1.0:

* ``replace_with_one``  -- ``where(std < 1e-12, 1.0, std)``
  used by the canonical all-source preprocessing
  (``scripts/train_and_freeze_strict_dg.py``) and by the raw source
  preprocessing (``neutral_data._fit_raw_preprocessing``);
* ``clip_at_minimum``   -- ``clip(std, 1e-12, None)``
  used by the C1 selection-stage preprocessing
  (``phase3b_execution.fit_first_difference_preprocessing``).

The two rules differ only for features whose training standard deviation falls
below 1e-12. On the canonical Strict-DG data the smallest observed feature
standard deviation is approximately 0.35, so the fallback is never taken and the
two rules produce bitwise identical scale vectors. The historical behaviour of
every call site is therefore preserved exactly by passing its existing mode.

``CANONICAL_MODE`` is the rule to use for any future Strict-DG execution. It is
recorded in the comprehensive release seal so that the choice is sealed rather
than implicit.
"""

from __future__ import annotations

import numpy as np


MINIMUM_SCALE = 1e-12
MINIMUM_SCALE_REPLACEMENT = 1.0

MODE_REPLACE_WITH_ONE = "replace_with_one"
MODE_CLIP_AT_MINIMUM = "clip_at_minimum"
SUPPORTED_MODES = (MODE_REPLACE_WITH_ONE, MODE_CLIP_AT_MINIMUM)

#: The rule future Strict-DG execution must use. Chosen to match the canonical
#: all-source preprocessing that produced the five released final checkpoints.
CANONICAL_MODE = MODE_REPLACE_WITH_ONE


def apply_scale_policy(raw_scale: np.ndarray, *, mode: str) -> np.ndarray:
    """Return the standardization denominator under an explicit, declared mode."""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown degenerate-scale mode: {mode!r}; expected one of {SUPPORTED_MODES}")
    values = np.ascontiguousarray(raw_scale, dtype=np.float64)
    if mode == MODE_REPLACE_WITH_ONE:
        return np.ascontiguousarray(
            np.where(values < MINIMUM_SCALE, MINIMUM_SCALE_REPLACEMENT, values)
        )
    return np.ascontiguousarray(np.clip(values, MINIMUM_SCALE, None))


def fallback_report(raw_scale: np.ndarray) -> dict:
    """Describe whether the low-variance fallback is reachable on real data."""

    values = np.ascontiguousarray(raw_scale, dtype=np.float64)
    triggered = int(np.count_nonzero(values < MINIMUM_SCALE))
    replace = apply_scale_policy(values, mode=MODE_REPLACE_WITH_ONE)
    clip = apply_scale_policy(values, mode=MODE_CLIP_AT_MINIMUM)
    return {
        "feature_count": int(values.size),
        "minimum_observed_standard_deviation": float(values.min()) if values.size else None,
        "maximum_observed_standard_deviation": float(values.max()) if values.size else None,
        "minimum_scale_threshold": MINIMUM_SCALE,
        "features_below_threshold": triggered,
        "fallback_triggered": triggered > 0,
        "modes_agree_bitwise_on_this_data": bool(np.array_equal(replace, clip)),
    }
