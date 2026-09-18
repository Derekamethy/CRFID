"""Larger P4 labelled-target calibration curve."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FACTOR_AWARE_WORKFLOW = REPOSITORY_ROOT / "workflows/11_p4_factor_aware_few_shot"
if str(FACTOR_AWARE_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(FACTOR_AWARE_WORKFLOW))
