"""Run the isolated IRMv1 strict source-only benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.irm_invariant_risk.runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
