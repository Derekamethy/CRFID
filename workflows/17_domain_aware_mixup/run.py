"""Run the isolated condition-matched domain-aware manifold mixup benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.domain_aware_mixup.runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
