"""Run the isolated GroupDRO worst-source-position benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.groupdro_worst_source.runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
