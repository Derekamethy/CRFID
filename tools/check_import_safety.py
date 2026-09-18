from __future__ import annotations

import importlib
import json
import pkgutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import crfid


def main() -> int:
    modules = sorted(module.name for module in pkgutil.walk_packages(crfid.__path__, crfid.__name__ + "."))
    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"module_count": len(modules), "failures": failures, "passed": not failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
