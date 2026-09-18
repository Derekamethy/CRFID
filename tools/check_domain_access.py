from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crfid.exceptions import DomainAccessViolation
from crfid.protocols.common import AccessRequest, Purpose, Resource
from crfid.protocols.few_shot import FewShotProtocol
from crfid.protocols.strict_dg import StrictDGProtocol


def main() -> int:
    checks: dict[str, bool] = {}
    strict = StrictDGProtocol()
    try:
        strict.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION))
        checks["strict_target_blocked_before_freeze"] = False
    except DomainAccessViolation:
        checks["strict_target_blocked_before_freeze"] = True
    recipe = {"method": "first_difference"}
    seal = strict.freeze_recipe(recipe)
    strict.authorize_final_evaluation(recipe, seal)
    strict.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION))
    checks["strict_target_allowed_after_freeze"] = True
    few_shot = FewShotProtocol()
    try:
        few_shot.authorize(AccessRequest("P4", Resource.LABELS, Purpose.ADAPTATION, "query"))
        checks["few_shot_query_labels_sealed"] = False
    except DomainAccessViolation:
        checks["few_shot_query_labels_sealed"] = True
    print(json.dumps({"checks": checks, "passed": all(checks.values())}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
