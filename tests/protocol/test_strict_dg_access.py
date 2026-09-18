from __future__ import annotations

import unittest

from crfid.exceptions import DomainAccessViolation, RecipeFreezeError
from crfid.protocols.common import AccessRequest, Purpose, Resource
from crfid.protocols.strict_dg import StrictDGProtocol


class StrictAccessTests(unittest.TestCase):
    def test_target_blocked_before_freeze(self) -> None:
        protocol = StrictDGProtocol()
        with self.assertRaises(DomainAccessViolation):
            protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION))

    def test_target_allowed_only_after_matching_freeze(self) -> None:
        protocol = StrictDGProtocol()
        recipe = {"method": "first_difference"}
        seal = protocol.freeze_recipe(recipe)
        protocol.authorize_final_evaluation(recipe, seal)
        protocol.authorize(AccessRequest("P4", Resource.LABELS, Purpose.EVALUATION))
        with self.assertRaises(DomainAccessViolation):
            protocol.authorize(AccessRequest("P4", Resource.LABELS, Purpose.SELECTION))

    def test_changed_recipe_rejected(self) -> None:
        protocol = StrictDGProtocol()
        seal = protocol.freeze_recipe({"method": "a"})
        with self.assertRaises(RecipeFreezeError):
            protocol.authorize_final_evaluation({"method": "b"}, seal)


if __name__ == "__main__":
    unittest.main()
