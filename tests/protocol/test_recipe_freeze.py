from __future__ import annotations

import unittest

from crfid.governance.recipe_freeze import create_recipe_seal, verify_recipe_seal


class RecipeFreezeTests(unittest.TestCase):
    def test_seal_verifies_identical_recipe(self) -> None:
        recipe = {"seed": 42, "model": {"name": "cnn"}}
        self.assertTrue(verify_recipe_seal(recipe, create_recipe_seal(recipe)))

    def test_seal_rejects_changed_recipe(self) -> None:
        seal = create_recipe_seal({"seed": 42})
        self.assertFalse(verify_recipe_seal({"seed": 43}, seal))


if __name__ == "__main__":
    unittest.main()
