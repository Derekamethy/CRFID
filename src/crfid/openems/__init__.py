"""Assumption-bounded physical redesign helpers without import-time solver access."""

from .confirmation import assess_confirmation
from .geometry import candidate_geometry

__all__ = ["assess_confirmation", "candidate_geometry"]
