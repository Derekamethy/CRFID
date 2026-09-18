"""Explicit training utilities; no training occurs on import."""

from .reproducibility import set_deterministic_seed

__all__ = ["set_deterministic_seed"]
