"""Analysis utilities kept separate from training and adaptation."""

from .embedding_analysis import separability_summary
from .failure_analysis import collapse_summary

__all__ = ["separability_summary", "collapse_summary"]
