"""Target-assisted and few-shot adaptation methods."""

from .prototypes import build_target_prototypes, squared_euclidean_predict

__all__ = ["build_target_prototypes", "squared_euclidean_predict"]
