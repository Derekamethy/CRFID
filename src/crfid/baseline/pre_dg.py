"""Convenience exports for the complete Pre-DG implementation."""

from .aggregation import aggregate_seed_results, compare_with_authoritative
from .evaluation import evaluate_predictions, load_predictions, save_predictions
from .model import PreDGDeepCNN, build_pre_dg_model
from .preprocessing import TrainOnlyChannelScaler
from .training import (
    CandidateTraining,
    predict_reloaded_model,
    reload_checkpoint_model,
    select_candidate,
    train_candidate,
)

__all__ = [
    "CandidateTraining",
    "PreDGDeepCNN",
    "TrainOnlyChannelScaler",
    "aggregate_seed_results",
    "build_pre_dg_model",
    "compare_with_authoritative",
    "evaluate_predictions",
    "load_predictions",
    "predict_reloaded_model",
    "reload_checkpoint_model",
    "save_predictions",
    "select_candidate",
    "train_candidate",
]
