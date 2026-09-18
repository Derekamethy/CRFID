"""Frozen training-only PCA plus multinomial logistic reference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PcaLogisticMetadata:
    retained_components: int
    explained_variance_ratio_sum: float
    converged: bool
    n_iter: tuple[int, ...]
    warnings: tuple[dict[str, str], ...]


def fit_and_save(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    state_path: str | Path,
) -> PcaLogisticMetadata:
    """Fit the frozen estimator on set 3 only and serialize all fitted state."""

    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    inputs = np.asarray(train_inputs, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(inputs)
    pca = PCA(n_components=0.95, svd_solver="full")
    projected = pca.fit_transform(scaled)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        classifier.fit(projected, labels)
    destination = Path(state_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        scaler_mean=np.asarray(scaler.mean_, dtype="<f8"),
        scaler_scale=np.asarray(scaler.scale_, dtype="<f8"),
        scaler_var=np.asarray(scaler.var_, dtype="<f8"),
        pca_mean=np.asarray(pca.mean_, dtype="<f8"),
        pca_components=np.asarray(pca.components_, dtype="<f8"),
        pca_explained_variance=np.asarray(pca.explained_variance_, dtype="<f8"),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype="<f8"
        ),
        pca_singular_values=np.asarray(pca.singular_values_, dtype="<f8"),
        classes=np.asarray(classifier.classes_, dtype="<i8"),
        coefficients=np.asarray(classifier.coef_, dtype="<f8"),
        intercept=np.asarray(classifier.intercept_, dtype="<f8"),
        n_iter=np.asarray(classifier.n_iter_, dtype="<i8"),
    )
    warning_rows = tuple(
        {"category": item.category.__name__, "message": str(item.message)}
        for item in captured
    )
    return PcaLogisticMetadata(
        retained_components=int(pca.n_components_),
        explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
        converged=not any(issubclass(item.category, ConvergenceWarning) for item in captured),
        n_iter=tuple(int(value) for value in classifier.n_iter_),
        warnings=warning_rows,
    )


def predict_from_saved_state(
    state_path: str | Path, test_inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reload the frozen estimator state before held-out inference."""

    inputs = np.asarray(test_inputs, dtype=np.float64)
    with np.load(Path(state_path), allow_pickle=False) as payload:
        scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        pca_mean = np.asarray(payload["pca_mean"], dtype=np.float64)
        components = np.asarray(payload["pca_components"], dtype=np.float64)
        classes = np.asarray(payload["classes"], dtype=np.int64)
        coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        intercept = np.asarray(payload["intercept"], dtype=np.float64)
    scaled = (inputs - scaler_mean) / scaler_scale
    projected = (scaled - pca_mean) @ components.T
    logits = projected @ coefficients.T + intercept
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    predictions = classes[np.argmax(scores, axis=1)]
    return (
        np.ascontiguousarray(predictions, dtype=np.int64),
        np.ascontiguousarray(scores, dtype=np.float64),
    )


def specification() -> dict[str, Any]:
    return {
        "input": "1601_raw_ordered_signal_values",
        "standard_scaler": {"fit_partition": "set_3", "with_mean": True, "with_std": True},
        "pca": {"n_components": 0.95, "svd_solver": "full", "fit_partition": "set_3"},
        "logistic_regression": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": None,
            "random_state": 42,
            "multiclass_behavior": "scikit-learn-1.9-default-multinomial",
        },
        "hyperparameter_tuning": False,
        "test_fit": False,
    }

