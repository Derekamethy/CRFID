"""Prediction-first held-out-label access for External Data1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import DomainAccessViolation, ProtocolViolation
from ..external_data1.artifacts import canonical_json_sha256
from .integrity import sha256_file


class ExternalLabelVault:
    """Seal test labels until a run's predictions exist and are hashed."""

    def __init__(
        self,
        labels: np.ndarray,
        sample_ids: tuple[str, ...],
        *,
        protocol_sha256: str,
    ) -> None:
        values = np.ascontiguousarray(labels, dtype=np.int64)
        if values.shape != (len(sample_ids),):
            raise ValueError("Sealed labels and sample identifiers are not aligned")
        self.__labels = values
        self.__sample_ids = sample_ids
        self.protocol_sha256 = protocol_sha256
        self._candidate_registry_sha256: str | None = None
        self._preprocessing: dict[str, str] = {}
        self._models: dict[str, dict[str, Any]] = {}
        self._predictions: dict[str, dict[str, Any]] = {}
        self._accessed: set[str] = set()
        self.log_rows: list[dict[str, Any]] = []

    def freeze_candidate_registry(self, registry: dict[str, Any]) -> str:
        if self._candidate_registry_sha256 is not None:
            raise ProtocolViolation("Candidate registry was already frozen")
        self._candidate_registry_sha256 = canonical_json_sha256(registry)
        return self._candidate_registry_sha256

    def register_preprocessing(
        self, method: str, state_sha256: str, *, fit_partition: str
    ) -> None:
        if self._candidate_registry_sha256 is None:
            raise ProtocolViolation("Candidate registry must be frozen first")
        if fit_partition != "set_3":
            raise DomainAccessViolation("Preprocessing may be fit only on set_3")
        self._preprocessing[method] = state_sha256

    def register_frozen_model(
        self,
        run_id: str,
        *,
        method: str,
        model_sha256: str,
        preprocessing_sha256: str | None,
    ) -> None:
        if self._candidate_registry_sha256 is None:
            raise ProtocolViolation("Candidate registry must be frozen first")
        if preprocessing_sha256 is not None:
            if self._preprocessing.get(method) != preprocessing_sha256:
                raise ProtocolViolation("Model does not reference frozen preprocessing")
        self._models[run_id] = {
            "method": method,
            "model_sha256": model_sha256,
            "preprocessing_sha256": preprocessing_sha256,
        }

    def freeze_predictions(self, run_id: str, path: str | Path) -> dict[str, Any]:
        if run_id not in self._models:
            raise ProtocolViolation("Model or deterministic rule must be frozen first")
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            if "labels" in payload.files or "truth" in payload.files:
                raise ProtocolViolation("Prediction artifact must precede test-label access")
            required = {"sample_ids", "predictions", "scores", "measurement_sets"}
            if not required.issubset(payload.files):
                raise ProtocolViolation("Prediction artifact is incomplete")
            sample_ids = tuple(str(value) for value in payload["sample_ids"])
            predictions = np.asarray(payload["predictions"], dtype=np.int64)
            if sample_ids != self.__sample_ids or predictions.shape != self.__labels.shape:
                raise ProtocolViolation("Prediction order does not match the sealed test order")
        record = {
            "run_id": run_id,
            "prediction_sha256": sha256_file(source),
            "sample_count": int(predictions.size),
            "frozen_before_label_access": True,
        }
        self._predictions[run_id] = record
        self.log_rows.append(
            {
                "sequence": len(self.log_rows) + 1,
                "run_id": run_id,
                "event": "PREDICTIONS_FROZEN",
                "prediction_sha256": record["prediction_sha256"],
                "selection_influence": False,
            }
        )
        return record

    def labels_for_scoring(self, run_id: str) -> np.ndarray:
        if run_id not in self._predictions:
            raise DomainAccessViolation(
                "Test labels remain sealed until predictions are serialized and hashed"
            )
        if run_id in self._accessed:
            raise DomainAccessViolation("Each run receives one final test-label access")
        self._accessed.add(run_id)
        self.log_rows.append(
            {
                "sequence": len(self.log_rows) + 1,
                "run_id": run_id,
                "event": "TEST_LABELS_UNSEALED_FOR_FINAL_SCORING",
                "prediction_sha256": self._predictions[run_id]["prediction_sha256"],
                "selection_influence": False,
            }
        )
        return self.__labels.copy()

    @property
    def scored_run_count(self) -> int:
        return len(self._accessed)

