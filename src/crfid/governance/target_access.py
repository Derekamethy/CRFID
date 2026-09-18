"""Machine-enforced access contract for P4 Target-Assisted Adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..exceptions import TargetAccessViolation
from ..protocols.common import AccessRequest, Purpose, Resource


@dataclass(frozen=True)
class TargetAccessContract:
    allow_features: bool
    allow_labels: bool
    allowed_adaptation_partition: str
    allowed_selection_partition: str
    allowed_evaluation_partition: str
    metrics_may_influence_selection: bool
    source_parameters_may_update: bool
    layers_may_update: tuple[str, ...]
    source_preprocessing_may_refit: bool
    target_preprocessing_may_fit: bool
    transductive_query_features_allowed: bool
    query_labels_allowed: bool
    final_claim_type: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TargetAccessContract":
        required = {
            "allow_features",
            "allow_labels",
            "allowed_adaptation_partition",
            "allowed_selection_partition",
            "allowed_evaluation_partition",
            "metrics_may_influence_selection",
            "source_parameters_may_update",
            "layers_may_update",
            "source_preprocessing_may_refit",
            "target_preprocessing_may_fit",
            "transductive_query_features_allowed",
            "query_labels_allowed",
            "final_claim_type",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise TargetAccessViolation(f"Target-access fields missing: {missing}")
        return cls(
            allow_features=bool(value["allow_features"]),
            allow_labels=bool(value["allow_labels"]),
            allowed_adaptation_partition=str(value["allowed_adaptation_partition"]),
            allowed_selection_partition=str(value["allowed_selection_partition"]),
            allowed_evaluation_partition=str(value["allowed_evaluation_partition"]),
            metrics_may_influence_selection=bool(value["metrics_may_influence_selection"]),
            source_parameters_may_update=bool(value["source_parameters_may_update"]),
            layers_may_update=tuple(str(item) for item in value["layers_may_update"]),
            source_preprocessing_may_refit=bool(value["source_preprocessing_may_refit"]),
            target_preprocessing_may_fit=bool(value["target_preprocessing_may_fit"]),
            transductive_query_features_allowed=bool(value["transductive_query_features_allowed"]),
            query_labels_allowed=bool(value["query_labels_allowed"]),
            final_claim_type=str(value["final_claim_type"]),
        )

    def authorize(self, request: AccessRequest) -> None:
        if request.domain != "P4":
            return
        partition = request.partition or ""
        allowed = {
            Purpose.ADAPTATION: self.allowed_adaptation_partition,
            Purpose.SELECTION: self.allowed_selection_partition,
            Purpose.EVALUATION: self.allowed_evaluation_partition,
        }.get(request.purpose)
        if allowed is not None and partition != allowed:
            raise TargetAccessViolation(f"Partition {partition!r} is not authorized for {request.purpose.value}")
        if request.resource is Resource.FEATURES and not self.allow_features:
            raise TargetAccessViolation("P4 features are forbidden")
        if request.resource is Resource.LABELS:
            if not self.allow_labels or (request.purpose is Purpose.EVALUATION and not self.query_labels_allowed):
                raise TargetAccessViolation("P4 labels are forbidden at this stage")
        if request.resource is Resource.OUTCOMES:
            if request.purpose is Purpose.SELECTION and not self.metrics_may_influence_selection:
                raise TargetAccessViolation("P4 metrics cannot influence method selection")

    def authorize_parameter_update(self, parameter_names: tuple[str, ...]) -> None:
        if parameter_names and not self.source_parameters_may_update:
            raise TargetAccessViolation("Source parameter updates are forbidden")
        disallowed = sorted(set(parameter_names) - set(self.layers_may_update))
        if disallowed:
            raise TargetAccessViolation(f"Layers are not authorized for update: {disallowed}")
