"""Self-supervised objectives screened by this branch.

Four candidates, deliberately few, so that the source-only screen does not become a
large uncontrolled search:

* ``O1_MASKED_SPAN`` - contiguous span masking with reconstruction loss on the masked
  positions only, so full-signal identity reconstruction is not a trivial solution.
  Physics-aware family: recovering a masked span of a chipless-RFID response requires
  modelling resonance shape rather than local smoothness alone.
* ``O2_MULTI_VIEW_CONTRASTIVE`` - NT-Xent between two augmented views. Generic family.
* ``O3_RAW_DIFFERENCE_DUAL_VIEW`` - NT-Xent between the amplitude view and the
  first-difference (slope) view of the same window. Physics-aware family; the first
  difference is treated as a *second view*, not as an assumed superior representation.
* ``O4_HYBRID_MASKED_CONTRASTIVE`` - weighted sum of O1 and O2, admissible only if both
  are individually stable on the source-only screen.

None of these objectives uses any label, from Paper4 or from Data1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]

from .augmentations import AugmentationConfig, augment, contiguous_span_mask

O1_MASKED_SPAN = "O1_MASKED_SPAN"
O2_MULTI_VIEW_CONTRASTIVE = "O2_MULTI_VIEW_CONTRASTIVE"
O3_RAW_DIFFERENCE_DUAL_VIEW = "O3_RAW_DIFFERENCE_DUAL_VIEW"
O4_HYBRID_MASKED_CONTRASTIVE = "O4_HYBRID_MASKED_CONTRASTIVE"

OBJECTIVE_IDS = (
    O1_MASKED_SPAN,
    O2_MULTI_VIEW_CONTRASTIVE,
    O3_RAW_DIFFERENCE_DUAL_VIEW,
    O4_HYBRID_MASKED_CONTRASTIVE,
)

GENERIC_FAMILY = (O2_MULTI_VIEW_CONTRASTIVE,)
PHYSICS_AWARE_FAMILY = (
    O1_MASKED_SPAN,
    O3_RAW_DIFFERENCE_DUAL_VIEW,
    O4_HYBRID_MASKED_CONTRASTIVE,
)

REQUIRES_DECODER = (O1_MASKED_SPAN, O4_HYBRID_MASKED_CONTRASTIVE)
REQUIRES_PROJECTION = (
    O2_MULTI_VIEW_CONTRASTIVE,
    O3_RAW_DIFFERENCE_DUAL_VIEW,
    O4_HYBRID_MASKED_CONTRASTIVE,
)


@dataclass(frozen=True)
class ObjectiveConfig:
    objective_id: str
    mask_span_count: int = 4
    mask_span_length: int = 16
    temperature: float = 0.2
    hybrid_contrastive_weight: float = 0.5
    augmentation: AugmentationConfig = AugmentationConfig()

    def as_record(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "family": "GENERIC" if self.objective_id in GENERIC_FAMILY else "PHYSICS_AWARE",
            "mask_span_count": self.mask_span_count,
            "mask_span_length": self.mask_span_length,
            "masked_fraction_of_window": self.mask_span_count * self.mask_span_length,
            "temperature": self.temperature,
            "hybrid_contrastive_weight": self.hybrid_contrastive_weight,
            "augmentation": self.augmentation.as_record(),
            "uses_any_label": False,
        }


def nt_xent(first: Any, second: Any, temperature: float) -> Any:
    """Normalised temperature-scaled cross entropy over ``2B`` views."""

    batch = first.shape[0]
    embeddings = functional.normalize(torch.cat([first, second], dim=0), dim=1)
    similarity = embeddings @ embeddings.t() / float(temperature)
    similarity.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        [torch.arange(batch, 2 * batch), torch.arange(0, batch)], dim=0
    ).to(similarity.device)
    return functional.cross_entropy(similarity, targets)


def compute_loss(
    *,
    config: ObjectiveConfig,
    backbone: Any,
    projection: Any,
    decoder: Any,
    difference_window: np.ndarray,
    raw_window: np.ndarray,
    generator: np.random.Generator,
) -> Any:
    """Return the differentiable SSL loss for one batch.

    ``difference_window`` and ``raw_window`` are standardized ``[B,1,W]`` arrays cropped
    from the same positions of the same samples.
    """

    objective = config.objective_id
    if objective == O1_MASKED_SPAN:
        return _masked_span_loss(config, backbone, decoder, difference_window, generator)
    if objective == O2_MULTI_VIEW_CONTRASTIVE:
        return _contrastive_loss(config, backbone, projection, difference_window, generator)
    if objective == O3_RAW_DIFFERENCE_DUAL_VIEW:
        return _dual_view_loss(config, backbone, projection, difference_window, raw_window, generator)
    if objective == O4_HYBRID_MASKED_CONTRASTIVE:
        masked = _masked_span_loss(config, backbone, decoder, difference_window, generator)
        contrastive = _contrastive_loss(config, backbone, projection, difference_window, generator)
        weight = float(config.hybrid_contrastive_weight)
        return (1.0 - weight) * masked + weight * contrastive
    raise ValueError(f"Unknown SSL objective: {objective}")


def _masked_span_loss(
    config: ObjectiveConfig,
    backbone: Any,
    decoder: Any,
    difference_window: np.ndarray,
    generator: np.random.Generator,
) -> Any:
    batch, _, length = difference_window.shape
    mask = contiguous_span_mask(
        batch,
        length,
        span_count=config.mask_span_count,
        span_length=config.mask_span_length,
        generator=generator,
    )
    target = torch.from_numpy(np.ascontiguousarray(difference_window[:, 0, :], dtype=np.float32))
    corrupted = np.array(difference_window, dtype=np.float32, copy=True)
    corrupted[:, 0, :][mask] = 0.0
    inputs = torch.from_numpy(corrupted)
    reconstruction = decoder(backbone.feature_map(inputs))
    if reconstruction.shape != target.shape:
        reconstruction = functional.interpolate(
            reconstruction.unsqueeze(1), size=target.shape[1], mode="linear", align_corners=False
        ).squeeze(1)
    selector = torch.from_numpy(mask)
    if not bool(selector.any()):  # pragma: no cover - configuration guarantees spans
        raise ValueError("Masked-span objective produced an empty mask")
    return functional.mse_loss(reconstruction[selector], target[selector])


def _contrastive_loss(
    config: ObjectiveConfig,
    backbone: Any,
    projection: Any,
    difference_window: np.ndarray,
    generator: np.random.Generator,
) -> Any:
    view_one = torch.from_numpy(augment(difference_window, config.augmentation, generator))
    view_two = torch.from_numpy(augment(difference_window, config.augmentation, generator))
    return nt_xent(
        projection(backbone.encode(view_one)),
        projection(backbone.encode(view_two)),
        config.temperature,
    )


def _dual_view_loss(
    config: ObjectiveConfig,
    backbone: Any,
    projection: Any,
    difference_window: np.ndarray,
    raw_window: np.ndarray,
    generator: np.random.Generator,
) -> Any:
    difference_view = torch.from_numpy(augment(difference_window, config.augmentation, generator))
    raw_view = torch.from_numpy(augment(raw_window, config.augmentation, generator))
    return nt_xent(
        projection(backbone.encode(difference_view)),
        projection(backbone.encode(raw_view)),
        config.temperature,
    )
