"""Canonical C1 adapter plus the single preregistered DANN domain head."""

from __future__ import annotations

import math

import torch
from torch import nn

from crfid.strict_runtime.neutral_model import initialize_model


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(inputs: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReversal.apply(inputs, float(strength))


class PositionDomainHead(nn.Module):
    """Fixed 256 -> 128 -> K source-position classifier."""

    def __init__(self, output_dimension: int, *, seed: int) -> None:
        super().__init__()
        if output_dimension not in {2, 3}:
            raise ValueError("The domain head must represent two LOPO or three final positions")
        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, output_dimension),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 9_000_001)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight,
                    a=0.0,
                    mode="fan_in",
                    nonlinearity="relu",
                    generator=generator,
                )
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


class C1PositionDANN(nn.Module):
    """Exact canonical C1 encoder/head, optionally augmented with a domain head."""

    def __init__(self, *, seed: int, domain_count: int | None) -> None:
        super().__init__()
        canonical = initialize_model(int(seed))
        self.encoder = nn.Sequential(*list(canonical.network.children())[:-1])
        self.tag_head = canonical.network[-1]
        self.domain_head = (
            PositionDomainHead(int(domain_count), seed=int(seed))
            if domain_count is not None
            else None
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tag_head(self.encode(inputs))

    def forward_dann(
        self, inputs: torch.Tensor, *, grl_strength: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.domain_head is None:
            raise RuntimeError("ERM has no active source-position domain head")
        embeddings = self.encode(inputs)
        tag_logits = self.tag_head(embeddings)
        domain_logits = self.domain_head(gradient_reverse(embeddings, grl_strength))
        return tag_logits, domain_logits, embeddings


def canonical_tag_state(model: C1PositionDANN) -> dict[str, torch.Tensor]:
    """State limited to the canonical encoder and TagID head."""

    return {
        **{f"encoder.{key}": value for key, value in model.encoder.state_dict().items()},
        **{f"tag_head.{key}": value for key, value in model.tag_head.state_dict().items()},
    }
