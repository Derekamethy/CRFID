"""Optimizer configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdamWConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("AdamW scalar parameters are invalid")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("AdamW beta values must lie in (0,1)")
