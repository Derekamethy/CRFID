"""Deterministic seed utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def set_deterministic_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
