"""Deterministic seeding for the fine-tuning runs.

Seed scheme (FINE_TUNING_PLAN §8):
  * global base seed (config) for torch/numpy/cuda,
  * per-outer-fold derived seed = base + fold_index,
  * per-permutation seed = base + permutation_offset + perm_index.
Full CUDA determinism (slower, and some SGConv/FFT ops lack deterministic
kernels) is opt-in via ``deterministic``.
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

log = logging.getLogger(__name__)

__all__ = ["seed_everything", "fold_seed"]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fully enable deterministic algorithms: %s", exc)
    else:
        torch.backends.cudnn.benchmark = True


def fold_seed(base: int, fold_index: int) -> int:
    return base + fold_index
