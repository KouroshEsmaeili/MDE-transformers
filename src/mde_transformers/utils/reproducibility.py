"""Utilities for reproducible pseudo-random number generation.

Deterministic algorithms can reduce performance and may reject operations for which PyTorch
does not provide a deterministic implementation. Seeding controls the generators in the current
process; data-loader workers and distributed processes must each initialize their own generators.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Runtime settings for pseudo-random seeding and deterministic PyTorch operations.

    Args:
        seed: Seed shared by Python, NumPy, and PyTorch. NumPy's global generator requires a
            value between 0 and ``2**32 - 1``.
        deterministic_algorithms: Ask PyTorch to use deterministic implementations where
            available. This can be slower and can raise at runtime for unsupported operations.
        warn_only: Warn instead of raising when a deterministic implementation is unavailable.
    """

    seed: int = 0
    deterministic_algorithms: bool = False
    warn_only: bool = False

    def apply(self) -> None:
        """Apply this configuration to the current process."""
        seed_everything(
            self.seed,
            deterministic_algorithms=self.deterministic_algorithms,
            warn_only=self.warn_only,
        )


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = False,
    warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    CUDA generators are seeded only when CUDA is available, so importing and calling this
    function is safe in a CPU-only environment. When deterministic algorithms are enabled,
    cuDNN benchmarking is disabled to avoid run-to-run algorithm selection differences.

    Args:
        seed: Integer in NumPy's supported seed range, ``0`` through ``2**32 - 1``.
        deterministic_algorithms: Enable PyTorch's deterministic algorithm mode.
        warn_only: Warn instead of raising for operations without a deterministic implementation.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` is outside NumPy's supported range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_NUMPY_SEED}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=warn_only)
    if deterministic_algorithms and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
