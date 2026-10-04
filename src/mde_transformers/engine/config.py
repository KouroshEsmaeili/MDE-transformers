"""Small immutable configuration for supervised optimization loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

DeviceSpec = Literal["auto", "cpu", "cuda"]
_MAX_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training-loop settings, deliberately separate from model and dataset configuration."""

    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: DeviceSpec = "auto"
    seed: int = 0
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        _validate_positive_integer(self.epochs, "epochs")
        _validate_positive_integer(self.batch_size, "batch_size")
        _validate_positive_real(self.learning_rate, "learning_rate")
        _validate_non_negative_real(self.weight_decay, "weight_decay")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= _MAX_SEED:
            raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
        if self.gradient_clip_norm is not None:
            _validate_positive_real(self.gradient_clip_norm, "gradient_clip_norm")


def resolve_device(device: DeviceSpec) -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise.

    An explicit unavailable CUDA request raises instead of silently changing the experiment.
    """
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_positive_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_non_negative_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
