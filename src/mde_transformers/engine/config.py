"""Small immutable configuration for supervised optimization loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

DeviceSpec = Literal["auto", "cpu", "cuda"]
EncoderPolicy = Literal["frozen", "trainable"]
StageWeights = tuple[float, float, float, float]
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
    encoder_policy: EncoderPolicy = "frozen"
    gradient_clip_norm: float | None = None
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1
    stage_weights: StageWeights = (0.125, 0.25, 0.5, 1.0)

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
        if self.encoder_policy not in ("frozen", "trainable"):
            raise ValueError("encoder_policy must be 'frozen' or 'trainable'")
        if self.gradient_clip_norm is not None:
            _validate_positive_real(self.gradient_clip_norm, "gradient_clip_norm")
        _validate_non_negative_integer(self.warmup_epochs, "warmup_epochs")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        _validate_warmup_factor(self.warmup_start_factor)
        _validate_stage_weights(self.stage_weights)


@dataclass(frozen=True, slots=True)
class NYUExperimentConfig:
    """Minimal reproducible NYU data and evaluation configuration.

    ``max_train_samples`` and ``max_dev_samples`` are explicit diagnostic limits applied only
    after the complete official training split has been partitioned. They default to ``None``.
    The only currently grounded crop policy is ``none``; no NYU crop constants are inferred.
    """

    validation_fraction: float = 0.1
    image_height: int = 448
    image_width: int = 576
    depth_source: Literal["depths", "rawDepths"] = "depths"
    validity_source: Literal["target", "rawDepths"] = "target"
    num_workers: int = 0
    alignment: Literal["none", "median"] = "none"
    min_depth: float = 0.1
    max_depth: float = 10.0
    crop: Literal["none"] = "none"
    max_train_samples: int | None = None
    max_dev_samples: int | None = None

    def __post_init__(self) -> None:
        _validate_fraction(self.validation_fraction, "validation_fraction")
        _validate_positive_integer(self.image_height, "image_height")
        _validate_positive_integer(self.image_width, "image_width")
        if self.depth_source not in ("depths", "rawDepths"):
            raise ValueError("depth_source must be 'depths' or 'rawDepths'")
        if self.validity_source not in ("target", "rawDepths"):
            raise ValueError("validity_source must be 'target' or 'rawDepths'")
        _validate_non_negative_integer(self.num_workers, "num_workers")
        if self.alignment not in ("none", "median"):
            raise ValueError("alignment must be 'none' or 'median'")
        _validate_positive_real(self.min_depth, "min_depth")
        _validate_positive_real(self.max_depth, "max_depth")
        if self.max_depth <= self.min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        if self.crop != "none":
            raise ValueError("crop must be 'none' until a sourced NYU crop is implemented")
        _validate_optional_positive_integer(self.max_train_samples, "max_train_samples")
        _validate_optional_positive_integer(self.max_dev_samples, "max_dev_samples")


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


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


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


def _validate_fraction(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between zero and one")


def _validate_warmup_factor(value: float) -> None:
    _validate_positive_real(value, "warmup_start_factor")
    if value >= 1:
        raise ValueError("warmup_start_factor must be smaller than one")


def _validate_stage_weights(weights: StageWeights) -> None:
    if not isinstance(weights, tuple) or len(weights) != 4:
        raise ValueError("stage_weights must be a four-element tuple")
    for weight in weights:
        _validate_non_negative_real(weight, "stage weight")
    if not any(weight > 0 for weight in weights):
        raise ValueError("at least one stage weight must be positive")


def _validate_optional_positive_integer(value: int | None, name: str) -> None:
    if value is None:
        return
    _validate_positive_integer(value, name)
