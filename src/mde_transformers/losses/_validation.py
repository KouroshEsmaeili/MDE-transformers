"""Internal validation shared by architecture-independent depth losses."""

from __future__ import annotations

import math

import torch


def select_valid_depths(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    require_4d: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate compatible physical-depth tensors and return selected values."""
    validate_depth_triplet(prediction, target, valid_mask, require_4d=require_4d)
    prediction_valid = prediction[valid_mask]
    target_valid = target[valid_mask]
    if target_valid.numel() == 0:
        raise ValueError("valid_mask selects no supervised depth pixels")
    if not torch.isfinite(target_valid).all().item():
        raise ValueError("selected target depths must be finite")
    if not (target_valid > 0).all().item():
        raise ValueError("selected target depths must be positive")
    if not torch.isfinite(prediction_valid).all().item():
        raise ValueError("selected predictions must be finite")
    if not (prediction_valid > 0).all().item():
        raise ValueError("selected predictions must be positive")
    return prediction_valid, target_valid


def validate_depth_triplet(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    require_4d: bool = False,
) -> None:
    """Validate exact shape, dtype, and device compatibility without broadcasting."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    validate_mask(valid_mask, target)
    if require_4d:
        validate_batched_depth(prediction, "prediction")


def validate_batched_depth(depth: torch.Tensor, name: str) -> None:
    """Validate a floating-point ``[B, 1, H, W]`` depth tensor."""
    if not depth.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W]")
    if depth.shape[0] == 0 or depth.shape[2] == 0 or depth.shape[3] == 0:
        raise ValueError(f"{name} dimensions must be non-empty")


def validate_mask(valid_mask: torch.Tensor, reference: torch.Tensor) -> None:
    """Validate an authoritative boolean mask against a reference depth tensor."""
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have boolean dtype")
    if valid_mask.shape != reference.shape:
        raise ValueError("valid_mask and depth tensors must have identical shapes")
    if valid_mask.device != reference.device:
        raise ValueError("valid_mask and depth tensors must be on the same device")


def validate_non_negative_weight(value: float, name: str) -> float:
    """Return a finite non-negative scalar coefficient."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def validate_positive_scalar(value: float, name: str) -> float:
    """Return a finite strictly positive scalar parameter."""
    result = validate_non_negative_weight(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result
