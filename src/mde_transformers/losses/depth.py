"""Masked pointwise losses for positive physical-depth predictions.

Every function reduces globally over the pixels selected by the explicit boolean mask. Selected
target and prediction values must be finite and strictly positive; invalid predictions are never
silently removed. Depth values are not clipped or modified in place.
"""

from __future__ import annotations

import torch

from mde_transformers.losses._validation import select_valid_depths, validate_positive_scalar


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``mean(abs(prediction - target))`` over valid pixels.

    The result has the same physical unit as the input depth tensors.
    """
    prediction_valid, target_valid = select_valid_depths(prediction, target, valid_mask)
    return (prediction_valid - target_valid).abs().mean()


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Return the classical Huber loss over valid pixels.

    For absolute error ``a``, the per-pixel loss is ``0.5 * a^2`` when ``a <= delta`` and
    ``delta * (a - 0.5 * delta)`` otherwise. ``delta`` is an explicit positive transition in the
    same unit as depth; no PyTorch default is assumed.
    """
    transition = validate_positive_scalar(delta, "delta")
    prediction_valid, target_valid = select_valid_depths(prediction, target, valid_mask)
    absolute_error = (prediction_valid - target_valid).abs()
    delta_tensor = absolute_error.new_tensor(transition)
    quadratic = 0.5 * absolute_error.square()
    linear = delta_tensor * (absolute_error - 0.5 * delta_tensor)
    return torch.where(absolute_error <= delta_tensor, quadratic, linear).mean()


def berhu_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold_ratio: float = 0.2,
) -> torch.Tensor:
    """Return reverse Huber (BerHu) loss over valid pixels.

    The threshold is ``c = threshold_ratio * max(abs(error_valid))``, where the maximum is global
    over every selected pixel in the supplied tensor or batch. Errors no larger than ``c`` use
    L1; larger errors use ``(error^2 + c^2) / (2c)``. The ratio must lie in ``(0, 1]``. Perfect
    prediction takes a separate differentiable-zero path, avoiding division by zero.
    """
    ratio = validate_positive_scalar(threshold_ratio, "threshold_ratio")
    if ratio > 1:
        raise ValueError("threshold_ratio must be no greater than one")
    prediction_valid, target_valid = select_valid_depths(prediction, target, valid_mask)
    absolute_error = (prediction_valid - target_valid).abs()
    maximum_error = absolute_error.max()
    if maximum_error.detach().item() == 0:
        return absolute_error.mean()

    threshold = maximum_error * ratio
    reverse_huber = (absolute_error.square() + threshold.square()) / (2 * threshold)
    return torch.where(absolute_error <= threshold, absolute_error, reverse_huber).mean()


def scale_invariant_log_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the scale-invariant log-error variance used for training.

    With ``d = log(prediction) - log(target)``, the result is
    ``mean(d^2) - mean(d)^2``. Unlike :func:`mde_transformers.metrics.silog`, this training loss
    does **not** take a square root and is not multiplied by 100. Only a negative result within a
    dtype-relative round-off tolerance is replaced by zero; a larger negative value raises.
    """
    prediction_valid, target_valid = select_valid_depths(prediction, target, valid_mask)
    log_error = prediction_valid.log() - target_valid.log()
    mean_error = log_error.mean()
    mean_square = log_error.square().mean()
    variance = mean_square - mean_error.square()

    scale = torch.maximum(mean_square.abs(), mean_error.square().abs())
    scale = torch.maximum(scale, torch.ones_like(scale))
    tolerance = 8 * torch.finfo(variance.dtype).eps * scale
    if variance.detach().item() < -tolerance.detach().item():
        raise FloatingPointError("SILog training variance is negative beyond round-off tolerance")
    return torch.where(variance < 0, torch.zeros_like(variance), variance)
