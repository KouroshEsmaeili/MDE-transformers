"""Architecture-independent metrics for positive metric-depth tensors.

All metric functions aggregate globally over the valid pixels selected by ``mask``. They never
derive validity from predictions: a selected prediction that is non-finite or non-positive raises
``ValueError``. This fail-fast policy prevents invalid predictions from silently improving a
reported score. Invalid predictions outside the target-derived mask are not evaluated.

These low-level reductions give every selected pixel equal weight over the supplied tensor.
Standard dataset benchmark reporting will instead compute metrics per image and then average the
per-image values with equal image weight; ``imagewise_mean`` makes that distinction explicit.

Inputs must have identical shapes and devices; broadcasting is intentionally rejected. Returned
metric tensors remain on the input device. AbsRel, RMSElog, SILog, and delta accuracy are
dimensionless; SqRel and RMSE retain the input depth unit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, overload

import torch

DepthMetric = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def valid_depth_mask(
    target: torch.Tensor,
    min_depth: float,
    max_depth: float,
    external_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return target validity for an inclusive metric-depth range.

    A target is valid exactly when it is finite, strictly positive, greater than or equal to
    ``min_depth``, less than or equal to ``max_depth``, and selected by ``external_mask`` when one
    is supplied. The external mask must be boolean and exactly match the target shape and device;
    it is never broadcast.
    """
    if not target.is_floating_point():
        raise TypeError("target must be a floating-point tensor")
    _validate_depth_range(min_depth, max_depth)

    mask = torch.isfinite(target) & (target > 0) & (target >= min_depth) & (target <= max_depth)
    if external_mask is not None:
        _validate_mask(external_mask, target)
        mask = mask & external_mask
    return mask


def abs_rel(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute global AbsRel: ``mean(abs(prediction - target) / target)``."""
    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    return ((prediction_valid - target_valid).abs() / target_valid).mean()


def sq_rel(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute global SqRel: ``mean((prediction - target)^2 / target)``."""
    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    return ((prediction_valid - target_valid).square() / target_valid).mean()


def rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute global RMSE: ``sqrt(mean((prediction - target)^2))``."""
    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    return (prediction_valid - target_valid).square().mean().sqrt()


def rmse_log(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute global RMSElog: ``sqrt(mean((log(prediction) - log(target))^2))``."""
    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    log_difference = prediction_valid.log() - target_valid.log()
    return log_difference.square().mean().sqrt()


def silog(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute dimensionless SILog without percentage scaling.

    With ``d = log(prediction) - log(target)``, this implementation returns
    ``sqrt(mean(d^2) - mean(d)^2)``. It does not multiply by 100. A negative variance is set to
    zero only when its magnitude is within a small dtype-relative round-off tolerance; a larger
    negative value raises ``FloatingPointError``.
    """
    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    log_difference = prediction_valid.log() - target_valid.log()
    mean_difference = log_difference.mean()
    mean_square = log_difference.square().mean()
    variance = mean_square - mean_difference.square()

    scale = torch.maximum(mean_square.abs(), mean_difference.square().abs())
    scale = torch.maximum(scale, torch.ones_like(scale))
    tolerance = 8 * torch.finfo(variance.dtype).eps * scale
    if variance.item() < -tolerance.item():
        raise FloatingPointError("SILog variance is negative beyond round-off tolerance")

    variance = torch.where(variance < 0, torch.zeros_like(variance), variance)
    return variance.sqrt()


def delta_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    power: int,
) -> torch.Tensor:
    """Compute the fraction satisfying ``max(pred/target, target/pred) < 1.25**power``.

    The threshold comparison is strict. ``power`` must be a positive integer.
    """
    if isinstance(power, bool) or not isinstance(power, int):
        raise TypeError("power must be an integer")
    if power < 1:
        raise ValueError("power must be positive")

    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    ratio = torch.maximum(prediction_valid / target_valid, target_valid / prediction_valid)
    return (ratio < 1.25**power).to(dtype=ratio.dtype).mean()


def delta1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute threshold accuracy for the strict ``1.25`` threshold."""
    return delta_accuracy(prediction, target, mask, power=1)


def delta2(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute threshold accuracy for the strict ``1.25**2`` threshold."""
    return delta_accuracy(prediction, target, mask, power=2)


def delta3(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute threshold accuracy for the strict ``1.25**3`` threshold."""
    return delta_accuracy(prediction, target, mask, power=3)


def imagewise_mean(
    metric: DepthMetric,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return an equal-weight mean of per-image metric values.

    Dimension zero is interpreted as the image dimension. This is the reduction intended for
    future standard dataset benchmark reporting. It differs from calling ``metric`` once on the
    full tensors, which gives every valid pixel equal weight. Every image must contain at least
    one valid pixel; images are never silently skipped.
    """
    _validate_tensor_shapes_and_devices(prediction, target, mask)
    if prediction.ndim == 0 or prediction.shape[0] == 0:
        raise ValueError("image-wise aggregation requires a non-empty image dimension")

    values: list[torch.Tensor] = []
    for index in range(prediction.shape[0]):
        try:
            value = metric(prediction[index], target[index], mask[index])
        except (TypeError, ValueError) as error:
            raise ValueError(f"cannot evaluate image at index {index}: {error}") from error
        if value.ndim != 0:
            raise ValueError("metric must return a scalar tensor")
        values.append(value)
    return torch.stack(values).mean()


@overload
def align_median(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    return_scale: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def align_median(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    return_scale: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def align_median(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    return_scale: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Scale a prediction by the ratio of valid target and prediction medians.

    The median is the linearly interpolated 0.5 quantile. Alignment is explicit and never applied
    by a metric function. Only masked values determine the scale, but the returned full prediction
    is multiplied by it. Inputs are not modified. The scale is returned as a scalar tensor when
    ``return_scale=True``.
    """
    prediction_valid, target_valid = _select_valid_depths(prediction, target, mask)
    prediction_median = torch.quantile(prediction_valid, 0.5)
    target_median = torch.quantile(target_valid, 0.5)

    if not torch.isfinite(prediction_median).item() or prediction_median.item() <= 0:
        raise ValueError("valid prediction median must be finite and positive")
    scale = target_median / prediction_median
    if not torch.isfinite(scale).item() or scale.item() <= 0:
        raise ValueError("median-alignment scale must be finite and positive")

    aligned = prediction * scale
    if return_scale:
        return aligned, scale
    return aligned


def _validate_depth_range(min_depth: float, max_depth: float) -> None:
    if isinstance(min_depth, bool) or not isinstance(min_depth, int | float):
        raise TypeError("min_depth must be a real number")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int | float):
        raise TypeError("max_depth must be a real number")
    if not math.isfinite(min_depth):
        raise ValueError("min_depth must be finite")
    if not math.isfinite(max_depth):
        raise ValueError("max_depth must be finite")
    if min_depth < 0:
        raise ValueError("min_depth must be non-negative")
    if max_depth < min_depth or max_depth <= 0:
        raise ValueError("max_depth must be positive and no smaller than min_depth")


def _validate_mask(mask: torch.Tensor, reference: torch.Tensor) -> None:
    if mask.dtype != torch.bool:
        raise TypeError("mask must have boolean dtype")
    if mask.shape != reference.shape:
        raise ValueError("mask and depth tensors must have identical shapes")
    if mask.device != reference.device:
        raise ValueError("mask and depth tensors must be on the same device")


def _validate_tensor_shapes_and_devices(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes; broadcasting is disabled"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    _validate_mask(mask, target)


def _select_valid_depths(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_tensor_shapes_and_devices(prediction, target, mask)
    prediction_valid = prediction[mask]
    target_valid = target[mask]

    if target_valid.numel() == 0:
        raise ValueError("mask selects no valid target pixels")
    if not torch.isfinite(target_valid).all().item():
        raise ValueError("selected target depths must be finite")
    if not (target_valid > 0).all().item():
        raise ValueError("selected target depths must be positive")
    if not torch.isfinite(prediction_valid).all().item():
        raise ValueError("selected predictions must be finite")
    if not (prediction_valid > 0).all().item():
        raise ValueError("selected predictions must be positive")
    return prediction_valid, target_valid
