"""Architecture-neutral multi-scale and composite depth supervision."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from mde_transformers.losses._validation import (
    select_valid_depths,
    validate_batched_depth,
    validate_non_negative_weight,
)
from mde_transformers.losses.depth import masked_l1_loss
from mde_transformers.losses.regularization import (
    GradientRepresentation,
    SmoothnessRepresentation,
    depth_gradient_consistency_loss,
    edge_aware_smoothness_loss,
    surface_normal_consistency_loss,
)

DepthLoss = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class MultiScaleLossResult:
    """Weighted multi-scale total plus raw and weighted per-stage scalar losses."""

    total: torch.Tensor
    stage_losses: tuple[torch.Tensor, ...]
    weighted_stage_losses: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class CompositeLossWeights:
    """Explicit non-negative coefficients for optional spatial loss terms."""

    smoothness: float
    gradient: float
    normal: float

    def __post_init__(self) -> None:
        validate_non_negative_weight(self.smoothness, "smoothness weight")
        validate_non_negative_weight(self.gradient, "gradient weight")
        validate_non_negative_weight(self.normal, "normal weight")


@dataclass(frozen=True, slots=True)
class CompositeLossResult:
    """Weighted loss breakdown whose four main components sum to ``total``.

    ``per_stage_depth`` contains the weighted stage terms and therefore sums to ``depth``.
    Disabled regularizers are represented by scalar zeros.
    """

    total: torch.Tensor
    depth: torch.Tensor
    smoothness: torch.Tensor
    gradient: torch.Tensor
    normal: torch.Tensor
    per_stage_depth: tuple[torch.Tensor, ...]


def multi_scale_depth_loss(
    predictions: Sequence[torch.Tensor],
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    stage_weights: Sequence[float],
    *,
    loss_fn: DepthLoss = masked_l1_loss,
) -> MultiScaleLossResult:
    """Supervise predictions at the original target resolution.

    Each ``[B, 1, h, w]`` prediction is bilinearly upsampled to target ``[H, W]`` with
    ``align_corners=False``. The original target and its authoritative mask are never
    downsampled, which preserves sparse supervision. Stage weights are applied exactly as
    supplied and are not normalized or inferred.
    """
    prediction_tuple = tuple(predictions)
    weight_tuple = tuple(stage_weights)
    if not prediction_tuple:
        raise ValueError("predictions must not be empty")
    if len(prediction_tuple) != len(weight_tuple):
        raise ValueError("predictions and stage_weights must have the same length")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")

    validate_batched_depth(target, "target")
    select_valid_depths(target, target, valid_mask, require_4d=True)
    weights = tuple(
        _validate_stage_weight(weight, index) for index, weight in enumerate(weight_tuple)
    )
    if not any(weight > 0 for weight in weights):
        raise ValueError("at least one stage weight must be positive")

    stage_losses: list[torch.Tensor] = []
    weighted_losses: list[torch.Tensor] = []
    for index, (prediction, weight) in enumerate(zip(prediction_tuple, weights, strict=True)):
        _validate_stage_prediction(prediction, target, index)
        if prediction.shape[-2:] == target.shape[-2:]:
            full_resolution = prediction
        else:
            full_resolution = functional.interpolate(
                prediction,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        stage_loss = loss_fn(full_resolution, target, valid_mask)
        if not isinstance(stage_loss, torch.Tensor) or stage_loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar torch.Tensor")
        if not torch.isfinite(stage_loss).item():
            raise FloatingPointError(f"loss_fn returned a non-finite value for stage {index}")
        stage_losses.append(stage_loss)
        weighted_losses.append(stage_loss * weight)

    total = torch.stack(weighted_losses).sum()
    return MultiScaleLossResult(total, tuple(stage_losses), tuple(weighted_losses))


def composite_depth_loss(
    predictions: Sequence[torch.Tensor],
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    stage_weights: Sequence[float],
    *,
    weights: CompositeLossWeights,
    regularization_prediction: torch.Tensor | None = None,
    image: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
    depth_loss_fn: DepthLoss = masked_l1_loss,
    smoothness_representation: SmoothnessRepresentation = "log",
    smoothness_beta: float = 1.0,
    gradient_representation: GradientRepresentation = "metric",
) -> CompositeLossResult:
    """Combine multi-scale depth supervision with explicitly weighted regularizers.

    ``regularization_prediction`` is deliberately separate from the stage sequence and must
    already match the target resolution. This prevents the loss from guessing which decoder
    output is final. It is required only when at least one optional coefficient is positive.
    ``image`` is required only for smoothness, and ``intrinsics`` only for surface normals.
    """
    depth_result = multi_scale_depth_loss(
        predictions,
        target,
        valid_mask,
        stage_weights,
        loss_fn=depth_loss_fn,
    )
    zero = target.new_zeros(())
    smoothness_component = zero
    gradient_component = zero
    normal_component = zero

    any_regularization = weights.smoothness > 0 or weights.gradient > 0 or weights.normal > 0
    if any_regularization:
        if regularization_prediction is None:
            raise ValueError("regularization_prediction is required for enabled regularizers")
        if regularization_prediction.shape != target.shape:
            raise ValueError("regularization_prediction must have the same shape as target")

    if weights.smoothness > 0:
        if regularization_prediction is None or image is None:
            raise ValueError("smoothness requires regularization_prediction and image")
        smoothness_component = weights.smoothness * edge_aware_smoothness_loss(
            regularization_prediction,
            image,
            valid_mask,
            representation=smoothness_representation,
            beta=smoothness_beta,
        )
    if weights.gradient > 0:
        if regularization_prediction is None:
            raise ValueError("gradient consistency requires regularization_prediction")
        gradient_component = weights.gradient * depth_gradient_consistency_loss(
            regularization_prediction,
            target,
            valid_mask,
            representation=gradient_representation,
        )
    if weights.normal > 0:
        if regularization_prediction is None or intrinsics is None:
            raise ValueError("surface-normal consistency requires prediction and intrinsics")
        normal_component = weights.normal * surface_normal_consistency_loss(
            regularization_prediction,
            target,
            valid_mask,
            intrinsics,
        )

    total = depth_result.total + smoothness_component + gradient_component + normal_component
    return CompositeLossResult(
        total=total,
        depth=depth_result.total,
        smoothness=smoothness_component,
        gradient=gradient_component,
        normal=normal_component,
        per_stage_depth=depth_result.weighted_stage_losses,
    )


def _validate_stage_weight(weight: float, index: int) -> float:
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        raise TypeError(f"stage weight {index} must be a real number")
    result = float(weight)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"stage weight {index} must be finite and non-negative")
    return result


def _validate_stage_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    index: int,
) -> None:
    validate_batched_depth(prediction, f"prediction stage {index}")
    if prediction.shape[0] != target.shape[0]:
        raise ValueError(f"prediction stage {index} batch size does not match target")
    if prediction.device != target.device:
        raise ValueError(f"prediction stage {index} and target must be on the same device")
    if not torch.isfinite(prediction).all().item():
        raise ValueError(f"prediction stage {index} must contain only finite values")
    if not (prediction > 0).all().item():
        raise ValueError(f"prediction stage {index} must contain only positive depth values")
