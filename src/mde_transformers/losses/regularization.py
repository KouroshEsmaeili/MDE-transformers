"""Spatial depth regularizers with explicit validity and camera geometry."""

from __future__ import annotations

from typing import Literal

import torch

from mde_transformers.losses._validation import (
    select_valid_depths,
    validate_batched_depth,
    validate_mask,
    validate_non_negative_weight,
    validate_positive_scalar,
)

SmoothnessRepresentation = Literal["log", "inverse"]
GradientRepresentation = Literal["metric", "log"]


def edge_aware_smoothness_loss(
    prediction: torch.Tensor,
    image: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    representation: SmoothnessRepresentation,
    beta: float = 1.0,
) -> torch.Tensor:
    """Penalize depth-representation gradients away from RGB edges.

    ``prediction`` is positive depth shaped ``[B, 1, H, W]`` and ``image`` is a finite floating
    RGB tensor shaped ``[B, 3, H, W]``. The depth representation is explicitly either ``log(z)``
    or ``1/z``. For each horizontal or vertical adjacent pair, the penalty is
    ``abs(depth_gradient) * exp(-beta * mean_c(abs(image_gradient)))``. The returned value is the
    mean over all eligible adjacent pairs in both directions; RGB is not normalized internally.

    When supplied, ``valid_mask`` is authoritative and a pair is eligible only when both depth
    endpoints are valid. Invalid depth values outside that mask are not evaluated. At least one
    valid adjacent pair is required.
    """
    validate_batched_depth(prediction, "prediction")
    _validate_image(image, prediction)
    edge_scale = validate_non_negative_weight(beta, "beta")
    if representation not in ("log", "inverse"):
        raise ValueError("representation must be 'log' or 'inverse'")

    mask = torch.ones_like(prediction, dtype=torch.bool) if valid_mask is None else valid_mask
    validate_mask(mask, prediction)
    select_valid_depths(prediction, prediction, mask, require_4d=True)

    safe_prediction = torch.where(mask, prediction, torch.ones_like(prediction))
    depth_representation = (
        safe_prediction.log() if representation == "log" else safe_prediction.reciprocal()
    )
    depth_dx, depth_dy = _spatial_gradients(depth_representation)
    image_dx, image_dy = _spatial_gradients(image)
    image_dx = image_dx.abs().mean(dim=1, keepdim=True)
    image_dy = image_dy.abs().mean(dim=1, keepdim=True)
    penalty_x = depth_dx.abs() * torch.exp(-edge_scale * image_dx)
    penalty_y = depth_dy.abs() * torch.exp(-edge_scale * image_dy)
    mask_x, mask_y = _neighbor_masks(mask)
    return _mean_valid_pairs(penalty_x, penalty_y, mask_x, mask_y)


def depth_gradient_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    representation: GradientRepresentation = "metric",
) -> torch.Tensor:
    """Compare predicted and target depth gradients over valid adjacent pairs.

    The default operates directly in the input metric-depth unit. ``representation="log"``
    instead compares dimensionless log-depth gradients. Horizontal and vertical residuals are
    pooled so every eligible adjacent pair has equal weight. A pair is eligible only when both
    target-mask endpoints are valid; prediction validity is never inferred or used to alter the
    mask.
    """
    prediction_valid, target_valid = select_valid_depths(
        prediction,
        target,
        valid_mask,
        require_4d=True,
    )
    del prediction_valid, target_valid
    if representation not in ("metric", "log"):
        raise ValueError("representation must be 'metric' or 'log'")

    safe_prediction = torch.where(valid_mask, prediction, torch.ones_like(prediction))
    safe_target = torch.where(valid_mask, target, torch.ones_like(target))
    if representation == "log":
        safe_prediction = safe_prediction.log()
        safe_target = safe_target.log()

    prediction_dx, prediction_dy = _spatial_gradients(safe_prediction)
    target_dx, target_dy = _spatial_gradients(safe_target)
    residual_x = (prediction_dx - target_dx).abs()
    residual_y = (prediction_dy - target_dy).abs()
    mask_x, mask_y = _neighbor_masks(valid_mask)
    return _mean_valid_pairs(residual_x, residual_y, mask_x, mask_y)


def surface_normal_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compare local surface normals derived from pinhole back-projected 3D points.

    Depth tensors use ``[B, 1, H, W]`` and camera matrices use ``[3, 3]`` or ``[B, 3, 3]``.
    Zero-based pixel centers ``(u, v)`` are back-projected as
    ``P = depth * inverse(K) @ [u, v, 1]``. Each top-left-anchored neighborhood uses the anchor,
    right neighbor, and lower neighbor; its normal is their horizontal/vertical cross product.
    All three target-mask pixels must be valid. The loss is ``mean(1 - cosine(n_pred, n_target))``.

    Intrinsics are mandatory and never guessed. Degenerate selected normals and empty valid
    neighborhood sets raise rather than being silently removed.
    """
    select_valid_depths(prediction, target, valid_mask, require_4d=True)
    normal_epsilon = validate_positive_scalar(epsilon, "epsilon")
    batch_size, _, height, width = prediction.shape
    if height < 2 or width < 2:
        raise ValueError("surface normals require depth height and width of at least two")
    batched_intrinsics = _validate_and_batch_intrinsics(intrinsics, prediction)

    safe_prediction = torch.where(valid_mask, prediction, torch.ones_like(prediction))
    safe_target = torch.where(valid_mask, target, torch.ones_like(target))
    prediction_points = _backproject(safe_prediction, batched_intrinsics)
    target_points = _backproject(safe_target, batched_intrinsics)
    prediction_normals = _local_normals(prediction_points)
    target_normals = _local_normals(target_points)

    cell_mask = (
        valid_mask[:, :, :-1, :-1] & valid_mask[:, :, :-1, 1:] & valid_mask[:, :, 1:, :-1]
    ).squeeze(1)
    if not cell_mask.any().item():
        raise ValueError("valid_mask selects no complete normal neighborhoods")

    prediction_norm = torch.linalg.vector_norm(prediction_normals, dim=1)
    target_norm = torch.linalg.vector_norm(target_normals, dim=1)
    if (prediction_norm[cell_mask] <= normal_epsilon).any().item():
        raise ValueError("predicted 3D neighborhood produces a degenerate normal")
    if (target_norm[cell_mask] <= normal_epsilon).any().item():
        raise ValueError("target 3D neighborhood produces a degenerate normal")

    prediction_unit = prediction_normals / prediction_norm.clamp_min(normal_epsilon).unsqueeze(1)
    target_unit = target_normals / target_norm.clamp_min(normal_epsilon).unsqueeze(1)
    cosine = (prediction_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cosine[cell_mask]).mean()


def _validate_image(image: torch.Tensor, depth: torch.Tensor) -> None:
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor")
    expected_shape = (depth.shape[0], 3, depth.shape[2], depth.shape[3])
    if image.shape != expected_shape:
        raise ValueError("image must have shape [B, 3, H, W] matching prediction")
    if image.device != depth.device:
        raise ValueError("image and prediction must be on the same device")
    if not torch.isfinite(image).all().item():
        raise ValueError("image must contain only finite values")


def _spatial_gradients(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tensor[..., 1:] - tensor[..., :-1], tensor[..., 1:, :] - tensor[..., :-1, :]


def _neighbor_masks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mask[..., 1:] & mask[..., :-1], mask[..., 1:, :] & mask[..., :-1, :]


def _mean_valid_pairs(
    values_x: torch.Tensor,
    values_y: torch.Tensor,
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
) -> torch.Tensor:
    selected_x = values_x[mask_x]
    selected_y = values_y[mask_y]
    if selected_x.numel() + selected_y.numel() == 0:
        raise ValueError("valid_mask selects no valid adjacent depth pairs")
    return torch.cat((selected_x, selected_y)).mean()


def _validate_and_batch_intrinsics(
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    batch_size = depth.shape[0]
    if not intrinsics.is_floating_point():
        raise TypeError("intrinsics must be a floating-point tensor")
    if intrinsics.device != depth.device:
        raise ValueError("intrinsics and depth must be on the same device")
    if intrinsics.dtype != depth.dtype:
        raise ValueError("intrinsics and depth must have the same dtype")
    if intrinsics.shape == (3, 3):
        batched = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
    elif intrinsics.shape == (batch_size, 3, 3):
        batched = intrinsics
    else:
        raise ValueError("intrinsics must have shape [3, 3] or [B, 3, 3]")
    if not torch.isfinite(batched).all().item():
        raise ValueError("intrinsics must contain only finite values")
    if not (batched[:, 0, 0] > 0).all().item() or not (batched[:, 1, 1] > 0).all().item():
        raise ValueError("intrinsic focal lengths fx and fy must be positive")

    expected_last_row = batched.new_tensor((0.0, 0.0, 1.0)).expand(batch_size, -1)
    if not torch.allclose(batched[:, 2], expected_last_row):
        raise ValueError("intrinsics must use the canonical pinhole last row [0, 0, 1]")
    try:
        inverse = torch.linalg.inv(batched)
    except RuntimeError as error:
        raise ValueError("intrinsics must be invertible") from error
    if not torch.isfinite(inverse).all().item():
        raise ValueError("intrinsics inverse must be finite")
    return batched


def _backproject(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = depth.shape
    vertical, horizontal = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    pixels = torch.stack((horizontal, vertical, torch.ones_like(horizontal)), dim=0)
    rays = torch.linalg.solve(
        intrinsics,
        pixels.reshape(3, -1).unsqueeze(0).expand(batch_size, -1, -1),
    )
    return rays.reshape(batch_size, 3, height, width) * depth


def _local_normals(points: torch.Tensor) -> torch.Tensor:
    anchor = points[:, :, :-1, :-1]
    horizontal = points[:, :, :-1, 1:] - anchor
    vertical = points[:, :, 1:, :-1] - anchor
    return torch.linalg.cross(horizontal, vertical, dim=1)
