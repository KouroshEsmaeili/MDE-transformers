"""Pinhole-camera and rectified-stereo geometry primitives.

Image coordinates use zero-based pixel coordinates: the top-left pixel center is ``(u, v) =
(0, 0)``. Horizontal mirroring follows the discrete transform ``u' = (width - 1) - u``.
"""

from __future__ import annotations

import math

import torch

Scalar = float | int | torch.Tensor


def resize_intrinsics(
    intrinsics: torch.Tensor,
    original_size: tuple[int, int],
    new_size: tuple[int, int],
) -> torch.Tensor:
    """Return intrinsics for an independently resized image without modifying the input.

    Sizes are ``(height, width)``. For ``sx = new_width / width`` and
    ``sy = new_height / height``, focal lengths transform as ``fx' = sx * fx`` and
    ``fy' = sy * fy``. Principal points follow the half-pixel mapping
    ``cx' = sx * (cx + 0.5) - 0.5`` and ``cy' = sy * (cy + 0.5) - 0.5``. This is the
    ``align_corners=False``-style coordinate convention expected by the future image
    preprocessing pipeline.
    """
    _validate_intrinsics(intrinsics)
    original_height, original_width = _validate_image_size(original_size, "original_size")
    new_height, new_width = _validate_image_size(new_size, "new_size")
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    resized = intrinsics.clone()
    resized[0, 0] = intrinsics[0, 0] * scale_x
    resized[1, 1] = intrinsics[1, 1] * scale_y
    resized[0, 2] = scale_x * (intrinsics[0, 2] + 0.5) - 0.5
    resized[1, 2] = scale_y * (intrinsics[1, 2] + 0.5) - 0.5
    return resized


def crop_intrinsics(intrinsics: torch.Tensor, *, left: int, top: int) -> torch.Tensor:
    """Return intrinsics for a crop whose integer origin is ``(left, top)``.

    The transformed principal point is ``(cx - left, cy - top)``. Focal lengths are unchanged,
    and the input matrix is not modified. Crop extents are deliberately outside this primitive;
    dataset code must validate them against the associated image.
    """
    _validate_intrinsics(intrinsics)
    _validate_non_negative_integer(left, "left")
    _validate_non_negative_integer(top, "top")

    cropped = intrinsics.clone()
    cropped[0, 2] = intrinsics[0, 2] - left
    cropped[1, 2] = intrinsics[1, 2] - top
    return cropped


def flip_intrinsics_horizontal(intrinsics: torch.Tensor, image_width: int) -> torch.Tensor:
    """Return intrinsics for a horizontally mirrored image.

    Under zero-based discrete pixel coordinates, mirroring maps ``u`` to
    ``u' = (image_width - 1) - u``. Therefore ``cx' = (image_width - 1) - cx``. The focal lengths
    remain positive and unchanged, corresponding to reversal of the camera-frame horizontal
    coordinate. The input matrix is not modified.
    """
    _validate_intrinsics(intrinsics)
    _validate_positive_integer(image_width, "image_width")

    flipped = intrinsics.clone()
    flipped[0, 2] = (image_width - 1) - intrinsics[0, 2]
    return flipped


def disparity_to_depth(
    disparity: torch.Tensor,
    *,
    fx: Scalar,
    baseline: Scalar,
    invalid_value: float = float("nan"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert rectified disparity to depth and return its validity mask.

    The equation is ``depth = fx * baseline / disparity``. When ``fx`` and disparity are measured
    in pixels and baseline is measured in meters, depth is in meters. No dataset-specific baseline
    unit is assumed.

    Valid disparities are finite and strictly positive. Invalid locations receive
    ``invalid_value`` (NaN by default) and are ``False`` in the returned boolean mask. Division is
    performed only with a safe nonzero denominator, so invalid disparities cannot cause a
    divide-by-zero. Output dtype and device match ``disparity``.
    """
    if not disparity.is_floating_point():
        raise TypeError("disparity must be a floating-point tensor")
    if isinstance(invalid_value, bool) or not isinstance(invalid_value, int | float):
        raise TypeError("invalid_value must be a real number")

    focal_length = _positive_scalar_like(fx, disparity, "fx")
    stereo_baseline = _positive_scalar_like(baseline, disparity, "baseline")
    numerator = focal_length * stereo_baseline
    if not torch.isfinite(numerator).item():
        raise ValueError("fx * baseline is not finite in the disparity dtype")

    valid = torch.isfinite(disparity) & (disparity > 0)
    safe_disparity = torch.where(valid, disparity, torch.ones_like(disparity))
    computed_depth = numerator / safe_disparity
    if not torch.isfinite(computed_depth[valid]).all().item():
        raise FloatingPointError("valid disparity produced non-finite depth")

    invalid_depth = torch.full_like(disparity, float(invalid_value))
    return torch.where(valid, computed_depth, invalid_depth), valid


def _validate_intrinsics(intrinsics: torch.Tensor) -> None:
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if not intrinsics.is_floating_point():
        raise TypeError("intrinsics must be a floating-point tensor")
    if not torch.isfinite(intrinsics).all().item():
        raise ValueError("intrinsics must be finite")
    if intrinsics[0, 0].item() <= 0 or intrinsics[1, 1].item() <= 0:
        raise ValueError("fx and fy must be positive")


def _validate_image_size(image_size: tuple[int, int], name: str) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError(f"{name} must contain (height, width)")
    height, width = image_size
    _validate_positive_integer(height, f"{name} height")
    _validate_positive_integer(width, f"{name} width")
    return height, width


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


def _positive_scalar_like(value: Scalar, reference: torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} tensor must be scalar")
        if not value.is_floating_point():
            raise TypeError(f"{name} tensor must be floating point")
        if value.device != reference.device:
            raise ValueError(f"{name} tensor and disparity must be on the same device")
        scalar = value.to(dtype=reference.dtype)
    else:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        scalar = reference.new_tensor(value)

    if not torch.isfinite(scalar).item() or scalar.item() <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar
