"""Synchronized functional transforms for :class:`DepthSample` components.

Geometric transforms update RGB, depth, the authoritative validity mask, and camera intrinsics
together. They preserve physical depth values and units; only spatial sample locations change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from mde_transformers.data.geometry import (
    crop_intrinsics,
    flip_intrinsics_horizontal,
    resize_intrinsics,
)
from mde_transformers.data.sample import DepthSample


def resize_depth_sample(sample: DepthSample, new_size: tuple[int, int]) -> DepthSample:
    """Resize all spatial sample components with synchronized coordinate semantics.

    RGB uses antialiased bilinear interpolation with ``align_corners=False``, matching the
    half-pixel convention used by :func:`resize_intrinsics`. Depth and the explicit validity mask
    are concatenated and sampled in one ``nearest-exact`` operation, ensuring that both select the
    same source pixel. Metric depth values are never multiplied by a spatial scale.
    """
    new_height, new_width = _validate_size(new_size, "new_size")
    target_size = (new_height, new_width)

    image = functional.interpolate(
        sample.image.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    depth_and_mask = torch.cat(
        (sample.depth, sample.valid_mask.to(dtype=sample.depth.dtype)),
        dim=0,
    )
    depth_and_mask = functional.interpolate(
        depth_and_mask.unsqueeze(0),
        size=target_size,
        mode="nearest-exact",
    ).squeeze(0)
    depth = depth_and_mask[:1]
    valid_mask = depth_and_mask[1:].to(dtype=torch.bool)

    intrinsics = (
        None
        if sample.intrinsics is None
        else resize_intrinsics(sample.intrinsics, sample.image_size, target_size)
    )
    return DepthSample(
        image=image,
        depth=depth,
        valid_mask=valid_mask,
        intrinsics=intrinsics,
        sample_id=sample.sample_id,
    )


def crop_depth_sample(
    sample: DepthSample,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
) -> DepthSample:
    """Crop every spatial component to exact integer bounds without padding.

    Returned slices are cloned so their storage does not alias the source sample. Intrinsic focal
    lengths are unchanged, while the principal point is translated by ``(-left, -top)``.
    """
    _validate_non_negative_integer(top, "top")
    _validate_non_negative_integer(left, "left")
    _validate_positive_integer(height, "height")
    _validate_positive_integer(width, "width")
    source_height, source_width = sample.image_size
    bottom = top + height
    right = left + width
    if bottom > source_height or right > source_width:
        raise ValueError("crop exceeds sample bounds")

    spatial_slice = (..., slice(top, bottom), slice(left, right))
    image = sample.image[spatial_slice].clone()
    depth = sample.depth[spatial_slice].clone()
    valid_mask = sample.valid_mask[spatial_slice].clone()
    intrinsics = (
        None
        if sample.intrinsics is None
        else crop_intrinsics(sample.intrinsics, left=left, top=top)
    )
    return DepthSample(
        image=image,
        depth=depth,
        valid_mask=valid_mask,
        intrinsics=intrinsics,
        sample_id=sample.sample_id,
    )


def horizontal_flip_depth_sample(sample: DepthSample) -> DepthSample:
    """Reverse every spatial component horizontally and update the principal point.

    The flip changes only spatial ordering: metric depth values and validity values are preserved.
    Intrinsics use the discrete zero-based mapping ``cx' = (width - 1) - cx``.
    """
    _, width = sample.image_size
    intrinsics = (
        None if sample.intrinsics is None else flip_intrinsics_horizontal(sample.intrinsics, width)
    )
    return DepthSample(
        image=torch.flip(sample.image, dims=(-1,)),
        depth=torch.flip(sample.depth, dims=(-1,)),
        valid_mask=torch.flip(sample.valid_mask, dims=(-1,)),
        intrinsics=intrinsics,
        sample_id=sample.sample_id,
    )


def _validate_size(size: tuple[int, int], name: str) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{name} must contain (height, width)")
    height, width = size
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
