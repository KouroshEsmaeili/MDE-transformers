"""Explicit NYU Depth V2 evaluation protocols and native crop geometry.

The standard NYU Eigen crop is a literature protocol option. It is not labelled as the
historical thesis crop because the currently available thesis text does not provide coordinates
that establish those protocols as identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from mde_transformers.metrics.crops import CropBounds

NYUAlignment = Literal["none", "median"]
NYUCrop = Literal["none", "nyu_eigen"]
NYU_NATIVE_IMAGE_SIZE = (480, 640)
NYU_EIGEN_CROP = CropBounds(top=45, left=41, bottom=471, right=601)


@dataclass(frozen=True, slots=True)
class NYUEvaluationProtocol:
    """NYU metric range, scale alignment, and crop as three independent choices.

    Crop coordinates, when selected, are defined in native ``480 x 640`` target coordinates.
    ``alignment='median'`` is performed per image and is never implied merely by selecting a crop.
    """

    depth_range: tuple[float, float] = (0.1, 10.0)
    alignment: NYUAlignment = "none"
    crop: NYUCrop = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.depth_range, tuple) or len(self.depth_range) != 2:
            raise ValueError("depth_range must contain (minimum, maximum)")
        minimum, maximum = self.depth_range
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in self.depth_range
        ):
            raise ValueError("depth_range values must be finite real numbers")
        if minimum <= 0 or maximum <= minimum:
            raise ValueError("depth_range must be positive and strictly increasing")
        if self.alignment not in ("none", "median"):
            raise ValueError("alignment must be 'none' or 'median'")
        if self.crop not in ("none", "nyu_eigen"):
            raise ValueError("crop must be 'none' or 'nyu_eigen'")

    @property
    def profile_name(self) -> str:
        """Return a descriptive label without conflating the Eigen and thesis crops."""
        if self == raw_metric_protocol():
            return "raw_metric_baseline"
        if self.crop == "nyu_eigen" and self.depth_range == (0.1, 10.0):
            return "standard_nyu_eigen"
        return "explicit_custom"


def raw_metric_protocol() -> NYUEvaluationProtocol:
    """Return the primary raw metric-scale protocol: range, no alignment, no crop."""
    return NYUEvaluationProtocol(
        depth_range=(0.1, 10.0),
        alignment="none",
        crop="none",
    )


def nyu_eigen_protocol(*, alignment: NYUAlignment = "median") -> NYUEvaluationProtocol:
    """Return the standard NYU Eigen crop protocol with explicit alignment.

    Median alignment is the named profile's default but remains an independently configurable
    operation. This helper does not assert that the crop matches the thesis's unspecified crop.
    """
    return NYUEvaluationProtocol(
        depth_range=(0.1, 10.0),
        alignment=alignment,
        crop="nyu_eigen",
    )


def nyu_native_crop_mask(
    crop: NYUCrop,
    image_size: tuple[int, int],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a boolean crop mask in native NYU coordinates.

    The Eigen constants must never be applied to an already resized tensor. Selecting
    ``nyu_eigen`` therefore requires the exact native ``(480, 640)`` size. The returned mask may
    subsequently be resized with the target validity mask using nearest-exact interpolation.
    """
    if crop == "none":
        height, width = _validate_image_size(image_size)
        return torch.ones((height, width), dtype=torch.bool, device=device)
    if crop != "nyu_eigen":
        raise ValueError("crop must be 'none' or 'nyu_eigen'")
    if image_size != NYU_NATIVE_IMAGE_SIZE:
        raise ValueError(
            "nyu_eigen crop must be constructed in native NYU 480x640 coordinates before resize"
        )
    return NYU_EIGEN_CROP.to_mask(image_size, device=device)


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise ValueError("image_size must contain (height, width)")
    height, width = image_size
    if any(isinstance(value, bool) or not isinstance(value, int) for value in image_size):
        raise TypeError("image dimensions must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width
