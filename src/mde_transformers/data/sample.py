"""Typed representation of one supervised monocular-depth sample."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DepthSample:
    """One unbatched RGB and depth-supervision sample.

    ``image`` is a floating-point RGB tensor shaped ``(3, H, W)``. Before any independent
    normalization step, its expected range is ``[0, 1]``. ``depth`` is a floating-point tensor
    shaped ``(1, H, W)`` and represents meters when metric supervision is available.
    ``valid_mask`` is an authoritative boolean tensor shaped ``(1, H, W)``; depth values outside
    it need not be finite or positive. Optional pinhole intrinsics are shaped ``(3, 3)``.

    Construction validates but does not clone caller-owned tensors. Functional transforms in this
    package do not mutate the sample and return independently owned outputs. A future collate layer
    is responsible for batching samples; this class deliberately represents only one sample.
    """

    image: torch.Tensor
    depth: torch.Tensor
    valid_mask: torch.Tensor
    intrinsics: torch.Tensor | None = None
    sample_id: str | None = None

    def __post_init__(self) -> None:
        """Validate tensor ranks, dtypes, spatial agreement, and optional intrinsics."""
        _validate_tensor(self.image, "image")
        _validate_tensor(self.depth, "depth")
        _validate_tensor(self.valid_mask, "valid_mask")

        if self.image.ndim != 3 or self.image.shape[0] != 3:
            raise ValueError("image must have shape (3, H, W)")
        if not self.image.is_floating_point():
            raise TypeError("image must be a floating-point tensor")
        if self.depth.ndim != 3 or self.depth.shape[0] != 1:
            raise ValueError("depth must have shape (1, H, W)")
        if not self.depth.is_floating_point():
            raise TypeError("depth must be a floating-point tensor")
        if self.valid_mask.ndim != 3 or self.valid_mask.shape[0] != 1:
            raise ValueError("valid_mask must have shape (1, H, W)")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have boolean dtype")
        if self.depth.shape != self.valid_mask.shape:
            raise ValueError("depth and valid_mask must have identical shapes")
        if self.image.shape[-2:] != self.depth.shape[-2:]:
            raise ValueError("image and depth must have identical spatial dimensions")
        if self.image.shape[-2] == 0 or self.image.shape[-1] == 0:
            raise ValueError("sample spatial dimensions must be positive")
        if not (self.image.device == self.depth.device == self.valid_mask.device):
            raise ValueError("image, depth, and valid_mask must be on the same device")

        if self.intrinsics is not None:
            _validate_tensor(self.intrinsics, "intrinsics")
            if self.intrinsics.shape != (3, 3):
                raise ValueError("intrinsics must have shape (3, 3)")
            if not self.intrinsics.is_floating_point():
                raise TypeError("intrinsics must be a floating-point tensor")
            if self.intrinsics.device != self.image.device:
                raise ValueError("intrinsics and spatial tensors must be on the same device")

        if self.sample_id is not None and not isinstance(self.sample_id, str):
            raise TypeError("sample_id must be a string or None")

    @property
    def image_size(self) -> tuple[int, int]:
        """Return ``(height, width)`` for this sample."""
        return self.image.shape[-2], self.image.shape[-1]


def _validate_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
