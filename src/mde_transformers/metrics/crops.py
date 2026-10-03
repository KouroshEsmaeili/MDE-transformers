"""Protocol-neutral spatial crop-mask representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CropBounds:
    """Integer half-open image bounds ``[top, bottom) x [left, right)``.

    This type represents already-grounded crop coordinates without encoding unverified Eigen,
    Garg, or NYU constants. Dataset-specific protocols can construct these bounds later.
    """

    top: int
    left: int
    bottom: int
    right: int

    def __post_init__(self) -> None:
        values = (self.top, self.left, self.bottom, self.right)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("crop bounds must be integers")
        if self.top < 0 or self.left < 0:
            raise ValueError("crop origins must be non-negative")
        if self.bottom <= self.top or self.right <= self.left:
            raise ValueError("crop bounds must have positive height and width")

    def to_mask(
        self,
        image_size: tuple[int, int],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Create a boolean ``(height, width)`` mask on the requested device."""
        height, width = _validate_image_size(image_size)
        if self.bottom > height or self.right > width:
            raise ValueError("crop bounds extend beyond the image size")
        rows = torch.arange(height, device=device)
        columns = torch.arange(width, device=device)
        row_mask = (rows >= self.top) & (rows < self.bottom)
        column_mask = (columns >= self.left) & (columns < self.right)
        return row_mask[:, None] & column_mask[None, :]


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must contain (height, width)")
    height, width = image_size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
    ):
        raise TypeError("image dimensions must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width
