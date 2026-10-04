"""Typed fixed-resolution batches for supervised monocular depth."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mde_transformers.data.sample import DepthSample


@dataclass(frozen=True, slots=True)
class DepthBatch:
    """A fixed-size batch produced from :class:`DepthSample` objects.

    Spatial tensors have shapes ``image: [B, 3, H, W]`` and
    ``depth/valid_mask: [B, 1, H, W]``. Intrinsics are either absent for the whole batch or a
    tensor shaped ``[B, 3, 3]``. Collation does not perform padding or device transfer.
    """

    image: torch.Tensor
    depth: torch.Tensor
    valid_mask: torch.Tensor
    intrinsics: torch.Tensor | None
    sample_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.image.ndim != 4 or self.image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if not self.image.is_floating_point():
            raise TypeError("image must be a floating-point tensor")
        if self.depth.ndim != 4 or self.depth.shape[1] != 1:
            raise ValueError("depth must have shape [B, 1, H, W]")
        if not self.depth.is_floating_point():
            raise TypeError("depth must be a floating-point tensor")
        if self.valid_mask.shape != self.depth.shape:
            raise ValueError("valid_mask and depth must have identical shapes")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have boolean dtype")
        if self.image.shape[0] == 0:
            raise ValueError("batch must not be empty")
        if self.image.shape[0] != self.depth.shape[0]:
            raise ValueError("image and depth batch dimensions must agree")
        if self.image.shape[-2:] != self.depth.shape[-2:]:
            raise ValueError("image and depth spatial dimensions must agree")
        if not (self.image.device == self.depth.device == self.valid_mask.device):
            raise ValueError("image, depth, and valid_mask must be on the same device")
        if len(self.sample_ids) != self.image.shape[0]:
            raise ValueError("sample_ids length must equal the batch size")
        if any(
            identifier is not None and not isinstance(identifier, str)
            for identifier in self.sample_ids
        ):
            raise TypeError("sample_ids entries must be strings or None")
        if self.intrinsics is not None:
            if self.intrinsics.shape != (self.image.shape[0], 3, 3):
                raise ValueError("intrinsics must have shape [B, 3, 3]")
            if not self.intrinsics.is_floating_point():
                raise TypeError("intrinsics must be a floating-point tensor")
            if self.intrinsics.device != self.image.device:
                raise ValueError("intrinsics and spatial tensors must be on the same device")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> DepthBatch:
        """Return a new batch with tensors transferred to ``device``."""
        intrinsics = (
            None
            if self.intrinsics is None
            else self.intrinsics.to(device=device, non_blocking=non_blocking)
        )
        return DepthBatch(
            image=self.image.to(device=device, non_blocking=non_blocking),
            depth=self.depth.to(device=device, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(device=device, non_blocking=non_blocking),
            intrinsics=intrinsics,
            sample_ids=self.sample_ids,
        )


def collate_depth_samples(samples: Sequence[DepthSample]) -> DepthBatch:
    """Stack same-sized samples without padding, normalization, or device movement.

    Intrinsics must be present for every sample or absent for every sample. Mixed calibration
    state is rejected because silently inserting placeholder matrices would corrupt geometry.
    """
    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("samples must not be empty")
    if not all(isinstance(sample, DepthSample) for sample in sample_tuple):
        raise TypeError("every item must be a DepthSample")

    intrinsics_values = tuple(sample.intrinsics for sample in sample_tuple)
    has_intrinsics = tuple(value is not None for value in intrinsics_values)
    if any(has_intrinsics) and not all(has_intrinsics):
        raise ValueError("intrinsics must be present for every sample or absent for every sample")
    intrinsics = None
    if all(has_intrinsics):
        intrinsics = torch.stack(
            tuple(value for value in intrinsics_values if value is not None),
            dim=0,
        )

    try:
        image = torch.stack(tuple(sample.image for sample in sample_tuple), dim=0)
        depth = torch.stack(tuple(sample.depth for sample in sample_tuple), dim=0)
        valid_mask = torch.stack(tuple(sample.valid_mask for sample in sample_tuple), dim=0)
    except RuntimeError as error:
        raise ValueError("all samples must have matching tensor shapes and dtypes") from error

    return DepthBatch(
        image=image,
        depth=depth,
        valid_mask=valid_mask,
        intrinsics=intrinsics,
        sample_ids=tuple(sample.sample_id for sample in sample_tuple),
    )
