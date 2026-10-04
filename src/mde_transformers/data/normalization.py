"""Explicit image-only normalization for supervised depth samples."""

from __future__ import annotations

from mde_transformers.data.sample import DepthSample

IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


def normalize_imagenet_sample(sample: DepthSample) -> DepthSample:
    """Return a sample whose RGB image uses ImageNet channel normalization.

    The input image is expected to be floating-point RGB in nominal range ``[0, 1]`` with shape
    ``[3, H, W]``. Normalization is ``(image - mean) / std`` channel by channel. Depth in meters,
    the authoritative validity mask, intrinsics, and sample identifier are preserved unchanged.
    The caller-owned image is not modified.
    """
    mean = sample.image.new_tensor(IMAGENET_RGB_MEAN).view(3, 1, 1)
    std = sample.image.new_tensor(IMAGENET_RGB_STD).view(3, 1, 1)
    image = (sample.image - mean) / std
    return DepthSample(
        image=image,
        depth=sample.depth,
        valid_mask=sample.valid_mask,
        intrinsics=sample.intrinsics,
        sample_id=sample.sample_id,
    )
