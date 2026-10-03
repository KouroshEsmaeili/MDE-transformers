from __future__ import annotations

import pytest
import torch

from mde_transformers.data import DepthSample


def _components() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.zeros((3, 2, 4), dtype=torch.float32)
    depth = torch.ones((1, 2, 4), dtype=torch.float32)
    valid_mask = torch.ones((1, 2, 4), dtype=torch.bool)
    intrinsics = torch.tensor(
        [[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return image, depth, valid_mask, intrinsics


def test_valid_depth_sample_contract() -> None:
    image, depth, valid_mask, intrinsics = _components()

    sample = DepthSample(image, depth, valid_mask, intrinsics, sample_id="scene/frame")

    assert sample.image.shape == (3, 2, 4)
    assert sample.depth.shape == (1, 2, 4)
    assert sample.valid_mask.dtype == torch.bool
    assert sample.image_size == (2, 4)
    assert sample.sample_id == "scene/frame"


def test_depth_sample_accepts_missing_intrinsics() -> None:
    image, depth, valid_mask, _ = _components()
    sample = DepthSample(image, depth, valid_mask, intrinsics=None)
    assert sample.intrinsics is None


def test_depth_sample_rejects_wrong_image_channel_count() -> None:
    image, depth, valid_mask, intrinsics = _components()
    with pytest.raises(ValueError, match=r"image must have shape \(3, H, W\)"):
        DepthSample(image[:2], depth, valid_mask, intrinsics)


def test_depth_sample_rejects_wrong_depth_shape() -> None:
    image, depth, valid_mask, intrinsics = _components()
    with pytest.raises(ValueError, match=r"depth must have shape \(1, H, W\)"):
        DepthSample(image, depth.squeeze(0), valid_mask, intrinsics)


def test_depth_sample_rejects_wrong_mask_shape() -> None:
    image, depth, valid_mask, intrinsics = _components()
    with pytest.raises(ValueError, match=r"valid_mask must have shape \(1, H, W\)"):
        DepthSample(image, depth, valid_mask.squeeze(0), intrinsics)


def test_depth_sample_rejects_non_boolean_mask() -> None:
    image, depth, valid_mask, intrinsics = _components()
    with pytest.raises(TypeError, match="boolean"):
        DepthSample(image, depth, valid_mask.float(), intrinsics)


def test_depth_sample_rejects_depth_mask_spatial_mismatch() -> None:
    image, depth, valid_mask, intrinsics = _components()
    with pytest.raises(ValueError, match="depth and valid_mask"):
        DepthSample(image, depth, valid_mask[..., :3], intrinsics)


def test_depth_sample_rejects_image_depth_spatial_mismatch() -> None:
    image, depth, valid_mask, intrinsics = _components()
    with pytest.raises(ValueError, match="spatial dimensions"):
        DepthSample(image[..., :3], depth, valid_mask, intrinsics)


def test_depth_sample_rejects_wrong_intrinsics_shape() -> None:
    image, depth, valid_mask, _ = _components()
    with pytest.raises(ValueError, match=r"intrinsics must have shape \(3, 3\)"):
        DepthSample(image, depth, valid_mask, torch.eye(4))


@pytest.mark.parametrize("field", ["image", "depth"])
def test_depth_sample_rejects_non_floating_spatial_values(field: str) -> None:
    image, depth, valid_mask, intrinsics = _components()
    if field == "image":
        image = image.to(torch.uint8)
    else:
        depth = depth.to(torch.int64)

    with pytest.raises(TypeError, match="floating-point"):
        DepthSample(image, depth, valid_mask, intrinsics)
