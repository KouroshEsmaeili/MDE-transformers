from __future__ import annotations

import torch

from mde_transformers.data import DepthSample, normalize_imagenet_sample


def test_imagenet_normalization_is_exact_and_image_only() -> None:
    image = torch.tensor(
        [
            [[0.485, 0.714]],
            [[0.456, 0.680]],
            [[0.406, 0.631]],
        ],
        dtype=torch.float64,
    )
    depth = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)
    mask = torch.tensor([[[True, False]]])
    intrinsics = torch.eye(3, dtype=torch.float64)
    sample = DepthSample(image, depth, mask, intrinsics, sample_id="sample")
    image_before = image.clone()

    normalized = normalize_imagenet_sample(sample)

    expected = torch.tensor(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(normalized.image, expected)
    torch.testing.assert_close(sample.image, image_before)
    assert normalized.image.data_ptr() != sample.image.data_ptr()
    assert normalized.depth is sample.depth
    assert normalized.valid_mask is sample.valid_mask
    assert normalized.intrinsics is sample.intrinsics
    assert normalized.sample_id == sample.sample_id
