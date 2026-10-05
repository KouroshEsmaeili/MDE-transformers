from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from mde_transformers.data import DepthSample, crop_intrinsics, resize_intrinsics
from mde_transformers.engine import NYUEvaluationPreprocess, NYUPreprocess
from mde_transformers.metrics import NYU_EIGEN_CROP, nyu_native_crop_mask


def _native_sample() -> DepthSample:
    image = torch.zeros((3, 480, 640), dtype=torch.float32)
    depth = torch.arange(480 * 640, dtype=torch.float32).reshape(1, 480, 640) + 1.0
    mask = torch.ones_like(depth, dtype=torch.bool)
    intrinsics = torch.tensor(
        [[500.0, 0.0, 319.5], [0.0, 510.0, 239.5], [0.0, 0.0, 1.0]],
    )
    return DepthSample(image, depth, mask, intrinsics, "native")


def test_evaluation_crop_mask_is_built_native_then_resized_nearest_exact() -> None:
    sample = _native_sample()
    transformed = NYUEvaluationPreprocess(240, 320, crop="nyu_eigen")(sample)
    native_mask = nyu_native_crop_mask("nyu_eigen", (480, 640))
    expected_mask = (
        functional.interpolate(
            native_mask[None, None].to(torch.float32),
            size=(240, 320),
            mode="nearest-exact",
        )
        .squeeze(0)
        .to(torch.bool)
    )

    assert transformed.image.shape == (3, 240, 320)
    torch.testing.assert_close(transformed.valid_mask, expected_mask)
    assert int(transformed.valid_mask.sum()) != 426 * 560
    assert transformed.intrinsics is not None
    assert sample.intrinsics is not None
    torch.testing.assert_close(
        transformed.intrinsics,
        resize_intrinsics(sample.intrinsics, (480, 640), (240, 320)),
    )


def test_training_crop_is_synchronized_and_occurs_before_resize() -> None:
    sample = _native_sample()
    transformed = NYUPreprocess(213, 280, training_crop="nyu_eigen")(sample)
    cropped_intrinsics = crop_intrinsics(
        sample.intrinsics,
        left=NYU_EIGEN_CROP.left,
        top=NYU_EIGEN_CROP.top,
    )
    expected_intrinsics = resize_intrinsics(
        cropped_intrinsics,
        (426, 560),
        (213, 280),
    )

    assert transformed.image.shape == (3, 213, 280)
    assert transformed.depth.shape == (1, 213, 280)
    assert transformed.valid_mask.shape == (1, 213, 280)
    assert transformed.intrinsics is not None
    torch.testing.assert_close(transformed.intrinsics, expected_intrinsics)
    assert torch.all(transformed.valid_mask)
    assert sample.image.shape == (3, 480, 640)
    assert sample.depth.shape == (1, 480, 640)


def test_eigen_preprocessing_rejects_non_native_input_instead_of_guessing_coordinates() -> None:
    sample = DepthSample(
        torch.zeros((3, 240, 320)),
        torch.ones((1, 240, 320)),
        torch.ones((1, 240, 320), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="native NYU 480x640"):
        NYUEvaluationPreprocess(128, 160, crop="nyu_eigen")(sample)
    with pytest.raises(ValueError, match="native NYU 480x640"):
        NYUPreprocess(128, 160, training_crop="nyu_eigen")(sample)
