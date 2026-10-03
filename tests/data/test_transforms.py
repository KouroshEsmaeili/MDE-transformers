from __future__ import annotations

import pytest
import torch

from mde_transformers.data import (
    DepthSample,
    crop_depth_sample,
    horizontal_flip_depth_sample,
    resize_depth_sample,
)


def _sample(*, with_intrinsics: bool = True) -> DepthSample:
    depth = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 4)
    image = torch.cat((depth / 12.0, depth / 24.0, depth / 48.0), dim=0)
    valid_mask = depth.remainder(2) == 0
    intrinsics = None
    if with_intrinsics:
        intrinsics = torch.tensor([[4.0, 0.0, 1.0], [0.0, 6.0, 1.0], [0.0, 0.0, 1.0]])
    return DepthSample(image, depth, valid_mask, intrinsics, sample_id="tiny")


def _snapshot(sample: DepthSample) -> tuple[torch.Tensor, ...]:
    tensors = (sample.image.clone(), sample.depth.clone(), sample.valid_mask.clone())
    if sample.intrinsics is None:
        return tensors
    return (*tensors, sample.intrinsics.clone())


def _assert_unchanged(sample: DepthSample, snapshot: tuple[torch.Tensor, ...]) -> None:
    torch.testing.assert_close(sample.image, snapshot[0])
    torch.testing.assert_close(sample.depth, snapshot[1])
    torch.testing.assert_close(sample.valid_mask, snapshot[2])
    if sample.intrinsics is not None:
        torch.testing.assert_close(sample.intrinsics, snapshot[3])


def test_resize_uses_nearest_exact_for_depth_and_authoritative_mask() -> None:
    image = torch.tensor(
        [
            [[0.0, 0.2], [0.4, 0.6]],
            [[0.1, 0.3], [0.5, 0.7]],
            [[0.2, 0.4], [0.6, 0.8]],
        ]
    )
    depth = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    valid_mask = torch.tensor([[[True, False], [False, True]]])
    intrinsics = torch.tensor([[2.0, 0.0, 0.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]])
    sample = DepthSample(image, depth, valid_mask, intrinsics)
    snapshot = _snapshot(sample)

    resized = resize_depth_sample(sample, (4, 4))

    expected_depth = depth.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    expected_mask = valid_mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    expected_intrinsics = torch.tensor([[4.0, 0.0, 1.5], [0.0, 8.0, 1.5], [0.0, 0.0, 1.0]])
    assert resized.image.shape == (3, 4, 4)
    assert resized.depth.shape == (1, 4, 4)
    assert resized.valid_mask.shape == (1, 4, 4)
    assert resized.valid_mask.dtype == torch.bool
    torch.testing.assert_close(resized.depth, expected_depth)
    torch.testing.assert_close(resized.valid_mask, expected_mask)
    assert resized.intrinsics is not None
    torch.testing.assert_close(resized.intrinsics, expected_intrinsics)
    assert set(resized.depth.unique().tolist()) == {1.0, 2.0, 3.0, 4.0}
    _assert_unchanged(sample, snapshot)


def test_resize_preserves_missing_intrinsics() -> None:
    sample = _sample(with_intrinsics=False)
    resized = resize_depth_sample(sample, (6, 8))
    assert resized.intrinsics is None


def test_crop_slices_every_component_and_updates_intrinsics() -> None:
    sample = _sample()
    snapshot = _snapshot(sample)

    cropped = crop_depth_sample(sample, top=1, left=1, height=2, width=2)

    torch.testing.assert_close(cropped.image, sample.image[:, 1:3, 1:3])
    torch.testing.assert_close(cropped.depth, sample.depth[:, 1:3, 1:3])
    torch.testing.assert_close(cropped.valid_mask, sample.valid_mask[:, 1:3, 1:3])
    expected_intrinsics = torch.tensor([[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 1.0]])
    assert cropped.intrinsics is not None
    torch.testing.assert_close(cropped.intrinsics, expected_intrinsics)
    _assert_unchanged(sample, snapshot)

    cropped.image[0, 0, 0] = -1.0
    cropped.depth[0, 0, 0] = -1.0
    cropped.valid_mask[0, 0, 0] = ~cropped.valid_mask[0, 0, 0]
    cropped.intrinsics[0, 2] = -1.0
    _assert_unchanged(sample, snapshot)


@pytest.mark.parametrize(
    ("top", "left", "height", "width"),
    [
        (-1, 0, 1, 1),
        (0, -1, 1, 1),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (2, 0, 2, 1),
        (0, 3, 1, 2),
    ],
)
def test_crop_rejects_invalid_bounds(top: int, left: int, height: int, width: int) -> None:
    with pytest.raises(ValueError):
        crop_depth_sample(_sample(), top=top, left=left, height=height, width=width)


def test_horizontal_flip_reverses_spatial_components_and_updates_intrinsics() -> None:
    sample = _sample()
    snapshot = _snapshot(sample)

    flipped = horizontal_flip_depth_sample(sample)

    torch.testing.assert_close(flipped.image, torch.flip(sample.image, dims=(-1,)))
    torch.testing.assert_close(flipped.depth, torch.flip(sample.depth, dims=(-1,)))
    torch.testing.assert_close(flipped.valid_mask, torch.flip(sample.valid_mask, dims=(-1,)))
    assert flipped.intrinsics is not None
    expected_intrinsics = torch.tensor([[4.0, 0.0, 2.0], [0.0, 6.0, 1.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(flipped.intrinsics, expected_intrinsics)
    torch.testing.assert_close(flipped.depth.sort().values, sample.depth.sort().values)
    _assert_unchanged(sample, snapshot)


def test_resize_crop_flip_composition_keeps_components_aligned() -> None:
    sample = _sample()

    transformed = horizontal_flip_depth_sample(
        crop_depth_sample(
            resize_depth_sample(sample, (6, 8)),
            top=1,
            left=2,
            height=4,
            width=5,
        )
    )

    expected_depth = torch.tensor(
        [
            [
                [4.0, 3.0, 3.0, 2.0, 2.0],
                [8.0, 7.0, 7.0, 6.0, 6.0],
                [8.0, 7.0, 7.0, 6.0, 6.0],
                [12.0, 11.0, 11.0, 10.0, 10.0],
            ]
        ]
    )
    expected_intrinsics = torch.tensor([[8.0, 0.0, 3.5], [0.0, 12.0, 1.5], [0.0, 0.0, 1.0]])
    assert transformed.image.shape == (3, 4, 5)
    assert transformed.depth.shape == (1, 4, 5)
    assert transformed.valid_mask.shape == (1, 4, 5)
    torch.testing.assert_close(transformed.depth, expected_depth)
    torch.testing.assert_close(transformed.valid_mask, transformed.depth.remainder(2) == 0)
    assert transformed.intrinsics is not None
    torch.testing.assert_close(transformed.intrinsics, expected_intrinsics)
