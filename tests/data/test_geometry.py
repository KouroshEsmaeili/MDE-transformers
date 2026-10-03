from __future__ import annotations

import pytest
import torch

from mde_transformers.data import (
    crop_intrinsics,
    disparity_to_depth,
    flip_intrinsics_horizontal,
    resize_intrinsics,
)


def _intrinsics() -> torch.Tensor:
    return torch.tensor(
        [
            [100.0, 0.0, 50.0],
            [0.0, 200.0, 40.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def test_resize_intrinsics_uses_independent_scales_and_half_pixel_centers() -> None:
    intrinsics = _intrinsics()

    resized = resize_intrinsics(intrinsics, original_size=(100, 200), new_size=(50, 400))

    expected = torch.tensor(
        [
            [200.0, 0.0, 100.5],
            [0.0, 100.0, 19.75],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(resized, expected)


def test_resize_intrinsics_maps_top_left_center_with_half_pixel_offset() -> None:
    intrinsics = _intrinsics()
    intrinsics[0, 2] = 0.0
    intrinsics[1, 2] = 0.0

    resized = resize_intrinsics(intrinsics, original_size=(10, 10), new_size=(20, 5))

    expected = torch.tensor(
        [
            [50.0, 0.0, -0.25],
            [0.0, 400.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(resized, expected)


def test_crop_intrinsics_updates_only_principal_point() -> None:
    intrinsics = _intrinsics()

    cropped = crop_intrinsics(intrinsics, left=10, top=5)

    expected = torch.tensor(
        [
            [100.0, 0.0, 40.0],
            [0.0, 200.0, 35.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(cropped, expected)


def test_horizontal_flip_intrinsics_uses_discrete_pixel_coordinates() -> None:
    intrinsics = _intrinsics()

    flipped = flip_intrinsics_horizontal(intrinsics, image_width=200)

    expected = _intrinsics()
    expected[0, 2] = 149.0
    torch.testing.assert_close(flipped, expected)


def test_intrinsics_operations_do_not_mutate_source() -> None:
    intrinsics = _intrinsics()
    original = intrinsics.clone()

    resize_intrinsics(intrinsics, (100, 200), (50, 400))
    crop_intrinsics(intrinsics, left=10, top=5)
    flip_intrinsics_horizontal(intrinsics, 200)

    torch.testing.assert_close(intrinsics, original)


def test_disparity_to_depth_known_values_and_invalid_disparities() -> None:
    disparity = torch.tensor([1.0, 2.0, 0.0, -1.0, float("nan")], dtype=torch.float64)

    depth, valid = disparity_to_depth(disparity, fx=100.0, baseline=0.2)

    torch.testing.assert_close(depth[:2], torch.tensor([20.0, 10.0], dtype=torch.float64))
    assert torch.isnan(depth[2:]).all()
    torch.testing.assert_close(valid, torch.tensor([True, True, False, False, False]))
    assert depth.dtype == disparity.dtype
    assert depth.device == disparity.device


def test_disparity_to_depth_is_unit_consistent() -> None:
    disparity_pixels = torch.tensor([35.0], dtype=torch.float32)
    focal_length_pixels = torch.tensor(700.0)
    baseline_meters = 0.54

    depth_meters, valid = disparity_to_depth(
        disparity_pixels,
        fx=focal_length_pixels,
        baseline=baseline_meters,
    )

    assert depth_meters.item() == pytest.approx(10.8)
    assert valid.item()


def test_disparity_to_depth_supports_explicit_invalid_value() -> None:
    depth, valid = disparity_to_depth(
        torch.tensor([0.0, 2.0]),
        fx=10.0,
        baseline=1.0,
        invalid_value=0.0,
    )

    torch.testing.assert_close(depth, torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(valid, torch.tensor([False, True]))


@pytest.mark.parametrize(
    ("fx", "baseline"),
    [(0.0, 0.2), (float("nan"), 0.2), (100.0, 0.0), (100.0, float("inf"))],
)
def test_disparity_to_depth_rejects_invalid_geometry(fx: float, baseline: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        disparity_to_depth(torch.tensor([1.0]), fx=fx, baseline=baseline)


def test_intrinsics_shape_is_validated() -> None:
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        resize_intrinsics(torch.eye(4), (10, 10), (20, 20))
