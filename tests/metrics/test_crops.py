import pytest
import torch

from mde_transformers.metrics import CropBounds


def test_crop_bounds_create_half_open_mask() -> None:
    crop = CropBounds(top=1, left=2, bottom=3, right=5)

    mask = crop.to_mask((4, 6))

    expected = torch.zeros((4, 6), dtype=torch.bool)
    expected[1:3, 2:5] = True
    torch.testing.assert_close(mask, expected)


def test_crop_bounds_reject_out_of_image_extent() -> None:
    crop = CropBounds(top=0, left=0, bottom=5, right=2)
    with pytest.raises(ValueError, match="beyond"):
        crop.to_mask((4, 6))


def test_crop_bounds_do_not_encode_dataset_protocol_constants() -> None:
    crop = CropBounds(top=2, left=3, bottom=7, right=11)
    assert (crop.top, crop.left, crop.bottom, crop.right) == (2, 3, 7, 11)
