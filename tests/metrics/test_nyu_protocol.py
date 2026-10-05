from __future__ import annotations

import pytest
import torch

from mde_transformers.metrics import (
    NYU_EIGEN_CROP,
    NYU_NATIVE_IMAGE_SIZE,
    NYUEvaluationProtocol,
    nyu_eigen_protocol,
    nyu_native_crop_mask,
    raw_metric_protocol,
)


def test_standard_nyu_eigen_crop_has_exact_native_half_open_bounds() -> None:
    assert (
        NYU_EIGEN_CROP.top,
        NYU_EIGEN_CROP.bottom,
        NYU_EIGEN_CROP.left,
        NYU_EIGEN_CROP.right,
    ) == (45, 471, 41, 601)
    assert NYU_EIGEN_CROP.bottom - NYU_EIGEN_CROP.top == 426
    assert NYU_EIGEN_CROP.right - NYU_EIGEN_CROP.left == 560


def test_standard_nyu_eigen_crop_mask_has_exact_area_and_edges() -> None:
    mask = nyu_native_crop_mask("nyu_eigen", NYU_NATIVE_IMAGE_SIZE)

    assert mask.shape == (480, 640)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 426 * 560
    assert not mask[44, 41]
    assert mask[45, 41]
    assert mask[470, 600]
    assert not mask[471, 600]
    assert not mask[470, 601]


def test_nyu_eigen_crop_rejects_already_resized_coordinate_space() -> None:
    with pytest.raises(ValueError, match="native NYU 480x640"):
        nyu_native_crop_mask("nyu_eigen", (240, 320))


def test_raw_metric_protocol_is_uncropped_unaligned_metric_depth() -> None:
    protocol = raw_metric_protocol()

    assert protocol.depth_range == (0.1, 10.0)
    assert protocol.alignment == "none"
    assert protocol.crop == "none"
    assert protocol.profile_name == "raw_metric_baseline"


def test_eigen_crop_and_median_alignment_are_independent() -> None:
    unaligned_eigen = nyu_eigen_protocol(alignment="none")
    aligned_uncropped = NYUEvaluationProtocol(alignment="median", crop="none")

    assert unaligned_eigen.crop == "nyu_eigen"
    assert unaligned_eigen.alignment == "none"
    assert unaligned_eigen.profile_name == "standard_nyu_eigen"
    assert aligned_uncropped.crop == "none"
    assert aligned_uncropped.alignment == "median"
    assert aligned_uncropped.profile_name == "explicit_custom"
