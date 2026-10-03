from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from mde_transformers.metrics import (
    abs_rel,
    align_median,
    delta1,
    delta2,
    delta3,
    imagewise_mean,
    rmse,
    rmse_log,
    silog,
    sq_rel,
    valid_depth_mask,
)

DepthMetric = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def test_valid_depth_mask_checks_range_finiteness_and_positivity() -> None:
    target = torch.tensor([-1.0, 0.0, 0.099, 0.1, 1.0, 10.0, 10.001, float("nan"), float("inf")])

    mask = valid_depth_mask(target, min_depth=0.1, max_depth=10.0)

    expected = torch.tensor([False, False, False, True, True, True, False, False, False])
    torch.testing.assert_close(mask, expected)


def test_valid_depth_mask_combines_external_mask_without_broadcasting() -> None:
    target = torch.tensor([0.1, 1.0, 10.0])
    external = torch.tensor([True, False, True])

    mask = valid_depth_mask(target, 0.1, 10.0, external)

    torch.testing.assert_close(mask, external)
    with pytest.raises(ValueError, match="identical shapes"):
        valid_depth_mask(target, 0.1, 10.0, torch.tensor([[True, False, True]]))
    with pytest.raises(TypeError, match="boolean"):
        valid_depth_mask(target, 0.1, 10.0, torch.ones(3))


def _metric_example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.tensor([2.0, 2.0], dtype=torch.float64)
    target = torch.tensor([1.0, 4.0], dtype=torch.float64)
    mask = torch.tensor([True, True])
    return prediction, target, mask


def test_abs_rel_hand_calculation() -> None:
    prediction, target, mask = _metric_example()
    assert abs_rel(prediction, target, mask).item() == pytest.approx(0.75)


def test_sq_rel_hand_calculation() -> None:
    prediction, target, mask = _metric_example()
    assert sq_rel(prediction, target, mask).item() == pytest.approx(1.0)


def test_rmse_hand_calculation() -> None:
    prediction, target, mask = _metric_example()
    assert rmse(prediction, target, mask).item() == pytest.approx(math.sqrt(2.5))


def test_rmse_log_hand_calculation() -> None:
    prediction, target, mask = _metric_example()
    assert rmse_log(prediction, target, mask).item() == pytest.approx(math.log(2.0))


def test_silog_hand_calculation_without_percentage_scaling() -> None:
    prediction, target, mask = _metric_example()
    assert silog(prediction, target, mask).item() == pytest.approx(math.log(2.0))


def test_delta_thresholds_are_strict() -> None:
    target = torch.ones(4, dtype=torch.float64)
    prediction = torch.tensor([1.0, 1.25, 1.25**2, 1.25**3], dtype=torch.float64)
    mask = torch.ones(4, dtype=torch.bool)

    assert delta1(prediction, target, mask).item() == pytest.approx(0.25)
    assert delta2(prediction, target, mask).item() == pytest.approx(0.5)
    assert delta3(prediction, target, mask).item() == pytest.approx(0.75)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_metrics_support_required_floating_dtypes(dtype: torch.dtype) -> None:
    prediction = torch.tensor([1.0, 2.0], dtype=dtype)
    target = torch.tensor([1.0, 1.0], dtype=dtype)
    mask = torch.ones(2, dtype=torch.bool)

    value = abs_rel(prediction, target, mask)

    assert value.dtype == dtype
    assert value.device == prediction.device


def test_shape_mismatch_is_rejected_instead_of_broadcast() -> None:
    prediction = torch.ones((2, 1))
    target = torch.ones(2)
    mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="identical shapes"):
        abs_rel(prediction, target, mask)


def test_empty_mask_is_rejected() -> None:
    values = torch.ones(3)
    with pytest.raises(ValueError, match="no valid target pixels"):
        abs_rel(values, values, torch.zeros(3, dtype=torch.bool))


@pytest.mark.parametrize("metric", [abs_rel, sq_rel, rmse, rmse_log, silog, delta1, delta2, delta3])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_selected_prediction_is_rejected(
    metric: DepthMetric,
    invalid_value: float,
) -> None:
    prediction = torch.tensor([1.0, invalid_value])
    target = torch.ones(2)
    mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="selected predictions"):
        metric(prediction, target, mask)


def test_invalid_prediction_outside_target_mask_is_not_evaluated() -> None:
    prediction = torch.tensor([1.0, float("nan")])
    target = torch.tensor([1.0, float("nan")])
    mask = torch.tensor([True, False])

    assert abs_rel(prediction, target, mask).item() == 0.0


def test_invalid_selected_target_is_rejected() -> None:
    prediction = torch.ones(2)
    target = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="target depths must be positive"):
        abs_rel(prediction, target, mask)


def test_pixel_weighted_and_equal_weight_imagewise_aggregation_are_distinct() -> None:
    prediction = torch.tensor([[4.0, float("nan"), float("nan")], [2.0, 2.0, 2.0]])
    target = torch.ones_like(prediction)
    mask = torch.tensor([[True, False, False], [True, True, True]])

    global_value = abs_rel(prediction, target, mask)
    per_image_value = imagewise_mean(abs_rel, prediction, target, mask)

    assert global_value.item() == pytest.approx(1.5)
    assert per_image_value.item() == pytest.approx(2.0)


def test_imagewise_aggregation_rejects_an_empty_image_mask() -> None:
    values = torch.ones((2, 2))
    mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="image at index 1"):
        imagewise_mean(abs_rel, values, values, mask)


def test_median_alignment_returns_known_scale_without_mutation() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 4.0, 6.0])
    original = prediction.clone()
    mask = torch.ones(3, dtype=torch.bool)

    aligned, scale = align_median(prediction, target, mask, return_scale=True)

    torch.testing.assert_close(aligned, target)
    assert scale.item() == pytest.approx(2.0)
    torch.testing.assert_close(prediction, original)


def test_median_alignment_can_return_only_the_prediction() -> None:
    prediction = torch.tensor([1.0, 3.0])
    target = torch.tensor([2.0, 6.0])
    mask = torch.ones(2, dtype=torch.bool)

    aligned = align_median(prediction, target, mask)

    torch.testing.assert_close(aligned, target)


@pytest.mark.parametrize(
    "prediction",
    [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, float("nan"), 3.0])],
)
def test_median_alignment_rejects_invalid_prediction_median(prediction: torch.Tensor) -> None:
    target = torch.ones(3)
    mask = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="selected predictions"):
        align_median(prediction, target, mask)


def test_median_alignment_rejects_empty_mask() -> None:
    values = torch.ones(3)
    with pytest.raises(ValueError, match="no valid target pixels"):
        align_median(values, values, torch.zeros(3, dtype=torch.bool))
