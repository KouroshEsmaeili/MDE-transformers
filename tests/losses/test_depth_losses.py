"""Tests for masked pointwise depth-training losses."""

from __future__ import annotations

import math

import pytest
import torch

from mde_transformers.losses import (
    berhu_loss,
    masked_huber_loss,
    masked_l1_loss,
    scale_invariant_log_loss,
)
from mde_transformers.metrics import silog


def test_masked_l1_has_exact_valid_pixel_mean() -> None:
    prediction = torch.tensor([2.0, 3.0, 9.0], dtype=torch.float64)
    target = torch.tensor([1.0, 5.0, 1.0], dtype=torch.float64)
    mask = torch.tensor([True, True, False])

    assert masked_l1_loss(prediction, target, mask).item() == pytest.approx(1.5)


def test_masked_l1_does_not_evaluate_values_outside_mask() -> None:
    prediction = torch.tensor([2.0, float("nan")])
    target = torch.tensor([1.0, float("nan")])
    mask = torch.tensor([True, False])

    assert masked_l1_loss(prediction, target, mask).item() == pytest.approx(1.0)


def test_masked_huber_quadratic_regime() -> None:
    prediction = torch.tensor([1.5])
    target = torch.tensor([1.0])
    mask = torch.tensor([True])

    assert masked_huber_loss(prediction, target, mask, delta=1.0).item() == pytest.approx(0.125)


def test_masked_huber_linear_regime_and_mask() -> None:
    prediction = torch.tensor([3.0, 100.0])
    target = torch.tensor([1.0, 1.0])
    mask = torch.tensor([True, False])

    assert masked_huber_loss(prediction, target, mask, delta=1.0).item() == pytest.approx(1.5)


@pytest.mark.parametrize("delta", [0.0, -1.0, float("nan")])
def test_masked_huber_rejects_invalid_delta(delta: float) -> None:
    values = torch.ones(1)
    with pytest.raises(ValueError, match="delta"):
        masked_huber_loss(values, values, torch.ones(1, dtype=torch.bool), delta=delta)


def test_berhu_known_global_threshold_and_values() -> None:
    prediction = torch.tensor([2.0, 6.0], dtype=torch.float64)
    target = torch.ones(2, dtype=torch.float64)
    mask = torch.ones(2, dtype=torch.bool)

    # Errors [1, 5] give c = 0.2 * 5 = 1 and losses [1, (25 + 1) / 2] = [1, 13].
    assert berhu_loss(prediction, target, mask).item() == pytest.approx(7.0)


def test_berhu_mask_controls_maximum_error_threshold() -> None:
    prediction = torch.tensor([2.0, 6.0], dtype=torch.float64)
    target = torch.ones(2, dtype=torch.float64)
    mask = torch.tensor([True, False])

    # Only error 1 is selected: c = 0.2 and loss = (1 + 0.04) / 0.4 = 2.6.
    assert berhu_loss(prediction, target, mask).item() == pytest.approx(2.6)


def test_berhu_perfect_prediction_is_differentiable_zero() -> None:
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    target = prediction.detach().clone()
    mask = torch.ones(2, dtype=torch.bool)

    loss = berhu_loss(prediction, target, mask)
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, float("inf"), float("nan")])
def test_berhu_rejects_invalid_threshold_ratio(ratio: float) -> None:
    values = torch.ones(1)
    with pytest.raises(ValueError, match="threshold_ratio"):
        berhu_loss(values, values, torch.ones(1, dtype=torch.bool), threshold_ratio=ratio)


def test_scale_invariant_log_loss_is_zero_for_constant_scale_relation() -> None:
    target = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    prediction = 3.0 * target
    mask = torch.ones(3, dtype=torch.bool)

    assert scale_invariant_log_loss(prediction, target, mask).item() == pytest.approx(
        0.0, abs=1e-15
    )


def test_scale_invariant_log_loss_is_variance_without_square_root() -> None:
    prediction = torch.tensor([2.0, 2.0], dtype=torch.float64)
    target = torch.tensor([1.0, 4.0], dtype=torch.float64)
    mask = torch.ones(2, dtype=torch.bool)
    training_loss = scale_invariant_log_loss(prediction, target, mask)
    metric_value = silog(prediction, target, mask)

    assert training_loss.item() == pytest.approx(math.log(2.0) ** 2)
    assert metric_value.item() == pytest.approx(math.log(2.0))
    assert training_loss.item() == pytest.approx(metric_value.square().item())


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_depth_losses_reject_invalid_selected_predictions(invalid: float) -> None:
    prediction = torch.tensor([1.0, invalid])
    target = torch.ones(2)
    mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="selected predictions"):
        masked_l1_loss(prediction, target, mask)


def test_depth_losses_reject_empty_mask_and_broadcasting() -> None:
    values = torch.ones(2)
    with pytest.raises(ValueError, match="no supervised"):
        masked_l1_loss(values, values, torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="identical shapes"):
        masked_l1_loss(values.reshape(2, 1), values, torch.ones(2, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        masked_l1_loss(values, values, torch.ones(2))


def test_depth_losses_reject_invalid_selected_targets() -> None:
    prediction = torch.ones(2)
    target = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="target depths must be positive"):
        scale_invariant_log_loss(prediction, target, mask)


@pytest.mark.parametrize(
    "loss_fn",
    [
        masked_l1_loss,
        lambda prediction, target, mask: masked_huber_loss(prediction, target, mask, delta=0.5),
        berhu_loss,
        scale_invariant_log_loss,
    ],
)
def test_pointwise_losses_produce_finite_gradients(loss_fn: object) -> None:
    prediction = torch.tensor([1.2, 1.8, 3.4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    mask = torch.ones(3, dtype=torch.bool)

    loss = loss_fn(prediction, target, mask)  # type: ignore[operator]
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
