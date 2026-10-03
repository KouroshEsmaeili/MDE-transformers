from __future__ import annotations

import math

import pytest
import torch

from mde_transformers.losses import (
    depth_gradient_consistency_loss,
    edge_aware_smoothness_loss,
    surface_normal_consistency_loss,
)


def _image(height: int, width: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.zeros((1, 3, height, width), dtype=dtype)


def test_smoothness_is_zero_for_constant_depth() -> None:
    depth = torch.full((1, 1, 2, 3), 2.0, dtype=torch.float64)

    loss = edge_aware_smoothness_loss(depth, _image(2, 3), representation="log")

    assert loss.item() == pytest.approx(0.0)


def test_smoothness_inverse_depth_edge_with_constant_image() -> None:
    depth = torch.tensor([[[[1.0, 2.0, 4.0]]]], dtype=torch.float64)

    loss = edge_aware_smoothness_loss(depth, _image(1, 3), representation="inverse")

    # Inverse-depth edge magnitudes are 1/2 and 1/4.
    assert loss.item() == pytest.approx(0.375)


def test_rgb_edge_exponentially_suppresses_smoothness() -> None:
    depth = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
    image = torch.tensor([[[[0.0, 2.0]], [[0.0, 2.0]], [[0.0, 2.0]]]], dtype=torch.float64)

    loss = edge_aware_smoothness_loss(
        depth,
        image,
        representation="inverse",
        beta=1.0,
    )

    assert loss.item() == pytest.approx(0.5 * math.exp(-2.0))


def test_smoothness_requires_both_mask_endpoints() -> None:
    depth = torch.tensor([[[[1.0, 2.0, 0.0]]]], dtype=torch.float64)
    mask = torch.tensor([[[[True, True, False]]]])

    loss = edge_aware_smoothness_loss(
        depth,
        _image(1, 3),
        mask,
        representation="inverse",
    )

    assert loss.item() == pytest.approx(0.5)


def test_smoothness_rejects_no_valid_neighbor_pairs() -> None:
    depth = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    mask = torch.tensor([[[[True, False, True]]]])

    with pytest.raises(ValueError, match="no valid adjacent"):
        edge_aware_smoothness_loss(
            depth, _image(1, 3, dtype=torch.float32), mask, representation="log"
        )


@pytest.mark.parametrize("representation", ["metric", "depth", ""])
def test_smoothness_rejects_unknown_representation(representation: str) -> None:
    depth = torch.ones((1, 1, 1, 2))
    with pytest.raises(ValueError, match="representation"):
        edge_aware_smoothness_loss(
            depth,
            _image(1, 2, dtype=torch.float32),
            representation=representation,  # type: ignore[arg-type]
        )


def test_gradient_consistency_is_zero_for_identical_maps() -> None:
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = depth_gradient_consistency_loss(target, target, mask)

    assert loss.item() == pytest.approx(0.0)


def test_gradient_consistency_known_metric_depth_value() -> None:
    prediction = torch.tensor([[[[1.0, 2.0, 4.0]]]], dtype=torch.float64)
    target = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)

    loss = depth_gradient_consistency_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(1.5)


def test_gradient_consistency_requires_both_mask_endpoints() -> None:
    prediction = torch.tensor([[[[1.0, 2.0, float("nan")]]]])
    target = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    mask = torch.tensor([[[[True, True, False]]]])

    loss = depth_gradient_consistency_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(1.0)


def test_log_gradient_consistency_is_invariant_to_constant_scale() -> None:
    target = torch.tensor([[[[1.0, 2.0, 4.0]]]], dtype=torch.float64)
    prediction = 3.0 * target
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = depth_gradient_consistency_loss(
        prediction,
        target,
        mask,
        representation="log",
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-15)


def test_gradient_consistency_rejects_no_valid_neighbor_pairs() -> None:
    values = torch.ones((1, 1, 1, 3))
    mask = torch.tensor([[[[True, False, True]]]])
    with pytest.raises(ValueError, match="no valid adjacent"):
        depth_gradient_consistency_loss(values, values, mask)


def _intrinsics(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def test_surface_normal_loss_is_zero_for_identical_plane() -> None:
    depth = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    mask = torch.ones_like(depth, dtype=torch.bool)

    loss = surface_normal_consistency_loss(depth, depth, mask, _intrinsics())

    assert loss.item() == pytest.approx(0.0, abs=1e-15)


def test_surface_normal_loss_detects_changed_plane() -> None:
    target = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    prediction = torch.tensor(
        [[[[1.0, 1.2, 1.4], [1.0, 1.2, 1.4], [1.0, 1.2, 1.4]]]],
        dtype=torch.float64,
    )
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = surface_normal_consistency_loss(prediction, target, mask, _intrinsics())

    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_surface_normal_supports_batched_intrinsics() -> None:
    depth = torch.ones((2, 1, 2, 2), dtype=torch.float64)
    mask = torch.ones_like(depth, dtype=torch.bool)
    intrinsics = torch.stack((_intrinsics(), _intrinsics()))

    loss = surface_normal_consistency_loss(depth, depth, mask, intrinsics)

    assert loss.item() == pytest.approx(0.0, abs=1e-15)


def test_surface_normal_requires_complete_valid_neighborhood() -> None:
    depth = torch.ones((1, 1, 2, 2), dtype=torch.float64)
    mask = torch.tensor([[[[True, True], [False, False]]]])

    with pytest.raises(ValueError, match="no complete normal"):
        surface_normal_consistency_loss(depth, depth, mask, _intrinsics())


def test_surface_normal_rejects_missing_or_malformed_intrinsics() -> None:
    depth = torch.ones((1, 1, 2, 2), dtype=torch.float64)
    mask = torch.ones_like(depth, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        surface_normal_consistency_loss(depth, depth, mask, torch.eye(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="canonical"):
        bad = _intrinsics().clone()
        bad[2, 0] = 1.0
        surface_normal_consistency_loss(depth, depth, mask, bad)


@pytest.mark.parametrize("loss_name", ["smoothness", "gradient", "normal"])
def test_spatial_losses_produce_finite_gradients(loss_name: str) -> None:
    prediction = torch.tensor(
        [[[[1.0, 1.3, 1.6], [1.1, 1.5, 1.8], [1.2, 1.6, 2.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    if loss_name == "smoothness":
        loss = edge_aware_smoothness_loss(
            prediction,
            _image(3, 3),
            mask,
            representation="log",
        )
    elif loss_name == "gradient":
        loss = depth_gradient_consistency_loss(prediction, target, mask)
    else:
        loss = surface_normal_consistency_loss(prediction, target, mask, _intrinsics())

    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
