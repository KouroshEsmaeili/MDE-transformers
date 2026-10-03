from __future__ import annotations

import pytest
import torch

from mde_transformers.losses import (
    CompositeLossWeights,
    composite_depth_loss,
    masked_l1_loss,
    multi_scale_depth_loss,
)


def _target() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.ones((1, 1, 2, 2), dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    return target, mask


def test_multiscale_known_stage_weights_are_not_normalized() -> None:
    target, mask = _target()
    predictions = [torch.full_like(target, 2.0), torch.full_like(target, 4.0)]

    result = multi_scale_depth_loss(predictions, target, mask, [0.5, 2.0])

    assert result.stage_losses[0].item() == pytest.approx(1.0)
    assert result.stage_losses[1].item() == pytest.approx(3.0)
    assert result.weighted_stage_losses[0].item() == pytest.approx(0.5)
    assert result.weighted_stage_losses[1].item() == pytest.approx(6.0)
    assert result.total.item() == pytest.approx(6.5)


def test_predictions_are_upsampled_while_sparse_target_mask_is_unchanged() -> None:
    target = torch.tensor([[[[1.0, 10.0], [20.0, 30.0]]]], dtype=torch.float64)
    mask = torch.tensor([[[[True, False], [False, False]]]])
    prediction = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    observed: list[tuple[tuple[int, ...], int, int]] = []

    def checking_loss(
        full_prediction: torch.Tensor,
        full_target: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> torch.Tensor:
        observed.append(
            (tuple(full_prediction.shape), full_target.data_ptr(), full_mask.data_ptr())
        )
        return masked_l1_loss(full_prediction, full_target, full_mask)

    result = multi_scale_depth_loss([prediction], target, mask, [1.0], loss_fn=checking_loss)

    assert observed == [((1, 1, 2, 2), target.data_ptr(), mask.data_ptr())]
    assert result.total.item() == pytest.approx(1.0)


def test_multiscale_gradients_reach_every_positive_stage() -> None:
    target, mask = _target()
    coarse = torch.full((1, 1, 1, 1), 1.5, dtype=torch.float64, requires_grad=True)
    full = torch.full((1, 1, 2, 2), 2.0, dtype=torch.float64, requires_grad=True)

    result = multi_scale_depth_loss([coarse, full], target, mask, [0.25, 2.0])
    result.total.backward()

    assert coarse.grad is not None and torch.isfinite(coarse.grad).all()
    assert full.grad is not None and torch.isfinite(full.grad).all()
    assert coarse.grad.abs().sum().item() > 0
    assert full.grad.abs().sum().item() > 0


def test_zero_weight_stage_is_allowed_but_all_zero_is_rejected() -> None:
    target, mask = _target()
    predictions = [target.clone(), target.clone()]
    result = multi_scale_depth_loss(predictions, target, mask, [0.0, 1.0])
    assert result.total.item() == pytest.approx(0.0)

    with pytest.raises(ValueError, match="at least one"):
        multi_scale_depth_loss(predictions, target, mask, [0.0, 0.0])


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_multiscale_rejects_invalid_stage_weight(weight: float) -> None:
    target, mask = _target()
    with pytest.raises(ValueError, match="stage weight"):
        multi_scale_depth_loss([target], target, mask, [weight])


def test_multiscale_rejects_empty_or_mismatched_sequences() -> None:
    target, mask = _target()
    with pytest.raises(ValueError, match="must not be empty"):
        multi_scale_depth_loss([], target, mask, [])
    with pytest.raises(ValueError, match="same length"):
        multi_scale_depth_loss([target], target, mask, [1.0, 2.0])


def test_multiscale_rejects_bad_prediction_contract_and_invalid_values() -> None:
    target, mask = _target()
    with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
        multi_scale_depth_loss([torch.ones((1, 2, 2, 2))], target, mask, [1.0])
    invalid = target.clone()
    invalid[..., 1, 1] = float("nan")
    mask[..., 1, 1] = False
    with pytest.raises(ValueError, match="only finite"):
        multi_scale_depth_loss([invalid], target, mask, [1.0])


def test_composite_breakdown_is_exact_weighted_sum() -> None:
    target = torch.ones((1, 1, 1, 2), dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    prediction = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
    image = torch.zeros((1, 3, 1, 2), dtype=torch.float64)

    result = composite_depth_loss(
        [prediction],
        target,
        mask,
        [1.0],
        weights=CompositeLossWeights(smoothness=2.0, gradient=3.0, normal=0.0),
        regularization_prediction=prediction,
        image=image,
        smoothness_representation="inverse",
    )

    assert result.depth.item() == pytest.approx(0.5)
    assert result.smoothness.item() == pytest.approx(1.0)
    assert result.gradient.item() == pytest.approx(3.0)
    assert result.normal.item() == pytest.approx(0.0)
    assert result.total.item() == pytest.approx(4.5)
    torch.testing.assert_close(
        result.total,
        result.depth + result.smoothness + result.gradient + result.normal,
    )
    torch.testing.assert_close(sum(result.per_stage_depth), result.depth)


def test_composite_zero_weights_disable_optional_inputs() -> None:
    target, mask = _target()

    result = composite_depth_loss(
        [target],
        target,
        mask,
        [1.0],
        weights=CompositeLossWeights(smoothness=0.0, gradient=0.0, normal=0.0),
    )

    assert result.total.item() == 0.0
    assert result.smoothness.item() == 0.0
    assert result.gradient.item() == 0.0
    assert result.normal.item() == 0.0


def test_composite_normal_requires_intrinsics_only_when_enabled() -> None:
    target, mask = _target()
    weights = CompositeLossWeights(smoothness=0.0, gradient=0.0, normal=1.0)

    with pytest.raises(ValueError, match="intrinsics"):
        composite_depth_loss(
            [target],
            target,
            mask,
            [1.0],
            weights=weights,
            regularization_prediction=target,
        )


def test_composite_rejects_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="smoothness weight"):
        CompositeLossWeights(smoothness=-1.0, gradient=0.0, normal=0.0)


def test_multiscale_requires_scalar_loss_result() -> None:
    target, mask = _target()

    def bad_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del target, mask
        return prediction

    with pytest.raises(ValueError, match="scalar"):
        multi_scale_depth_loss([target], target, mask, [1.0], loss_fn=bad_loss)
