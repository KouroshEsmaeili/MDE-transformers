"""No-gradient batch evaluation with explicit scale-alignment semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from mde_transformers.data import DepthBatch
from mde_transformers.losses import masked_l1_loss, multi_scale_depth_loss
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
from mde_transformers.models import MonocularDepthModel
from mde_transformers.models.decoders import DecoderOutput

EvaluationAlignment = Literal["none", "median"]


@dataclass(frozen=True, slots=True)
class DepthMetricResult:
    """Equal-image-weight metric values for one evaluated batch."""

    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    silog: float
    delta1: float
    delta2: float
    delta3: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Raw multi-scale loss plus explicitly aligned full-resolution metrics."""

    total_loss: float
    stage_losses: tuple[float, ...]
    metrics: DepthMetricResult
    alignment: EvaluationAlignment
    depth_range: tuple[float, float] | None


@torch.no_grad()
def evaluate_depth_batch(
    model: MonocularDepthModel,
    batch: DepthBatch,
    *,
    device: torch.device,
    stage_weights: Sequence[float],
    alignment: EvaluationAlignment,
    depth_range: tuple[float, float] | None = None,
) -> EvaluationResult:
    """Evaluate a batch without changing parameter trainability.

    The multi-scale masked-L1 loss always uses raw D1--D4 physical depths. Metrics use the
    full-resolution output and equal-weight per-image aggregation. ``alignment='median'`` applies
    a separate scale to each image for metrics only; ``'none'`` performs metric-scale evaluation.
    When supplied, ``depth_range=(minimum, maximum)`` intersects the authoritative dataset mask
    with finite-positive inclusive target range validity for both validation loss and metrics.
    The target tensor is never clipped or mutated.
    """
    if alignment not in ("none", "median"):
        raise ValueError("alignment must be 'none' or 'median'")
    model.eval()
    batch_on_device = batch.to(device)
    output = model(batch_on_device.image)
    if not isinstance(output, DecoderOutput):
        raise TypeError("model must return DecoderOutput")
    evaluation_mask = batch_on_device.valid_mask
    if depth_range is not None:
        if not isinstance(depth_range, tuple) or len(depth_range) != 2:
            raise ValueError("depth_range must contain (minimum, maximum)")
        evaluation_mask = valid_depth_mask(
            batch_on_device.depth,
            depth_range[0],
            depth_range[1],
            batch_on_device.valid_mask,
        )
    loss_result = multi_scale_depth_loss(
        output.stage_depths(),
        batch_on_device.depth,
        evaluation_mask,
        stage_weights,
        loss_fn=masked_l1_loss,
    )

    metric_prediction = output.full_resolution
    if alignment == "median":
        metric_prediction = torch.stack(
            tuple(
                align_median(
                    metric_prediction[index],
                    batch_on_device.depth[index],
                    evaluation_mask[index],
                )
                for index in range(metric_prediction.shape[0])
            ),
            dim=0,
        )

    def metric_value(
        metric: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        value = imagewise_mean(
            metric,
            metric_prediction,
            batch_on_device.depth,
            evaluation_mask,
        )
        return float(value.item())

    metrics = DepthMetricResult(
        abs_rel=metric_value(abs_rel),
        sq_rel=metric_value(sq_rel),
        rmse=metric_value(rmse),
        rmse_log=metric_value(rmse_log),
        silog=metric_value(silog),
        delta1=metric_value(delta1),
        delta2=metric_value(delta2),
        delta3=metric_value(delta3),
    )
    return EvaluationResult(
        total_loss=float(loss_result.total.item()),
        stage_losses=tuple(float(loss.item()) for loss in loss_result.stage_losses),
        metrics=metrics,
        alignment=alignment,
        depth_range=depth_range,
    )
