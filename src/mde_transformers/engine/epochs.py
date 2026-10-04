"""Sample-weighted supervised training and validation epochs."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from mde_transformers.data import DepthBatch
from mde_transformers.engine.evaluation import (
    DepthMetricResult,
    EvaluationAlignment,
    evaluate_depth_batch,
)
from mde_transformers.engine.scheduler import WarmupCosineScheduler
from mde_transformers.engine.training import supervised_train_step
from mde_transformers.models import MonocularDepthModel


@dataclass(frozen=True, slots=True)
class EpochLosses:
    """Sample-weighted mean total and raw D1--D4 losses."""

    total: float
    stages: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """Aggregated state from one optimization epoch."""

    epoch: int
    losses: EpochLosses
    samples: int
    optimizer_steps: int
    global_step: int
    learning_rate: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ValidationEpochResult:
    """Sample-weighted dev loss and equal-image-weight depth metrics."""

    epoch: int
    losses: EpochLosses
    metrics: DepthMetricResult
    samples: int
    alignment: EvaluationAlignment
    depth_range: tuple[float, float]
    elapsed_seconds: float


def train_one_epoch(
    model: MonocularDepthModel,
    loader: Iterable[DepthBatch],
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    *,
    device: torch.device,
    stage_weights: tuple[float, float, float, float],
    epoch: int,
    global_step: int,
    gradient_clip_norm: float | None = None,
) -> TrainingEpochResult:
    """Optimize one epoch and aggregate batch means by batch sample count."""
    _validate_epoch_and_step(epoch, global_step)
    started = time.perf_counter()
    total_sum = 0.0
    stage_sums = [0.0, 0.0, 0.0, 0.0]
    sample_count = 0
    optimizer_steps = 0

    for batch in loader:
        if not isinstance(batch, DepthBatch):
            raise TypeError("loader must yield DepthBatch instances")
        batch_size = batch.image.shape[0]
        result = supervised_train_step(
            model,
            batch,
            optimizer,
            device=device,
            stage_weights=stage_weights,
            gradient_clip_norm=gradient_clip_norm,
        )
        scheduler.step()
        total_sum += result.total_loss * batch_size
        for index, loss in enumerate(result.stage_losses):
            stage_sums[index] += loss * batch_size
        sample_count += batch_size
        optimizer_steps += 1
        global_step += 1

    if sample_count == 0:
        raise ValueError("training loader yielded no samples")
    losses = EpochLosses(
        total=total_sum / sample_count,
        stages=(
            stage_sums[0] / sample_count,
            stage_sums[1] / sample_count,
            stage_sums[2] / sample_count,
            stage_sums[3] / sample_count,
        ),
    )
    return TrainingEpochResult(
        epoch=epoch,
        losses=losses,
        samples=sample_count,
        optimizer_steps=optimizer_steps,
        global_step=global_step,
        learning_rate=scheduler.get_last_lr()[0],
        elapsed_seconds=time.perf_counter() - started,
    )


def validate_one_epoch(
    model: MonocularDepthModel,
    loader: Iterable[DepthBatch],
    *,
    device: torch.device,
    stage_weights: tuple[float, float, float, float],
    epoch: int,
    alignment: EvaluationAlignment,
    depth_range: tuple[float, float],
) -> ValidationEpochResult:
    """Evaluate one train-derived dev epoch with explicit NYU range and alignment."""
    _validate_epoch_and_step(epoch, 0)
    started = time.perf_counter()
    total_sum = 0.0
    stage_sums = [0.0, 0.0, 0.0, 0.0]
    metric_sums = {
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "silog": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }
    sample_count = 0

    for batch in loader:
        if not isinstance(batch, DepthBatch):
            raise TypeError("loader must yield DepthBatch instances")
        batch_size = batch.image.shape[0]
        result = evaluate_depth_batch(
            model,
            batch,
            device=device,
            stage_weights=stage_weights,
            alignment=alignment,
            depth_range=depth_range,
        )
        total_sum += result.total_loss * batch_size
        for index, loss in enumerate(result.stage_losses):
            stage_sums[index] += loss * batch_size
        for name in metric_sums:
            metric_sums[name] += getattr(result.metrics, name) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("validation loader yielded no samples")
    losses = EpochLosses(
        total=total_sum / sample_count,
        stages=(
            stage_sums[0] / sample_count,
            stage_sums[1] / sample_count,
            stage_sums[2] / sample_count,
            stage_sums[3] / sample_count,
        ),
    )
    metrics = DepthMetricResult(
        **{name: value / sample_count for name, value in metric_sums.items()}
    )
    return ValidationEpochResult(
        epoch=epoch,
        losses=losses,
        metrics=metrics,
        samples=sample_count,
        alignment=alignment,
        depth_range=depth_range,
        elapsed_seconds=time.perf_counter() - started,
    )


def _validate_epoch_and_step(epoch: int, global_step: int) -> None:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
