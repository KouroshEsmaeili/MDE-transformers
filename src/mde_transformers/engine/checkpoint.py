"""Minimal in-memory checkpoint state for reproducible supervised continuation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer

from mde_transformers.engine.config import TrainingConfig
from mde_transformers.models import DepthModelConfig, MonocularDepthModel


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Independent model/optimizer snapshot without dataset or machine-specific paths."""

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    epoch: int
    step: int
    model_config: DepthModelConfig
    training_config: TrainingConfig
    seed: int
    current_loss: float | None = None


def create_checkpoint_state(
    model: MonocularDepthModel,
    optimizer: Optimizer,
    training_config: TrainingConfig,
    *,
    epoch: int,
    step: int,
    current_loss: float | None = None,
) -> CheckpointState:
    """Create a deep in-memory snapshot suitable for ordinary ``torch.save`` later."""
    _validate_non_negative_integer(epoch, "epoch")
    _validate_non_negative_integer(step, "step")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")
    if current_loss is not None:
        if isinstance(current_loss, bool) or not isinstance(current_loss, int | float):
            raise TypeError("current_loss must be a real number or None")
        if not math.isfinite(float(current_loss)):
            raise ValueError("current_loss must be finite")
    return CheckpointState(
        model_state_dict=copy.deepcopy(model.state_dict()),
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
        epoch=epoch,
        step=step,
        model_config=model.config,
        training_config=training_config,
        seed=training_config.seed,
        current_loss=None if current_loss is None else float(current_loss),
    )


def restore_checkpoint_state(
    checkpoint: CheckpointState,
    model: MonocularDepthModel,
    optimizer: Optimizer,
) -> None:
    """Strictly restore a checkpoint into an identically configured model and optimizer."""
    if not isinstance(checkpoint, CheckpointState):
        raise TypeError("checkpoint must be a CheckpointState")
    if model.config != checkpoint.model_config:
        raise ValueError("model configuration does not match checkpoint")
    model.load_state_dict(checkpoint.model_state_dict, strict=True)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
