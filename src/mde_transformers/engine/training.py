"""Single-batch supervised optimization primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from mde_transformers.data import DepthBatch
from mde_transformers.losses import masked_l1_loss, multi_scale_depth_loss
from mde_transformers.models import MonocularDepthModel
from mde_transformers.models.decoders import DecoderOutput


@dataclass(frozen=True, slots=True)
class SupervisedStepResult:
    """Detached scalar loss values from one optimizer update."""

    total_loss: float
    stage_losses: tuple[float, ...]


def configure_supervised_training_mode(
    model: MonocularDepthModel,
    *,
    encoder_trainable: bool,
) -> None:
    """Configure requires-grad flags and stochastic-module modes for supervised training.

    A frozen encoder is explicitly placed in evaluation mode after ``model.train()`` so Swin
    stochastic depth is disabled while the decoder remains in training mode. A trainable encoder
    remains in training mode. This policy is orchestration and is not hidden in model forward.
    """
    if not isinstance(encoder_trainable, bool):
        raise TypeError("encoder_trainable must be a boolean")
    model.set_encoder_trainable(encoder_trainable)
    model.train()
    if not encoder_trainable:
        model.encoder.eval()


def create_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Create AdamW over exactly the parameters currently marked trainable."""
    _validate_positive_real(learning_rate, "learning_rate")
    _validate_non_negative_real(weight_decay, "weight_decay")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))


def supervised_train_step(
    model: MonocularDepthModel,
    batch: DepthBatch,
    optimizer: Optimizer,
    *,
    device: torch.device,
    stage_weights: Sequence[float],
    gradient_clip_norm: float | None = None,
) -> SupervisedStepResult:
    """Run one multi-scale masked-L1 update using only decoder stage depths D1--D4.

    The model must already be on ``device``. The batch is transferred here. Encoder mode follows
    its current requires-grad state: fully frozen means evaluation mode; fully trainable means
    training mode. Partial encoder freezing is rejected because no stable stage policy exists yet.
    ``full_resolution`` is intentionally excluded from the loss.
    """
    _apply_training_mode_from_flags(model)
    batch_on_device = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch_on_device.image)
    if not isinstance(output, DecoderOutput):
        raise TypeError("model must return DecoderOutput")
    loss_result = multi_scale_depth_loss(
        output.stage_depths(),
        batch_on_device.depth,
        batch_on_device.valid_mask,
        stage_weights,
        loss_fn=masked_l1_loss,
    )
    loss_result.total.backward()

    if gradient_clip_norm is not None:
        _validate_positive_real(gradient_clip_norm, "gradient_clip_norm")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(gradient_clip_norm))
    optimizer.step()
    return SupervisedStepResult(
        total_loss=float(loss_result.total.detach().item()),
        stage_losses=tuple(float(loss.detach().item()) for loss in loss_result.stage_losses),
    )


def _apply_training_mode_from_flags(model: MonocularDepthModel) -> None:
    encoder_flags = {parameter.requires_grad for parameter in model.encoder.parameters()}
    if len(encoder_flags) != 1:
        raise ValueError("partially trainable encoders are not supported by this training policy")
    encoder_trainable = encoder_flags.pop()
    model.train()
    if not encoder_trainable:
        model.encoder.eval()


def _validate_positive_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_non_negative_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
