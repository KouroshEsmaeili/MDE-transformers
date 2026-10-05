"""Portable supervised-training checkpoint state and file serialization."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from mde_transformers.engine.config import NYUExperimentConfig, TrainingConfig
from mde_transformers.engine.scheduler import WarmupCosineScheduler
from mde_transformers.models import DepthModelConfig, MonocularDepthModel

_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Complete continuation state without dataset contents or filesystem paths.

    ``epoch`` is the one-based epoch just completed, and ``step`` is the total number of optimizer
    updates already performed. Resume therefore begins at ``epoch + 1`` without repeating or
    skipping an epoch; scheduler state describes the LR after exactly ``step`` updates.
    """

    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    epoch: int
    step: int
    model_config: DepthModelConfig
    training_config: TrainingConfig
    seed: int
    current_loss: float | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    experiment_config: NYUExperimentConfig | None = None
    best_dev_loss: float = math.inf
    best_epoch: int | None = None
    torch_rng_state: torch.Tensor | None = None
    cuda_rng_states: tuple[torch.Tensor, ...] = ()


def create_checkpoint_state(
    model: MonocularDepthModel,
    optimizer: Optimizer,
    training_config: TrainingConfig,
    *,
    epoch: int,
    step: int,
    current_loss: float | None = None,
    scheduler: WarmupCosineScheduler | None = None,
    experiment_config: NYUExperimentConfig | None = None,
    best_dev_loss: float = math.inf,
    best_epoch: int | None = None,
) -> CheckpointState:
    """Create an independent snapshot including scheduler and PyTorch RNG state."""
    _validate_non_negative_integer(epoch, "epoch")
    _validate_non_negative_integer(step, "step")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")
    if experiment_config is not None and not isinstance(experiment_config, NYUExperimentConfig):
        raise TypeError("experiment_config must be an NYUExperimentConfig or None")
    _validate_optional_loss(current_loss, "current_loss")
    _validate_best(best_dev_loss, best_epoch)
    if scheduler is not None and scheduler.step_count != step:
        raise ValueError("scheduler step_count must equal completed optimizer updates")
    cuda_states = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return CheckpointState(
        model_state_dict=copy.deepcopy(model.state_dict()),
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
        epoch=epoch,
        step=step,
        model_config=model.config,
        training_config=training_config,
        seed=training_config.seed,
        current_loss=None if current_loss is None else float(current_loss),
        scheduler_state_dict=None if scheduler is None else copy.deepcopy(scheduler.state_dict()),
        experiment_config=experiment_config,
        best_dev_loss=float(best_dev_loss),
        best_epoch=best_epoch,
        torch_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=tuple(state.clone() for state in cuda_states),
    )


def restore_checkpoint_state(
    checkpoint: CheckpointState,
    model: MonocularDepthModel,
    optimizer: Optimizer,
    *,
    scheduler: WarmupCosineScheduler | None = None,
) -> None:
    """Strictly restore model, optimizer, scheduler, and framework RNG state."""
    if not isinstance(checkpoint, CheckpointState):
        raise TypeError("checkpoint must be a CheckpointState")
    if not model.config.is_architecturally_compatible(checkpoint.model_config):
        raise ValueError("model architecture does not match checkpoint")
    model.load_state_dict(checkpoint.model_state_dict, strict=True)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    if checkpoint.scheduler_state_dict is not None:
        if scheduler is None:
            raise ValueError("checkpoint contains scheduler state but no scheduler was supplied")
        scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        if scheduler.step_count != checkpoint.step:
            raise ValueError("restored scheduler step_count does not match global step")
    elif scheduler is not None:
        raise ValueError("scheduler was supplied but checkpoint contains no scheduler state")

    if checkpoint.torch_rng_state is not None:
        torch.set_rng_state(checkpoint.torch_rng_state)
    if checkpoint.cuda_rng_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(checkpoint.cuda_rng_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA device count does not match this runtime")
        torch.cuda.set_rng_state_all(list(checkpoint.cuda_rng_states))


def save_checkpoint(checkpoint: CheckpointState, path: str | Path) -> None:
    """Save a versioned dictionary payload with ordinary :func:`torch.save`."""
    if not isinstance(checkpoint, CheckpointState):
        raise TypeError("checkpoint must be a CheckpointState")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_payload(checkpoint), destination)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    """Load and validate a checkpoint dictionary using tensor-only safe loading."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    if payload.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    required = {
        "format_version",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "step",
        "model_config",
        "training_config",
        "experiment_config",
        "seed",
        "current_loss",
        "best_dev_loss",
        "best_epoch",
        "torch_rng_state",
        "cuda_rng_states",
    }
    if set(payload) != required:
        raise ValueError("checkpoint has unexpected or missing fields")

    model_values = dict(payload["model_config"])
    model_values["decoder_channels"] = tuple(model_values["decoder_channels"])
    training_values = dict(payload["training_config"])
    training_values["stage_weights"] = tuple(training_values["stage_weights"])
    experiment_payload = payload["experiment_config"]
    experiment_config = None
    if experiment_payload is not None:
        experiment_values = dict(experiment_payload)
        # Milestone-10 checkpoints stored the only supported dev crop as ``crop='none'``.
        # Preserve that meaning while making training and evaluation preprocessing independent.
        legacy_crop = experiment_values.pop("crop", None)
        experiment_values.setdefault("training_crop", "none")
        experiment_values.setdefault(
            "evaluation_crop",
            "none" if legacy_crop is None else legacy_crop,
        )
        experiment_config = NYUExperimentConfig(**experiment_values)
    checkpoint = CheckpointState(
        model_state_dict=dict(payload["model_state_dict"]),
        optimizer_state_dict=dict(payload["optimizer_state_dict"]),
        scheduler_state_dict=(
            None
            if payload["scheduler_state_dict"] is None
            else dict(payload["scheduler_state_dict"])
        ),
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        model_config=DepthModelConfig(**model_values),
        training_config=TrainingConfig(**training_values),
        experiment_config=experiment_config,
        seed=int(payload["seed"]),
        current_loss=(None if payload["current_loss"] is None else float(payload["current_loss"])),
        best_dev_loss=float(payload["best_dev_loss"]),
        best_epoch=None if payload["best_epoch"] is None else int(payload["best_epoch"]),
        torch_rng_state=payload["torch_rng_state"],
        cuda_rng_states=tuple(payload["cuda_rng_states"]),
    )
    _validate_loaded_checkpoint(checkpoint)
    return checkpoint


def _to_payload(checkpoint: CheckpointState) -> dict[str, Any]:
    return {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": checkpoint.model_state_dict,
        "optimizer_state_dict": checkpoint.optimizer_state_dict,
        "scheduler_state_dict": checkpoint.scheduler_state_dict,
        "epoch": checkpoint.epoch,
        "step": checkpoint.step,
        "model_config": asdict(checkpoint.model_config),
        "training_config": asdict(checkpoint.training_config),
        "experiment_config": (
            None if checkpoint.experiment_config is None else asdict(checkpoint.experiment_config)
        ),
        "seed": checkpoint.seed,
        "current_loss": checkpoint.current_loss,
        "best_dev_loss": checkpoint.best_dev_loss,
        "best_epoch": checkpoint.best_epoch,
        "torch_rng_state": checkpoint.torch_rng_state,
        "cuda_rng_states": checkpoint.cuda_rng_states,
    }


def _validate_loaded_checkpoint(checkpoint: CheckpointState) -> None:
    _validate_non_negative_integer(checkpoint.epoch, "epoch")
    _validate_non_negative_integer(checkpoint.step, "step")
    if checkpoint.seed != checkpoint.training_config.seed:
        raise ValueError("checkpoint seed does not match training configuration")
    _validate_optional_loss(checkpoint.current_loss, "current_loss")
    _validate_best(checkpoint.best_dev_loss, checkpoint.best_epoch)


def _validate_optional_loss(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number or None")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_best(best_dev_loss: float, best_epoch: int | None) -> None:
    if isinstance(best_dev_loss, bool) or not isinstance(best_dev_loss, int | float):
        raise TypeError("best_dev_loss must be a real number")
    if math.isnan(float(best_dev_loss)):
        raise ValueError("best_dev_loss must not be NaN")
    if best_epoch is None:
        if not math.isinf(float(best_dev_loss)):
            raise ValueError("best_dev_loss must be infinity when best_epoch is None")
        return
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch <= 0:
        raise ValueError("best_epoch must be a positive integer or None")
    if not math.isfinite(float(best_dev_loss)):
        raise ValueError("best_dev_loss must be finite once a best epoch exists")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
