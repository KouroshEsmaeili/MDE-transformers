"""Run metadata, best-dev tracking, and strict resume coordination."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.optim import Optimizer

from mde_transformers.engine.checkpoint import CheckpointState, restore_checkpoint_state
from mde_transformers.engine.config import NYUExperimentConfig, TrainingConfig
from mde_transformers.engine.scheduler import WarmupCosineScheduler
from mde_transformers.models import DepthModelConfig, MonocularDepthModel


@dataclass(frozen=True, slots=True)
class BestDevState:
    """Lowest observed train-derived dev loss and its one-based epoch."""

    loss: float
    epoch: int | None


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Training counters restored for continuation at ``next_epoch``."""

    next_epoch: int
    global_step: int
    best: BestDevState


def update_best_dev(
    *,
    epoch: int,
    dev_loss: float,
    current: BestDevState,
) -> tuple[BestDevState, bool]:
    """Update model-selection state using only a supplied train-derived dev loss."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    if not math.isfinite(dev_loss):
        raise ValueError("dev_loss must be finite")
    if dev_loss < current.loss:
        return BestDevState(loss=dev_loss, epoch=epoch), True
    return current, False


def resume_training_state(
    checkpoint: CheckpointState,
    model: MonocularDepthModel,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    *,
    training_config: TrainingConfig,
    experiment_config: NYUExperimentConfig,
) -> ResumeState:
    """Validate configs and restore all state for the epoch after the checkpoint."""
    if checkpoint.training_config != training_config:
        raise ValueError("training configuration does not match checkpoint")
    if checkpoint.experiment_config != experiment_config:
        raise ValueError("NYU experiment configuration does not match checkpoint")
    restore_checkpoint_state(
        checkpoint,
        model,
        optimizer,
        scheduler=scheduler,
    )
    return ResumeState(
        next_epoch=checkpoint.epoch + 1,
        global_step=checkpoint.step,
        best=BestDevState(checkpoint.best_dev_loss, checkpoint.best_epoch),
    )


def write_run_manifest(
    output_dir: str | Path,
    *,
    model_config: DepthModelConfig,
    training_config: TrainingConfig,
    experiment_config: NYUExperimentConfig,
    resolved_device: torch.device,
    split_sizes: dict[str, int],
) -> Path:
    """Write auditable, human-readable configuration without paths or environment variables."""
    destination = Path(output_dir) / "run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "nyu_experiment": asdict(experiment_config),
        "resolved_device": str(resolved_device),
        "split_sizes": dict(split_sizes),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "mde_transformers": _package_version(),
        },
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _package_version() -> str:
    try:
        return version("mde-transformers")
    except PackageNotFoundError:
        return "0.1.0"
