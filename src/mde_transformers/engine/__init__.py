"""Small supervised training, evaluation, and checkpoint interfaces."""

from mde_transformers.engine.checkpoint import (
    CheckpointState,
    create_checkpoint_state,
    restore_checkpoint_state,
)
from mde_transformers.engine.config import DeviceSpec, TrainingConfig, resolve_device
from mde_transformers.engine.evaluation import (
    DepthMetricResult,
    EvaluationAlignment,
    EvaluationResult,
    evaluate_depth_batch,
)
from mde_transformers.engine.training import (
    SupervisedStepResult,
    configure_supervised_training_mode,
    create_adamw_optimizer,
    supervised_train_step,
)

__all__ = [
    "CheckpointState",
    "DepthMetricResult",
    "DeviceSpec",
    "EvaluationAlignment",
    "EvaluationResult",
    "SupervisedStepResult",
    "TrainingConfig",
    "configure_supervised_training_mode",
    "create_adamw_optimizer",
    "create_checkpoint_state",
    "evaluate_depth_batch",
    "resolve_device",
    "restore_checkpoint_state",
    "supervised_train_step",
]
