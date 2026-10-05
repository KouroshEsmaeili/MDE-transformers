"""Small supervised training, evaluation, and checkpoint interfaces."""

from mde_transformers.engine.checkpoint import (
    CheckpointState,
    create_checkpoint_state,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
)
from mde_transformers.engine.config import (
    DeviceSpec,
    EncoderPolicy,
    NYUExperimentConfig,
    TrainingConfig,
    resolve_device,
)
from mde_transformers.engine.epochs import (
    EpochLosses,
    TrainingEpochResult,
    ValidationEpochResult,
    train_one_epoch,
    validate_one_epoch,
)
from mde_transformers.engine.evaluation import (
    DepthMetricResult,
    EvaluationAlignment,
    EvaluationResult,
    evaluate_depth_batch,
)
from mde_transformers.engine.experiment import (
    BestDevState,
    ResumeState,
    resume_training_state,
    update_best_dev,
    write_run_manifest,
)
from mde_transformers.engine.nyu_data import (
    EpochShuffleSampler,
    NYUEvaluationPreprocess,
    NYUPreprocess,
    NYUTrainDevLoaders,
    create_nyu_evaluation_loader,
    create_nyu_train_dev_loaders,
    seed_data_worker,
)
from mde_transformers.engine.scheduler import WarmupCosineScheduler
from mde_transformers.engine.training import (
    SupervisedStepResult,
    configure_supervised_training_mode,
    create_adamw_optimizer,
    supervised_train_step,
)

__all__ = [
    "CheckpointState",
    "BestDevState",
    "DepthMetricResult",
    "DeviceSpec",
    "EncoderPolicy",
    "EpochLosses",
    "EpochShuffleSampler",
    "EvaluationAlignment",
    "EvaluationResult",
    "NYUExperimentConfig",
    "NYUEvaluationPreprocess",
    "NYUPreprocess",
    "NYUTrainDevLoaders",
    "ResumeState",
    "SupervisedStepResult",
    "TrainingConfig",
    "TrainingEpochResult",
    "ValidationEpochResult",
    "WarmupCosineScheduler",
    "configure_supervised_training_mode",
    "create_adamw_optimizer",
    "create_checkpoint_state",
    "create_nyu_evaluation_loader",
    "create_nyu_train_dev_loaders",
    "evaluate_depth_batch",
    "load_checkpoint",
    "resolve_device",
    "restore_checkpoint_state",
    "resume_training_state",
    "save_checkpoint",
    "seed_data_worker",
    "supervised_train_step",
    "train_one_epoch",
    "update_best_dev",
    "validate_one_epoch",
    "write_run_manifest",
]
