from __future__ import annotations

import json

import torch

from mde_transformers.engine import NYUExperimentConfig, TrainingConfig, write_run_manifest
from mde_transformers.models import DepthModelConfig


def test_run_manifest_contains_reproducible_configs_without_dataset_paths(tmp_path) -> None:
    path = write_run_manifest(
        tmp_path,
        model_config=DepthModelConfig(decoder_channels=(8, 8, 8, 8)),
        training_config=TrainingConfig(epochs=2, seed=17),
        experiment_config=NYUExperimentConfig(
            validation_fraction=0.2,
            image_height=128,
            image_width=160,
            training_crop="nyu_eigen",
            evaluation_crop="nyu_eigen",
            max_train_samples=2,
            max_dev_samples=1,
        ),
        resolved_device=torch.device("cpu"),
        split_sizes={
            "official_train": 795,
            "optimization_partition": 636,
            "train_derived_dev_partition": 159,
            "effective_optimization": 2,
            "effective_dev": 1,
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["training"]["seed"] == 17
    assert payload["nyu_experiment"]["validation_fraction"] == 0.2
    assert payload["nyu_experiment"]["training_crop"] == "nyu_eigen"
    assert payload["nyu_experiment"]["evaluation_crop"] == "nyu_eigen"
    assert payload["split_sizes"]["official_train"] == 795
    assert payload["resolved_device"] == "cpu"
    serialized = path.read_text(encoding="utf-8")
    assert "mat_path" not in serialized
    assert "split_path" not in serialized
