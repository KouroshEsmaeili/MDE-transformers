#!/usr/bin/env python3
"""Final-only evaluation on the reserved official NYU test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from mde_transformers.data import NYUDepthV2
from mde_transformers.engine import (
    NYUEvaluationPreprocess,
    create_nyu_evaluation_loader,
    load_checkpoint,
    resolve_device,
    validate_one_epoch,
)
from mde_transformers.metrics import NYUCrop, NYUEvaluationProtocol
from mde_transformers.models import MonocularDepthModel
from mde_transformers.utils import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on the reserved official NYU test split only."
    )
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--alignment", choices=("none", "median"), default=None)
    parser.add_argument(
        "--crop",
        choices=("none", "nyu-eigen"),
        default=None,
        help="explicit evaluation crop; defaults to the checkpoint's dev protocol",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="explicit diagnostic truncation; default evaluates all 654 official test samples",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.experiment_config is None:
        raise ValueError("checkpoint does not contain NYU experiment configuration")
    experiment = checkpoint.experiment_config
    num_workers = experiment.num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    alignment = experiment.alignment if args.alignment is None else args.alignment
    crop = (
        experiment.evaluation_crop
        if args.crop is None
        else cast(NYUCrop, args.crop.replace("-", "_"))
    )
    protocol = NYUEvaluationProtocol(
        depth_range=(experiment.min_depth, experiment.max_depth),
        alignment=alignment,
        crop=crop,
    )
    seed_everything(checkpoint.seed)
    device = resolve_device(args.device)

    model = MonocularDepthModel.from_checkpoint_config(checkpoint.model_config).to(device)
    model.load_state_dict(checkpoint.model_state_dict, strict=True)
    model.eval()
    preprocess = NYUEvaluationPreprocess(
        experiment.image_height,
        experiment.image_width,
        crop=protocol.crop,
    )
    dataset = NYUDepthV2(
        args.mat_path,
        args.split_path,
        split="test",
        depth_source=experiment.depth_source,
        validity_source=experiment.validity_source,
        transform=preprocess,
        max_samples=args.max_samples,
    )
    try:
        loader = create_nyu_evaluation_loader(
            dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            seed=checkpoint.seed,
        )
        result = validate_one_epoch(
            model,
            loader,
            device=device,
            stage_weights=checkpoint.training_config.stage_weights,
            epoch=max(checkpoint.epoch, 1),
            alignment=protocol.alignment,
            depth_range=protocol.depth_range,
        )
        scope = "TRUNCATED_DIAGNOSTIC" if args.max_samples is not None else "FULL_OFFICIAL_TEST"
        print(
            f"evaluation_scope={scope} samples={result.samples} split=official_test "
            f"protocol={protocol.profile_name} alignment={result.alignment} "
            f"range={result.depth_range} crop={protocol.crop}"
        )
        print(
            f"loss={result.losses.total:.6f} abs_rel={result.metrics.abs_rel:.6f} "
            f"sq_rel={result.metrics.sq_rel:.6f} rmse={result.metrics.rmse:.6f} "
            f"rmse_log={result.metrics.rmse_log:.6f} silog={result.metrics.silog:.6f} "
            f"delta1={result.metrics.delta1:.6f} delta2={result.metrics.delta2:.6f} "
            f"delta3={result.metrics.delta3:.6f}"
        )
        if args.max_samples is not None:
            print(
                "diagnostic_warning=metrics are truncated and are not full-test benchmark results"
            )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
