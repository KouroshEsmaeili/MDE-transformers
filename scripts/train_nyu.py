#!/usr/bin/env python3
"""Train a supervised NYU baseline using only the official training split."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import cast

from mde_transformers.data import NYUDepthV2
from mde_transformers.engine import (
    BestDevState,
    NYUEvaluationPreprocess,
    NYUExperimentConfig,
    NYUPreprocess,
    TrainingConfig,
    TrainingEpochResult,
    ValidationEpochResult,
    WarmupCosineScheduler,
    configure_supervised_training_mode,
    create_adamw_optimizer,
    create_checkpoint_state,
    create_nyu_train_dev_loaders,
    load_checkpoint,
    resolve_device,
    resume_training_state,
    save_checkpoint,
    train_one_epoch,
    update_best_dev,
    validate_one_epoch,
    write_run_manifest,
)
from mde_transformers.metrics import NYUCrop
from mde_transformers.models import DepthModelConfig, MonocularDepthModel
from mde_transformers.utils import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised NYU trainer: official train split -> train-derived train/dev only."
    )
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("tiny", "base"), default="tiny")
    parser.add_argument(
        "--encoder-pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--encoder-policy", choices=("frozen", "trainable"), default="frozen")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--minimum-depth", type=float, default=1e-6)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument(
        "--stage-weights",
        type=float,
        nargs=4,
        default=(0.125, 0.25, 0.5, 1.0),
        metavar=("D1", "D2", "D3", "D4"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--depth-source", choices=("depths", "rawDepths"), default="depths")
    parser.add_argument("--validity-source", choices=("target", "rawDepths"), default="target")
    parser.add_argument("--alignment", choices=("none", "median"), default="none")
    parser.add_argument(
        "--training-crop",
        choices=("none", "nyu-eigen"),
        default="none",
        help="native-space synchronized training crop; default preserves uncropped behavior",
    )
    parser.add_argument(
        "--evaluation-crop",
        choices=("none", "nyu-eigen"),
        default="none",
        help="native-space mask for train-derived dev evaluation",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="explicit post-partition diagnostic limit; default uses the full train partition",
    )
    parser.add_argument(
        "--max-dev-samples",
        type=int,
        default=None,
        help="explicit post-partition diagnostic limit; default uses the full dev partition",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="diagnostic interruption point for resume testing; total schedule remains --epochs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decoder_channels = (args.decoder_channels,) * 4
    model_config = DepthModelConfig(
        encoder_variant=args.encoder,
        encoder_pretrained=args.encoder_pretrained,
        decoder_channels=decoder_channels,
        minimum_depth=args.minimum_depth,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        encoder_policy=args.encoder_policy,
        gradient_clip_norm=args.gradient_clip_norm,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        stage_weights=tuple(args.stage_weights),
    )
    experiment_config = NYUExperimentConfig(
        validation_fraction=args.validation_fraction,
        image_height=args.height,
        image_width=args.width,
        depth_source=args.depth_source,
        validity_source=args.validity_source,
        num_workers=args.num_workers,
        alignment=args.alignment,
        training_crop=_parse_crop(args.training_crop),
        evaluation_crop=_parse_crop(args.evaluation_crop),
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
    )
    _validate_runtime_args(args, training_config)
    seed_everything(training_config.seed)
    device = resolve_device(training_config.device)

    checkpoint = None if args.resume is None else load_checkpoint(args.resume, map_location="cpu")
    if checkpoint is not None and not checkpoint.model_config.is_architecturally_compatible(
        model_config
    ):
        raise ValueError("CLI model configuration does not match resume checkpoint")
    if checkpoint is not None:
        model_config = checkpoint.model_config
    model = (
        MonocularDepthModel.from_config(model_config)
        if checkpoint is None
        else MonocularDepthModel.from_checkpoint_config(checkpoint.model_config)
    ).to(device)
    configure_supervised_training_mode(
        model,
        encoder_trainable=training_config.encoder_policy == "trainable",
    )

    train_preprocess = NYUPreprocess(
        experiment_config.image_height,
        experiment_config.image_width,
        training_crop=experiment_config.training_crop,
    )
    dev_preprocess = NYUEvaluationPreprocess(
        experiment_config.image_height,
        experiment_config.image_width,
        crop=experiment_config.evaluation_crop,
    )
    dataset = NYUDepthV2(
        args.mat_path,
        args.split_path,
        split="train",
        depth_source=experiment_config.depth_source,
        validity_source=experiment_config.validity_source,
        transform=train_preprocess,
    )
    dev_dataset = NYUDepthV2(
        args.mat_path,
        args.split_path,
        split="train",
        depth_source=experiment_config.depth_source,
        validity_source=experiment_config.validity_source,
        transform=dev_preprocess,
    )
    try:
        loaders = create_nyu_train_dev_loaders(
            dataset,
            training_config=training_config,
            experiment_config=experiment_config,
            dev_dataset=dev_dataset,
        )
        optimizer = create_adamw_optimizer(
            model,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        steps_per_epoch = len(loaders.train)
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=training_config.epochs * steps_per_epoch,
            warmup_steps=training_config.warmup_epochs * steps_per_epoch,
            warmup_start_factor=training_config.warmup_start_factor,
        )
        start_epoch = 1
        global_step = 0
        best = BestDevState(loss=math.inf, epoch=None)
        if checkpoint is not None:
            resumed = resume_training_state(
                checkpoint,
                model,
                optimizer,
                scheduler,
                training_config=training_config,
                experiment_config=experiment_config,
            )
            start_epoch = resumed.next_epoch
            global_step = resumed.global_step
            best = resumed.best
        if start_epoch > training_config.epochs:
            raise ValueError("checkpoint has already completed the configured number of epochs")

        split_sizes = {
            "official_train": loaders.official_train_samples,
            "optimization_partition": loaders.optimization_partition_samples,
            "train_derived_dev_partition": loaders.dev_partition_samples,
            "effective_optimization": loaders.effective_train_samples,
            "effective_dev": loaders.effective_dev_samples,
        }
        write_run_manifest(
            args.output_dir,
            model_config=model_config,
            training_config=training_config,
            experiment_config=experiment_config,
            resolved_device=device,
            split_sizes=split_sizes,
        )
        _print_configuration(model_config, training_config, experiment_config, split_sizes, device)
        if checkpoint is not None:
            print(
                f"resumed_from_epoch={checkpoint.epoch} next_epoch={start_epoch} "
                f"global_step={global_step} best_epoch={best.epoch} best_dev_loss={best.loss:.6f}"
                f" scheduler_step={scheduler.step_count} lr={scheduler.get_last_lr()[0]:.8g}"
            )

        for epoch in range(start_epoch, training_config.epochs + 1):
            loaders.train_sampler.set_epoch(epoch)
            train_result = train_one_epoch(
                model,
                loaders.train,
                optimizer,
                scheduler,
                device=device,
                stage_weights=training_config.stage_weights,
                epoch=epoch,
                global_step=global_step,
                gradient_clip_norm=training_config.gradient_clip_norm,
            )
            global_step = train_result.global_step
            dev_result = validate_one_epoch(
                model,
                loaders.dev,
                device=device,
                stage_weights=training_config.stage_weights,
                epoch=epoch,
                alignment=experiment_config.alignment,
                depth_range=(experiment_config.min_depth, experiment_config.max_depth),
            )
            best, improved = update_best_dev(
                epoch=epoch,
                dev_loss=dev_result.losses.total,
                current=best,
            )
            state = create_checkpoint_state(
                model,
                optimizer,
                training_config,
                epoch=epoch,
                step=global_step,
                current_loss=dev_result.losses.total,
                scheduler=scheduler,
                experiment_config=experiment_config,
                best_dev_loss=best.loss,
                best_epoch=best.epoch,
            )
            save_checkpoint(state, args.output_dir / "last.pt")
            if improved:
                save_checkpoint(state, args.output_dir / "best.pt")
            _print_epoch(train_result, dev_result, improved, best)
            if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
                print(f"diagnostic_stop_after_epoch={epoch}")
                break
    finally:
        dataset.close()
        dev_dataset.close()


def _validate_runtime_args(args: argparse.Namespace, training_config: TrainingConfig) -> None:
    if args.decoder_channels <= 0:
        raise ValueError("--decoder-channels must be positive")
    if (
        args.stop_after_epoch is not None
        and not 1 <= args.stop_after_epoch <= training_config.epochs
    ):
        raise ValueError("--stop-after-epoch must lie within configured epochs")


def _parse_crop(value: str) -> NYUCrop:
    return cast(NYUCrop, value.replace("-", "_"))


def _print_configuration(
    model_config: DepthModelConfig,
    training_config: TrainingConfig,
    experiment_config: NYUExperimentConfig,
    split_sizes: dict[str, int],
    device: object,
) -> None:
    payload = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "nyu_experiment": asdict(experiment_config),
        "resolved_device": str(device),
        "split_sizes": split_sizes,
        "model_selection": "lowest train-derived dev loss",
        "official_test_used": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_epoch(
    train: TrainingEpochResult,
    dev: ValidationEpochResult,
    improved: bool,
    best: BestDevState,
) -> None:
    train_stages = ",".join(f"{value:.6f}" for value in train.losses.stages)
    dev_stages = ",".join(f"{value:.6f}" for value in dev.losses.stages)
    print(
        f"epoch={train.epoch} train_loss={train.losses.total:.6f} "
        f"train_stages=[{train_stages}] optimizer_steps={train.optimizer_steps} "
        f"global_step={train.global_step} lr={train.learning_rate:.8g} "
        f"train_seconds={train.elapsed_seconds:.3f}"
    )
    print(
        f"epoch={dev.epoch} dev_loss={dev.losses.total:.6f} dev_stages=[{dev_stages}] "
        f"abs_rel={dev.metrics.abs_rel:.6f} sq_rel={dev.metrics.sq_rel:.6f} "
        f"rmse={dev.metrics.rmse:.6f} rmse_log={dev.metrics.rmse_log:.6f} "
        f"silog={dev.metrics.silog:.6f} delta1={dev.metrics.delta1:.6f} "
        f"delta2={dev.metrics.delta2:.6f} delta3={dev.metrics.delta3:.6f} "
        f"dev_seconds={dev.elapsed_seconds:.3f}"
    )
    print(f"best={improved} best_epoch={best.epoch} best_dev_loss={best.loss:.6f}")


if __name__ == "__main__":
    main()
