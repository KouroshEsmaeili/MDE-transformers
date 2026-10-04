#!/usr/bin/env python3
"""Overfit a tiny NYU training subset as a supervised-pipeline diagnostic."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

from mde_transformers.data import (
    DepthBatch,
    DepthSample,
    NYUDepthV2,
    collate_depth_samples,
    normalize_imagenet_sample,
    resize_depth_sample,
)
from mde_transformers.engine import (
    TrainingConfig,
    configure_supervised_training_mode,
    create_adamw_optimizer,
    evaluate_depth_batch,
    resolve_device,
    supervised_train_step,
)
from mde_transformers.models import MonocularDepthModel
from mde_transformers.utils import seed_everything

_STAGE_WEIGHTS = (0.125, 0.25, 0.5, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit only a tiny official-NYU training subset; never reads the test split."
    )
    parser.add_argument("--mat-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--decoder-channels",
        type=int,
        default=32,
        help="constant diagnostic decoder width (not a thesis hyperparameter)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--decoder-only", action="store_true", help="freeze the encoder (default)")
    mode.add_argument("--encoder-trainable", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("samples", "steps", "height", "width", "decoder_channels"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    encoder_trainable = bool(args.encoder_trainable)
    config = TrainingConfig(
        epochs=1,
        batch_size=args.samples,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )
    seed_everything(config.seed)
    device = resolve_device(config.device)

    def transform(sample: DepthSample) -> DepthSample:
        resized = resize_depth_sample(sample, (args.height, args.width))
        return normalize_imagenet_sample(resized)

    dataset = NYUDepthV2(
        args.mat_path,
        args.split_path,
        split="train",
        transform=transform,
        max_samples=args.samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.samples,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_depth_samples,
    )
    batch = next(iter(loader))
    if not isinstance(batch, DepthBatch):
        raise TypeError("NYU DataLoader must produce DepthBatch")

    decoder_channels = (args.decoder_channels,) * 4
    model = MonocularDepthModel(
        encoder_variant="tiny",
        encoder_pretrained=False,
        decoder_channels=decoder_channels,
    ).to(device)
    configure_supervised_training_mode(model, encoder_trainable=encoder_trainable)
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(f"device={device}")
    identifiers = ",".join(identifier or "<none>" for identifier in batch.sample_ids)
    print(f"samples={len(dataset)} ids={identifiers}")
    print(f"resolution={args.height}x{args.width}")
    print(f"encoder={'trainable' if encoder_trainable else 'frozen'}")
    print(
        f"decoder_channels={decoder_channels[0]} stage_weights={_STAGE_WEIGHTS} seed={config.seed}"
    )

    started = time.perf_counter()
    initial = evaluate_depth_batch(
        model,
        batch,
        device=device,
        stage_weights=_STAGE_WEIGHTS,
        alignment="none",
    ).total_loss
    best = initial
    log_interval = max(1, args.steps // 10)
    for step in range(1, args.steps + 1):
        result = supervised_train_step(
            model,
            batch,
            optimizer,
            device=device,
            stage_weights=_STAGE_WEIGHTS,
        )
        best = min(best, result.total_loss)
        if step == 1 or step % log_interval == 0 or step == args.steps:
            components = ",".join(f"{value:.6f}" for value in result.stage_losses)
            print(f"step={step} total_loss={result.total_loss:.6f} stage_losses=[{components}]")

    final = evaluate_depth_batch(
        model,
        batch,
        device=device,
        stage_weights=_STAGE_WEIGHTS,
        alignment="none",
    ).total_loss
    best = min(best, final)
    elapsed = time.perf_counter() - started
    relative_reduction = (initial - final) / initial if initial > 0 else 0.0
    print(
        f"initial_loss={initial:.6f} final_loss={final:.6f} best_loss={best:.6f} "
        f"relative_reduction={relative_reduction:.2%} elapsed_seconds={elapsed:.3f}"
    )
    dataset.close()


if __name__ == "__main__":
    main()
