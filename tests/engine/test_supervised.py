"""CPU-only tests for supervised orchestration and checkpoint state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from mde_transformers.data import DepthBatch
from mde_transformers.engine import (
    BestDevState,
    NYUExperimentConfig,
    TrainingConfig,
    WarmupCosineScheduler,
    configure_supervised_training_mode,
    create_adamw_optimizer,
    create_checkpoint_state,
    evaluate_depth_batch,
    load_checkpoint,
    resolve_device,
    restore_checkpoint_state,
    resume_training_state,
    save_checkpoint,
    supervised_train_step,
    update_best_dev,
)
from mde_transformers.models import DepthModelConfig
from mde_transformers.models.decoders import DecoderOutput
from mde_transformers.utils import seed_everything

_WEIGHTS = (0.125, 0.25, 0.5, 1.0)


class _ScalarEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channel_scale = nn.Parameter(torch.ones((1, 3, 1, 1)))
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.dropout(image * self.channel_scale).mean(dim=1, keepdim=True)


class _ScalarDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature * self.scale + self.bias


class _TinyDepthModel(nn.Module):
    """Small model obeying the production output/mode contract for fast engine tests."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ScalarEncoder()
        self.decoder = _ScalarDecoder()
        self.config = DepthModelConfig(decoder_channels=(1, 1, 1, 1))

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, image: torch.Tensor) -> DecoderOutput:
        depth = functional.softplus(self.decoder(self.encoder(image))) + 1e-4
        return DecoderOutput(depth, depth, depth, depth, depth)


class _FixedDepthModel(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, image: torch.Tensor) -> DecoderOutput:
        del image
        depth = self.prediction
        return DecoderOutput(depth, depth, depth, depth, depth)


def _batch(*, target: torch.Tensor | None = None) -> DepthBatch:
    image = torch.full((1, 3, 4, 5), 0.25)
    depth = torch.full((1, 1, 4, 5), 2.0) if target is None else target
    mask = torch.ones_like(depth, dtype=torch.bool)
    return DepthBatch(image, depth, mask, None, ("synthetic",))


def test_training_config_and_device_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TrainingConfig(
        epochs=2,
        batch_size=3,
        learning_rate=1e-3,
        weight_decay=1e-2,
        device="cpu",
        seed=7,
        gradient_clip_norm=1.0,
    )
    assert config.device == "cpu"
    assert resolve_device(config.device) == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


def test_frozen_encoder_policy_keeps_decoder_training() -> None:
    model = _TinyDepthModel()

    configure_supervised_training_mode(model, encoder_trainable=False)  # type: ignore[arg-type]

    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not model.encoder.training
    assert model.decoder.training

    configure_supervised_training_mode(model, encoder_trainable=True)  # type: ignore[arg-type]
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.encoder.training
    assert model.decoder.training


def test_train_step_is_finite_and_updates_only_optimizer_parameters() -> None:
    seed_everything(4)
    model = _TinyDepthModel()
    configure_supervised_training_mode(model, encoder_trainable=False)  # type: ignore[arg-type]
    optimizer = create_adamw_optimizer(model, learning_rate=0.05, weight_decay=0.0)
    encoder_before = [parameter.detach().clone() for parameter in model.encoder.parameters()]
    decoder_before = [parameter.detach().clone() for parameter in model.decoder.parameters()]

    result = supervised_train_step(
        model,  # type: ignore[arg-type]
        _batch(),
        optimizer,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
        gradient_clip_norm=10.0,
    )

    assert result.total_loss > 0
    assert len(result.stage_losses) == 4
    assert all(torch.isfinite(torch.tensor(value)) for value in result.stage_losses)
    assert all(
        torch.equal(before, after)
        for before, after in zip(encoder_before, model.encoder.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(decoder_before, model.decoder.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_evaluation_is_no_grad_and_alignment_is_explicit() -> None:
    seed_everything(5)
    model = _TinyDepthModel().eval()
    source = _batch()
    with torch.no_grad():
        prediction = model(source.image).full_resolution.detach()
    batch = _batch(target=prediction * 2.0)
    model.zero_grad(set_to_none=True)

    unaligned = evaluate_depth_batch(
        model,  # type: ignore[arg-type]
        batch,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
        alignment="none",
    )
    aligned = evaluate_depth_batch(
        model,  # type: ignore[arg-type]
        batch,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
        alignment="median",
    )

    assert unaligned.metrics.abs_rel == pytest.approx(0.5)
    assert aligned.metrics.abs_rel == pytest.approx(0.0, abs=1e-6)
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="alignment"):
        evaluate_depth_batch(
            model,  # type: ignore[arg-type]
            batch,
            device=torch.device("cpu"),
            stage_weights=_WEIGHTS,
            alignment="automatic",  # type: ignore[arg-type]
        )


def test_evaluation_intersects_mask_with_range_without_mutating_target() -> None:
    target = torch.tensor([[[[0.05, 1.0, 11.0]]]])
    prediction = torch.tensor([[[[100.0, 2.0, 100.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    batch = DepthBatch(
        image=torch.ones((1, 3, 1, 3)),
        depth=target,
        valid_mask=mask,
        intrinsics=None,
        sample_ids=("range",),
    )
    target_before = target.clone()
    mask_before = mask.clone()

    result = evaluate_depth_batch(
        _FixedDepthModel(prediction),  # type: ignore[arg-type]
        batch,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
        alignment="none",
        depth_range=(0.1, 10.0),
    )

    assert result.stage_losses == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert result.total_loss == pytest.approx(sum(_WEIGHTS))
    assert result.metrics.abs_rel == pytest.approx(1.0)
    assert result.metrics.rmse == pytest.approx(1.0)
    assert result.depth_range == (0.1, 10.0)
    torch.testing.assert_close(target, target_before)
    torch.testing.assert_close(mask, mask_before)


def test_checkpoint_state_restores_model_and_optimizer_in_memory() -> None:
    seed_everything(6)
    model = _TinyDepthModel()
    configure_supervised_training_mode(model, encoder_trainable=True)  # type: ignore[arg-type]
    optimizer = create_adamw_optimizer(model, learning_rate=0.01, weight_decay=0.001)
    supervised_train_step(
        model,  # type: ignore[arg-type]
        _batch(),
        optimizer,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
    )
    config = TrainingConfig(learning_rate=0.01, weight_decay=0.001, seed=6)
    checkpoint = create_checkpoint_state(
        model,  # type: ignore[arg-type]
        optimizer,
        config,
        epoch=2,
        step=9,
        current_loss=1.25,
    )
    expected = deepcopy(checkpoint.model_state_dict)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    restore_checkpoint_state(checkpoint, model, optimizer)  # type: ignore[arg-type]

    assert checkpoint.epoch == 2
    assert checkpoint.step == 9
    assert checkpoint.seed == 6
    assert checkpoint.training_config == config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_deterministic_synthetic_batch_overfits() -> None:
    seed_everything(23)
    model = _TinyDepthModel()
    configure_supervised_training_mode(model, encoder_trainable=False)  # type: ignore[arg-type]
    optimizer = create_adamw_optimizer(model, learning_rate=0.05, weight_decay=0.0)
    batch = _batch()
    initial = evaluate_depth_batch(
        model,  # type: ignore[arg-type]
        batch,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
        alignment="none",
    ).total_loss

    for _ in range(40):
        supervised_train_step(
            model,  # type: ignore[arg-type]
            batch,
            optimizer,
            device=torch.device("cpu"),
            stage_weights=_WEIGHTS,
        )
    final = evaluate_depth_batch(
        model,  # type: ignore[arg-type]
        batch,
        device=torch.device("cpu"),
        stage_weights=_WEIGHTS,
        alignment="none",
    ).total_loss

    assert final < initial * 0.25


def test_file_checkpoint_restores_next_epoch_steps_optimizer_and_scheduler(tmp_path) -> None:
    seed_everything(11)
    model = _TinyDepthModel()
    configure_supervised_training_mode(model, encoder_trainable=False)  # type: ignore[arg-type]
    optimizer = create_adamw_optimizer(model, learning_rate=0.01, weight_decay=0.001)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=4,
        warmup_steps=1,
        warmup_start_factor=0.1,
    )
    for _ in range(2):
        supervised_train_step(
            model,  # type: ignore[arg-type]
            _batch(),
            optimizer,
            device=torch.device("cpu"),
            stage_weights=_WEIGHTS,
        )
        scheduler.step()
    training_config = TrainingConfig(
        epochs=2,
        learning_rate=0.01,
        weight_decay=0.001,
        seed=11,
        warmup_epochs=0,
    )
    experiment_config = NYUExperimentConfig(image_height=4, image_width=5)
    checkpoint = create_checkpoint_state(
        model,  # type: ignore[arg-type]
        optimizer,
        training_config,
        epoch=1,
        step=2,
        current_loss=0.75,
        scheduler=scheduler,
        experiment_config=experiment_config,
        best_dev_loss=0.75,
        best_epoch=1,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, path)
    loaded = load_checkpoint(path)

    restored_model = _TinyDepthModel()
    configure_supervised_training_mode(
        restored_model,
        encoder_trainable=False,  # type: ignore[arg-type]
    )
    restored_optimizer = create_adamw_optimizer(
        restored_model,
        learning_rate=0.01,
        weight_decay=0.001,
    )
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer,
        total_steps=4,
        warmup_steps=1,
        warmup_start_factor=0.1,
    )
    resumed = resume_training_state(
        loaded,
        restored_model,  # type: ignore[arg-type]
        restored_optimizer,
        restored_scheduler,
        training_config=training_config,
        experiment_config=experiment_config,
    )

    assert resumed.next_epoch == 2
    assert resumed.global_step == 2
    assert resumed.best == BestDevState(loss=0.75, epoch=1)
    assert restored_scheduler.step_count == 2
    assert restored_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
    assert restored_optimizer.state_dict()["state"]
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

    for incompatible in (
        replace(experiment_config, training_crop="nyu_eigen"),
        replace(experiment_config, evaluation_crop="nyu_eigen"),
    ):
        with pytest.raises(ValueError, match="experiment configuration"):
            resume_training_state(
                loaded,
                restored_model,  # type: ignore[arg-type]
                restored_optimizer,
                restored_scheduler,
                training_config=training_config,
                experiment_config=incompatible,
            )


def test_best_dev_tie_keeps_earlier_checkpoint() -> None:
    current = BestDevState(loss=1.0, epoch=2)

    tied, improved = update_best_dev(epoch=3, dev_loss=1.0, current=current)
    lower, lower_improved = update_best_dev(epoch=3, dev_loss=0.9, current=current)

    assert tied == current
    assert not improved
    assert lower == BestDevState(loss=0.9, epoch=3)
    assert lower_improved


def test_checkpoint_rejects_scheduler_global_step_mismatch() -> None:
    model = _TinyDepthModel()
    configure_supervised_training_mode(model, encoder_trainable=False)  # type: ignore[arg-type]
    optimizer = create_adamw_optimizer(model, learning_rate=0.01, weight_decay=0.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=2,
        warmup_steps=0,
    )

    with pytest.raises(ValueError, match="completed optimizer updates"):
        create_checkpoint_state(
            model,  # type: ignore[arg-type]
            optimizer,
            TrainingConfig(epochs=2),
            epoch=1,
            step=1,
            scheduler=scheduler,
        )
