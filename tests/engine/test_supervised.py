"""CPU-only tests for supervised orchestration and checkpoint state."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from mde_transformers.data import DepthBatch
from mde_transformers.engine import (
    TrainingConfig,
    configure_supervised_training_mode,
    create_adamw_optimizer,
    create_checkpoint_state,
    evaluate_depth_batch,
    resolve_device,
    restore_checkpoint_state,
    supervised_train_step,
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
