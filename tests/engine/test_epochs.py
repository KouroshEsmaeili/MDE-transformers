from __future__ import annotations

import pytest
import torch

from mde_transformers.data import DepthBatch
from mde_transformers.engine import WarmupCosineScheduler, train_one_epoch
from mde_transformers.engine.training import SupervisedStepResult


def _batch(size: int) -> DepthBatch:
    image = torch.ones((size, 3, 1, 1))
    depth = torch.ones((size, 1, 1, 1))
    mask = torch.ones_like(depth, dtype=torch.bool)
    return DepthBatch(image, depth, mask, None, tuple(f"sample-{index}" for index in range(size)))


def test_train_epoch_weights_unequal_final_batch_by_sample_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=2,
        warmup_steps=0,
    )

    def fake_step(*args: object, **kwargs: object) -> SupervisedStepResult:
        del kwargs
        batch = args[1]
        assert isinstance(batch, DepthBatch)
        loss = 2.0 if batch.image.shape[0] == 2 else 8.0
        return SupervisedStepResult(loss, (loss, loss + 1, loss + 2, loss + 3))

    monkeypatch.setattr("mde_transformers.engine.epochs.supervised_train_step", fake_step)
    result = train_one_epoch(
        model,  # type: ignore[arg-type]
        [_batch(2), _batch(1)],
        optimizer,
        scheduler,
        device=torch.device("cpu"),
        stage_weights=(0.125, 0.25, 0.5, 1.0),
        epoch=1,
        global_step=5,
    )

    assert result.losses.total == pytest.approx((2 * 2 + 8) / 3)
    assert result.losses.stages == pytest.approx((4.0, 5.0, 6.0, 7.0))
    assert result.samples == 3
    assert result.optimizer_steps == 2
    assert result.global_step == 7
