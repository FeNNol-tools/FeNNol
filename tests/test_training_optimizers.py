"""Optimizers and learning-rate schedules for training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.config import TrainingConfig
from fennol.training.optimizers import get_lr_schedule, get_optimizer


def test_stochastic_scheduler_initializes_without_external_rng_key():
    config = TrainingConfig(
        max_epochs=2,
        nbatch_per_epoch=2,
        lr=1e-3,
        schedule_type="cosine_onecycle",
        stochastic_scheduler=True,
    )
    schedule, state, _, _ = get_lr_schedule(config)

    assert "rng_key" in state
    assert callable(schedule)


def test_stochastic_reduce_on_plateau_applies_metric_updates():
    config = TrainingConfig(
        schedule_type="reduce_on_plateau",
        lr=0.1,
        final_lr=0.001,
        lr_factor=0.5,
        patience=1,
        stochastic_scheduler=True,
        scheduler_seed=7,
    )
    schedule, state, _, adaptive = get_lr_schedule(config)
    assert adaptive

    _, state = schedule(state)
    _, state = schedule(state, rmse=1.0)
    epoch_lr, state = schedule(state, rmse=2.0)
    batch_lr, state = schedule(state)

    assert float(epoch_lr) == pytest.approx(0.05)
    assert float(state["lr_max"]) == pytest.approx(0.05)
    assert 0.001 <= float(batch_lr) <= 0.05


def test_stochastic_reduce_on_plateau_clamps_at_final_lr():
    config = TrainingConfig(
        schedule_type="reduce_on_plateau",
        lr=0.1,
        final_lr=0.04,
        lr_factor=0.5,
        patience=1,
        stochastic_scheduler=True,
        scheduler_seed=7,
    )
    schedule, state, _, _ = get_lr_schedule(config)
    _, state = schedule(state, rmse=1.0)
    for worse_rmse in (2.0, 3.0, 4.0):
        _, state = schedule(state, rmse=worse_rmse)

    assert float(state["lr_max"]) == pytest.approx(0.04)
    batch_lr, state = schedule(state)
    assert float(batch_lr) == pytest.approx(0.04)
    assert float(state["lr_max"]) == pytest.approx(0.04)


def test_reduce_on_plateau_clamps_at_final_lr():
    config = TrainingConfig(
        schedule_type="reduce_on_plateau",
        lr=0.1,
        final_lr=0.04,
        lr_factor=0.5,
        patience=1,
    )
    schedule, state, _, _ = get_lr_schedule(config)

    for metric in (1.0, 2.0, 3.0, 4.0):
        lr, state = schedule(state, rmse=metric)

    assert float(lr) == pytest.approx(0.04)
    assert float(state["lr"]) == pytest.approx(0.04)


def test_stochastic_scheduler_sequence_repeats_for_the_same_seed():
    config = TrainingConfig(
        max_epochs=2,
        nbatch_per_epoch=2,
        lr=1e-3,
        schedule_type="cosine_onecycle",
        stochastic_scheduler=True,
        scheduler_seed=47,
    )

    def learning_rates():
        schedule, state, _, _ = get_lr_schedule(config)
        rates = []
        for _ in range(5):
            lr, state = schedule(state)
            rates.append(float(lr))
        return rates

    assert learning_rates() == learning_rates()


def test_optimizer_accepts_optax_contrib_adopt():
    config = TrainingConfig(optimizer="adopt")
    variables = {"params": {"weight": jnp.array([[1.0, 2.0], [3.0, 4.0]])}}

    optimizer, _ = get_optimizer(config, variables, initial_lr=1e-3)
    state = optimizer.init(variables)
    updates, _ = optimizer.update(
        jax.tree_util.tree_map(jnp.ones_like, variables), state, params=variables
    )

    assert np.isfinite(np.asarray(updates["params"]["weight"])).all()


def test_optimizer_consumes_training_config_model():
    config = TrainingConfig()
    variables = {"params": {"weight": jnp.ones((1,))}}

    optimizer, _ = get_optimizer(config, variables, initial_lr=1e-3)

    assert callable(optimizer.init)
