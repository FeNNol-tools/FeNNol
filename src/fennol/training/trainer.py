"""Single training loop implementations.

- Trainer: class that orchestrate the training loop.
- TrainState: Flax structure design to be updated by core functions.
- CoreFunctions: Container for the JIT-compiled step functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import freeze

from ..models import FENNIX
from .callbacks import Callback, DivergenceError
from .config import TrainingConfig


class Trainer:
    """Manages the training loop execution.

    Responsible for iterating over epochs, running training/validation steps,
    invoking callbacks, and handling state updates.

    Args:
        model (FENNIX): initial model. For the trainer, the model is like a
            shell. Model variables, the parameters, are updated before saving
            checkpoints to correspond to an up to date version of FENNIX.
        config (TrainingConfig): training configuration.
        core_functions (CoreFunctions): compiled functions required
            for training.
        callbacks (Sequence[Callback]): Callbacks to customize Trainer
            behavior.
        loss_definition (dict[str, Any]): Dictionary defining loss weights.
        schedule (Callable[[Any], tuple[float, Any]]): learning rate schedule
            function.
        index_lr (int): index of the learning rate in the Optimizer Optax
            object. Used to manually update lr.
        schedule_metrics (str):
        is_sch_adaptative (bool default False): whether the learning rate
            scheduler is based on performance (reduce on plateau) or not.
    """

    def __init__(
        self,
        model: FENNIX,
        config: TrainingConfig,
        core_functions: CoreFunctions,
        callbacks: Sequence[Callback],
        loss_definition: dict[str, Any],
        schedule: Callable,
        index_lr: int,
        schedule_metrics: str,
        is_sch_adaptative: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.train_step = core_functions.train_step
        self.validation_step = core_functions.validation_step
        self.init_opt = core_functions.init_opt
        self.init_ema = core_functions.init_ema
        self.callbacks = callbacks
        self.loss_definition = loss_definition
        self.schedule = schedule
        self.index_lr = index_lr
        self.schedule_metrics = schedule_metrics
        self.is_sch_adaptative = is_sch_adaptative
        self.metrics = {}

    def fit(
        self,
        state: TrainState,
        training_iterator: Iterator,
        validation_iterator: Iterator,
    ) -> TrainState:
        """Run the full training loop for the specified number of epochs.

        Args:
            state (TrainState): Initial state (fresh or restored).
            training_iterator (Iterator): Infinite iterator for training data.
            validation_iterator (Iterator): Infinite iterator for
                validation data.

        Returns:
            TrainState: The final state after training completes.
        """
        self.should_stop = False

        for cb in self.callbacks:
            cb.on_train_start(self, state)

        backup_state = state  # rely on flax structure immutability.
        epoch = state.epoch + 1
        while epoch <= self.config.max_epochs:
            state = state.replace(epoch=epoch)
            for cb in self.callbacks:
                cb.on_epoch_start(self, state)
            try:
                state = self._run_epoch(
                    state, training_iterator, validation_iterator, epoch
                )
                for cb in self.callbacks:
                    cb.on_epoch_end(self, state)
                state = state.replace(restore_count=0)

                if self.should_stop:
                    break

            except DivergenceError as divergence:
                print(f"Divergence detected: {divergence.reason}")
                if state.restore_count >= self.config.max_restore_count:
                    failed_restores = (
                        f"{state.restore_count} unsuccessful restores, "
                    )
                    if divergence.action_to_take == "reinit":
                        raise RuntimeError(
                            failed_restores
                            + "model could not be re initialized."
                        )
                    else:
                        print(
                            failed_restores
                            + "resuming training with reinitialization."
                        )
                        state = state.replace(restore_count=-1)

                new_scale = state.restore_scale * self.config.restore_scaling
                state = state.replace(
                    variables=backup_state.variables,
                    variables_ema=backup_state.variables_ema,
                    restore_scale=new_scale,
                    restore_count=state.restore_count + 1,
                )
                print(
                    f"Restored previous model after divergence."
                    f" Restore scale: {new_scale:.3f}"
                )

                if divergence.action_to_take == "reinit":
                    print("Reinitialized optimizer after divergence.")
                    new_opt_state = self.init_opt(state.variables)
                    new_ema_state = self.init_ema(state.variables)
                    state = state.replace(
                        opt_state=new_opt_state,
                        ema_state=new_ema_state,
                    )
                for cb in self.callbacks:
                    cb.on_restart(self, state)
                # Retry this epoch number. The iterator has advanced, but the
                # training state and epoch schedule are restored consistently.
                continue

            backup_state = state  # rely on flax structure immutability.
            epoch += 1
            # Update LR based on metric.
            if self.schedule_metrics not in self.metrics:
                raise RuntimeError(
                    f"Error: cannot update lr, '{self.schedule_metrics}'"
                    " not in metrics"
                )

            scaled_sch_metric = self.get_scaled_metrics([self.schedule_metrics])
            current_lr, sch_state = self.schedule(
                state.sch_state, scaled_sch_metric[self.schedule_metrics]
            )
            self.metrics["lr"] = current_lr
            state = state.replace(sch_state=sch_state)

        for cb in self.callbacks:
            cb.on_train_end(self, state)

        return state

    def _run_epoch(
        self,
        state: TrainState,
        training_iterator: Iterator,
        validation_iterator: Iterator,
        epoch: int,
    ) -> TrainState:
        """Inner loop for training and validation batches."""
        self.metrics = {}
        acc_train_loss = 0
        n_train_batch = self.config.nbatch_per_epoch
        # Training batches
        for _ in range(n_train_batch):
            # Adjust learning rate
            current_lr, sch_state = self.schedule(state.sch_state)
            current_lr *= state.restore_scale
            self.metrics["lr"] = current_lr
            # Todo could be replace with optax.
            opt_state = state.opt_state
            opt_state.inner_states["trainable"].inner_state[
                self.index_lr
            ].hyperparams["step_size"] = current_lr
            state = state.replace(
                sch_state=sch_state,
                opt_state=state.opt_state,
            )

            # fetch and preprocess data
            inputs_raw, data = next(training_iterator)
            inputs = self.model.preprocess(
                verbose=True,
                use_gpu=self.config.gpu_preprocessing,
                **inputs_raw,
            )
            rng_key, subkey = jax.random.split(state.rng_key)
            state = state.replace(rng_key=rng_key)
            inputs["rng_key"] = subkey
            inputs["training_epoch"] = epoch

            train_loss, new_state_dict, _ = self.train_step(
                data, inputs, state.to_dict()
            )
            state = state.replace(
                variables=new_state_dict["variables"],
                step=new_state_dict["step"],
                opt_state=new_state_dict["opt_state"],
                ema_state=new_state_dict["ema_state"],
                variables_ema=new_state_dict["variables_ema"],
            )
            acc_train_loss += float(train_loss)

        self.metrics["train_loss"] = acc_train_loss / n_train_batch

        # Validation batches
        rmses_acc = None
        maes_acc = None
        for _ in range(self.config.nbatch_per_validation):
            # fetch and preprocess data
            inputs_raw, data = next(validation_iterator)
            inputs = self.model.preprocess(
                verbose=True,
                use_gpu=self.config.gpu_preprocessing,
                **inputs_raw,
            )
            rng_key, subkey = jax.random.split(state.rng_key)
            state = state.replace(rng_key=rng_key)
            inputs["rng_key"] = subkey
            inputs["training_epoch"] = epoch

            rmses_batch, maes_batch, output_val = self.validation_step(
                data=data,
                inputs=inputs,
                variables=state.variables_ema[self.config.ema_validate],
            )
            if rmses_acc is None:
                rmses_acc, maes_acc = rmses_batch, maes_batch
            else:
                rmses_acc = jax.tree_util.tree_map(
                    jnp.add, rmses_acc, rmses_batch
                )
                maes_acc = jax.tree_util.tree_map(jnp.add, maes_acc, maes_batch)

        jax.block_until_ready(output_val)
        norm_fn = lambda x: x / self.config.nbatch_per_validation
        rmses_epoch = jax.tree_util.tree_map(norm_fn, rmses_acc)
        maes_epoch = jax.tree_util.tree_map(norm_fn, maes_acc)
        metrics_epoch = {f"rmse_{k}": float(v) for k, v in rmses_epoch.items()}
        metrics_epoch.update({
            f"mae_{k}": float(v) for k, v in maes_epoch.items()
        })
        self.metrics.update(metrics_epoch)

        return state

    def get_callback[T](
        self, callback_type: type[T], raise_error: bool = False
    ) -> T | None:
        """Find the first callback of a specific type.

        Args:
            callback_type (type): Type of the callback we want to access.
            raise_error (bool): Whether or not an error is raise if we do not
                find the callback. Defaults to False.
        """
        for cb in self.callbacks:
            if isinstance(cb, callback_type):
                return cb
        if raise_error:
            raise ValueError(
                f"Callback of type {callback_type.__name__} not found."
            )

    def get_scaled_metrics(
        self, metric_keys: list[str] | None = None
    ) -> dict[str, float]:
        """Return the scaled metrics.

        Args:
            metric_key (optional list[str]): optional specific key that
                will be filtered on.

        Returns:
            (dict[str, float]): the scaled metrics
                (optionally for the requested keys)
        """
        # scaled_metrics = self.metrics.copy()
        scaled_metrics = {
            k: v
            for k, v in self.metrics.items()
            if (metric_keys is None) or (k in metric_keys)
        }
        if metric_keys is not None:
            for missing_key in set(metric_keys).difference(set(scaled_metrics)):
                print(
                    f"Warning: {missing_key} not present in metrics, "
                    "can not be scaled."
                )

        for key, params in self.loss_definition.items():
            mult = params.get("display_mult", 1.0)
            if mult == 1.0:
                continue
            # Scale both raw and smoothed versions
            targets = [f"rmse_{key}", f"mae_{key}"]
            targets += [f"{t}_smooth" for t in targets]
            for t in targets:
                if t in scaled_metrics:
                    scaled_metrics[t] /= mult

        return scaled_metrics


class TrainState(struct.PyTreeNode):
    """Immutable state of the training loop."""

    # Counters
    step: int
    epoch: int

    # Model
    variables: dict
    preproc_state: Any

    # Randomness
    rng_key: jax.Array

    # Optimization
    opt_state: optax.OptState
    sch_state: Any
    variables_ema: Any
    ema_state: Any

    # Control Flow
    restore_scale: float = 1.0
    restore_count: int = 0
    best_metric: float = float("inf")

    @classmethod
    def create(
        cls,
        variables: Any,
        variables_ema: Any,
        ema_state: optax.OptState,
        opt_state: optax.OptState,
        sch_state: dict[str, Any],
        preproc_state: Any,
        rng_key: jax.Array,
        **kwargs,
    ) -> TrainState:
        """Build initialize state."""
        return cls(
            step=0,
            epoch=0,
            variables=variables,
            variables_ema=variables_ema,
            ema_state=ema_state,
            opt_state=opt_state,
            sch_state=sch_state,
            preproc_state=preproc_state,
            rng_key=rng_key,
            **kwargs,
        )

    @classmethod
    def init_from_checkpoint(
        cls, checkpoint: dict[str, Any] | TrainState
    ) -> TrainState:
        """Initialize a TrainState from a given checkpoint."""
        if isinstance(checkpoint, TrainState):
            return checkpoint

        train_state = TrainState(
            preproc_state=freeze(checkpoint["preproc_state"]),
            step=checkpoint["training"].get("step", 0),
            epoch=checkpoint["training"].get("epoch", 0),
            variables=checkpoint["training"].get("variables"),
            variables_ema=checkpoint["training"].get("variables_ema"),
            opt_state=checkpoint["training"].get("opt_state"),
            sch_state=checkpoint["training"].get("sch_state"),
            ema_state=checkpoint["training"].get("ema_state"),
            restore_scale=checkpoint["training"].get("restore_scale", 1.0),
            restore_count=checkpoint["training"].get("restore_count", 0),
            best_metric=checkpoint["training"].get("best_metric", float("inf")),
            rng_key=checkpoint["training"].get("rng_key"),
        )

        return train_state

    def to_dict(
        self, best_metric_override: float | None = None
    ) -> dict[str, Any]:
        """Export state for restart checkpoint."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "rng_key": self.rng_key,
            "variables": self.variables,
            "variables_ema": self.variables_ema,
            "opt_state": self.opt_state,
            "ema_state": self.ema_state,
            "sch_state": self.sch_state,
            "restore_count": self.restore_count,
            "restore_scale": self.restore_scale,
            "best_metric": (
                best_metric_override
                if best_metric_override is not None
                else self.best_metric
            ),
        }


@dataclass
class CoreFunctions:
    """Container for compiled functions use in training.

    Args:
        train_step (Callable): JIT-compiled training step function.
        validation_step (Callable): JIT-compiled validation step function.
        init_opt (Callable): Optimizer initialization function.
        init_ema (Callable): EMA initialization function.
    """

    train_step: Callable
    validation_step: Callable
    init_opt: Callable
    init_ema: Callable
