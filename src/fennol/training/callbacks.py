"""Callbacks that add features during training."""

from __future__ import annotations

import pickle
import time
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, override

import jax
import numpy as np

from ..utils.io import human_time_duration
from .config import TrainingConfig
from .constants import LATEST_MODEL, RESTART_CHECKPOINT
from .mlflow_interface import DisabledMlFlowLogger, MlFlowLogger
from .utils import linear_schedule, resolve_checkpoint_epochs

if TYPE_CHECKING:
    from .trainer import Trainer, TrainState


class Callback(ABC):
    """Callback to add process in Trainer hooks."""

    def on_train_start(self, trainer: Trainer, state: TrainState) -> None:
        """Define an action for the train start hook."""
        return

    def on_train_end(self, trainer: Trainer, state: TrainState) -> None:
        """Define an action for the train end hook."""
        return

    def on_epoch_start(self, trainer: Trainer, state: TrainState) -> None:
        """Define an action for the epoch end hook."""
        return

    def on_epoch_end(self, trainer: Trainer, state: TrainState) -> None:
        """Define an action for the epoch end hook."""
        return

    def on_restart(self, trainer: Trainer, state: TrainState) -> None:
        """Define an action for the restart hook."""
        return


class MetricTracker(Callback):
    """Handles tracking, aggregating and smoothing of metrics.

    Args:
        beta (float): the decay rate of EMA for smoothing metrics.
            Between 1 and 0.
        initial_metrics_state (dict[str, Any] | None): Optional metrics state
            from a restart checkpoint to resume updating. Default to None.
    """

    def __init__(
        self,
        beta: float,
        initial_metrics_state: dict[str, Any] | None,
    ):
        initial_metrics_state = initial_metrics_state or {}
        self.beta = beta
        self.nsmooth = initial_metrics_state.get("nsmooth", 0)
        self.internal_smooth_values = defaultdict(lambda: 0.0)
        self.internal_smooth_values.update(
            self._load_smooth_metrics_dict(initial_metrics_state)
        )

    def on_epoch_end(self, trainer: Trainer, state: TrainState) -> None:
        """Add smoothed and total metrics to metrics state."""
        metrics = trainer.metrics
        # compute weighted sum to get total metrics
        rmse_tot = 0.0
        mae_tot = 0.0
        for key, params in trainer.loss_definition.items():
            rmse_key = metrics.get(f"rmse_{key}")
            mae_key = metrics.get(f"mae_{key}")

            if rmse_key is not None:
                rmse_tot += rmse_key * (params["weight"] ** 0.5)
            if mae_key is not None:
                mae_tot += mae_key * params["weight"]
        metrics["rmse_tot"] = rmse_tot
        metrics["mae_tot"] = mae_tot

        # smooth metric calculation
        if self.beta <= 0 or self.beta >= 1:
            return

        self.nsmooth += 1
        correction = 1.0 - self.beta**self.nsmooth  # EMA bias correction
        smoothed_output = {}
        for k, v in metrics.items():
            skip_prefix = ("epoch", "step", "lr")
            if isinstance(v, (float, int)) and not k.startswith(skip_prefix):
                self.internal_smooth_values[k] = (
                    self.beta * self.internal_smooth_values[k]
                    + (1.0 - self.beta) * v
                )
                smoothed_output[f"{k}_smooth"] = (
                    self.internal_smooth_values[k] / correction
                )

        metrics.update(smoothed_output)
        return

    def state_dict(self) -> dict[str, Any]:
        """Export state for checkpointing (flat format)."""
        return {
            "smooth_values": dict(self.internal_smooth_values),
            "nsmooth": self.nsmooth,
        }

    def _load_smooth_metrics_dict(
        self, initial_smooth_values: dict[str, Any]
    ) -> dict[str, float]:
        """Flatten hypothetical restart nested smooth metrics dict.

        Ingests the checkpoint state.
        Handles strictly converting Legacy Nested Dicts -> Flat Internal Keys.

        Args:
            initial_smooth_values ( dict[str, Any]): metric_state of the last
                restart checkpoint in case of restart or empty dict.

        Return:
            dict[str, float]: A flattened dictionary of internal EMA values.
        """
        if "smooth_values" in initial_smooth_values:
            return initial_smooth_values["smooth_values"]

        smooth_values = deepcopy(initial_smooth_values)
        for k, v in initial_smooth_values.items():
            if isinstance(v, dict) and k.endswith("_smooth"):
                for key, smooth_value in v.items():
                    metric = k.split("_")[0]
                    metric = metric[:-1] if metric.endswith("s") else metric
                    new_key = f"{metric}_{key}"
                    smooth_values[new_key] = float(smooth_value)
            elif k.endswith("_smooth"):
                smooth_values[k.replace("_smooth", "")] = v
                smooth_values.pop(k)
        return smooth_values


class Checkpointer(Callback):
    """Callback responsible for saving model files and the restart checkpoint.

    Saves the latest model at every epoch, periodically saves EMA variants,
    tracks the best model based on a specified metric, and saves a comprehensive
    restart checkpoint (including optimizer and metric states).

    Args:
        output_dir (Path): path to output directory.
        save_at_epochs (list[int]): sequence of index of epochs when a saving
            of the ema variants should occur.
        ema_val_idx (int): Index of the EMA variables to use for validation
            and saving.
        metric_best (str): name of the metric used to determine the best model.
        metric_mode (Literal["min", "max"]): whether the metric best has to be
            minimize or maximize.
        keep_all_bests (bool). Whether to overwrite the best model file or add
            a sequentially numbered best model file. Defaulted to False.
        epoch_format (int): Number of digits for formatting epoch numbers.
            Defaults to 4.
        stage_idx (int | None): Index of the current training stage used
            in saved file names. Defaults to None.
        stage_prefix (str): Prefix string for the current stage e.g. '_stage_1'.
            Used in filenames. Defaults to "".
    """

    _compare_mode = {
        "min": lambda metric, best_metric: metric < best_metric,
        "max": lambda metric, best_metric: metric > best_metric,
    }

    def __init__(
        self,
        output_dir: Path,
        save_at_epochs: list[int],
        ema_val_idx: int,
        metric_best: str,
        metric_mode: Literal["min", "max"] = "min",
        keep_all_bests: bool = False,
        epoch_format: int = 4,
        stage_idx: int | None = None,
        stage_prefix: str = "",
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.save_at_epochs = save_at_epochs
        self.ema_val_idx = ema_val_idx
        self.epoch_format = epoch_format
        self.stage_idx = stage_idx
        self.metric_best = metric_best
        self.metric_mode = metric_mode
        self.keep_all_bests = keep_all_bests
        default_best = float("inf") if metric_mode == "min" else -float("inf")
        self.best_metric_val = default_best
        self.stage_prefix = stage_prefix
        self.final_model_file = f"final_model{self.stage_prefix}.fnx"
        self.best_model_file = f"best_model{self.stage_prefix}.fnx"

    @classmethod
    def init_from_config(
        cls,
        output_dir: Path,
        config: TrainingConfig,
        stage_idx: int,
        stage_prefix: str,
    ) -> Checkpointer:
        """Initialize a Checkpointer instance from a configuration object.

        Args:
            output_dir (Path): path to output directory.
            config (TrainingConfig): The training configuration.
        stage_idx (int | None): Index of the current training stage used
            in saved file names. Defaults to None.
        stage_prefix (str): Prefix string for the current stage e.g. '_stage_1'.
            Used in filenames. Defaults to "".

        Returns:
            Checkpointer: A configured instance of the class.
        """
        save_at_epochs = resolve_checkpoint_epochs(config)
        epoch_format = len(str(config.max_epochs))

        return cls(
            output_dir=output_dir,
            save_at_epochs=save_at_epochs,
            metric_best=config.metric_best,
            ema_val_idx=config.ema_validate,
            keep_all_bests=config.keep_all_bests,
            epoch_format=epoch_format,
            stage_idx=stage_idx,
            stage_prefix=stage_prefix,
        )

    def on_epoch_end(
        self,
        trainer: Trainer,
        state: TrainState,
    ) -> None:
        """Save all required model files and the restart checkpoint at the end of an epoch.

        - Save all EMA variants for scheduled epochs.
        - Replace latest model.
        - If model's metric beat last best, save as best model.
        - Save a complete restart checkpoint as pkl.
        """
        epoch_timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

        if state.epoch in self.save_at_epochs:
            self._save_model_ema_variants(trainer, state, epoch_timestamp)

        trainer.model.variables = state.variables_ema[self.ema_val_idx]
        self._save_model(
            trainer,
            filename=LATEST_MODEL,
        )

        self._save_best_model_if_needed(trainer, state, epoch_timestamp)

        self._save_restart_checkpoint(
            trainer,
            state,
        )

    def on_train_end(self, trainer: Trainer, state: TrainState) -> None:
        """Save the final model at the end of training."""
        trainer.model.variables = state.variables_ema[self.ema_val_idx]
        self._save_model(
            trainer,
            filename=self.final_model_file,
        )
        final_model_path = str(self.output_dir / self.final_model_file)
        print("Final model saved to:", final_model_path)

    def _save_best_model_if_needed(
        self, trainer: Trainer, state: TrainState, epoch_timestamp: str
    ) -> None:
        """Compare metric_best to best value and possibly save model."""
        if self.metric_best not in trainer.metrics:
            raise ValueError(
                f"Metric for selecting best model '{self.metric_best}'"
                "not in metrics"
            )

        epoch_metric = trainer.metrics[self.metric_best]
        if self._compare_mode[self.metric_mode](
            epoch_metric, self.best_metric_val
        ):
            self.best_metric_val = epoch_metric
            trainer.model.variables = state.variables_ema[self.ema_val_idx]
            epoch_identifier_str = (
                f"_epoch{state.epoch:0{self.epoch_format}}_{epoch_timestamp}"
                if self.keep_all_bests
                else ""
            )
            filename = (
                f"best_model{self.stage_prefix}{epoch_identifier_str}.fnx"
            )
            self.best_model_file = filename
            self._save_model(
                trainer,
                filename=filename,
            )

    def _save_model_ema_variants(
        self, trainer: Trainer, state: TrainState, epoch_timestamp: str
    ):
        """Save model EMA variants."""
        for i, variables_ema in enumerate(state.variables_ema):
            ema_suffix = f"_ema{i + 1}" if len(state.variables_ema) > 1 else ""
            trainer.model.variables = variables_ema
            filename = (
                f"model{self.stage_prefix}_epoch{state.epoch:0{self.epoch_format}}"
                f"{ema_suffix}_{epoch_timestamp}.fnx"
            )
            self._save_model(
                trainer=trainer,
                filename=filename,
            )

    def _save_model(self, trainer: Trainer, filename: str):
        filepath = self.output_dir / filename
        trainer.model.save(str(filepath))

    def _save_restart_checkpoint(self, trainer: Trainer, state: TrainState):

        metrics_state = {}
        for c in trainer.callbacks:
            if isinstance(c, MetricTracker):
                metrics_state = c.state_dict()
                break
        checkpoint_data = {
            "training": state.to_dict(
                best_metric_override=self.best_metric_val
            ),
            "preproc_state": trainer.model.preproc_state,
            "metrics_state": metrics_state,
            "stage_idx": self.stage_idx,
            "best_model_file": self.best_model_file,
        }
        filepath = self.output_dir / RESTART_CHECKPOINT
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)


class ThresholdStopper(Callback):
    """Early stop based on the comparison of a metric with a threshold.

    Args:
        metric_to_monitor (str): metric used for early stopping e.g. 'rmse_tot'
        threshold (float): Threshold on the metric that define the stopping
            criteria.
        mode (Literal["min", "max"]): whether we want the score to be greater
            (max) than or the error to be lower than the threshold.
    """

    _compare_mode = {
        "min": lambda metric, threshold: metric <= threshold,
        "max": lambda metric, threshold: metric >= threshold,
    }

    def __init__(
        self,
        metric_to_monitor: str,
        threshold: float,
        mode: Literal["min", "max"],
    ) -> None:
        super().__init__()
        self.metric_to_monitor = metric_to_monitor
        self.threshold = threshold
        self.mode = mode

    @override
    def on_epoch_end(self, trainer: Trainer, state: TrainState) -> None:
        """Compare monitored metric to actual for early stop."""
        current_value = trainer.metrics.get(self.metric_to_monitor)
        if current_value is None:
            return

        if self._compare_mode[self.mode](current_value, self.threshold):
            trainer.should_stop = True


class TrainingLogger(Callback):
    """Logs training metrics to a file and the console.

    Handles creating a `.traj` log file, printing formatted epoch summaries
    to stdout, and tracking elapsed time.

    Args:
        output_dir (Path): Directory where the log file will be created.
        writing_mode (Literal["a", "w"]): File open mode. 'w' (overwrite)
            for new runs, 'a' (append) for restarts. Defaults to "w".
        stage_prefix (str): Prefix string for the current stage, used in the
            logfile name. Defaults to "".
    """

    def __init__(
        self,
        output_dir: Path,
        writing_mode: Literal["a", "w"] = "w",
        stage_prefix: str = "",
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.writing_mode = writing_mode
        self.stage_prefix = stage_prefix
        self.train_start_time = 0.0
        self.epoch_start_time = 0.0
        self._file_handle = None

    @property
    def file_handle(self) -> IO[Any]:
        """Return the active file handle for the log file.

        Raises:
            RuntimeError: If the file handle has not been initialized.
        """
        if self._file_handle is None:
            raise RuntimeError("'file_handle' is not set.")
        return self._file_handle

    @file_handle.setter
    def file_handle(self, file_handle: IO[Any]) -> None:
        self._file_handle = file_handle

    @override
    def on_train_start(self, trainer: Trainer, state: TrainState):
        """Launch timer and create logging file."""
        print("Starting training...")
        self.train_start_time = time.time()
        metric_logging_file_path = (
            self.output_dir / f"metrics{self.stage_prefix}.traj"
        )
        metric_logging_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(metric_logging_file_path, self.writing_mode)

    @override
    def on_epoch_start(self, trainer: Trainer, state: TrainState) -> None:
        """Launch epoch timer."""
        self.epoch_start_time = time.time()

    @override
    def on_epoch_end(self, trainer: Trainer, state: TrainState) -> None:
        """Log epoch in print and in file."""
        elapsed = time.time() - self.train_start_time
        trainer.metrics["elapsed_time"] = elapsed

        scaled_metrics = trainer.get_scaled_metrics()
        self._print_epoch_logs(trainer, state, scaled_metrics.copy())
        self._write_log_file(trainer, state, scaled_metrics)
        return

    @override
    def on_train_end(self, trainer: Trainer, state: TrainState):
        """Log train time and close log file."""
        train_time = human_time_duration(time.time() - self.train_start_time)
        print(f"Training time: {train_time}")

        if self.file_handle:
            self.file_handle.close()
            self._file_handle = None

    def _print_epoch_logs(
        self,
        trainer: Trainer,
        state: TrainState,
        scaled_metrics: dict[str, Any],
    ) -> None:
        """Print epoch and metrics summary."""
        # log epoch summary
        epoch_time = time.time() - self.epoch_start_time
        epoch_time_display = human_time_duration(epoch_time)
        elapsed_time = trainer.metrics["elapsed_time"]
        lr = trainer.metrics["lr"]
        loss = trainer.metrics["train_loss"]
        elapsed_time_str = human_time_duration(elapsed_time)
        batch_time = human_time_duration(
            epoch_time
            / (
                trainer.config.nbatch_per_epoch
                + trainer.config.nbatch_per_validation
            )
        )
        remain_log = ""
        if not trainer.is_sch_adaptative:
            max_epochs = trainer.config.max_epochs
            epoch = state.epoch
            remain_glob = elapsed_time * (max_epochs - epoch + 1) / epoch
            remain_last = epoch_time * (max_epochs - epoch + 1)
            # estimate remaining time via weighted average
            # (put more weight on last at the beginning)
            w_remain = np.sin(0.5 * np.pi * epoch / max_epochs)
            remaining_time = human_time_duration(
                remain_glob * w_remain + remain_last * (1 - w_remain)
            )
            remain_log = f", est. remaining time = {remaining_time}"

        epoch_log_line = (
            f"\nEpoch {state.epoch}, lr={lr:.3e}, loss = {loss:.3e}, "
            f"epoch time = {epoch_time_display}, batch time = {batch_time}, "
            f"elapsed time = {elapsed_time_str}" + remain_log
        )
        print(epoch_log_line)

        # log metrics summary
        for key, params in trainer.loss_definition.items():
            rmse_norm = scaled_metrics.get(f"rmse_{key}")
            mae_norm = scaled_metrics.get(f"mae_{key}")
            if rmse_norm is None:
                continue
            unit = (
                "(" + params["display_unit"] + ")"
                if "display_unit" in params
                else ""
            )
            weight_str = ""
            if "weight_schedule" in params:
                w = linear_schedule(epoch, *params["weight_schedule"])
                weight_str = f"(w={w:.3f})"

            fmt = ".3e" if rmse_norm < 1e-2 else ".3f"
            print(
                f"    rmse_{key}= {rmse_norm:{fmt}} ;"
                f" mae_{key}= {mae_norm:{fmt}}   {unit}  {weight_str}"
            )

    def _write_log_file(
        self,
        trainer: Trainer,
        state: TrainState,
        scaled_metrics: dict[str, Any],
    ) -> None:
        """Append a new line of metrics to the log file."""
        ordered_keys = []
        for key in trainer.loss_definition:
            ordered_keys.extend([f"rmse_{key}", f"mae_{key}"])
        ordered_keys.extend(["rmse_tot", "mae_tot"])

        metrics_log = {
            "epoch": state.epoch,
            "step": state.step,
            "data_count": state.step * trainer.config.batch_size,
            "elapsed_time": trainer.metrics["elapsed_time"],
            "lr": trainer.metrics["lr"],
            "loss": trainer.metrics["train_loss"],
        }

        checkpointer = trainer.get_callback(Checkpointer)
        if checkpointer is not None:
            metrics_log["best_metric"] = checkpointer.best_metric_val
            ordered_keys.append("best_metric")

        for k in ordered_keys:
            if k in scaled_metrics:
                metrics_log[k] = scaled_metrics[k]
        metrics_log.update({
            k: v for k, v in scaled_metrics.items() if k.endswith("_smooth")
        })

        # On first write, write header
        if self.file_handle.tell() == 0:
            headers = [f"{i + 1}:{k}" for i, k in enumerate(metrics_log.keys())]
            self.file_handle.write("# " + " ".join(headers) + "\n")

        values = [str(v) for v in metrics_log.values()]
        self.file_handle.write(" ".join(values) + "\n")
        self.file_handle.flush()


class DivergenceChecker(Callback):
    """Monitors training metrics for divergence (NaNs or exploding loss).

    Raises a DivergenceError to trigger the Trainer's restoration logic if
    instability is detected. Maintains a history of metrics from the previous
    epoch to detect sudden spikes.
    """

    def __init__(self):
        self.prev_maes: dict[str, float] = {}

    @override
    def on_train_start(self, trainer: Trainer, state: TrainState) -> None:
        """Reset the history of previous metrics."""
        self.prev_maes = {}  # reset previous value.
        return

    @override
    def on_epoch_end(self, trainer: Trainer, state: TrainState) -> None:
        """Check current metrics for NaNs or loss spike.

        Raises:
            DivergenceError: If NaNs are found (action: reinit) or if a metric
                raise beyond its defined threshold relative to the previous
                epoch (action: restore).
        """
        # reinit if Nan detected in metrics.
        for key_metric, value in trainer.metrics.items():
            if key_metric.startswith("mae_") and np.isnan(value):
                raise DivergenceError(
                    reason=f"NaN detected in metric '{key_metric}'.",
                    action_to_take="reinit",
                )

        # restore if epoch metrics spike.
        if self.prev_maes:
            for key, params in trainer.loss_definition.items():
                threshold = params.get("threshold")
                if threshold is None:
                    continue
                metric_key = f"mae_{key}"
                current_metric = trainer.metrics.get(metric_key)
                prev_metric = self.prev_maes.get(metric_key)
                if current_metric is None or prev_metric is None:
                    continue

                limit = threshold * prev_metric
                if current_metric > limit:
                    raise DivergenceError(
                        reason=(
                            f"Exploding loss detected in '{metric_key}'. "
                            f"Current: {current_metric:.4e} > "
                            f"Limit: {limit:.4e} "
                            f"({threshold} * {prev_metric:.4e})"
                        ),
                        action_to_take="restore",
                    )

        # update previous maes.
        self.prev_maes = {
            k: v for k, v in trainer.metrics.items() if k.startswith("mae_")
        }

        return


class DivergenceError(Exception):
    """
    Signal model divergence to request state restoration.

    Caught by the Trainer to initiate restoration or re-initialization routines.

    Args:
        reason (str): Description of the divergence (e.g., "NaN in mae_energy").
        action_to_take (Literal["restore", "reinit"]): The recovery strategy
            to apply. "restore" reverts to the in-memory training state from the
            end of the previous epoch,
            "reinit" resets the optimizer states.
    """

    def __init__(
        self, reason: str, action_to_take: Literal["restore", "reinit"]
    ):
        self.reason = reason
        self.action_to_take = action_to_take


class MLflowHandler(Callback):
    """Logs training metrics to MLflow."""

    def __init__(
        self,
        logger: MlFlowLogger,
    ):
        self._logger = logger

    @property
    def logger(self) -> MlFlowLogger:
        """ML flow custom logger object."""
        if self._logger is None:
            self._logger = DisabledMlFlowLogger()
        return self._logger

    @logger.setter
    def logger(self, logger: MlFlowLogger) -> None:
        self._logger = logger

    @override
    def on_train_start(self, trainer: Trainer, state: TrainState) -> None:
        """Defer model logging until the orchestrator has completed training."""

    @override
    def on_epoch_end(self, trainer: Trainer, state: TrainState) -> None:
        """Log relevant training and validation metrics to MLflow."""
        metrics = trainer.metrics.copy()

        # --- Training metrics
        train_epoch_metrics = {
            "loss_epoch_end": float(metrics["train_loss"]),
            "learning_rate_epoch_end": metrics["lr"],
        }
        self.logger.log_epoch_metrics(
            train_epoch_metrics, epoch=state.epoch, split="train"
        )

        # --- Validation metrics
        scaled_metrics = trainer.get_scaled_metrics()
        val_epoch_metrics = {}
        for key in trainer.loss_definition:
            rmse_key, mae_key = f"rmse_{key}", f"mae_{key}"
            val_epoch_metrics[rmse_key] = scaled_metrics[rmse_key]
            val_epoch_metrics[mae_key] = scaled_metrics[mae_key]

            if trainer.config.smoothen_metrics:
                rmse_key_smooth = f"rmse_{key}_smooth"
                mae_key_smooth = f"mae_{key}_smooth"
                rmse_smooth_value = scaled_metrics.get(rmse_key_smooth)
                mae_smooth_value = scaled_metrics.get(mae_key_smooth)
                if rmse_smooth_value is None:
                    raise ValueError(
                        "RMSE smooth missing at mlfow logging time."
                    )
                if mae_smooth_value is None:
                    raise ValueError(
                        "MAE smooth missing at mlfow logging time."
                    )
                self.logger.log_metric(
                    rmse_key_smooth,
                    rmse_smooth_value,
                    step=state.epoch,
                    split="val",
                )
                self.logger.log_metric(
                    mae_key_smooth,
                    mae_smooth_value,
                    step=state.epoch,
                    split="val",
                )

        val_epoch_metrics["rmse_tot"] = metrics["rmse_tot"]
        val_epoch_metrics["mae_tot"] = metrics["mae_tot"]
        if trainer.config.smoothen_metrics:
            self.logger.log_metric(
                "rmse_tot_smooth",
                scaled_metrics.get("rmse_tot_smooth"),
                step=state.epoch,
                split="val",
            )
            self.logger.log_metric(
                "mae_tot_smooth",
                scaled_metrics.get("mae_tot_smooth"),
                step=state.epoch,
                split="val",
            )

        self.logger.log_epoch_metrics(
            val_epoch_metrics, epoch=state.epoch, split="val"
        )

    @override
    def on_restart(self, trainer: Trainer, state: TrainState) -> None:
        """Log restart with restart count and epoch number to MLflow."""
        self.logger.log_restart(state.restore_count, step=state.epoch + 1)
