"""Entry point script for FeNNol training."""

from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from pathlib import Path

import jax

from ..utils.device import configure_device
from .config import RunConfig
from .io import TeeLogger
from .mlflow_interface import DisabledMlFlowLogger, MlFlowLogger
from .orchestrator import RunContext, TrainingOrchestrator
from .utils import (
    backup_config_file,
    backup_restarted_output_dir,
    setup_float_precision,
    setup_random_numbers,
)

def main():
    """Entry point for the training script.

    Parses CLI arguments, initializes the configuration, sets up the runtime
    environment (logging, devices, random seeds), and launches the
    TrainingOrchestrator.
    """
    parser = argparse.ArgumentParser(
        prog="fennol_train",
        description="Train or restart a FeNNix model run.",
    )
    parser.add_argument(
        "config_file",
        type=str,
        default=None,
        help="Path to config.yaml OR output directory to restart",
    )
    parser.add_argument("--model_file", type=str, default=None)
    args = parser.parse_args()
    config_file = args.config_file
    model_file = args.model_file

    try:
        config = RunConfig.from_cli_args(config_file, model_file)
        dataset_path = config.training.dspath
        is_remote_dataset = isinstance(dataset_path, str) and dataset_path.lower().startswith(
            ("http://", "https://")
        )
        if not is_remote_dataset and not Path(dataset_path).exists():
            raise ValueError(f"Dataset file not found at {dataset_path}")
    except Exception as e:
        print(f"Error loading configuration from {config_file}:\n{e}")
        sys.exit(1)

    if config.is_restart:
        backup_restarted_output_dir(config.output_directory)
    else:
        config.output_directory.mkdir(parents=True, exist_ok=True)
        print("Output directory:", config.output_directory)

    path_backuped_config = backup_config_file(
        config.path_config_file, config.output_directory
    )
    run_context = setup_run_context(config)
    run_context.mlflow_logger.config_path = path_backuped_config

    orchestrator = TrainingOrchestrator(run_context)
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(0)
    except Exception:
        print("Training failed.")
        traceback.print_exc()
        sys.exit(1)
    finally:
        run_context.logger.unbind_stdout()
        run_context.logger.close()


def setup_run_context(config: RunConfig) -> RunContext:
    """Set up technical configuration for the run.

    - Configures environment variables.
    - Setup logging.
    - Apply device configuration to JAX and the OS environment.
        **Must be called before any JAX computation.**
    - Set float precision.
    - Set up random number generators.

    Args:
        config (RunConfig): the configuration of the training run.

    Returns:
        RunContext: configuration and runtime state of the run.

    """
    mlflow_client_logger = None
    if config.mlflow.logging_enabled:
        try:
            from .mlflow import MlFlowLogger as mlflow_client_logger
        except ModuleNotFoundError as exc:
            if exc.name == "mlflow" or (exc.name or "").startswith("mlflow."):
                raise RuntimeError(
                    "MLflow logging is enabled, but the optional dependency "
                    "is not installed. Install fennol[train,mlflow]."
                ) from exc
            raise

    os.environ["OMP_NUM_THREADS"] = "1"
    config.output_directory.mkdir(parents=True, exist_ok=True)
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    logger = TeeLogger(str(config.output_directory) + "/train.log")
    logger.bind_stdout()

    jax_device = configure_device(config.device)
    print(f"# Using device: {jax_device} (requested {config.device})")

    setup_float_precision(config)
    rng_key, np_rng = setup_random_numbers(config)

    mlflow_logger: MlFlowLogger
    if mlflow_client_logger is not None:
        try:
            mlflow_logger = mlflow_client_logger(config)
        except Exception as exc:
            if config.mlflow.strict:
                raise
            print(f"Could not initialize MLflow; continuing without it: {exc}")
            mlflow_logger = DisabledMlFlowLogger()
    else:
        mlflow_logger = DisabledMlFlowLogger()

    run_context = RunContext(
        config=config,
        rng_key=rng_key,
        np_rng=np_rng,
        jax_device=jax_device,
        logger=logger,
        mlflow_logger=mlflow_logger,
    )

    return run_context


if __name__ == "__main__":
    main()
