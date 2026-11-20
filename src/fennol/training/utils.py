"""Helper methods for the training module.

- Checkpoint management.
- File system operations (backups).
- JAX/Randomness configuration.
- Precision casting.
"""
import pickle
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Literal

try:
    import torch
except ImportError:
    error_msg = (
        "PyTorch is required for training models. Install the CPU "
        "version following instructions at https://pytorch.org/get-started/locally/"
    )
    raise ImportError(error_msg)

import jax
import jax.numpy as jnp
import numpy as np

from .config import RunConfig, TrainingConfig
from .constants import RESTART_CHECKPOINT


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def linear_schedule(step, start_value, end_value, start_step, duration):
    return start_value + jnp.clip((step - start_step) / duration, 0.0, 1.0) * (
        end_value - start_value
    )

def resolve_checkpoint_epochs(config: TrainingConfig) -> list[int]:
    """Convert checkpoints configuration to list of int (epochs).

    Parses explicit epochs, periodic intervals, and custom schedules defined
    in the configuration.

    Args:
        config (TrainingConfig): training configuration.

    Returns
    -------
        list[int]: Sorted list of epoch integers when checkpoints
            must be saved.
    """
    # explicit epochs
    epochs = set(config.save_model_at_epochs)

    # periodic saves
    if config.save_model_every_epoch > 0:
        freq = config.save_model_every_epoch
        epochs.update(range(freq, config.max_epochs + 1, freq))

    # custom schedule
    if config.save_model_schedule:
        schedule_def = list(config.save_model_schedule)
        while schedule_def:
            val = schedule_def.pop(0)
            if val > 0:
                epochs.add(val)
            elif val < 0:
                # Syntax: [..., -start_epoch, frequency, ...]
                start = -val
                if not schedule_def:
                    raise ValueError(
                        "Invalid save_model_schedule: "
                        "missing frequency after negative value"
                    )
                freq = schedule_def.pop(0)
                if freq <= 0:
                    raise ValueError(
                        f"Invalid frequency {freq} in save_model_schedule"
                    )
                epochs.update(range(start, config.max_epochs + 1, freq))

    return sorted(epochs)


def load_restart_checkpoint(output_directory: Path) -> dict[str, Any]:
    """Load restart checkpoints.

    Args:
        output_directory (Path): The directory containing the checkpoint.

    Returns:
        dict[str, Any]: The loaded checkpoint dictionary containing state.

    Raises:
        FileNotFoundError: If RESTART_CHECKPOINT does not exist.
    """
    restart_checkpoint_file = output_directory / RESTART_CHECKPOINT
    if not restart_checkpoint_file.exists():
        raise FileNotFoundError(
            f"Training state file not found: {restart_checkpoint_file}"
        )
    with open(restart_checkpoint_file, "rb") as f:
        restart_checkpoint = pickle.load(f)
    return restart_checkpoint


def backup_restarted_output_dir(output_dir: Path):
    """Create a copy of output dir at the parent level."""
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"
    print(f"Backing up existing run to: {backup_dir}")
    try:
        shutil.copytree(output_dir, backup_dir, dirs_exist_ok=True)
    except Exception as e:
        error_msg = (
            f"Backup of the restarted output dir {str(output_dir)} failed: {e}"
        )
        raise BlockingIOError(error_msg)


def setup_float_precision(config: RunConfig) -> None:
    """Set JAX float precision and matmul precision."""
    jax.config.update("jax_enable_x64", config.double_precision)
    if config.double_precision:
        print("Double precision enabled.")
    jax.config.update("jax_default_matmul_precision", config.matmul_prec)


def setup_random_numbers(
    config: RunConfig,
) -> tuple[jax.Array, np.random.Generator]:
    """Seed torch, numpy and create initial Jax rnd key."""
    print(f"rng_seed: {config.rng_seed}")
    rng_key = jax.random.PRNGKey(config.rng_seed)
    torch.manual_seed(config.rng_seed)
    np.random.seed(config.rng_seed)
    random.seed(config.rng_seed)
    np_rng = np.random.Generator(np.random.PCG64(config.rng_seed))

    return rng_key, np_rng


def backup_config_file(config_path: Path, output_directory: Path) -> Path:
    """Copy config_file to output directory.

    Args:
        config_path (Path): Path to the source config file.
        output_directory (Path): Path to the destination directory.
    """
    destination_file = output_directory / f"config{config_path.suffix}"

    if config_path.exists():
        if destination_file != config_path:
            shutil.copy2(config_path, destination_file)
    else:
        print(f"Warning, config at {config_path} do not exist",
              " could not be backuped")

    return destination_file


def cast_variables_precision(
    variables: Any, precision: Literal["float32", "float64"]
) -> Any:
    """Recursively casts all floating-point leaves in a PyTree."""
    def _convert(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(precision)
        return x

    return jax.tree_util.tree_map(_convert, variables)
