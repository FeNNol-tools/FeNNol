"""Selection of the JAX computation device.

Devices are given as ``cpu``, ``gpu``, ``cuda``, ``gpu:N`` or ``cuda:N``.
The GPU index ``N`` is relative to the GPUs exposed by ``CUDA_VISIBLE_DEVICES``,
which is never modified so that job schedulers keep control of the allocation.
JAX only creates a client (context and memory pool) on the selected GPU.

The device is taken, by decreasing priority, from the command line, the
``FENNOL_DEVICE`` environment variable and the input file, defaulting to CPU.
"""

import os
import re
import warnings
from typing import Any

import jax

_DEVICE_PATTERN = re.compile(r"^(cpu|(gpu|cuda)(:\d+)?)$")
_ALLOWED_DEVICES = ("cpu", "gpu", "cuda", "gpu|cuda:<CUDA_VISIBLE_DEVICES>")


def normalize_device(device: str) -> str:
    """Return the lowercase device string; raise ValueError if invalid."""
    device = device.lower().strip()
    if not _DEVICE_PATTERN.match(device):
        raise ValueError(
            f"Device must be one of {_ALLOWED_DEVICES}, got '{device}'"
        )
    return device


def resolve_device(cli: Any = None, input_file: Any = None) -> str:
    """Pick the device: CLI argument > FENNOL_DEVICE > input file > cpu.

    Only the selected value is validated, so an invalid value in a
    lower-priority source does not block the run. Prints where the device
    comes from and the values it overrides.
    """
    sources = [
        ("command line", cli),
        ("FENNOL_DEVICE", os.environ.get("FENNOL_DEVICE")),
        ("input file", input_file),
    ]
    given = [
        (name, value) for name, value in sources
        if value is not None and value != ""
    ]
    if not given:
        print("# No device specified, using cpu")
        return "cpu"

    (source, value), *lower_priority = given
    device = normalize_device(str(value))
    overridden = [
        f"'{other}' from {name}" for name, other in lower_priority
        if str(other).lower().strip() != device
    ]
    message = f"# Device {device} from {source}"
    if overridden:
        message += f" (overrides {', '.join(overridden)})"
    print(message)
    return device


def _backends_are_initialized() -> bool:
    from jax._src import xla_bridge

    return xla_bridge.backends_are_initialized()


def configure_device(device: str) -> jax.Device:
    """Restrict JAX to the requested device and make it the default device.

    Should be called before any JAX computation: once JAX backends are
    initialized, the device is only selected among the already visible ones
    and other GPUs may keep a context and preallocated memory.
    """
    device = normalize_device(device)
    backend = "cpu" if device == "cpu" else "gpu"
    index = int(device.split(":")[1]) if ":" in device else 0

    try:
        if _backends_are_initialized():
            warnings.warn(
                "JAX backends are already initialized: selecting device "
                f"'{device}' without restricting the GPUs visible to JAX",
                stacklevel=2,
            )
            jax_device = jax.devices(backend)[index]
        elif backend == "cpu":
            # never load the CUDA backend
            jax.config.update("jax_platforms", "cpu")
            jax_device = jax.devices("cpu")[0]
        else:
            jax.config.update("jax_cuda_visible_devices", str(index))
            jax_device = jax.devices("gpu")[0]
    except (RuntimeError, IndexError) as exc:
        if backend == "cpu":
            raise
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        exposed = "unset" if visible is None else f"'{visible}'"
        raise RuntimeError(
            f"Device '{device}' is not available. GPU indices count among the "
            f"GPUs exposed by CUDA_VISIBLE_DEVICES (currently {exposed}), and "
            "GPUs require the CUDA version of JAX "
            "(pip install \"fennol[cuda]\")."
        ) from exc

    jax.config.update("jax_default_device", jax_device)
    return jax_device
