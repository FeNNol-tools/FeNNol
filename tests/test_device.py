"""Device selection must restrict JAX without touching CUDA_VISIBLE_DEVICES."""

import json
import os
import subprocess
import sys

import pytest

from fennol.utils.device import normalize_device, resolve_device

# configure_device only works before JAX backends are initialized, so each
# scenario runs in a fresh interpreter.
_SCRIPT = """
import json, os, sys
import jax
from jax._src import xla_bridge
from fennol import configure_device

mode, device = sys.argv[1], sys.argv[2]
if mode == "fake_gpu":
    # no GPU in CI: serve CPU devices for the "gpu" backend
    real_devices = jax.devices
    jax.devices = lambda backend=None: real_devices("cpu")
if mode == "initialized":
    jax.devices()
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        dev = configure_device(device)
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit()
def name(d):
    # platform:id, stable across JAX versions (str() changed in JAX 0.12)
    return None if d is None else f"{d.platform}:{d.id}"
print(json.dumps({
    "warnings": [str(w.message) for w in caught],
    "device": name(dev),
    "default_device": name(jax.config.jax_default_device),
    "platforms": jax.config.jax_platforms,
    "cuda_visible": xla_bridge.CUDA_VISIBLE_DEVICES.value,
    "env": os.environ.get("CUDA_VISIBLE_DEVICES"),
}))
"""


def _run(mode, device):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2,3"}
    env.pop("JAX_PLATFORMS", None)
    if mode == "fake_gpu":
        # GPUs 2,3 may not exist: keep a real CUDA backend out of the test
        env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT, mode, device],
        env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "cpu"), (" GPU ", "gpu"), ("cuda", "cuda"), ("CUDA:3", "cuda:3")],
)
def test_normalize_device_accepts_valid_devices(device, expected):
    assert normalize_device(device) == expected


@pytest.mark.parametrize("device", ["tpu", "cuda:foo", "gpu:", "cuda:1,2"])
def test_normalize_device_rejects_invalid_devices(device):
    with pytest.raises(ValueError, match="Device must be one of"):
        normalize_device(device)


def test_cpu_disables_cuda_backend_and_keeps_scheduler_gpus():
    result = _run("plain", "cpu")
    assert result["platforms"] == "cpu"
    assert result["device"] == result["default_device"] == "cpu:0"
    assert result["env"] == "2,3"


@pytest.mark.parametrize(("device", "index"), [("gpu", "0"), ("cuda:1", "1")])
def test_gpu_restricts_jax_to_one_visible_gpu(device, index):
    result = _run("fake_gpu", device)
    assert result["cuda_visible"] == index
    assert result["default_device"] == result["device"]
    assert result["env"] == "2,3"


def test_configure_after_backend_initialization_warns_and_selects():
    result = _run("initialized", "cpu")
    assert any("already initialized" in w for w in result["warnings"])
    assert result["device"] == result["default_device"] == "cpu:0"



@pytest.mark.parametrize("mode", ["plain", "initialized"])
def test_unavailable_gpu_explains_index_and_visible_gpus(mode):
    # GPUs 2,3 do not exist on test machines, and CI has no CUDA JAX
    result = _run(mode, "cuda:7")
    assert result["error"].startswith("Device 'cuda:7' is not available.")
    assert "CUDA_VISIBLE_DEVICES (currently '2,3')" in result["error"]

@pytest.mark.parametrize(
    ("cli", "env", "input_file", "expected"),
    [
        ("cuda:1", "gpu", "cpu", "cuda:1"),
        (None, "GPU", "cpu", "gpu"),
        (None, None, "cuda", "cuda"),
        (None, "", None, "cpu"),
    ],
)
def test_resolve_device_precedence(
    monkeypatch, cli, env, input_file, expected
):
    if env is None:
        monkeypatch.delenv("FENNOL_DEVICE", raising=False)
    else:
        monkeypatch.setenv("FENNOL_DEVICE", env)
    assert resolve_device(cli=cli, input_file=input_file) == expected


def test_resolve_device_reports_overridden_values(monkeypatch, capsys):
    monkeypatch.setenv("FENNOL_DEVICE", "gpu")
    resolve_device(input_file="cpu")
    assert capsys.readouterr().out == (
        "# Device gpu from FENNOL_DEVICE (overrides 'cpu' from input file)\n"
    )


def test_resolve_device_does_not_report_identical_values(monkeypatch, capsys):
    monkeypatch.setenv("FENNOL_DEVICE", "GPU")
    resolve_device(cli="gpu", input_file="gpu")
    assert capsys.readouterr().out == "# Device gpu from command line\n"


def test_resolve_device_rejects_invalid_selected_device(monkeypatch):
    monkeypatch.delenv("FENNOL_DEVICE", raising=False)
    with pytest.raises(ValueError, match="Device must be one of"):
        resolve_device(cli="tpu")


def test_resolve_device_ignores_invalid_lower_priority_values(
    monkeypatch, capsys
):
    monkeypatch.setenv("FENNOL_DEVICE", "tpu")
    assert resolve_device(cli="cpu", input_file=["cuda", 0]) == "cpu"
    assert capsys.readouterr().out == (
        "# Device cpu from command line "
        "(overrides 'tpu' from FENNOL_DEVICE, '['cuda', 0]' from input file)\n"
    )


@pytest.mark.parametrize("input_file", [0, ["cuda", 0]])
def test_resolve_device_rejects_non_string_selected_device(
    monkeypatch, input_file
):
    monkeypatch.delenv("FENNOL_DEVICE", raising=False)
    with pytest.raises(ValueError, match="Device must be one of"):
        resolve_device(input_file=input_file)
