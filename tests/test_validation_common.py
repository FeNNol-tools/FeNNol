"""Checks for the shared runner of the quick validation suites."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from validation.common import compare_states, effective_samples, scientific_environment

REPOSITORY = Path(__file__).resolve().parents[1]


def test_scientific_environment_prepends_checkout_and_module_paths(tmp_path, monkeypatch):
    repository = tmp_path / "checkout"
    modules = repository / "modules"
    monkeypatch.setenv("PYTHONPATH", "/other/source")
    monkeypatch.setenv("FENNOL_MODULES_PATH", "/other/modules")

    environment = scientific_environment(repository, modules)

    assert environment["PYTHONPATH"].split(":") == [str(repository / "src"), "/other/source"]
    assert environment["FENNOL_MODULES_PATH"].split(":") == [str(modules), "/other/modules"]
    assert scientific_environment(repository, None)["FENNOL_MODULES_PATH"] == "/other/modules"


def test_restart_state_comparison_is_exact():
    state = {"rng_key": np.array([1, 2], dtype=np.uint32), "x": np.array([10.0, np.nan])}
    assert compare_states(state, {**state, "x": np.array([10.0, np.nan])}) == []
    assert compare_states(state, {**state, "x": np.array([np.nextafter(10.0, 11.0), np.nan])})
    assert compare_states(state, {**state, "rng_key": np.array([1, 3], dtype=np.uint32)})
    assert compare_states(state, {"x": state["x"]}) == ["/rng_key: missing"]


def test_effective_samples_of_independent_and_correlated_series():
    rng = np.random.default_rng(0)
    white = rng.normal(size=20000)
    correlated = np.repeat(white[:2000], 10)
    assert effective_samples(white) > 15000
    assert effective_samples(correlated) < 3000


def test_custom_module_discovery_serialization_and_analytic_force(tmp_path):
    env = os.environ.copy()
    env["FENNOL_MODULES_PATH"] = str(REPOSITORY / "validation" / "thermostat" / "modules")
    env["PYTHONPATH"] = os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)])
    program = f"""
import yaml
from pathlib import Path
from fennol.models.modules import MODULES
assert "VALIDATION_HARMONIC_LATTICE" in MODULES
from validation.thermostat.run import build_model
cfg = yaml.safe_load(Path({str(REPOSITORY / 'validation/thermostat/config.yaml')!r}).read_text())
build_model(cfg, Path({str(tmp_path / 'harmonic.fnx')!r}))
"""
    result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
