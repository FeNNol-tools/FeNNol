"""Barostat cell updates, pressure estimate and variable-cell restart."""

import jax
import jax.numpy as jnp
import numpy as np
from fennol.md import integrate as integrate_module
from fennol.md.barostats import get_barostat


def _barostat_step(kind, pressure, *, target=0.0, anisotropic=False, mask=None, restart=None, key=0):
    cell = jnp.eye(3, dtype=jnp.float32) * 10.0
    params = {
        "barostat": kind,
        "target_pressure": target,
        "compressibility": 1.0,
        "tau_piston": 1.0,
        "aniso_barostat": anisotropic,
    }
    if mask is not None:
        params["aniso_mask"] = mask
    thermostat = lambda vel, state: (vel, state)
    barostat, _, state, _ = get_barostat(
        thermostat,
        params,
        dt=0.1,
        simulation_setup={
            "kT": 1.0,
            "pbc": {"cell": cell, "volume": 1000.0},
            "nat": 3,
            "nfree": 9,
        },
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(key),
        restart_data={} if restart is None else {"barostat": restart},
    )
    dynamics_state = {
        "cell": cell,
        "pressure_tensor": jnp.diag(jnp.asarray(pressure, dtype=jnp.float32)),
        "barostat": state,
        "thermostat": {},
    }
    x, vel, out = barostat(jnp.zeros((3, 3)), jnp.zeros((3, 3)), dynamics_state)
    return np.asarray(out["cell"]), out["barostat"]


def test_berendsen_cell_moves_with_pressure_and_respects_anisotropic_mask():
    high_pressure_cell, _ = _barostat_step("BERENDSEN", [3.0, 3.0, 3.0])
    low_pressure_cell, _ = _barostat_step("BERENDSEN", [-3.0, -3.0, -3.0])
    assert np.linalg.det(high_pressure_cell) > 1000.0
    assert np.linalg.det(low_pressure_cell) < 1000.0

    masked_cell, _ = _barostat_step(
        "BERENDSEN", [3.0, 0.0, 0.0], anisotropic=True,
        mask=[True, False, False, False, False, False],
    )
    assert masked_cell[0, 0] > 10.0
    np.testing.assert_allclose(masked_cell[1:, 1:], np.eye(2) * 10.0, atol=1e-6)


def test_c_rescale_cell_moves_respect_anisotropic_mask():
    masked_cell, _ = _barostat_step(
        "C-RESCALE",
        [3.0, 0.0, 0.0],
        anisotropic=True,
        mask=[True, False, False, False, False, False],
        key=12,
    )
    assert masked_cell[0, 0] != 10.0
    np.testing.assert_allclose(masked_cell[1:, 1:], np.eye(2) * 10.0, atol=1e-6)
    np.testing.assert_allclose(masked_cell[np.triu_indices(3, 1)], 0.0, atol=1e-6)


def test_bussi_restart_repeats_the_next_stochastic_cell_step():
    _, first_state = _barostat_step("BUSSI", [1.0, 1.0, 1.0], key=19)
    direct_cell, direct_state = _barostat_step(
        "BUSSI", [1.0, 1.0, 1.0], restart=first_state, key=99
    )
    resumed_cell, resumed_state = _barostat_step(
        "BUSSI", [1.0, 1.0, 1.0], restart=first_state, key=3
    )
    np.testing.assert_allclose(direct_cell, resumed_cell)
    np.testing.assert_array_equal(direct_state["rng_key"], resumed_state["rng_key"])


def test_initial_pressure_excludes_velocity_of_frozen_atoms(monkeypatch, tmp_path):
    from fennol.md import integrate as integrate_module

    nat = 2
    cell = np.eye(3, dtype=np.float32) * 5.0
    simulation_setup = {
        "nat": nat,
        "nreplicas": 1,
        "nfree": 3,
        "mass": np.ones(nat, dtype=np.float32),
        "mass_Da": np.ones(nat, dtype=np.float32),
        "totmass_Da": float(nat),
        "frozen_atoms": np.array([0]),
        "kT": 1.0,
        "species": np.array([1, 1], dtype=np.int32),
        "pbc": {"cell": cell, "volume": 125.0, "estimate_pressure": True},
        "name": str(tmp_path / "frozen-pressure"),
    }
    batch = {
        "coordinates": np.zeros((nat, 3), dtype=np.float32),
        "species": simulation_setup["species"],
        "batch_index": np.zeros(nat, dtype=np.int32),
        "natoms": np.array([nat], dtype=np.int32),
        "cells": cell[None],
        "reciprocal_cells": np.linalg.inv(cell)[None],
    }

    class Model:
        energy_unit = "KCALPERMOL"
        variables = {}
        preproc_state = None

        def get_gradient_function(self, *keys, **kwargs):
            def energy_gradient(_variables, inputs):
                return jnp.zeros((1,), dtype=jnp.float32), {
                    "coordinates": jnp.zeros_like(inputs["coordinates"]),
                    "strain": jnp.zeros((1, 3, 3), dtype=jnp.float32),
                }, {}

            return energy_gradient

    monkeypatch.setattr(integrate_module, "load_model", lambda _params: Model())
    monkeypatch.setattr(
        integrate_module, "load_simulation_setup", lambda _params, _fprec: (simulation_setup, batch)
    )
    monkeypatch.setattr(
        integrate_module,
        "initialize_preprocessing",
        lambda _params, _model, conf, _data: ({}, conf),
    )
    monkeypatch.setattr(
        integrate_module,
        "get_restart_file",
        lambda _data: str(tmp_path / "no-restart"),
    )
    initial_velocities = jnp.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    monkeypatch.setattr(
        integrate_module,
        "get_thermostat",
        lambda *_args: (
            lambda vel, state: (vel, state), None, {}, initial_velocities, "TEST", "O"
        ),
    )

    _, _, _, _, _, dynamics_state = integrate_module.initialize_dynamics(
        {
            "dt": 0.001,
            "thermostat": "NONE",
            "barostat": "LGV",
            "target_pressure": 0.0,
            "nblist_skin": -1.0,
        },
        "float32",
        jax.random.PRNGKey(7),
    )
    np.testing.assert_allclose(dynamics_state["pressure_tensor"], 0.0, atol=1e-6)


def test_barostat_getter_restores_saved_runtime_state():
    saved_state = {
        "extvol": jnp.array([2.0]),
        "vextvol": jnp.array([3.0]),
        "rng_key": jax.random.PRNGKey(1),
        "istep": 7,
    }
    _, _, state, _ = get_barostat(
        lambda vel, thermostat_state: (vel, thermostat_state),
        {"barostat": "LGV", "target_pressure": 1.0},
        dt=0.001,
        simulation_setup={
            "kT": 1.0,
            "pbc": {"cell": jnp.eye(3), "volume": 1.0},
            "nat": 1,
            "nfree": 1,
        },
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(0),
        restart_data={"barostat": saved_state},
    )

    assert int(state["istep"]) == saved_state["istep"]
    np.testing.assert_array_equal(state["vextvol"], saved_state["vextvol"])


def test_variable_cell_restart_restores_saved_cell_and_barostat_state(
    monkeypatch, tmp_path
):
    original_cell = np.eye(3, dtype=np.float32) * 4.0
    saved_cell = np.eye(3, dtype=np.float32) * 5.0
    simulation_setup = {
        "nat": 1,
        "nreplicas": 1,
        "species": np.array([1], dtype=np.int32),
        "mass": np.ones(1, dtype=np.float32),
        "mass_Da": np.ones(1, dtype=np.float32),
        "totmass_Da": 1.0,
        "kT": 1.0,
        "nfree": 3,
        "frozen_atoms": None,
        "pbc": {
            "cell": original_cell,
            "volume": float(np.linalg.det(original_cell)),
            "estimate_pressure": True,
        },
        "name": str(tmp_path / "npt"),
    }
    batch = {
        "coordinates": np.zeros((1, 3), dtype=np.float32),
        "species": simulation_setup["species"],
        "batch_index": np.array([0], dtype=np.int32),
        "natoms": np.array([1], dtype=np.int32),
        "cells": original_cell[None, :, :],
        "reciprocal_cells": np.linalg.inv(original_cell)[None, :, :],
    }
    restart_data = {
        "coordinates": batch["coordinates"],
        "cells": saved_cell[None, :, :],
        "vel": np.zeros((1, 3), dtype=np.float32),
        "preproc_state": {},
        "simulation_time_ps": 1.0,
        "integration_step": 10,
        "thermostat": {},
        "barostat": {
            "extvol": jnp.asarray([np.linalg.det(saved_cell)]),
            "vextvol": jnp.asarray([0.25]),
            "rng_key": jax.random.PRNGKey(3),
            "istep": 10,
        },
    }

    class Model:
        energy_unit = "KCALPERMOL"
        variables = {}
        preproc_state = None

        def get_gradient_function(self, *gradient_keys, **kwargs):
            def energy_gradient(_variables, inputs):
                gradients = {
                    "coordinates": jnp.zeros_like(inputs["coordinates"]),
                    "strain": jnp.zeros((1, 3, 3), dtype=jnp.float32),
                }
                return jnp.zeros((1,), dtype=jnp.float32), gradients, {}

            return energy_gradient

    monkeypatch.setattr(integrate_module, "load_model", lambda _params: Model())
    monkeypatch.setattr(
        integrate_module,
        "load_simulation_setup",
        lambda _params, _fprec: (simulation_setup, batch),
    )
    restart_file = tmp_path / "npt.dyn.restart"
    restart_file.touch()
    monkeypatch.setattr(
        integrate_module, "get_restart_file", lambda _data: str(restart_file)
    )
    monkeypatch.setattr(
        integrate_module,
        "load_dynamics_restart",
        lambda _data: restart_data,
    )
    monkeypatch.setattr(
        integrate_module,
        "initialize_preprocessing",
        lambda _params, _model, conf, _data: ({}, conf),
    )

    _, _, _, _, restored_batch, dynamics_state = integrate_module.initialize_dynamics(
        {
            "dt": 0.001,
            "thermostat": "NONE",
            "barostat": "LGV",
            "target_pressure": 1.0,
            "nblist_skin": -1.0,
        },
        "float32",
        jnp.array([0, 1], dtype=jnp.uint32),
    )

    np.testing.assert_allclose(dynamics_state["cell"], saved_cell)
    np.testing.assert_allclose(restored_batch["cells"][0], saved_cell)
    np.testing.assert_allclose(
        restored_batch["reciprocal_cells"][0], np.linalg.inv(saved_cell)
    )
    np.testing.assert_array_equal(
        dynamics_state["barostat"]["vextvol"], restart_data["barostat"]["vextvol"]
    )
