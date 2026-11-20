"""Thermostat and integrator state saved to and restored from MD restart files."""

import pickle

import jax
import numpy as np
from fennol.md.thermostats import get_thermostat
from fennol.md.utils import save_dynamics_restart, us


def test_thermostat_getter_restores_saved_runtime_state():
    saved_state = {"nose_s": 4.0, "nose_v": 0.5}
    _, _, state, _, _, _ = get_thermostat(
        {"thermostat": "NOSE", "gamma": 1.0},
        dt=0.001,
        simulation_setup={
            "kT": 1.0,
            "mass": np.array([1.0]),
            "species": np.array([1]),
            "mass_Da": np.array([1.0]),
            "totmass_Da": 1.0,
            "nat": 1,
            "nfree": 1,
            "nreplicas": 1,
        },
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(0),
        restart_data={"thermostat": saved_state},
    )

    assert state["nose_s"] == saved_state["nose_s"]
    assert state["nose_v"] == saved_state["nose_v"]


def test_qtb_restart_does_not_restore_unsynchronized_generic_segment_state(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    parameters = {
        "thermostat": "QTB",
        "gamma": 1.0 / us.THZ,
        "qtb": {
            "tseg": 0.1,
            "write_spectra": False,
            "omegacut": 1000.0 / us.CM1,
            "classical_kernel": True,
        },
    }
    simulation_setup = {
        "name": str(tmp_path / "system"),
        "kT": 1.0,
        "mass": np.array([1.0]),
        "species": np.array([1]),
        "mass_Da": np.array([1.0]),
        "totmass_Da": 1.0,
        "nat": 1,
        "nfree": 1,
        "nreplicas": 1,
    }
    rng_key = jax.random.PRNGKey(0)
    thermostat, postprocess, state, vel, _, _ = get_thermostat(
        parameters,
        dt=0.01,
        simulation_setup=simulation_setup,
        fprec=np.float32,
        rng_key=rng_key,
    )

    # Reach an arbitrary mid-segment checkpoint: the QTB force index is 9,
    # while the segment counter is also at offset 9 inside the closure.
    for _ in range(9):
        vel, state = thermostat(vel, state)
        state, postprocess[0](state, postprocess[1])
    assert int(state["istep"]) == 9

    _, resumed_postprocess, resumed_state, resumed_vel, _, _ = get_thermostat(
        parameters,
        dt=0.01,
        simulation_setup=simulation_setup,
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(1),
        restart_data={
            "thermostat": state,
            "thermostat_post_state": postprocess[1],
        },
    )
    assert int(resumed_state["istep"]) == 0
    resumed_post_state = resumed_postprocess[1]
    for _ in range(11):
        resumed_vel, resumed_state = thermostat(resumed_vel, resumed_state)
        resumed_state, resumed_post_state = resumed_postprocess[0](
            resumed_state, resumed_post_state
        )

    assert int(resumed_state["istep"]) == 1


def test_dynamics_restart_saves_extended_integrator_state(tmp_path):
    simulation_setup = {
        "name": str(tmp_path / "state"),
        "nat": 1,
        "species": np.array([1]),
        "start_time_ps": 0.0,
        "dt": 0.001,
    }
    batch = {"coordinates": np.zeros((1, 3))}
    step_state = {
        "preproc_state": {},
        "istep": 0,
        "thermostat_post_state": {"counter": 4},
    }
    dynamics_state = {
        "vel": np.zeros((1, 3)),
        "istep": 2,
        "thermostat": {"rng_key": np.array([1, 2]), "nose_v": 3.0},
        "barostat": {"rng_key": np.array([3, 4]), "vextvol": 5.0},
    }

    save_dynamics_restart(simulation_setup, batch, step_state, dynamics_state)

    with open(f"{simulation_setup['name']}.dyn.restart", "rb") as restart_file:
        restart = pickle.load(restart_file)
    np.testing.assert_array_equal(
        restart["thermostat"]["rng_key"], dynamics_state["thermostat"]["rng_key"]
    )
    assert restart["thermostat"]["nose_v"] == dynamics_state["thermostat"]["nose_v"]
    np.testing.assert_array_equal(
        restart["barostat"]["rng_key"], dynamics_state["barostat"]["rng_key"]
    )
    assert restart["barostat"]["vextvol"] == dynamics_state["barostat"]["vextvol"]
    assert restart["thermostat_post_state"] == step_state["thermostat_post_state"]
