"""OPES and metadynamics bias: kernels, normalization and restart state."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from fennol.md.opes import initialize_opes

# Non-empty dynamics restart data, as returned when `.dyn.restart` is loaded.
DYN_RESTART = {"integration_step": 1}


def _initialize_opes(tmp_path, bounds=None, sigma=0.5, restart_data={}):
    parameters = {
        "opes": {
            "colvars": ["cv"],
            "pace": 1,
            "sigma": sigma,
            "gamma": 2.0,
            "epsilon": 0.01,
            "tseg": 1.0,
            "kernel_chunk": 11,
        }
    }
    if bounds is not None:
        parameters["opes"]["bounds"] = {"cv": bounds}
    return initialize_opes(
        parameters,
        0.1,
        {"kT": 1.0, "name": str(tmp_path / "system")},
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: coordinates[0]},
        restart_data,
    )


def _step(step_function, state, cv):
    dynamics_state = {
        "opes": state,
        "colvars": {"cv": jnp.asarray(cv, dtype=jnp.float32)},
    }
    _, _, updated = step_function(
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
        dynamics_state,
    )
    return updated["opes"]


def test_opes_running_weight_moments_are_returned_and_neff_grows(tmp_path):
    step, _, _, state = _initialize_opes(tmp_path)

    state = _step(step, state, 0.0)
    first_mean = float(state["wk_mean"])
    assert int(state["count"]) == 2

    state = _step(step, state, 1.5)
    assert int(state["count"]) == 3
    assert float(state["wk_mean"]) != first_mean
    neff = float(state["count"] * state["wk_mean"] ** 2 / state["wk2_mean"])
    assert neff > 1.0


def test_opes_out_of_bounds_first_hill_keeps_finite_normalization(tmp_path):
    step, _, _, state = _initialize_opes(tmp_path, bounds=[-1.0, 1.0])

    state = _step(step, state, 2.0)

    assert int(state["nkernels"]) == 0
    assert np.isfinite(float(state["Zn"]))


def test_opes_accepts_integer_scalar_sigma(tmp_path):
    step, _, _, state = _initialize_opes(tmp_path, sigma=1)

    state = _step(step, state, 0.0)

    assert int(state["nkernels"]) == 1
    assert np.isfinite(float(state["Zn"]))


def test_opes_pimd_bias_force_acts_on_centroid_mode(tmp_path):
    parameters = {
        "opes": {
            "colvars": ["cv"],
            "pace": 1,
            "sigma": 0.5,
            "gamma": 2.0,
            "epsilon": 0.01,
            "tseg": 1.0,
            "kernel_chunk": 11,
        }
    }
    step, bias, _, state = initialize_opes(
        parameters,
        0.1,
        {"kT": 1.0, "name": str(tmp_path / "pimd"), "nbeads": 2},
        np.float32,
        jax.random.PRNGKey(0),
        # The CV reads every coordinate it is given, so only the bias's own
        # centroid selection can keep force off the non-centroid modes.
        {"cv": lambda coordinates, cell: jnp.sum(coordinates[..., 0])},
    )
    state = _step(step, state, 0.0)
    coordinates = jnp.array([[[0.2, 0.0, 0.0]], [[8.0, 0.0, 0.0]]])

    _, gradients = bias({"coordinates": coordinates, "opes": state})

    assert float(jnp.linalg.norm(gradients["coordinates"][0])) > 0.0
    np.testing.assert_allclose(gradients["coordinates"][1], 0.0, atol=1e-7)


def test_opes_restart_restores_bias_state_and_appends_history(tmp_path):
    step, _, postprocess, state = _initialize_opes(tmp_path)
    state = _step(step, state, 0.0)
    trajectory = tmp_path / "system.opes.traj"
    trajectory.write_text("existing history\n")
    with open(tmp_path / "system.opes.restart", "wb") as restart:
        pickle.dump(state, restart)

    _, _, _, restored = _initialize_opes(tmp_path, restart_data=DYN_RESTART)

    assert int(restored["istep"]) == int(state["istep"])
    assert int(restored["nkernels"]) == int(state["nkernels"])
    np.testing.assert_array_equal(restored["kernel_params"], state["kernel_params"])
    assert trajectory.read_text() == "existing history\n"


def test_opes_fresh_run_ignores_stale_restart_and_rewrites_history(tmp_path):
    step, _, _, state = _initialize_opes(tmp_path)
    state = _step(step, state, 0.0)
    trajectory = tmp_path / "system.opes.traj"
    trajectory.write_text("stale history\n")
    with open(tmp_path / "system.opes.restart", "wb") as restart:
        pickle.dump(state, restart)

    _, _, _, fresh = _initialize_opes(tmp_path)

    assert int(fresh["istep"]) == 0
    assert int(fresh["nkernels"]) == 0
    np.testing.assert_array_equal(fresh["kernel_params"], 0.0)
    assert "stale history" not in trajectory.read_text()


def test_opes_dynamics_restart_without_opes_restart_warns(tmp_path, capsys):
    _, _, _, state = _initialize_opes(tmp_path, restart_data=DYN_RESTART)

    assert int(state["nkernels"]) == 0
    assert "warning" in capsys.readouterr().out.lower()


def test_opes_normalization_is_independent_of_kernel_buffer_capacity(tmp_path):
    def initial_state(name, kernel_chunk):
        parameters = {
            "opes": {
                "colvars": ["cv"],
                "pace": 1,
                "sigma": 0.5,
                "gamma": 2.0,
                "epsilon": 0.01,
                "tseg": 1.0,
                "kernel_chunk": kernel_chunk,
            }
        }
        return initialize_opes(
            parameters,
            0.1,
            {"kT": 1.0, "name": str(tmp_path / name)},
            np.float32,
            jax.random.PRNGKey(0),
            {"cv": lambda coordinates, cell: coordinates[0]},
        )

    def add_first_hill(step, state):
        dynamics_state = {"opes": state, "colvars": {"cv": jnp.asarray(0.0)}}
        _, _, updated = step(jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state)
        return updated["opes"]

    small_step, _, _, small_state = initial_state("small", 11)
    large_step, _, _, large_state = initial_state("large", 21)
    small_state = add_first_hill(small_step, small_state)
    large_state = add_first_hill(large_step, large_state)

    np.testing.assert_allclose(small_state["Zn"], large_state["Zn"], atol=1e-7)


def test_opes_adaptive_sigma_is_finite_for_constant_initial_cv(tmp_path):
    parameters = {
        "opes": {
            "colvars": ["cv"],
            "pace": 1,
            "sigma": "adaptive",
            "sigma_tau": 1,
            "gamma": 2.0,
            "epsilon": 0.01,
            "tseg": 1.0,
            "kernel_chunk": 11,
        }
    }
    step, _, _, state = initialize_opes(
        parameters,
        0.1,
        {"kT": 1.0, "name": str(tmp_path / "adaptive")},
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: jnp.asarray(0.0)},
    )
    dynamics_state = {"opes": state, "colvars": {"cv": jnp.asarray(0.0)}}

    _, _, updated = step(jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state)

    assert np.isfinite(float(updated["opes"]["Zn"]))
    assert np.all(np.isfinite(np.asarray(updated["opes"]["kernel_params"])))


def test_opes_merging_retains_positive_width_for_narrow_kernels_far_from_zero(
    tmp_path,
):
    parameters = {
        "opes": {
            "colvars": ["cv"],
            "pace": 1,
            "sigma": 0.001,
            "decay_sigma": False,
            "gamma": 2.0,
            "epsilon": 0.01,
            "tseg": 1.0,
            "kernel_chunk": 11,
        }
    }
    step, _, _, state = initialize_opes(
        parameters,
        0.1,
        {"kT": 1.0, "name": str(tmp_path / "narrow")},
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: coordinates[0]},
    )
    for cv in (100.0, 100.0005):
        dynamics_state = {"opes": state, "colvars": {"cv": jnp.asarray(cv)}}
        _, _, updated = step(jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state)
        state = updated["opes"]

    assert int(state["nkernels"]) == 1
    assert np.all(np.isfinite(np.asarray(state["kernel_params"][0])))
    assert float(state["kernel_params"][0, -1]) > 0.0
