"""Unit checks for eABF initialization and restart state."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from fennol.md.eabf import initialize_eabf


def test_nonperiodic_eabf_system_does_not_require_pbc_metadata(tmp_path):
    initialize_eabf(
        {
            "abf": {
                "colvar": "r",
                "sigma": 0.2,
                "bounds": [0.3, 1.8],
                "bin_width": 0.1,
            }
        },
        0.0005,
        {
            "kT": 1.0,
            "name": str(tmp_path / "nonperiodic"),
            "energy_unit": 1.0,
            "energy_unit_str": "energy",
            "initial_coordinates": np.zeros((2, 3)),
        },
        np.float32,
        jax.random.PRNGKey(0),
        {"r": lambda coordinates, cell: jnp.linalg.norm(coordinates[1] - coordinates[0])},
    )


def test_eabf_restart_saves_partial_block_after_boundary_reset(tmp_path):
    parameters = {
        "abf": {
            "colvar": "r",
            "sigma": 0.2,
            "bounds": [-1.0, 1.0],
            "bin_width": 0.5,
            "block_average": 0.2,
        }
    }
    state_data = {
        "kT": 1.0,
        "name": str(tmp_path / "block"),
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "initial_coordinates": np.zeros((1, 3)),
    }
    step, _, postprocess, state = initialize_eabf(
        parameters,
        0.1,
        state_data,
        np.float32,
        jax.random.PRNGKey(0),
        {"r": lambda coordinates, cell: coordinates[0, 0]},
    )
    for _ in range(2):
        _, _, updated = step(
            jnp.zeros((1, 3)),
            jnp.zeros((1, 3)),
            {"abf": state, "colvars": {"r": jnp.asarray(0.25)}},
        )
        state = updated["abf"]
    for _ in range(10):
        state = postprocess(state)

    with open(tmp_path / "block.abf.restart", "rb") as stream:
        restart = pickle.load(stream)

    assert int(state["block_step"]) == 0
    assert restart["block_step"] == 0
    assert restart["iblock"] == 1
    assert int(np.sum(restart["block_counts"])) == 0


# Non-empty dynamics restart data, as returned when `.dyn.restart` is loaded.
DYN_RESTART = {"integration_step": 1}


def _initialize_eabf(tmp_path, fullsamples=0, restart_data={}):
    parameters = {
        "abf": {
            "colvar": "cv",
            "sigma": 1.0,
            "bounds": [-1.0, 1.0],
            "bin_width": 0.5,
            "fullsamples": fullsamples,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "name": str(tmp_path / "system"),
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "nat": 1,
        "initial_coordinates": np.zeros((1, 3)),
        "pbc_data": None,
    }
    return initialize_eabf(
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: coordinates[0, 0]},
        restart_data,
    )


def _eabf_step(step, state, cv):
    dynamics_state = {"abf": state, "colvars": {"cv": jnp.asarray(cv)}}
    _, _, updated = step(
        jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state
    )
    return updated["abf"]


def test_eabf_uses_lambda_histogram_gradient_for_applied_bias(tmp_path):
    step, _, _, state = _initialize_eabf(tmp_path)
    state["l"] = jnp.asarray(0.75)
    state["l_counts"] = state["l_counts"].at[3].set(2)
    state["gradients_avg"] = state["gradients_avg"].at[3].set(4.0)
    state["cv_counts"] = state["cv_counts"].at[2].set(1)
    state["cv_gradients_avg"] = state["cv_gradients_avg"].at[2].set(10.0)

    state = _eabf_step(step, state, 0.0)

    # dedl is 0.75; its running mean in the lambda bin becomes 2.9166667.
    np.testing.assert_allclose(float(state["fbias"]), 35.0 / 12.0, rtol=1e-6)


def test_eabf_out_of_bounds_cv_does_not_wrap_to_last_bin(tmp_path):
    step, _, _, state = _initialize_eabf(tmp_path)

    state = _eabf_step(step, state, -1.5)

    assert int(state["cv_counts"][-1]) == 0


def test_eabf_fullsamples_zero_has_finite_bias(tmp_path):
    step, _, _, state = _initialize_eabf(tmp_path, fullsamples=0)
    state["l"] = jnp.asarray(0.25)

    state = _eabf_step(step, state, 0.0)

    assert np.isfinite(float(state["fbias"]))


def test_eabf_pimd_bias_force_acts_on_centroid_mode(tmp_path):
    parameters = {
        "abf": {
            "colvar": "cv",
            "sigma": 1.0,
            "bounds": [-1.0, 1.0],
            "bin_width": 0.5,
            "tseg": 1.0,
        }
    }
    step, bias, _, state = initialize_eabf(
        parameters,
        0.1,
        {
            "kT": 1.0,
            "name": str(tmp_path / "pimd"),
            "energy_unit": 1.0,
            "energy_unit_str": "energy",
            "nat": 1,
            "nbeads": 2,
            "initial_coordinates": np.zeros((1, 3)),
            "pbc_data": None,
        },
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: jnp.sum(coordinates[..., 0])},
    )
    coordinates = jnp.array([[[0.2, 0.0, 0.0]], [[8.0, 0.0, 0.0]]])

    _, gradients = bias({"coordinates": coordinates, "abf": state})

    assert float(jnp.linalg.norm(gradients["coordinates"][0])) > 0.0
    np.testing.assert_allclose(gradients["coordinates"][1], 0.0, atol=1e-7)


def test_eabf_restart_restores_extended_state_and_preserves_trajectory(tmp_path):
    step, _, postprocess, state = _initialize_eabf(tmp_path)
    state = _eabf_step(step, state, 0.25)
    for _ in range(10):
        state = postprocess(state)
    trajectory = tmp_path / "system.abf.traj"
    trajectory.write_text(trajectory.read_text() + "preserved history\n")

    _, _, _, restored = _initialize_eabf(tmp_path, restart_data=DYN_RESTART)

    assert int(restored["istep"]) == int(state["istep"])
    np.testing.assert_allclose(restored["l"], state["l"])
    np.testing.assert_allclose(restored["vl"], state["vl"])
    np.testing.assert_array_equal(restored["rng_key"], state["rng_key"])
    assert trajectory.read_text().endswith("preserved history\n")


def test_eabf_restarted_rng_key_matches_fresh_key_layout(tmp_path):
    step, _, postprocess, fresh = _initialize_eabf(tmp_path)
    state = _eabf_step(step, fresh, 0.25)
    for _ in range(10):
        state = postprocess(state)

    _, _, _, restored = _initialize_eabf(tmp_path, restart_data=DYN_RESTART)

    assert restored["rng_key"].dtype == fresh["rng_key"].dtype
    assert restored["rng_key"].shape == fresh["rng_key"].shape
    _eabf_step(step, restored, 0.25)


def test_eabf_fresh_run_ignores_stale_restart(tmp_path):
    step, _, postprocess, state = _initialize_eabf(tmp_path)
    state = _eabf_step(step, state, 0.25)
    for _ in range(10):
        state = postprocess(state)
    assert int(state["l_counts"].sum()) > 0

    _, _, _, fresh = _initialize_eabf(tmp_path)

    assert int(fresh["istep"]) == 0
    assert int(fresh["l_counts"].sum()) == 0


def test_eabf_restart_preserves_incomplete_block_and_gradient_variance(tmp_path):
    parameters = {
        "abf": {
            "colvar": "cv",
            "sigma": 1.0,
            "bounds": [-1.0, 1.0],
            "bin_width": 0.5,
            "fullsamples": 0,
            "block_average": 2.0,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "name": str(tmp_path / "system"),
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "nat": 1,
        "initial_coordinates": np.zeros((1, 3)),
        "pbc_data": None,
    }
    args = (
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: coordinates[0, 0]},
    )
    step, _, postprocess, state = initialize_eabf(*args)
    for _ in range(10):
        dynamics_state = {"abf": state, "colvars": {"cv": jnp.asarray(0.25)}}
        _, _, updated = step(jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state)
        state = updated["abf"]
    state["gradients_var"] = state["gradients_var"].at[2].set(3.0)
    for _ in range(10):
        state = postprocess(state)

    _, _, _, restored = initialize_eabf(*args, restart_data={"integration_step": 10})

    np.testing.assert_array_equal(restored["block_counts"], state["block_counts"])
    np.testing.assert_allclose(restored["block_gradients"], state["block_gradients"])
    assert int(restored["block_step"]) == int(state["block_step"])
    assert int(restored["iblock"]) == int(state["iblock"])
    np.testing.assert_allclose(restored["gradients_var"], state["gradients_var"])


def test_eabf_upper_bound_is_counted_in_last_histogram_bin(tmp_path):
    parameters = {
        "abf": {
            "colvar": "cv",
            "sigma": 1.0,
            "bounds": [-1.0, 1.0],
            "bin_width": 0.5,
            "fullsamples": 0,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "name": str(tmp_path / "upper"),
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "nat": 1,
        "initial_coordinates": np.zeros((1, 3)),
        "pbc_data": None,
    }
    step, _, _, state = initialize_eabf(
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(1),
        {"cv": lambda coordinates, cell: coordinates[0, 0]},
    )
    state["l"] = jnp.asarray(1.0)
    dynamics_state = {"abf": state, "colvars": {"cv": jnp.asarray(1.0)}}

    _, _, updated = step(jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state)

    assert int(updated["abf"]["l_counts"][-1]) == 1
    assert int(updated["abf"]["cv_counts"][-1]) == 1
