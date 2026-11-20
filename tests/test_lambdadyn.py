"""Unit checks for lambda dynamics: ABF switch and restart state."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fennol.md.integrate import _validate_replica_lambdadyn
from fennol.md.lambdadyn import initialize_lambdadyn
from fennol.md.utils import us


def _one_step(tmp_path, abf):
    from fennol.md.lambdadyn import initialize_lambdadyn

    parameters = {
        "lambdadyn": {
            "ligand": [1],
            "lambda": 0.4,
            "bin_width": 0.25,
            "fullsamples": 0,
            "tseg": 0.001,
            "abf": abf,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([6, 1], dtype=np.int32),
        "name": str(tmp_path / f"lambda-{abf}"),
    }
    step, _, _, state = initialize_lambdadyn(
        parameters,
        dt=0.001,
        simulation_setup=simulation_setup,
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(3),
    )
    coords = jnp.zeros((2, 3), dtype=jnp.float32)
    # A constant vdW gradient: with fullsamples=0 the ABF bias cancels it exactly.
    gradients = {"alch_elambda": 0.0, "alch_vlambda": 2.0}
    _, _, updated = step(
        coords, coords, {"lambdadyn": state, "energy_gradients": gradients}
    )
    return state, updated["lambdadyn"]


def test_abf_switch_removes_bias_but_keeps_statistics(tmp_path):
    initial, with_abf = _one_step(tmp_path, True)
    _, without_abf = _one_step(tmp_path, False)

    # Identical RNG streams: only the deterministic ABF kick differs.
    np.testing.assert_array_equal(with_abf["rng_key"], without_abf["rng_key"])
    np.testing.assert_array_equal(with_abf["lambda_counts"], without_abf["lambda_counts"])
    np.testing.assert_allclose(with_abf["gradients_avg"], without_abf["gradients_avg"])
    # dU/dlambda = 2 * d(vlambda)/dlambda = 2 * 2 at lambda=0.4 with bound_vlambda=0.5.
    assert float(np.max(with_abf["gradients_avg"])) == pytest.approx(4.0)

    velocity_gap = float(with_abf["alch_lambda_vel"] - without_abf["alch_lambda_vel"])
    assert velocity_gap > 0.0
    assert float(with_abf["alch_lambda"]) > float(without_abf["alch_lambda"])


def test_lambdadyn_restart_uses_restored_lambda_for_couplings(tmp_path):
    parameters = {
        "lambdadyn": {"ligand": [1], "lambda": 0.3, "bin_width": 0.25, "tseg": 0.001}
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([6, 1], dtype=np.int32),
        "name": str(tmp_path / "lambda"),
    }
    _, _, _, state = initialize_lambdadyn(
        parameters,
        dt=0.001,
        simulation_setup=simulation_setup,
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(0),
        restart_data={
            "integration_step": 10,
            "lambdadyn": {"alch_lambda": 0.8, "alch_lambda_vel": 0.0},
        },
    )
    assert float(state["alch_lambda"]) == pytest.approx(0.8)
    # bound_elambda = bound_vlambda = 0.5 by default.
    assert float(state["alch_elambda"]) == pytest.approx(0.6)
    assert float(state["alch_vlambda"]) == pytest.approx(1.0)


def test_lambdadyn_restart_restores_rng_key(tmp_path):
    from fennol.md.lambdadyn import initialize_lambdadyn

    system_name = str(tmp_path / "lambda")
    parameters = {
        "lambdadyn": {
            "ligand": [1],
            "lambda": 0.4,
            "bin_width": 0.25,
            "fullsamples": 0,
            "tseg": 0.001,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([6, 1], dtype=np.int32),
        "name": system_name,
    }
    original_key = jax.random.PRNGKey(12)
    step, _, postprocess, state = initialize_lambdadyn(
        parameters,
        dt=0.001,
        simulation_setup=simulation_setup,
        fprec=np.float32,
        rng_key=original_key,
    )
    coords = jnp.zeros((2, 3), dtype=jnp.float32)
    _, _, updated = step(
        coords,
        coords,
        {"lambdadyn": state, "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0}},
    )
    state = postprocess(updated["lambdadyn"])

    with open(f"{system_name}.lambdadyn.restart", "rb") as stream:
        saved = pickle.load(stream)
    np.testing.assert_array_equal(saved["rng_key"], np.asarray(state["rng_key"]))

    resumed_step, _, _, resumed = initialize_lambdadyn(
        parameters,
        dt=0.001,
        simulation_setup=simulation_setup,
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(99),
        restart_data={
            "integration_step": 1,
            "lambdadyn": {
                "alch_lambda": state["alch_lambda"],
                "alch_lambda_vel": state["alch_lambda_vel"],
            },
        },
    )
    np.testing.assert_array_equal(np.asarray(resumed["rng_key"]), np.asarray(state["rng_key"]))
    assert int(resumed["istep"]) == int(state["istep"])
    _, _, resumed_update = resumed_step(
        coords,
        coords,
        {"lambdadyn": resumed, "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0}},
    )
    _, _, continuous_update = step(
        coords,
        coords,
        {"lambdadyn": state, "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0}},
    )
    np.testing.assert_array_equal(
        np.asarray(resumed_update["lambdadyn"]["alch_lambda"]),
        np.asarray(continuous_update["lambdadyn"]["alch_lambda"]),
    )
    np.testing.assert_array_equal(
        np.asarray(resumed_update["lambdadyn"]["rng_key"]),
        np.asarray(continuous_update["lambdadyn"]["rng_key"]),
    )


def test_lambdadyn_is_rejected_with_independent_replicas():
    message = "not supported with independent replicas"
    with pytest.raises(ValueError, match=message):
        _validate_replica_lambdadyn(2, {"ligand": [1]})


def test_lambdadyn_is_allowed_with_pimd():
    _validate_replica_lambdadyn(2, {"ligand": [1]}, nbeads=2)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("bound_elambda", 1.0), ("bound_vlambda", 0.0)],
)
def test_lambdadyn_rejects_singular_bounds(parameter, value):
    with pytest.raises(AssertionError):
        initialize_lambdadyn(
            {"lambdadyn": {"ligand": [1], parameter: value}},
            dt=0.001,
            simulation_setup={
                "kT": 1.0,
                "species": np.array([6, 1]),
                "energy_unit": 1.0,
                "energy_unit_str": "unit",
                "name": "system",
            },
            fprec=np.float32,
            rng_key=jnp.array([0, 1], dtype=jnp.uint32),
        )


def test_pimd_lambdadyn_topology_restraint_only_forces_centroid_mode(tmp_path):
    from fennol.utils.periodic_table import COV_RADII

    _, bias, _, state = initialize_lambdadyn(
        {
            "lambdadyn": {
                "ligand": [1],
                "lambda": 0.5,
                "topology_wall_K": 2.0,
                "topology_wall_R": 0.1,
            }
        },
        dt=0.001,
        simulation_setup={
            "kT": 1.0,
            "species": np.array([1, 1]),
            "nbeads": 2,
            "topology": np.array([[0, 1]]),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
            "name": str(tmp_path / "system"),
        },
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(0),
    )
    rc = 2 * float(COV_RADII[1] / us.BOHR) + 0.1
    coordinates = jnp.array(
        [
            [[0.0, 0.0, 0.0], [rc + 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [rc + 5.0, 0.0, 0.0]],
        ]
    )

    energy, gradients = bias({"coordinates": coordinates, "lambdadyn": state})

    # sin(0.5 * pi)**2 == 1, so the energy is K * (r - rc)**2 on the centroid.
    assert float(energy) == pytest.approx(2.0, rel=1e-5)
    np.testing.assert_allclose(
        gradients["coordinates"],
        [
            [[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        rtol=1e-5,
    )


def _initialize_lambdadyn_with_stale_restart(tmp_path, restart_data):
    edges = np.linspace(0.0, 1.0, 101)
    with open(tmp_path / "system.lambdadyn.restart", "wb") as restart:
        pickle.dump(
            {
                "istep": 7,
                "centers": 0.5 * (edges[:-1] + edges[1:]),
                "ligand_indices": np.array([0]),
                "bound_elambda": 0.5,
                "bound_vlambda": 0.5,
                "lambda_counts": np.ones(100, dtype=np.int32),
                "gradients_avg": np.ones(100),
            },
            restart,
        )
    trajectory = tmp_path / "system.deltaG.traj"
    trajectory.write_text("stale history\n")
    _, _, _, state = initialize_lambdadyn(
        {"lambdadyn": {"ligand": [1]}},
        dt=0.001,
        simulation_setup={
            "kT": 1.0,
            "species": np.array([6, 1]),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
            "name": str(tmp_path / "system"),
        },
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(0),
        restart_data=restart_data,
    )
    return state, trajectory.read_text()


def test_lambdadyn_fresh_run_ignores_stale_restart(tmp_path):
    state, history = _initialize_lambdadyn_with_stale_restart(tmp_path, {})

    assert int(state["lambda_counts"].sum()) == 0
    assert "stale history" not in history


def test_lambdadyn_dynamics_restart_loads_histograms(tmp_path):
    state, history = _initialize_lambdadyn_with_stale_restart(
        tmp_path, {"integration_step": 7}
    )

    assert int(state["lambda_counts"].sum()) == 100
    assert history == "stale history\n"


def test_lambdadyn_dynamics_restart_without_restart_file_warns(
    tmp_path, capsys
):
    initialize_lambdadyn(
        {"lambdadyn": {"ligand": [1]}},
        dt=0.001,
        simulation_setup={
            "kT": 1.0,
            "species": np.array([6, 1]),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
            "name": str(tmp_path / "system"),
        },
        fprec=np.float32,
        rng_key=jax.random.PRNGKey(0),
        restart_data={"integration_step": 7},
    )

    assert "warning" in capsys.readouterr().out.lower()


def test_lambda_fullsamples_zero_keeps_extended_velocity_finite(tmp_path):
    parameters = {
        "lambdadyn": {
            "ligand": [1],
            "lambda": 0.4,
            "bin_width": 0.25,
            "fullsamples": 0,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([1, 1], dtype=np.int32),
        "name": str(tmp_path / "lambda"),
    }
    step, _, _, state = initialize_lambdadyn(
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(0),
    )
    dynamics_state = {
        "lambdadyn": state,
        "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0},
    }

    _, _, updated = step(jnp.zeros((2, 3)), jnp.zeros((2, 3)), dynamics_state)

    assert np.isfinite(float(updated["lambdadyn"]["alch_lambda_vel"]))


def test_lambda_upper_bound_is_counted_in_last_histogram_bin(tmp_path):
    parameters = {
        "lambdadyn": {
            "ligand": [1],
            "lambda": 1.0,
            "bin_width": 0.25,
            "fullsamples": 0,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([1, 1], dtype=np.int32),
        "name": str(tmp_path / "lambda-upper"),
    }
    step, _, _, state = initialize_lambdadyn(
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(0),
    )
    dynamics_state = {
        "lambdadyn": state,
        "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0},
    }

    _, _, updated = step(jnp.zeros((2, 3)), jnp.zeros((2, 3)), dynamics_state)

    assert int(updated["lambdadyn"]["lambda_counts"][-1]) == 1


def test_lambda_out_of_bounds_sample_does_not_clip_into_last_bin(tmp_path):
    parameters = {
        "lambdadyn": {
            "ligand": [1],
            "lambda": 0.4,
            "bin_width": 0.25,
            "fullsamples": 0,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([1, 1], dtype=np.int32),
        "name": str(tmp_path / "lambda-outside"),
    }
    step, _, _, state = initialize_lambdadyn(
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(2),
    )
    state["alch_lambda"] = jnp.asarray(1.1)
    dynamics_state = {
        "lambdadyn": state,
        "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0},
    }

    _, _, updated = step(jnp.zeros((2, 3)), jnp.zeros((2, 3)), dynamics_state)

    assert int(jnp.sum(updated["lambdadyn"]["lambda_counts"])) == 0
    assert np.isfinite(float(updated["lambdadyn"]["alch_lambda_vel"]))


def test_lambda_dynamics_with_opes_updates_finite_bias(tmp_path):
    parameters = {
        "lambdadyn": {
            "ligand": [1],
            "lambda": 0.4,
            "bin_width": 0.05,
            "fullsamples": 0,
            "barrier_opes": 2.0,
            "pace_opes": 1,
            "sigma_opes": 0.1,
            "tseg": 1.0,
        }
    }
    simulation_setup = {
        "kT": 1.0,
        "energy_unit": 1.0,
        "energy_unit_str": "energy",
        "species": np.array([1, 1], dtype=np.int32),
        "name": str(tmp_path / "lambda-opes"),
    }
    step, _, _, state = initialize_lambdadyn(
        parameters,
        0.1,
        simulation_setup,
        np.float32,
        jax.random.PRNGKey(0),
    )
    dynamics_state = {
        "lambdadyn": state,
        "energy_gradients": {"alch_elambda": 0.0, "alch_vlambda": 0.0},
    }

    _, _, updated = step(jnp.zeros((2, 3)), jnp.zeros((2, 3)), dynamics_state)
    updated_state = updated["lambdadyn"]

    assert int(updated_state["opes_count"]) == 1
    assert np.all(np.isfinite(np.asarray(updated_state["opes_bias"])))
    assert np.all(np.isfinite(np.asarray(updated_state["opes_fbias"])))
