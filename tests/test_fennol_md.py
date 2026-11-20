"""MD driver: trajectory output, frozen atoms, replicas and end-to-end `fennol_md` runs."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from fennol import FENNIX
from fennol.md import dynamic as dynamic_module
from fennol.md import integrate as integrate_module
from fennol.md.dynamic import config_and_run_dynamic
from fennol.md.integrate import initialize_dynamics


def _run_mocked_dynamics(monkeypatch, tmp_path, *, pimd, tdump, write_all_beads):
    nbeads = 2 if pimd else 1
    system_name = str(tmp_path / "toy")
    simulation_setup = {
        "nat": 1,
        "nbeads": nbeads,
        "nfree": 3,
        "name": system_name,
        "pbc": None,
        "model_energy_unit": 1.0,
        "model_energy_unit_str": "kcal/mol",
        "energy_unit": 1.0,
        "energy_unit_str": "kcal/mol",
        "symbols": ["H"],
        "dt": 0.1,
        "start_time_ps": 0.0,
        "estimate_pressure": False,
        "thermostat_name": "NONE",
        "pimd": pimd,
        "variable_cell": False,
    }
    step_state = {}
    batch = {"coordinates": np.zeros((nbeads, 3), dtype=np.float32)}
    dynamics_state = {
        "coordinates": np.zeros((nbeads, 1, 3), dtype=np.float32),
        "vel": np.zeros((nbeads, 1, 3), dtype=np.float32),
        "forces": np.zeros((nbeads, 1, 3), dtype=np.float32),
        "epot": 0.0,
        "ek": 0.0,
        "ek_c": 0.0,
        "thermostat": {},
    }

    def step(_istep, state, current_dynamics_state, current_batch, _force_preprocess):
        return state, current_dynamics_state, current_batch, {}

    monkeypatch.setattr(
        dynamic_module,
        "initialize_dynamics",
        lambda *_args: (
            step,
            lambda conf, _dynamics_state: conf,
            simulation_setup,
            step_state,
            batch,
            dynamics_state,
        ),
    )
    monkeypatch.setattr(dynamic_module, "save_dynamics_restart", lambda *args: None)

    return dynamic_module.dynamic(
        {
            "nsteps": 1,
            "tdump": tdump,
            "nprint": 1,
            "nsummary": 2,
            "check_temperature": False,
            "per_atom_energy": False,
            "traj_format": "xyz",
            "write_all_beads": write_all_beads,
        },
        "cpu",
        "float32",
    )


def test_short_dump_interval_writes_at_least_one_frame(monkeypatch, tmp_path):
    _run_mocked_dynamics(
        monkeypatch,
        tmp_path,
        pimd=False,
        tdump=0.01,
        write_all_beads=False,
    )

    trajectory = tmp_path / "toy.traj.xyz"
    assert trajectory.exists()
    assert trajectory.read_text().splitlines()[0] == "1"


def test_write_all_beads_closes_each_trajectory_on_shutdown(monkeypatch, tmp_path):
    _run_mocked_dynamics(
        monkeypatch,
        tmp_path,
        pimd=True,
        tdump=0.1,
        write_all_beads=True,
    )

    for bead in (1, 2):
        trajectory = tmp_path / f"toy_bead{bead:03d}.traj.xyz"
        assert trajectory.exists()
        assert trajectory.read_text().splitlines()[0] == "1"


def test_frozen_atoms_stay_frozen_in_each_independent_replica(monkeypatch, tmp_path):
    nat = 2
    nreplicas = 2
    coordinates = np.zeros((nreplicas * nat, 3), dtype=np.float32)
    simulation_setup = {
        "nat": nat,
        "nreplicas": nreplicas,
        "frozen_atoms": np.array([0]),
        "mass": np.ones(nreplicas * nat, dtype=np.float32),
        "mass_Da": np.ones(nat, dtype=np.float32),
        "totmass_Da": float(nat),
        "kT": 1.0,
        "pbc": None,
        "name": str(tmp_path / "replicas"),
        "species": np.ones(nreplicas * nat, dtype=np.int32),
    }
    batch = {
        "coordinates": coordinates,
        "species": simulation_setup["species"],
        "batch_index": np.repeat(np.arange(nreplicas), nat),
        "natoms": np.full(nreplicas, nat),
    }

    class Model:
        energy_unit = "Ha"
        variables = {}
        preproc_state = None

        preprocessing = SimpleNamespace(
            process=lambda state, inputs: inputs,
            check_reallocate=lambda state, inputs: (state, {}, inputs, False),
            update_skin=lambda inputs: inputs,
        )

        def get_gradient_function(self, *keys, **kwargs):
            def energy_gradient(_variables, inputs):
                return (
                    jnp.zeros((nreplicas,), dtype=jnp.float32),
                    {"coordinates": jnp.zeros_like(inputs["coordinates"])},
                    {},
                )

            return energy_gradient

    monkeypatch.setattr(integrate_module, "load_model", lambda _params: Model())
    monkeypatch.setattr(
        integrate_module,
        "load_simulation_setup",
        lambda _params, _fprec: (simulation_setup, batch),
    )
    monkeypatch.setattr(
        integrate_module,
        "initialize_preprocessing",
        lambda _params, _model, conf, _data: ({}, conf),
    )
    monkeypatch.setattr(
        integrate_module,
        "get_restart_file",
        lambda _data: str(tmp_path / "missing.restart"),
    )

    initial_velocities = jnp.ones_like(jnp.asarray(coordinates))
    monkeypatch.setattr(
        integrate_module,
        "get_thermostat",
        lambda *_args: (
            lambda vel, state: (vel, state),
            None,
            {},
            initial_velocities,
            "TEST",
            "O",
        ),
    )

    step, _, _, step_state, current_batch, dynamics_state = (
        integrate_module.initialize_dynamics(
            {"dt": 0.1, "nblist_skin": -1.0}, "float32", jnp.array([0, 1], dtype=jnp.uint32)
        )
    )
    _, dynamics_state, _, _ = step(0, step_state, dynamics_state, current_batch)

    # Atom 0 is frozen in each replica; atom 1 moves under its initial velocity.
    final_coordinates = np.asarray(dynamics_state["coordinates"])
    np.testing.assert_allclose(final_coordinates[[0, 2]], 0.0)
    assert np.all(final_coordinates[[1, 3], 0] > 0.0)


def test_graph_model_custom_forces_and_preserved_inputs(tmp_path):
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    model = FENNIX(
        cutoff=3.0,
        energy_unit="kcal/mol",
        modules={
            "predicted_forces": {
                "module_name": "APPLY_FUNCTION",
                "argument_keys": ["coordinates", "graph", "drive"],
                "function_str": (
                    "x,g,d: zeros_like(x).at[g['edge_src']].add("
                    "d*g['vec']*g['edge_mask'][:,None])"
                ),
            },
        },
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.array([1, 1]), "coordinates": coordinates,
            "natoms": np.array([2]), "batch_index": np.array([0, 0]),
            "drive": np.asarray(2., dtype=np.float32),
        },
    )
    model_file = tmp_path / "graph.fnx"
    model.save(model_file)
    xyz = tmp_path / "dimer.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 1 0 0\n")
    velocities = tmp_path / "velocities.dat"
    np.savetxt(velocities, np.zeros((2, 3)))
    step, _, _, state, conf, dynamics_state = initialize_dynamics(
        {
            "model_file": str(model_file), "xyz_input/file": str(xyz),
            "system_name": str(tmp_path / "dimer"), "thermostat": "NVE",
            "dt": 0.01, "initial_velocities": str(velocities),
            "forces_key": "predicted_forces", "additional_keys": {"drive": 2.},
            "input_keys_to_keep": ["drive"],
        },
        "float32", jax.random.PRNGKey(1),
    )
    # Directed graph edges give equal-and-opposite forces along the bond.
    np.testing.assert_allclose(dynamics_state["forces"], [[2., 0., 0.], [-2., 0., 0.]])
    state, dynamics_state, conf, _ = step(0, state, dynamics_state, conf)
    assert conf["drive"] == 2.
    assert np.isfinite(dynamics_state["coordinates"]).all()
    assert float(dynamics_state["coordinates"][0, 0]) > 0.
    assert float(dynamics_state["coordinates"][1, 0]) < 1.
    np.testing.assert_allclose(np.sum(dynamics_state["forces"], axis=0), 0., atol=1e-7)


def _write_static_dimer(tmp_path):
    """Save a zero-force model and an H2 dimer at rest, so coordinates never move."""
    model = FENNIX(
        cutoff=3.0,
        energy_unit="kcal/mol",
        modules={
            "zero_forces": {
                "module_name": "APPLY_FUNCTION",
                "argument_keys": ["coordinates"],
                "function_str": "x: zeros_like(x)",
            },
        },
        rng_key=jax.random.PRNGKey(0),
    )
    model.save(tmp_path / "static.fnx")
    (tmp_path / "dimer.xyz").write_text("2\n\nH 0 0 0\nH 1 0 0\n")
    np.savetxt(tmp_path / "velocities.dat", np.zeros((2, 3)))


def _run_fennol_md(tmp_path, nsteps):
    """Run fennol_md on a YAML input for the static dimer, recording the x coordinate of atom 2."""
    param_file = tmp_path / "md.yaml"
    param_file.write_text(yaml.safe_dump({
        "model_file": str(tmp_path / "static.fnx"),
        "xyz_input": {"file": str(tmp_path / "dimer.xyz")},
        "initial_velocities": str(tmp_path / "velocities.dat"),
        "system_name": str(tmp_path / "dimer"),
        "device": "cpu",
        "forces_key": "zero_forces",
        "thermostat": "NVE",
        "dt[fs]": 1.0,
        "tdump[fs]": 2.0,  # also writes the restart file
        "nsteps": nsteps,
        "nprint": 2,
        "nsummary": 4,
        "check_temperature": False,
        "traj_format": "xyz",
        "colvars": {"x": {"type": "coordinate", "atom": 2, "axis": "x"}},
    }))
    # fennol_md's main() parses argv, then hands the file to config_and_run_dynamic.
    config_and_run_dynamic(param_file)
    return np.loadtxt(tmp_path / "dimer.colvars.traj", ndmin=2)


def test_fennol_md_writes_colvars_trajectory(tmp_path):
    _write_static_dimer(tmp_path)
    rows = _run_fennol_md(tmp_path, nsteps=4)

    assert (tmp_path / "dimer.colvars.traj").read_text().splitlines()[0].split() == ["#time[ps]", "x"]
    np.testing.assert_allclose(rows, [[0.002, 1.0], [0.004, 1.0]], atol=1e-6)


def test_fennol_md_restart_continues_simulated_time(tmp_path):
    _write_static_dimer(tmp_path)
    _run_fennol_md(tmp_path, nsteps=4)
    rows = _run_fennol_md(tmp_path, nsteps=4)  # resumes from dimer.dyn.restart

    np.testing.assert_allclose(rows[:, 0], [0.002, 0.004, 0.006, 0.008], atol=1e-6)
