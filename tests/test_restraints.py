"""Atom, topology and wall restraints, including PIMD centroid and replica handling."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fennol.md import integrate as integrate_module
from fennol.md.restraints import build_atoms_restraints, build_wall_restraint
from fennol.md.utils import us


def test_pimd_colvar_wall_restraint_only_forces_centroid_mode():
    restraint = build_wall_restraint(
        "centroid_wall",
        {"colvars": ["x"], "center": [0.0], "force_constant": [1.0]},
        {"x": lambda coordinates, cell: coordinates[1, 0]},
        np.float32,
        {"nbeads": 2},
    )
    coordinates = jnp.array(
        [
            [[10.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        ]
    )

    energy, gradients = restraint({"coordinates": coordinates})

    assert float(energy) == pytest.approx(2.0)
    np.testing.assert_allclose(
        gradients["coordinates"],
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
    )


def test_pimd_atom_restraint_targets_selected_atom_in_centroid_mode():
    restraint = build_atoms_restraints(
        "centroid",
        {"atoms": [2], "ref_coords": [[0.0, 0.0, 0.0]], "force_constant": 2.0},
        {
            "nat": 2,
            "nbeads": 2,
            "nreplicas": 2,
            "species": np.array([6, 1]),
            "initial_coordinates": np.zeros((2, 3)),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
        },
        np.float32,
    )

    energy, gradients = restraint(
        {
            "coordinates": jnp.array(
                [
                    [[9.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[7.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                ]
            )
        }
    )
    assert float(energy) == pytest.approx(1.0)
    np.testing.assert_allclose(
        gradients["coordinates"],
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
    )


def test_topology_restraint_rejects_independent_replicas():
    from fennol.md.restraints import get_topology_bond_restraints

    with pytest.raises(ValueError, match="not supported with multiple replicas"):
        get_topology_bond_restraints(
            "bonds", {}, {"nat": 2, "nreplicas": 2}, np.float32
        )


def test_pimd_topology_restraint_only_forces_centroid_mode():
    from fennol.md.restraints import get_topology_bond_restraints
    from fennol.utils.periodic_table import COV_RADII

    restraint = get_topology_bond_restraints(
        "bonds",
        {"force_constant": 2.0, "upper_wall": 0.1},
        {
            "nat": 2,
            "nbeads": 2,
            # load_simulation_setup sets nreplicas to nbeads under PIMD.
            "nreplicas": 2,
            "species": np.array([1, 1]),
            "topology": np.array([[0, 1]]),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
        },
        np.float32,
    )
    rc = 2 * float(COV_RADII[1] / us.BOHR) + 0.1
    coordinates = jnp.array(
        [
            [[0.0, 0.0, 0.0], [rc + 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [rc + 5.0, 0.0, 0.0]],
        ]
    )

    energy, gradients = restraint({"coordinates": coordinates})

    assert float(energy) == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(
        gradients["coordinates"],
        [
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        rtol=1e-5,
    )


def test_atom_restraint_accepts_explicit_atom_indices():
    restraint = build_atoms_restraints(
        "selected",
        {"atoms": [2], "ref_coords": [[1.0, 0.0, 0.0]], "force_constant": 2.0},
        {
            "nat": 2,
            "species": np.array([6, 1]),
            "initial_coordinates": np.zeros((2, 3)),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
        },
        np.float32,
    )

    energy, gradients = restraint({"coordinates": jnp.zeros((2, 3))})
    assert float(energy) == pytest.approx(1.0)
    np.testing.assert_allclose(gradients["coordinates"][1], [-2.0, 0.0, 0.0])


def test_atom_restraint_applies_to_each_independent_replica():
    restraint = build_atoms_restraints(
        "replicated",
        {"atoms_species": ["H"], "ref_coords": [[0.0, 0.0, 0.0]], "force_constant": 2.0},
        {
            "nat": 2,
            "nreplicas": 2,
            "species": np.array([6, 1, 6, 1]),
            "initial_coordinates": np.zeros((4, 3)),
            "energy_unit": 1.0,
            "energy_unit_str": "unit",
        },
        np.float32,
    )

    coordinates = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    energy, gradients = restraint({"coordinates": coordinates})

    assert float(energy) == pytest.approx(5.0)
    np.testing.assert_allclose(
        gradients["coordinates"][:, 0], [0.0, 2.0, 0.0, 4.0]
    )


def test_single_sided_wall_energy_and_force_are_consistent():
    restraint = build_wall_restraint(
        "upper",
        {"colvars": ["x"], "upper_wall": [0.0], "force_constant_upper": [2.0]},
        {"x": lambda coordinates, cell: coordinates[0, 0]},
        np.float32,
        {},
    )
    above = {"coordinates": jnp.array([[1.0, 0.0, 0.0]])}
    below = {"coordinates": jnp.array([[-1.0, 0.0, 0.0]])}

    energy_above, gradient_above = restraint(above)
    energy_below, gradient_below = restraint(below)

    np.testing.assert_allclose(energy_above, 1.0, atol=1e-6)
    np.testing.assert_allclose(gradient_above["coordinates"], [[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(energy_below, 0.0, atol=1e-6)
    np.testing.assert_allclose(gradient_below["coordinates"], 0.0, atol=1e-6)


def test_pimd_model_and_centroid_restraint_forces_share_normal_mode_space(
    monkeypatch, tmp_path
):
    root_two = np.sqrt(2.0)
    eigmat = np.array(
        [[1.0 / root_two, -1.0 / root_two], [1.0 / root_two, 1.0 / root_two]],
        dtype=np.float32,
    )
    simulation_setup = {
        "nat": 1,
        "nreplicas": 2,
        "nbeads": 2,
        "species": np.array([1, 1], dtype=np.int32),
        "mass": np.ones(1, dtype=np.float32),
        "mass_Da": np.ones(1, dtype=np.float32),
        "totmass_Da": 1.0,
        "kT": 1.0,
        "nfree": 3,
        "pbc": None,
        "name": str(tmp_path / "pimd"),
        "energy_unit": 1.0,
        "energy_unit_str": "kcal/mol",
        "initial_coordinates": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        "frozen_atoms": None,
        "omk": np.array([0.0, 1.0], dtype=np.float32),
        "eigmat": eigmat,
    }
    batch = {
        "coordinates": np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        "species": simulation_setup["species"],
        "batch_index": np.array([0, 1], dtype=np.int32),
        "natoms": np.array([1, 1], dtype=np.int32),
    }

    class Model:
        energy_unit = "KCALPERMOL"
        variables = {}
        preproc_state = None
        preprocessing = SimpleNamespace(
            process=lambda state, inputs: inputs,
            check_reallocate=lambda state, inputs: (state, {}, inputs, False),
            update_skin=lambda inputs: inputs,
        )

        def get_gradient_function(self, *keys, **kwargs):
            def energy_gradient(_variables, inputs):
                energy_per_bead = 0.5 * jnp.sum(inputs["coordinates"] ** 2, axis=-1)
                energy = jax.ops.segment_sum(
                    energy_per_bead, inputs["batch_index"], inputs["natoms"].shape[0]
                )
                return energy, {"coordinates": inputs["coordinates"]}, {}

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
        lambda _data: str(tmp_path / "missing-pimd.restart"),
    )
    monkeypatch.setattr(
        integrate_module,
        "get_thermostat",
        lambda *_args: (
            lambda vel, state: (vel, state),
            None,
            {},
            jnp.zeros((2, 1, 3), dtype=jnp.float32),
            "TEST",
            "O",
        ),
    )

    _, _, _, _, _, dynamics_state = integrate_module.initialize_dynamics(
        {
            "dt": 0.001,
            "nblist_skin": -1.0,
            "restraints": {
                "centroid": {
                    "type": "atoms",
                    "atoms": [1],
                    "ref_coords": [[0.0, 0.0, 0.0]],
                    "force_constant": 2.0,
                }
            },
        },
        "float32",
        jnp.array([0, 1], dtype=jnp.uint32),
    )

    # The bead-space harmonic gradient transforms to [2, 1] in normal-mode
    # coordinates. The centroid restraint adds [4, 0] before the force sign.
    np.testing.assert_allclose(
        np.asarray(dynamics_state["energy_gradients"]["coordinates"])[:, 0, 0],
        [6.0, 1.0],
        rtol=1e-6,
    )
