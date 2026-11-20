"""Collective variables: definitions, values and replica support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fennol.md.colvars import (
    build_colvar_coordnum,
    build_colvar_dihedral,
    build_colvar_distance,
    build_colvar_function,
)
from fennol.md.integrate import (
    _validate_replica_colvars,
)


def test_function_colvar_accepts_cell_and_forwards_it_to_components():
    colvar = build_colvar_function(
        "twice_distance",
        {
            "lambda": "distance: 2 * distance",
            "distance": {"type": "distance", "atoms": [1, 2]},
        },
    )
    coordinates = jnp.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
    cell = jnp.eye(3) * 10

    assert float(colvar(coordinates, cell)) == pytest.approx(1.0)


def test_colvars_are_rejected_with_independent_replicas():
    with pytest.raises(ValueError, match="not supported with independent replicas"):
        _validate_replica_colvars(2, {"distance": {"type": "distance"}})


def test_colvars_are_allowed_with_pimd_centroid():
    _validate_replica_colvars(2, {"distance": {"type": "distance"}}, nbeads=2)


def test_dihedral_is_invariant_to_terminal_bond_length():
    dihedral = build_colvar_dihedral(
        "phi", {"atoms": [1, 2, 3, 4], "use_radians": True}
    )
    coordinates = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]]
    )
    scaled_terminal_bond = coordinates.at[3].set(jnp.array([3.0, 1.0, 2.0]))

    np.testing.assert_allclose(
        dihedral(coordinates), dihedral(scaled_terminal_bond), atol=1e-6
    )


def test_coordination_number_cutoff_has_analytic_value_and_continuous_gradient():
    coordination = build_colvar_coordnum(
        "cn", {"group1": "1", "group2": "2", "cutoff": 1.0}
    )

    def value_at_distance(distance):
        return coordination(jnp.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]))

    at_cutoff = value_at_distance(1.0)
    derivative_at_cutoff = jax.grad(value_at_distance)(1.0)
    expected_derivative = (6.0 / 12.0) * (6.0 - 12.0) / 2.0

    np.testing.assert_allclose(at_cutoff, 0.5, atol=1e-6)
    np.testing.assert_allclose(derivative_at_cutoff, expected_derivative, atol=1e-5)
    np.testing.assert_allclose(
        value_at_distance(1.0 - 1e-5),
        value_at_distance(1.0 + 1e-5),
        atol=3e-5,
    )


def test_periodic_group_distance_unwraps_group_before_com():
    distance = build_colvar_distance(
        "d", {"group1": "-1,2", "group2": "3"}
    )
    cell = jnp.eye(3) * 10.0
    coordinates = jnp.array(
        [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    one_member_shifted = coordinates.at[1, 0].add(10.0)

    np.testing.assert_allclose(distance(coordinates, cell), 5.0, atol=1e-6)
    np.testing.assert_allclose(
        distance(one_member_shifted, cell), distance(coordinates, cell), atol=1e-6
    )
