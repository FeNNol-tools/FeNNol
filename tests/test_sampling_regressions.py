"""Regressions for enhanced-sampling outputs checked by validation/sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fennol.md.colvars import setup_colvars

from fennol.md.opes import initialize_opes


def test_metadynamics_kernels_export_minus_free_energy(tmp_path):
    gamma, height = 4.0, 0.3
    parameters = {
        "opes": {
            "colvars": ["cv"],
            "pace": 1,
            "sigma": 0.5,
            "gamma": gamma,
            "epsilon": 0.01,
            "metadyn": True,
            "hill_height": height,
            "tseg": 1.0,
            "kernel_chunk": 11,
        }
    }
    step, _, postprocess, state = initialize_opes(
        parameters,
        0.1,
        {"kT": 1.0, "name": str(tmp_path / "system")},
        np.float32,
        jax.random.PRNGKey(0),
        {"cv": lambda coordinates, cell: coordinates[0]},
    )
    dynamics_state = {"opes": state, "colvars": {"cv": jnp.asarray(0.0, dtype=jnp.float32)}}
    _, _, updated = step(jnp.zeros((1, 3)), jnp.zeros((1, 3)), dynamics_state)
    state = updated["opes"]
    for _ in range(10):  # one segment of tseg/dt steps triggers the export
        state = postprocess(state)

    kernels = np.loadtxt(tmp_path / "system.opes.kernels", ndmin=2)
    # First hill sees no bias; well-tempered F = -gamma/(gamma-1) V.
    assert kernels[0, 0] == pytest.approx(height * gamma / (gamma - 1.0), rel=1e-5)




def test_analyze_opes_reads_metadynamics_heights_in_kcal_per_mol(tmp_path, monkeypatch):
    from scripts import analyze_opes

    kernels = tmp_path / "kernels.dat"
    np.savetxt(kernels, [[0.5, 0.0, 0.2]])  # height, center, sigma
    output = tmp_path / "fes.dat"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_opes", str(kernels), "--metadyn",
         "--bounds", "-2", "2", "--dx", "0.5", "-o", str(output)],
    )
    analyze_opes.main()
    grid, free_energy = np.loadtxt(output, unpack=True)
    expected = -0.5 * np.exp(-0.5 * (grid / 0.2) ** 2)
    np.testing.assert_allclose(free_energy, expected - expected.min(), atol=1e-5)


def test_coordinate_colvar_selects_one_based_atom_and_cartesian_axis():
    calculators, names = setup_colvars(
        {"reaction_coordinate": {"type": "coordinate", "atom": 1, "axis": "x"}}
    )

    coordinates = jnp.array([[0.75, -0.5, 0.25], [2.0, 3.0, 4.0]])
    value = calculators["reaction_coordinate"](coordinates)
    gradient = jax.grad(calculators["reaction_coordinate"])(coordinates)

    assert names == ["reaction_coordinate"]
    np.testing.assert_allclose(value, 0.75)
    np.testing.assert_allclose(
        gradient,
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        atol=1e-7,
    )


def test_analyze_opes_reconstructs_two_collective_variables(tmp_path, monkeypatch):
    from scripts import analyze_opes

    kernels = tmp_path / "kernels.dat"
    np.savetxt(kernels, [[0.5, -0.5, -0.5, 0.4, 0.3], [0.5, 0.5, 0.5, 0.4, 0.3]])
    output = tmp_path / "surface.dat"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_opes", str(kernels), "--cvs", "0", "1", "--bounds", "-1", "1", "-1", "1",
         "--dx", "0.2", "0.25", "-o", str(output)],
    )
    analyze_opes.main()
    table = np.loadtxt(output)
    x, y = np.unique(table[:, 0]), np.unique(table[:, 1])
    assert (len(x), len(y)) == (10, 8)
    surface = table[:, 2].reshape(len(x), len(y))
    # Symmetric kernels give a surface symmetric under (x, y) -> (-x, -y).
    np.testing.assert_allclose(surface, surface[::-1, ::-1], atol=1e-5)
