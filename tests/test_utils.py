"""Utilities: file readers, input parser, spherical harmonics, gradients and analysis scripts."""

import jax.numpy as jnp
import numpy as np
from fennol.utils import get_energy_gradient_function
from fennol.utils.input_parser import parse_input
from fennol.utils.io import last_xyz_frame
from scripts.compute_rdf import get_pairs


def test_last_xyz_frame_returns_final_frame(tmp_path):
    path = tmp_path / "two.xyz"
    path.write_text("1\nfirst\nH 0 0 0\n1\nlast\nH 2 0 0\n")
    symbols, xyz, comment = last_xyz_frame(path)
    assert symbols == ["H"]
    np.testing.assert_array_equal(xyz, [[2, 0, 0]])
    assert comment == "last"


def test_empty_fnl_section_is_stored_at_its_section_path(tmp_path):
    path = tmp_path / "input.fnl"
    path.write_text("outer {\n empty {}\n}\n")
    parsed = parse_input(path)
    assert parsed.get("outer/empty") == {}


def test_spherical_harmonics_lmax_zero_returns_constant_component():
    from fennol.utils.spherical_harmonics import generate_spherical_harmonics

    result = generate_spherical_harmonics(0)(np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(result, [[1.0]])


def test_same_species_pair_normalization_excludes_self_pairs():
    pairs, normalization = get_pairs(["H", "H", "H"], "H", "H")
    assert pairs.shape == (2, 3)
    assert normalization == 3


def test_energy_gradient_function_differentiates_its_input_dict():
    def energy_function(inputs):
        energy = jnp.sum(inputs["coordinates"] ** 2)
        return energy, {"total_energy": energy}

    gradient_function = get_energy_gradient_function(
        energy_function, ["coordinates"], jit=False
    )
    coordinates = jnp.array([[1.0, -2.0, 3.0]])

    gradients, output = gradient_function({"coordinates": coordinates})

    np.testing.assert_allclose(gradients["coordinates"], 2 * coordinates)
    np.testing.assert_allclose(output["dEd_coordinates"], 2 * coordinates)


def test_version_matches_installed_distribution():
    from importlib.metadata import version

    import fennol

    assert fennol.__version__ == version("FeNNol")
