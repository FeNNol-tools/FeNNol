"""Independent checks for the force-shifted LJ explicit-image model."""

from collections import OrderedDict
from pathlib import Path

import jax
import numpy as np
import pytest


def _direct_lattice_sum(coordinates, cell, epsilon, sigma, cutoff):
    """Reference total energy from an independent ordered lattice sum."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    repeats = int(np.ceil(cutoff / np.min(np.linalg.norm(cell, axis=1))))
    energy = 0.0

    def lj(radius):
        ratio6 = (sigma / radius) ** 6
        return 4.0 * epsilon * (ratio6**2 - ratio6)

    ratio6c = (sigma / cutoff) ** 6
    at_cutoff = 4.0 * epsilon * (ratio6c**2 - ratio6c)
    derivative_cutoff = 24.0 * epsilon / cutoff * (ratio6c - 2.0 * ratio6c**2)

    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            for a in range(-repeats, repeats + 1):
                for b in range(-repeats, repeats + 1):
                    for c in range(-repeats, repeats + 1):
                        shift = np.array([a, b, c])
                        if i == j and not np.any(shift):
                            continue
                        vector = coordinates[j] - coordinates[i] + shift @ cell
                        radius = np.linalg.norm(vector)
                        if 1e-12 < radius < cutoff:
                            energy += lj(radius) - at_cutoff - (radius - cutoff) * derivative_cutoff
    return 0.5 * energy


def _ev_to_ha(value):
    from fennol.utils.atomic_units import au

    return np.asarray(value) * au.get_multiplier("eV")


def _make_model(
    coordinates, cell, *, cutoff=3.0, epsilon=0.4, sigma=1.0, skin=None, energy_unit="eV"
):
    from fennol.models.fennix import FENNIX

    coordinates = np.asarray(coordinates, dtype=np.float32)
    cell = np.asarray(cell, dtype=np.float32)
    example = {
        "species": np.ones((len(coordinates),), dtype=np.int32),
        "coordinates": coordinates,
        "natoms": np.array([len(coordinates)], dtype=np.int32),
        "batch_index": np.zeros((len(coordinates),), dtype=np.int32),
        "cells": cell[None, ...],
        "reciprocal_cells": np.linalg.inv(cell)[None, ...],
    }
    preprocessing = OrderedDict()
    if skin is not None:
        preprocessing["graph"] = {"nblist_skin": skin}
    return FENNIX(
        cutoff=cutoff,
        modules=OrderedDict(
            [(
                "lj",
                {
                    "module_name": "LENNARD_JONES_FS",
                    "epsilon": epsilon,
                    "sigma": sigma,
                    "cutoff": cutoff,
                },
            )]
        ),
        preprocessing=preprocessing,
        energy_terms="lj",
        rng_key=jax.random.PRNGKey(0),
        example_data=example,
        energy_unit=energy_unit,
    )


def _input(coordinates, cell):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    cell = np.asarray(cell, dtype=np.float32)
    return {
        "species": np.ones((len(coordinates),), dtype=np.int32),
        "coordinates": coordinates,
        "natoms": np.array([len(coordinates)], dtype=np.int32),
        "batch_index": np.zeros((len(coordinates),), dtype=np.int32),
        "cells": cell[None, ...],
        "reciprocal_cells": np.linalg.inv(cell)[None, ...],
    }


@pytest.mark.parametrize(
    "cell,coordinates",
    [
        (np.diag([2.2, 2.4, 2.6]), [[0.2, 0.3, 0.4], [1.1, 1.0, 1.2]]),
        (np.array([[2.5, 0.0, 0.0], [0.4, 2.6, 0.0], [0.2, 0.3, 2.7]]), [[0.2, 0.3, 0.4], [1.1, 1.0, 1.2]]),
    ],
)
def test_explicit_image_lj_energy_forces_and_virial_match_direct_lattice_sum(cell, coordinates):
    model = _make_model(coordinates, cell)
    inputs = _input(coordinates, cell)
    energy, forces, virial, outputs = model.energy_and_forces_and_virial(**inputs)

    ref = _direct_lattice_sum(coordinates, cell, 0.4, 1.0, 3.0)
    np.testing.assert_allclose(energy[0], _ev_to_ha(ref), rtol=2e-5, atol=2e-5)
    assert outputs["graph"]["edge_src"].shape[0] > 2 * len(coordinates)
    graph = outputs["graph"]
    valid_edges = np.asarray(graph["edge_src"]) < len(coordinates)
    sources = np.asarray(graph["edge_src"])[valid_edges]
    destinations = np.asarray(graph["edge_dst"])[valid_edges]
    shifts = np.asarray(graph["pbc_shifts"])[valid_edges]
    directed_edges = {
        (int(src), int(dst), tuple(int(v) for v in shift))
        for src, dst, shift in zip(sources, destinations, shifts)
    }
    assert all(
        (dst, src, tuple(-v for v in shift)) in directed_edges
        for src, dst, shift in directed_edges
    )
    assert len({(src, dst) for src, dst, _ in directed_edges}) < len(directed_edges)

    step = 1e-4
    from fennol.utils.atomic_units import au

    au_factor = au.get_multiplier("eV")
    numerical_forces = np.zeros_like(coordinates, dtype=float)
    for atom in range(len(coordinates)):
        for axis in range(3):
            plus = np.array(coordinates, dtype=float)
            minus = np.array(coordinates, dtype=float)
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical_forces[atom, axis] = -(
                _direct_lattice_sum(plus, cell, 0.4, 1.0, 3.0)
                - _direct_lattice_sum(minus, cell, 0.4, 1.0, 3.0)
            ) / (2 * step) * au_factor
    np.testing.assert_allclose(forces, numerical_forces, rtol=2e-3, atol=2e-3)

    numerical_virial = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            strain = np.eye(3)
            strain[i, j] += step
            plus = _direct_lattice_sum(np.asarray(coordinates) @ strain, cell @ strain, 0.4, 1.0, 3.0)
            strain[i, j] -= 2 * step
            minus = _direct_lattice_sum(np.asarray(coordinates) @ strain, cell @ strain, 0.4, 1.0, 3.0)
            numerical_virial[i, j] = (plus - minus) / (2 * step) * au_factor
    np.testing.assert_allclose(virial[0], numerical_virial, rtol=4e-3, atol=4e-3)


def test_single_atom_self_images_have_energy_and_cell_virial_but_no_force():
    cell = np.diag([2.0, 2.2, 2.4])
    coordinates = np.array([[0.3, 0.4, 0.5]])
    model = _make_model(coordinates, cell)
    energy, forces, virial, outputs = model.energy_and_forces_and_virial(
        **_input(coordinates, cell)
    )
    expected = _direct_lattice_sum(coordinates, cell, 0.4, 1.0, 3.0)
    np.testing.assert_allclose(energy[0], _ev_to_ha(expected), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(forces, 0.0, atol=1e-6)
    assert np.linalg.norm(np.asarray(virial)) > 1e-4
    graph = outputs["graph"]
    valid = np.asarray(graph["edge_src"]) < 1
    shifts = np.asarray(graph["pbc_shifts"])[valid]
    assert np.any(np.any(shifts != 0, axis=1))
    assert not np.any(np.all(shifts == 0, axis=1))


def test_force_shift_has_zero_energy_and_slope_at_three_sigma_cutoff():
    from fennol.models.physics.lennard_jones import force_shifted_lj

    cutoff = 3.0
    energy, derivative = force_shifted_lj(np.asarray(cutoff), 0.4, 1.0, cutoff)
    np.testing.assert_allclose(energy, 0.0, atol=1e-12)
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)


def test_nist_force_shifted_three_sigma_pair_energy_and_virial():
    fixture = Path(__file__).parent / "fixtures/lennard_jones/nist_lj_sample_4.txt"
    with fixture.open() as source:
        cell_lengths = np.fromstring(source.readline(), sep=" ")
        natoms = int(source.readline())
    reference = np.loadtxt(fixture, skiprows=2)
    assert reference.shape == (natoms, 4)
    cell = np.diag(cell_lengths)
    coordinates = reference[:, 1:4].astype(np.float32)
    model = _make_model(
        coordinates,
        cell,
        cutoff=3.0,
        epsilon=1.0,
        sigma=1.0,
        energy_unit="Ha",
    )
    energy, _, virial, _ = model.energy_and_forces_and_virial(
        **_input(coordinates, cell)
    )

    # NIST reports four significant figures for configuration 4.
    np.testing.assert_allclose(energy[0], -15.001, atol=5e-3, rtol=0.0)
    np.testing.assert_allclose(-np.trace(virial[0]), -43.096, atol=5e-3, rtol=0.0)


def test_skin_neighbor_list_remains_correct_when_an_atom_wraps_across_boundary():
    import jax
    import jax.numpy as jnp
    from fennol.models.preprocessing import GraphGenerator, GraphProcessor

    cell = np.eye(3, dtype=np.float32) * 8.0
    initial_coordinates = np.array([[7.8, 0.2, 0.2], [0.5, 0.2, 0.2]], dtype=np.float32)
    updated_coordinates = np.array([[0.1, 0.2, 0.2], [0.5, 0.2, 0.2]], dtype=np.float32)
    initial = _input(initial_coordinates, cell)
    updated = _input(updated_coordinates, cell)
    generator = GraphGenerator(cutoff=1.5)
    state = generator.init().copy({"nblist_skin": 0.6})
    state, skin_graph = generator(state, initial)
    processor = GraphProcessor(cutoff=1.5)

    skin_inputs = {
        **jax.tree_util.tree_map(jnp.asarray, skin_graph),
        **jax.tree_util.tree_map(jnp.asarray, updated),
    }
    updated_graph = processor.apply({}, generator.update_skin(skin_inputs))["graph"]
    valid = np.asarray(updated_graph["edge_mask"])
    radii = np.asarray(updated_graph["distances"])[valid]
    actual = 0.5 * sum(
        force_shifted_lj_numpy(radius, 0.4, 1.0, 1.5) for radius in radii
    )
    expected = _direct_lattice_sum(updated_coordinates, cell, 0.4, 1.0, 1.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_skin_update_after_wrapping_and_jitted_rebuild_keeps_all_pairs():
    """Dynamics order: skin update, wrap into the box, full rebuild, skin update."""
    import jax
    import jax.numpy as jnp
    from fennol.models.preprocessing import GraphGenerator, GraphProcessor

    cell = np.eye(3, dtype=np.float32) * 8.0
    unwrapped = np.array([[8.1, 0.2, 0.2], [0.5, 0.2, 0.2]], dtype=np.float32)
    wrapped = np.array([[0.1, 0.2, 0.2], [0.5, 0.2, 0.2]], dtype=np.float32)
    generator = GraphGenerator(cutoff=1.5)
    state = generator.init().copy({"nblist_skin": 0.6})
    state, graph = generator(state, _input(unwrapped, cell))

    def as_jax(tree):
        return jax.tree_util.tree_map(jnp.asarray, tree)

    inputs = generator.update_skin({**as_jax(graph), **as_jax(_input(unwrapped, cell))})
    # The rebuild must reset the reference coordinates of the crossing tracker.
    inputs = generator.process(state, {**inputs, **as_jax(_input(wrapped, cell))})
    np.testing.assert_allclose(inputs["graph"]["coordinates_skin"], wrapped)
    inputs = generator.update_skin(inputs)
    updated_graph = GraphProcessor(cutoff=1.5).apply({}, inputs)["graph"]
    radii = np.asarray(updated_graph["distances"])[np.asarray(updated_graph["edge_mask"])]
    actual = 0.5 * sum(force_shifted_lj_numpy(radius, 0.4, 1.0, 1.5) for radius in radii)
    expected = _direct_lattice_sum(wrapped, cell, 0.4, 1.0, 1.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_rebuilt_neighbor_list_expands_image_range_after_cell_compression():
    import jax
    import jax.numpy as jnp
    from fennol.models.preprocessing import GraphGenerator, GraphProcessor

    coordinates = np.array([[0.3, 0.4, 0.5]], dtype=np.float32)
    initial_cell = np.eye(3, dtype=np.float32) * 8.0
    compressed_cell = np.eye(3, dtype=np.float32) * 2.8
    generator = GraphGenerator(cutoff=3.0)
    state = generator.init().copy({"nblist_skin": 0.5})
    state, _ = generator(state, _input(coordinates, initial_cell))
    state, rebuilt = generator(state, _input(coordinates, compressed_cell))
    graph = GraphProcessor(cutoff=3.0).apply(
        {}, jax.tree_util.tree_map(jnp.asarray, rebuilt)
    )["graph"]

    assert max(state["num_repeats_pbc"]) > 1
    valid = np.asarray(graph["edge_mask"])
    radii = np.asarray(graph["distances"])[valid]
    actual = 0.5 * sum(
        force_shifted_lj_numpy(radius, 0.4, 1.0, 3.0) for radius in radii
    )
    expected = _direct_lattice_sum(coordinates, compressed_cell, 0.4, 1.0, 3.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_explicit_image_results_match_replicated_supercell():
    base_cell = np.diag([3.6, 4.0, 4.2])
    base_coordinates = np.array([[0.4, 0.7, 0.9], [1.8, 2.2, 2.5]])
    base = _make_model(base_coordinates, base_cell)
    base_energy, base_forces, base_virial, _ = base.energy_and_forces_and_virial(
        **_input(base_coordinates, base_cell)
    )

    repetitions = (2, 2, 2)
    supercell = np.diag(np.asarray(repetitions) * np.diag(base_cell))
    super_coordinates = np.array(
        [
            coordinate + np.array([i, j, k]) * np.diag(base_cell)
            for i in range(repetitions[0])
            for j in range(repetitions[1])
            for k in range(repetitions[2])
            for coordinate in base_coordinates
        ]
    )
    repeated = _make_model(super_coordinates, supercell)
    repeated_energy, repeated_forces, repeated_virial, _ = (
        repeated.energy_and_forces_and_virial(**_input(super_coordinates, supercell))
    )

    np.testing.assert_allclose(
        repeated_energy[0] / len(super_coordinates),
        base_energy[0] / len(base_coordinates),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        repeated_forces.reshape((-1, len(base_coordinates), 3)),
        np.broadcast_to(base_forces, (np.prod(repetitions), *base_forces.shape)),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        repeated_virial[0] / np.linalg.det(supercell),
        base_virial[0] / np.linalg.det(base_cell),
        rtol=2e-5,
        atol=2e-5,
    )


def force_shifted_lj_numpy(distance, epsilon, sigma, cutoff):
    ratio6 = (sigma / distance) ** 6
    ratio6_cutoff = (sigma / cutoff) ** 6
    potential = 4 * epsilon * (ratio6**2 - ratio6)
    at_cutoff = 4 * epsilon * (ratio6_cutoff**2 - ratio6_cutoff)
    derivative_cutoff = 24 * epsilon / cutoff * (
        ratio6_cutoff - 2 * ratio6_cutoff**2
    )
    return potential - at_cutoff - (distance - cutoff) * derivative_cutoff
