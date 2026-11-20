"""Checks for the independent LJ NPT Monte Carlo reference of validation/barostat."""

from pathlib import Path

import numpy as np

from validation.lj import build_model, fcc
from validation.barostat.npt_mc import NPTMonteCarlo, force_shifted_lj_energy, total_energy


def test_independent_lattice_sum_matches_nist_force_shifted_configuration():
    fixture = Path(__file__).parent / "fixtures/lennard_jones/nist_lj_sample_4.txt"
    with fixture.open() as source:
        lengths = np.fromstring(source.readline(), sep=" ")
        natoms = int(source.readline())
    values = np.loadtxt(fixture, skiprows=2)
    coordinates = values[:, 1:4]
    energy = total_energy(
        coordinates,
        np.diag(lengths),
        epsilon=1.0,
        sigma=1.0,
        cutoff=3.0,
    )
    assert len(coordinates) == natoms
    np.testing.assert_allclose(energy, -15.001, atol=5e-3, rtol=0.0)


def test_cell_move_reenumerates_self_images_and_uses_log_volume_jacobian():
    coordinates = np.array([[0.2, 0.3, 0.4]])
    expanded = np.diag([3.2, 3.3, 3.4])
    compressed = np.diag([2.4, 2.5, 2.6])
    expanded_energy = total_energy(coordinates, expanded, 1.0, 1.0, 3.0)
    compressed_energy = total_energy(coordinates, compressed, 1.0, 1.0, 3.0)
    assert compressed_energy != expanded_energy

    sampler = NPTMonteCarlo(
        coordinates=coordinates,
        cell=expanded,
        beta=1.0,
        pressure=0.8,
        epsilon=1.0,
        sigma=1.0,
        cutoff=3.0,
        seed=17,
    )
    candidate = sampler.propose_log_volume(0.05)
    expected_jacobian = (len(coordinates) + 1) * np.log(
        candidate.volume / sampler.volume
    )
    np.testing.assert_allclose(candidate.log_volume_jacobian, expected_jacobian)
    expected_log_acceptance = -sampler.beta * (
        candidate.delta_energy + sampler.pressure * (candidate.volume - sampler.volume)
    ) + expected_jacobian
    np.testing.assert_allclose(candidate.log_acceptance, expected_log_acceptance)
    assert np.isfinite(candidate.delta_energy)


def test_force_shifted_pair_energy_is_zero_at_cutoff():
    assert force_shifted_lj_energy(3.0, 1.0, 1.0, 3.0) == 0.0


def test_mc_incremental_energy_matches_full_reference_after_translations_and_volume_moves():
    coordinates = np.array([[0.2, 0.3, 0.4], [1.1, 1.3, 1.4]])
    sampler = NPTMonteCarlo(
        coordinates=coordinates,
        cell=np.diag([2.8, 3.0, 3.2]),
        beta=1.0,
        pressure=0.6,
        epsilon=0.4,
        sigma=1.0,
        cutoff=2.5,
        seed=82,
    )
    sampler.run(
        burn_in=0,
        production=12,
        stride=1,
        max_displacement=0.03,
        max_log_volume=0.02,
    )
    exact = total_energy(
        sampler.coordinates, sampler.cell, sampler.epsilon, sampler.sigma, sampler.cutoff
    )
    np.testing.assert_allclose(sampler.energy, exact, rtol=1e-11, atol=1e-11)


def test_serialized_fennix_model_uses_same_physical_epsilon_as_mc_reference(tmp_path):
    from fennol import FENNIX
    from fennol.utils.atomic_units import au

    cfg = {"epsilon_kcal_per_mol": 0.238, "sigma_angstrom": 3.4, "cutoff_sigma": 3.0}
    reduced_coordinates, reduced_cell = fcc(32, 0.65)
    model_path = tmp_path / "lj.fnx"
    build_model(cfg, 32, model_path)
    model = FENNIX.load(model_path)
    coordinates = reduced_coordinates * cfg["sigma_angstrom"]
    cell = reduced_cell * cfg["sigma_angstrom"]
    inputs = {
        "species": np.full(32, 18, dtype=np.int32),
        "coordinates": coordinates.astype(np.float32),
        "natoms": np.asarray([32], dtype=np.int32),
        "batch_index": np.zeros(32, dtype=np.int32),
        "cells": cell[None].astype(np.float32),
        "reciprocal_cells": np.linalg.inv(cell)[None].astype(np.float32),
    }
    model_energy = model.energy_and_forces_and_virial(**inputs)[0][0]
    reference_kcal = total_energy(reduced_coordinates, reduced_cell, 1.0, 1.0, 3.0) * 0.238
    expected_ha = reference_kcal / au.get_multiplier("kcal/mol")
    np.testing.assert_allclose(model_energy, expected_ha, rtol=2e-4, atol=2e-5)
