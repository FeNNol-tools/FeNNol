"""Force-shifted Lennard-Jones argon systems shared by the thermostat and barostat suites."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


# ------------------------------------------------------------------ systems


def fcc(natoms, density_reduced):
    """FCC positions and cubic cell in reduced (sigma) units."""
    repetitions = round((natoms / 4) ** (1 / 3))
    if 4 * repetitions**3 != natoms:
        raise ValueError("FCC requires natoms = 4 * n**3")
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    cells = np.array([[i, j, k] for i in range(repetitions)
                      for j in range(repetitions) for k in range(repetitions)])
    fractions = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / repetitions
    side = (natoms / density_reduced) ** (1 / 3)
    return fractions * side, np.eye(3) * side


def build_model(cfg, natoms, path):
    import jax
    from fennol import FENNIX
    from fennol.utils.atomic_units import au

    sigma = cfg["sigma_angstrom"]
    cutoff = cfg["cutoff_sigma"] * sigma
    positions, cell = fcc(natoms, 1.0)
    lattice = cell * sigma
    model = FENNIX(
        cutoff=cutoff,
        energy_unit="Ha",
        modules={"lj": {
            "module_name": "LENNARD_JONES_FS",
            "epsilon": cfg["epsilon_kcal_per_mol"] / au.get_multiplier("kcal/mol"),
            "sigma": sigma,
            "cutoff": cutoff,
        }},
        energy_terms="lj",
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.full(natoms, 18, dtype=np.int32),
            "coordinates": (positions * sigma).astype(np.float32),
            "natoms": np.array([natoms], dtype=np.int32),
            "batch_index": np.zeros(natoms, dtype=np.int32),
            "cells": lattice[None].astype(np.float32),
            "reciprocal_cells": np.linalg.inv(lattice)[None].astype(np.float32),
        },
    )
    model.save(path)


def pressure_atm(cfg, reduced):
    from fennol.md.utils import us

    return reduced * cfg["epsilon_kcal_per_mol"] / cfg["sigma_angstrom"] ** 3 * us.get_multiplier("atm")


def write_xyz(path, positions_angstrom):
    lines = [str(len(positions_angstrom)), "LJ argon"]
    lines += [f"Ar {x:.8f} {y:.8f} {z:.8f}" for x, y, z in positions_angstrom]
    path.write_text("\n".join(lines) + "\n")


def md_config(cfg, system, name, model, xyz, cell, steps, seed, *, barostat=None,
              thermostat="LGV", aniso=False, mask=None):
    """fennol_md input; ``cell`` is the 3x3 lattice in Angstrom (rows are vectors)."""
    dt_fs = system["dt_fs"]
    every = system["output_interval_steps"]
    config = {
        "model_file": str(model),
        "system_name": str(name),
        "temperature": system["temperature_kelvin"],
        "energy_unit": "kcal/mol",
        "random_seed": int(seed),
        "nsteps": int(steps),
        "dt[fs]": dt_fs,
        "tdump[fs]": dt_fs * every,
        "nprint": every,
        "nsummary": 50 * every,
        "device": "cpu",
        "double_precision": bool(system.get("double_precision", False)),
        "thermostat": thermostat,
        "gamma[THz]": system["thermostat_friction_thz"],
        "minimum_image": False,
        "estimate_pressure": True,
        "print_aniso_pressure": True,
        "pressure_unit": cfg["pressure_unit"],
        "traj_format": "extxyz",
        "wrap_box": True,
        "check_temperature": False,
        "nblist_skin": 1.0,
        "cell": np.asarray(cell, dtype=float).ravel().tolist(),
        "xyz_input": {"file": str(xyz), "indexed": False, "has_comment_line": True},
    }
    if barostat is not None:
        config.update({
            "barostat": barostat["barostat"],
            "target_pressure[atm]": pressure_atm(cfg, system["pressure_reduced"]),
            "tau_piston[ps]": barostat["tau_piston_ps"],
        })
        if "gamma_piston_thz" in barostat:
            config["gamma_piston[THz]"] = barostat["gamma_piston_thz"]
        if "compressibility_atm" in barostat:
            config["compressibility[atm^-1]"] = barostat["compressibility_atm"]
        if aniso:
            config["aniso_barostat"] = True
            if mask is not None:
                config["aniso_mask"] = list(mask)
    return config


# ------------------------------------------------------------------ outputs


def read_frames(name, burn_in_steps, every):
    """Lattices (rows are vectors, Angstrom) and energies saved after burn-in."""
    lattices, energies = [], []
    text = Path(f"{name}.traj.extxyz").read_text().splitlines()
    cursor = 0
    while cursor < len(text):
        natoms = int(text[cursor])
        comment = text[cursor + 1]
        lattice = re.search(r'Lattice="([^"]+)"', comment).group(1)
        lattices.append(np.array(lattice.split(), dtype=float).reshape(3, 3))
        energies.append(float(re.search(r"energy=(\S+)", comment).group(1)))
        cursor += natoms + 2
    skip = int(np.ceil(burn_in_steps / every))
    return np.array(lattices[skip:]), np.array(energies[skip:])


def read_summaries(log, burn_in_steps, block_steps):
    """Per-block means of every printed property, after burn-in."""
    blocks, current = [], None
    for line in Path(log).read_text().splitlines():
        if line.startswith("# Averages over last"):
            current = {}
            blocks.append(current)
        elif current is not None:
            match = re.match(r"#\s+(\S+)\s+:\s+(\S+)\s+\+/-", line)
            if match:
                current[match.group(1)] = float(match.group(2))
            elif line.startswith("# Perf"):
                current = None
    blocks = blocks[int(np.ceil(burn_in_steps / block_steps)):]
    return {key: np.array([b[key] for b in blocks]) for key in blocks[0]} if blocks else {}
