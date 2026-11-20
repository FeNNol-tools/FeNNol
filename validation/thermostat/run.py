"""Quick thermostat checks on independent anisotropic harmonic wells.

From the repository root::

    python validation/thermostat/run.py
    python validation/thermostat/run.py --only sampling --methods LGV

512 Ar atoms sit in non-interacting harmonic wells with a different stiffness
along each axis, so one short run gives tens of thousands of independent
samples. Local thermostats must reproduce <k_a q_a**2> = kBT on every axis and
the target kinetic temperature, also at a large timestep.

Global thermostats (Bussi, Nose-Hoover) cannot equilibrate independent
oscillators. They are checked instead on a 32-atom Lennard-Jones liquid at
fixed volume, where their energy and pressure must match Langevin's (section
``lj_nvt``).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
MODULES = HERE / "modules"
WORK = HERE / "work"
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from validation import common  # noqa: E402
from validation.common import (  # noqa: E402
    block_standard_error, effective_samples, fresh_directory, seed_for,
)
from validation import lj  # noqa: E402

common.use_modules(MODULES)
KB_KCAL_PER_MOL_K = 0.00198720425864083


def sites(cfg):
    n = cfg["per_side"]
    index = np.arange(n**3)
    return cfg["spacing"] * np.stack([index % n, (index // n) % n, index // (n * n)], axis=-1)


def build_model(cfg, path):
    import jax
    from fennol import FENNIX

    centers = sites(cfg)
    natoms = len(centers)
    model = FENNIX(
        cutoff=1.0,
        energy_unit="kcal/mol",
        modules={"harmonic": {
            "module_name": "VALIDATION_HARMONIC_LATTICE",
            "stiffness": [float(k) for k in cfg["stiffness"]],
            "spacing": cfg["spacing"],
            "per_side": cfg["per_side"],
        }},
        energy_terms=["harmonic_energy"],
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.full(natoms, 18, dtype=np.int32),
            "coordinates": centers.astype(np.float32),
            "natoms": np.array([natoms], dtype=np.int32),
            "batch_index": np.zeros(natoms, dtype=np.int32),
        },
    )
    model.save(path)
    loaded = FENNIX.load(path)
    displacement = np.random.default_rng(0).normal(scale=0.3, size=centers.shape)
    energy, forces, _ = loaded.energy_and_forces(
        species=np.full(natoms, 18, dtype=np.int32),
        coordinates=(centers + displacement).astype(np.float32), unit="kcal/mol",
    )
    k = np.asarray(cfg["stiffness"])
    error = max(abs(float(np.sum(energy)) - 0.5 * np.sum(k * displacement**2)),
                float(np.max(np.abs(np.asarray(forces) + k * displacement))))
    if error > 1e-3:
        raise RuntimeError(f"harmonic model check failed (max error {error:g})")


def md_config(cfg, name, model, start, steps, seed, thermostat, dt_fs):
    every = cfg["output_interval_steps"]
    return {
        "model_file": str(model),
        "system_name": str(name),
        "temperature": cfg["temperature_kelvin"],
        "energy_unit": "kcal/mol",
        "random_seed": int(seed),
        "nsteps": int(steps),
        "dt[fs]": dt_fs,
        "tdump[fs]": dt_fs * every,
        "nprint": every,
        "nsummary": 50 * every,
        "device": "cpu",
        "double_precision": False,
        "thermostat": thermostat,
        "gamma[THz]": cfg["friction_thz"],
        "traj_format": "xyz",
        "check_temperature": False,
        "xyz_input": {"file": str(start), "indexed": False, "has_comment_line": True},
    }


def write_start(cfg, path):
    centers = sites(cfg)
    lines = [str(len(centers)), "harmonic lattice"] + [f"Ar {x:.6f} {y:.6f} {z:.6f}" for x, y, z in centers]
    path.write_text("\n".join(lines) + "\n")


def read_positions(path, natoms):
    lines = Path(path).read_text().splitlines()
    frames = len(lines) // (natoms + 2)
    rows = [lines[f * (natoms + 2) + 2 + i].split()[1:4] for f in range(frames) for i in range(natoms)]
    return np.asarray(rows, dtype=float).reshape(frames, natoms, 3)


def read_log_column(log, column, burn_in_steps):
    """Values of one printed column of the fennol_md log, after burn-in."""
    values = []
    header = None
    for line in Path(log).read_text().splitlines():
        if line.startswith("#") and column in line.split():
            header = line[1:].split()
        elif header and line.strip() and not line.startswith("#"):
            fields = line.split()
            if int(fields[0]) > burn_in_steps:
                values.append(float(fields[header.index(column)]))
    return np.asarray(values)


def sampling_case(cfg, model, thermostat, spec, directory):
    dt_fs = spec.get("dt_fs", cfg["dt_fs"])
    scale = cfg["dt_fs"] / dt_fs
    burn_in = int(cfg["burn_in_steps"] * scale)
    steps = burn_in + int(cfg["production_steps"] * scale)
    write_start(cfg, directory / "start.xyz")
    config = md_config(cfg, directory / "md", model, directory / "start.xyz", steps,
                       seed_for(f"sampling/{spec['label']}"), thermostat, dt_fs)
    config.update({"nprint": int(cfg["output_interval_steps"] * scale),
                   "tdump[fs]": dt_fs * int(cfg["output_interval_steps"] * scale)})
    config["nsummary"] = 50 * config["nprint"]
    common.run_cli(config, directory, MODULES)
    centers = sites(cfg)
    positions = read_positions(directory / "md.traj.xyz", len(centers))
    positions = positions[int(np.ceil(burn_in / config["nprint"])):]
    kbt = KB_KCAL_PER_MOL_K * cfg["temperature_kelvin"]
    k = np.asarray(cfg["stiffness"])
    # Per-atom time averages are independent samples of <k q**2>/kBT.
    per_atom = (k * (positions - centers) ** 2 / kbt).mean(axis=0)
    ratios = per_atom.mean(axis=0)
    standard_errors = per_atom.std(axis=0, ddof=1) / np.sqrt(len(centers))
    temperature = read_log_column(directory / "md.log", "Temp[K]", burn_in)
    temperature_error = temperature.mean() / cfg["temperature_kelvin"] - 1
    tolerance = cfg["tolerance"]
    passed = np.all(np.abs(ratios - 1) <= tolerance) and abs(temperature_error) <= tolerance
    result = {f"q2_ratio_{axis}": float(r) for axis, r in zip("xyz", ratios)}
    result.update({
        "max_standard_error": float(standard_errors.max()),
        "temperature_relative_error": float(temperature_error),
        "effective_samples_per_atom": effective_samples(positions[:, 0, 0]),
        "status": "pass" if passed else "fail",
    })
    return result


def lj_nvt_case(cfg, model, thermostat, directory):
    """NVT Lennard-Jones liquid: mean energy, pressure and kinetic temperature."""
    spec = cfg["lj_nvt"]
    positions, cell = lj.fcc(spec["natoms"], spec["density_reduced"])
    sigma = spec["sigma_angstrom"]
    lj.write_xyz(directory / "start.xyz", positions * sigma)
    burn_in, every = spec["burn_in_steps"], spec["output_interval_steps"]
    config = lj.md_config(spec, spec, directory / "nvt", model, directory / "start.xyz", cell * sigma,
                          burn_in + spec["production_steps"], seed_for(f"lj_nvt/{thermostat}"),
                          thermostat=thermostat)
    common.run_cli(config, directory, MODULES)
    blocks = lj.read_summaries(directory / "md.log", burn_in, 50 * every)
    _, energies = lj.read_frames(directory / "nvt", burn_in, every)
    energy = energies / spec["epsilon_kcal_per_mol"]
    pressure = blocks["Pressure"]
    temperature_error = blocks["Temper"].mean() / spec["temperature_kelvin"] - 1
    # Independent of the code's own degree-of-freedom count: Langevin also
    # thermalizes the centre of mass, global thermostats conserve its zero
    # momentum, leaving 3N-3 kinetic degrees of freedom.
    natoms = spec["natoms"]
    ndof = 3 * natoms - (0 if thermostat in ("LGV", "FFLGV") else 3)
    kinetic = read_log_column(directory / "md.log", "Ekin", burn_in)  # per atom
    expected = 0.5 * ndof / natoms * KB_KCAL_PER_MOL_K * spec["temperature_kelvin"]
    kinetic_error = kinetic.mean() / expected - 1
    kinetic_se = block_standard_error(kinetic) / expected
    passed = (abs(temperature_error) <= spec["temperature_tolerance"]
              and abs(kinetic_error) <= max(spec["kinetic_tolerance"], 3 * kinetic_se))
    return {
        "energy_mean_epsilon": float(energy.mean()),
        "energy_se_epsilon": block_standard_error(energy),
        "pressure_mean_atm": float(pressure.mean()),
        "pressure_se_atm": float(np.std(pressure, ddof=1) / np.sqrt(len(pressure))),
        "temperature_relative_error": float(temperature_error),
        "kinetic_relative_error": float(kinetic_error),
        "kinetic_relative_se": kinetic_se,
        "status": "pass" if passed else "fail",
    }


def lj_nvt_comparisons(cfg, results):
    """Global thermostats must sample the same NVT ensemble as Langevin."""
    reference = results.get("lj_nvt/lgv", {})
    checks = []
    if "energy_mean_epsilon" not in reference:
        return checks
    for thermostat in cfg["lj_nvt"]["thermostats"]:
        other = results.get(f"lj_nvt/{thermostat.lower()}", {})
        if thermostat == "LGV" or "energy_mean_epsilon" not in other:
            continue

        def compare(other=other):
            z_energy = (other["energy_mean_epsilon"] - reference["energy_mean_epsilon"]) / np.hypot(
                other["energy_se_epsilon"], reference["energy_se_epsilon"])
            z_pressure = (other["pressure_mean_atm"] - reference["pressure_mean_atm"]) / np.hypot(
                other["pressure_se_atm"], reference["pressure_se_atm"])
            limit = cfg["lj_nvt"]["maximum_z_score"]
            passed = abs(z_energy) <= limit and abs(z_pressure) <= limit
            return {"energy_z_score": float(z_energy), "pressure_z_score": float(z_pressure),
                    "status": "pass" if passed else "fail"}
        checks.append((f"lj_nvt/{thermostat.lower()}_vs_lgv", compare))
    return checks


SECTIONS = ("sampling", "lj_nvt", "restart")


def plan_jobs(cfg, sections, labels):
    model = WORK / "models" / "harmonic.fnx"
    build_model(cfg, model)
    jobs = []
    if "sampling" in sections:
        for label, spec in cfg["sampling"].items():
            if labels and label not in labels:
                continue
            spec = {**spec, "label": label}
            jobs.append((f"sampling/{label}", lambda spec=spec: sampling_case(
                cfg, model, spec["thermostat"], spec, fresh_directory(WORK / "sampling" / spec["label"]))))
    if "lj_nvt" in sections:
        lj_model = WORK / "models" / "lj.fnx"
        lj.build_model(cfg["lj_nvt"], cfg["lj_nvt"]["natoms"], lj_model)
        for thermostat in cfg["lj_nvt"]["thermostats"]:
            if labels and thermostat not in labels:
                continue
            jobs.append((f"lj_nvt/{thermostat.lower()}", lambda t=thermostat: lj_nvt_case(
                cfg, lj_model, t, fresh_directory(WORK / "lj_nvt" / t.lower()))))
    if "restart" in sections:
        for thermostat in cfg["restart"]["thermostats"]:
            if labels and thermostat not in labels:
                continue

            def job(thermostat=thermostat):
                directory = fresh_directory(WORK / "restart" / thermostat.lower())
                write_start(cfg, directory / "start.xyz")

                def configure(run_dir, steps):
                    config = md_config(cfg, run_dir / "md", model, directory / "start.xyz", steps,
                                       seed_for(f"restart/{thermostat}"), thermostat, cfg["dt_fs"])
                    return config, run_dir / "md"
                return common.run_restart(directory, configure, cfg["restart"]["steps"], MODULES)
            jobs.append((f"restart/{thermostat.lower()}", job))
    return jobs


def main():
    parser = common.argument_parser(__doc__.splitlines()[0], SECTIONS)
    args = parser.parse_args()
    common.SEED = args.seed
    cfg = yaml.safe_load((args.config or HERE / "config.yaml").read_text())
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(cfg, args.only, args.methods)
    sys.exit(common.run_suite(jobs, args.jobs, WORK / "report.json",
                              lambda results: lj_nvt_comparisons(cfg, results)))


if __name__ == "__main__":
    main()
