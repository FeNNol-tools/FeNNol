"""Quick barostat and thermostat checks on force-shifted Lennard-Jones argon.

From the repository root::

    python validation/barostat/run.py                    # every check
    python validation/barostat/run.py --only liquid crystal --jobs 8
    python validation/barostat/run.py --only reference   # regenerate the MC reference

* ``liquid``: 32-atom liquid. C-rescale and Langevin volume mean and variance
  against an independent NPT Monte Carlo reference (``npt_mc.py``), Langevin
  piston temperature, and Berendsen relaxation of an expanded box.
* ``crystal``: 108-atom FCC crystal with a fully flexible cell. The mean cell
  must stay cubic; C-rescale and Langevin must agree on the mean cell metric
  and its fluctuations; the isotropic barostat on the mean volume; masked cell
  components must stay fixed; Berendsen must relax a distorted cell.
* ``ideal_gas``: non-interacting atoms, whose NPT volume distribution is an
  exact Gamma distribution with shape N_f/3 + 1 (N_f kinetic degrees of
  freedom). This isolates the degree-of-freedom terms, the V**N measure and
  the kinetic pressure estimator, also at a 4x larger timestep.
* ``cell_spring``: the same ideal gas in a cell held by a spring on the cell
  metric. The six lower-triangular cell entries must match an exact Monte
  Carlo sample of p(h) ~ V**N exp(-(PV + E(h))/kBT), uniform in the entries.
  This checks the anisotropic measure and the shear fluctuations.
* ``restart``: split and uninterrupted runs leave identical states.

Generated files go to the ignored ``work/`` directory, the summary to
``work/report.json``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from validation import common  # noqa: E402
from validation.common import (  # noqa: E402
    effective_samples, fresh_directory, seed_for,
)
from validation.lj import (  # noqa: E402
    build_model, fcc, md_config, read_frames, read_summaries, write_xyz,
)

MODULES = HERE / "modules"
common.use_modules(MODULES)
KB_KCAL_PER_MOL_K = 0.00198720425864083
COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


# ------------------------------------------------------------------ systems


def run_cli(config, directory, label="md"):
    common.run_cli(config, directory, MODULES, label)


# ------------------------------------------------------------------ outputs


def metric(lattices):
    """Rotation-invariant cell metric G = h h^T as (n, 6) [xx, yy, zz, xy, xz, yz]."""
    g = np.einsum("nij,nkj->nik", lattices, lattices)
    return np.stack([g[:, 0, 0], g[:, 1, 1], g[:, 2, 2], g[:, 0, 1], g[:, 0, 2], g[:, 1, 2]], axis=1)


def cubic_asymmetry(g):
    """Largest deviation of a mean metric from cubic, relative to its diagonal."""
    diagonal = g[:3].mean()
    return float(max(np.max(np.abs(g[:3] - diagonal)), np.max(np.abs(g[3:]))) / diagonal)


def check(value, tolerance):
    return "pass" if abs(value) <= tolerance else "fail"


def piston_temperature_error(log, burn_in_steps, every, temperature):
    """Relative error of the Langevin piston temperature, or None for other barostats."""
    blocks = read_summaries(log, burn_in_steps, 50 * every)
    if "T_piston" not in blocks:
        return None
    return float(blocks["T_piston"].mean() / temperature - 1)


# ------------------------------------------------------------------ liquid


def liquid_case(cfg, model, key, barostat, directory):
    liquid = cfg["liquid"]
    sigma = cfg["sigma_angstrom"]
    every, burn_in = liquid["output_interval_steps"], liquid["burn_in_steps"]
    steps = burn_in + liquid["production_steps"]
    positions, cell = fcc(liquid["natoms"], liquid["start_density_reduced"])
    write_xyz(directory / "start.xyz", positions * sigma)
    config = md_config(cfg, liquid, directory / key, model, directory / "start.xyz",
                       cell * sigma, steps, seed_for(f"liquid/{key}"), barostat=barostat)
    run_cli(config, directory)
    lattices, energies = read_frames(directory / key, burn_in, every)
    volume = np.abs(np.linalg.det(lattices)) / sigma**3
    reference = liquid["reference"]
    mean_error = volume.mean() / reference["volume_mean_reduced"] - 1
    variance_error = volume.var() / reference["volume_variance_reduced"] - 1
    piston = piston_temperature_error(directory / "md.log", burn_in, every, liquid["temperature_kelvin"])
    passed = (abs(mean_error) <= liquid["volume_mean_tolerance"]
              and abs(variance_error) <= liquid["volume_variance_tolerance"]
              and (piston is None or abs(piston) <= liquid["piston_temperature_tolerance"]))
    return {
        "volume_mean": float(volume.mean()),
        "volume_variance": float(volume.var()),
        "mean_relative_error": float(mean_error),
        "variance_relative_error": float(variance_error),
        "energy_mean_epsilon": float(energies.mean() / cfg["epsilon_kcal_per_mol"]),
        "effective_samples": effective_samples(volume),
        "piston_temperature_relative_error": piston,
        "status": "pass" if passed else "fail",
    }


def liquid_berendsen(cfg, model, directory):
    liquid = cfg["liquid"]
    spec = liquid["berendsen"]
    sigma = cfg["sigma_angstrom"]
    positions, cell = fcc(liquid["natoms"], spec["start_density_reduced"])
    write_xyz(directory / "start.xyz", positions * sigma)
    config = md_config(cfg, liquid, directory / "berendsen", model, directory / "start.xyz",
                       cell * sigma, spec["steps"], seed_for("liquid/berendsen"), barostat=spec)
    run_cli(config, directory)
    lattices, _ = read_frames(directory / "berendsen", spec["steps"] // 2, liquid["output_interval_steps"])
    volume = np.abs(np.linalg.det(lattices)).mean() / sigma**3
    initial = liquid["natoms"] / spec["start_density_reduced"]
    error = volume / liquid["reference"]["volume_mean_reduced"] - 1
    return {"initial_volume": initial, "final_volume": float(volume),
            "relative_error": float(error), "status": check(error, spec["volume_tolerance"])}


def mc_reference(cfg):
    """Regenerate the independent NPT Monte Carlo reference (several minutes)."""
    from validation.barostat.npt_mc import NPTMonteCarlo

    liquid = cfg["liquid"]
    mc = liquid["mc"]
    positions, cell = fcc(liquid["natoms"], liquid["start_density_reduced"])
    kbt = KB_KCAL_PER_MOL_K * liquid["temperature_kelvin"]
    chains = []
    for seed in mc["seeds"]:
        sampler = NPTMonteCarlo(positions, cell, beta=cfg["epsilon_kcal_per_mol"] / kbt,
                                pressure=liquid["pressure_reduced"], epsilon=1.0, sigma=1.0,
                                cutoff=cfg["cutoff_sigma"], seed=seed)
        chains.append(sampler.run(burn_in=mc["burn_in_sweeps"], production=mc["production_sweeps"],
                                  stride=1, max_displacement=mc["max_displacement_sigma"],
                                  max_log_volume=mc["max_log_volume_step"]))
    result = {}
    for key, values in (("volume_mean_reduced", [c["volume"].mean() for c in chains]),
                        ("volume_variance_reduced", [c["volume"].var() for c in chains]),
                        ("energy_mean_epsilon", [c["energy"].mean() for c in chains])):
        result[key] = float(np.mean(values))
        result[key.replace("_reduced", "").replace("_epsilon", "") + "_se"] = float(
            np.std(values, ddof=1) / np.sqrt(len(values)))
    result["status"] = "pass"
    return result


# ------------------------------------------------------------------ crystal


def crystal_npt(cfg, model, key, case, directory):
    crystal = cfg["crystal"]
    sigma = cfg["sigma_angstrom"]
    barostat = crystal["barostats"][case["barostat"]]
    positions, cell = fcc(crystal["natoms"], crystal["density_reduced"])
    write_xyz(directory / "start.xyz", positions * sigma)
    burn_in, every = crystal["burn_in_steps"], crystal["output_interval_steps"]
    config = md_config(cfg, crystal, directory / key, model, directory / "start.xyz", cell * sigma,
                       burn_in + crystal["production_steps"], seed_for(f"crystal/{key}"),
                       barostat=barostat, aniso=not case.get("isotropic"), mask=case.get("mask"))
    run_cli(config, directory)
    lattices, _ = read_frames(directory / key, burn_in, every)
    g = metric(lattices)
    # FCC is cubic: equal diagonal metric components and no mean shear.
    asymmetry = cubic_asymmetry(g.mean(axis=0))
    piston = piston_temperature_error(directory / "md.log", burn_in, every, crystal["temperature_kelvin"])
    passed = (asymmetry <= crystal["symmetry_tolerance"]
              and (piston is None or abs(piston) <= crystal["piston_temperature_tolerance"]))
    result = {
        "volume_mean": float(np.abs(np.linalg.det(lattices)).mean()),
        "mean_lattice": lattices.mean(axis=0).tolist(),
        "metric_mean": g.mean(axis=0).tolist(),
        "metric_covariance": np.cov(g, rowvar=False).tolist(),
        "effective_samples": min(effective_samples(g[:, i]) or 0.0 for i in range(3)),
        "cubic_asymmetry": asymmetry,
        "piston_temperature_relative_error": piston,
        "status": "pass" if passed else "fail",
    }
    mask = case.get("mask")
    if mask is not None:
        # Masked shear components start at zero and must stay exactly there.
        off = np.array([lattices[:, 1, 0], lattices[:, 2, 0], lattices[:, 2, 1]])
        drift = float(np.max(np.abs(off)))
        moving = float(np.min(np.std(lattices[:, [0, 1, 2], [0, 1, 2]], axis=0)))
        result.update({"masked_max_abs_angstrom": drift, "free_min_std_angstrom": moving})
        if drift > 1e-3 or moving <= 1e-3:
            result["status"] = "fail"
    return result


def crystal_berendsen(cfg, model, directory):
    crystal = cfg["crystal"]
    spec = crystal["berendsen"]
    sigma = cfg["sigma_angstrom"]
    positions, cell = fcc(crystal["natoms"], crystal["density_reduced"])
    # Lower-triangular strain keeps the lattice rows in FeNNol's convention.
    strain = np.asarray(spec["strain"])
    distorted = cell @ strain
    write_xyz(directory / "start.xyz", (positions @ strain) * sigma)
    config = md_config(cfg, crystal, directory / "berendsen", model, directory / "start.xyz",
                       distorted * sigma, spec["steps"], seed_for("crystal/berendsen"),
                       barostat=spec, aniso=True)
    run_cli(config, directory)
    lattices, _ = read_frames(directory / "berendsen", spec["steps"] // 2, crystal["output_interval_steps"])
    mean = metric(lattices).mean(axis=0)
    # The sheared start is far from cubic; the relaxed cell must be cubic again.
    asymmetry = cubic_asymmetry(mean)
    return {"metric_mean": mean.tolist(), "cubic_asymmetry": asymmetry,
            "initial_asymmetry": cubic_asymmetry(metric(distorted[None])[0]),
            "status": check(asymmetry, crystal["symmetry_tolerance"])}


def crystal_comparisons(cfg, results):
    """Cross-run crystal checks, evaluated once the NPT runs have finished."""
    crystal = cfg["crystal"]
    get = lambda key: results.get(f"crystal/{key}", {})  # noqa: E731
    checks = []
    one, two = get("flexible_crescale"), get("flexible_langevin")
    if "metric_mean" in one and "metric_mean" in two:
        def compare():
            m1, m2 = np.array(one["metric_mean"]), np.array(two["metric_mean"])
            scale = m1[:3].mean()
            mean_error = float(np.max(np.abs(m1 - m2)) / scale)
            c1, c2 = np.diag(one["metric_covariance"]), np.diag(two["metric_covariance"])
            covariance_error = float(np.max(np.abs(c1 / c2 - 1)))
            passed = (mean_error <= crystal["metric_mean_tolerance"]
                      and covariance_error <= crystal["metric_covariance_tolerance"])
            return {"metric_mean_error": mean_error, "metric_variance_error": covariance_error,
                    "variance_ratios": (c1 / c2).tolist(), "status": "pass" if passed else "fail"}
        checks.append(("crystal/crescale_vs_langevin", compare))
    if "volume_mean" in one and "volume_mean" in get("isotropic_crescale"):
        def volume():
            error = get("isotropic_crescale")["volume_mean"] / one["volume_mean"] - 1
            return {"relative_error": float(error), "status": check(error, crystal["volume_tolerance"])}
        checks.append(("crystal/isotropic_vs_flexible_volume", volume))
    if "metric_mean" in one and "metric_mean" in get("berendsen"):
        def berendsen():
            m1, m2 = np.array(one["metric_mean"]), np.array(get("berendsen")["metric_mean"])
            error = float(np.max(np.abs(m2 - m1)) / m1[:3].mean())
            return {"metric_error": error, "status": check(error, crystal["berendsen"]["metric_tolerance"])}
        checks.append(("crystal/berendsen_relaxation", berendsen))
    return checks


# ---------------------------------------------------------------- ideal gas


def build_ideal_gas_model(natoms, path):
    import jax
    from fennol import FENNIX

    lattice = np.eye(3) * 10.0
    FENNIX(
        cutoff=1.0,
        energy_unit="kcal/mol",
        modules={"zero": {"module_name": "VALIDATION_ZERO_ENERGY"}},
        energy_terms=["zero_energy"],
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.full(natoms, 18, dtype=np.int32),
            "coordinates": np.random.default_rng(0).uniform(0, 10, (natoms, 3)).astype(np.float32),
            "natoms": np.array([natoms], dtype=np.int32),
            "batch_index": np.zeros(natoms, dtype=np.int32),
            "cells": lattice[None].astype(np.float32),
            "reciprocal_cells": np.linalg.inv(lattice)[None].astype(np.float32),
        },
    ).save(path)


def ideal_gas_case(cfg, model, key, case, directory):
    gas = cfg["ideal_gas"]
    natoms = gas["natoms"]
    thermostat = case.get("thermostat", "LGV")
    system = {**gas, "dt_fs": case.get("dt_fs", gas["dt_fs"])}
    scale = gas["dt_fs"] / system["dt_fs"]
    every = max(1, int(gas["output_interval_steps"] * scale))
    system["output_interval_steps"] = every
    burn_in = int(gas["burn_in_steps"] * scale)
    steps = burn_in + int(gas["production_steps"] * scale)
    kbt = KB_KCAL_PER_MOL_K * gas["temperature_kelvin"]
    pressure = gas["pressure_atm"]
    from fennol.md.utils import us

    theta = kbt / (pressure / us.get_multiplier("atm"))  # kBT/P in A**3
    side = ((natoms + 1) * theta) ** (1 / 3)
    positions = np.random.default_rng(seed_for(f"ideal_gas/{key}")).uniform(0, side, (natoms, 3))
    write_xyz(directory / "start.xyz", positions)
    barostat = gas["barostats"][case["barostat"]]
    config = md_config(cfg, {**system, "pressure_reduced": 0.0}, directory / key, model,
                       directory / "start.xyz", np.eye(3) * side, steps, seed_for(f"ideal_gas/{key}"),
                       barostat=barostat, thermostat=thermostat)
    config["target_pressure[atm]"] = pressure
    run_cli(config, directory)
    lattices, _ = read_frames(directory / key, burn_in, every)
    volume = np.abs(np.linalg.det(lattices))
    # Langevin thermalizes the centre of mass (N_f = 3N); global thermostats
    # conserve its zero momentum (N_f = 3N - 3).
    ndof = 3 * natoms - (0 if thermostat in ("LGV", "FFLGV") else 3)
    shape = ndof / 3 + 1
    ess = effective_samples(volume) or 1.0
    mean_error = volume.mean() / (shape * theta) - 1
    variance_error = volume.var() / (shape * theta**2) - 1
    mean_se = 1 / np.sqrt(shape * ess)
    passed = (abs(mean_error) <= max(gas["mean_tolerance"], 3 * mean_se)
              and abs(variance_error) <= gas["variance_tolerance"])
    return {
        "expected_shape": shape,
        "mean_relative_error": float(mean_error),
        "mean_relative_se": float(mean_se),
        "variance_relative_error": float(variance_error),
        "effective_samples": ess,
        "status": "pass" if passed else "fail",
    }


def build_cell_spring_model(spring, path):
    import jax
    from fennol import FENNIX

    natoms = spring["natoms"]
    lattice = np.eye(3) * spring["length_angstrom"]
    FENNIX(
        cutoff=1.0,
        energy_unit="kcal/mol",
        modules={"spring": {
            "module_name": "VALIDATION_CELL_SPRING",
            "stiffness": spring["stiffness_kcal_per_mol_a4"],
            "length": spring["length_angstrom"],
        }},
        energy_terms=["cell_spring_energy"],
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.full(natoms, 18, dtype=np.int32),
            "coordinates": np.random.default_rng(0).uniform(0, 10, (natoms, 3)).astype(np.float32),
            "natoms": np.array([natoms], dtype=np.int32),
            "batch_index": np.zeros(natoms, dtype=np.int32),
            "cells": lattice[None].astype(np.float32),
            "reciprocal_cells": np.linalg.inv(lattice)[None].astype(np.float32),
        },
    ).save(path)


def cell_spring_case(cfg, model, key, case, directory):
    """Lower-triangular cell entries of an ideal gas in a cell spring against exact MC."""
    from fennol.md.utils import us
    from validation.barostat.npt_mc import TRIL_ENTRIES, sample_cell_spring

    spring = cfg["cell_spring"]
    natoms = spring["natoms"]
    length = spring["length_angstrom"]
    every = spring["output_interval_steps"]
    burn_in = spring["burn_in_steps"]
    mask = case.get("mask", [1] * 6)
    kbt = KB_KCAL_PER_MOL_K * spring["temperature_kelvin"]
    pressure = spring["pressure_atm"] / us.get_multiplier("atm")  # kcal/mol/A**3
    reference_mean, reference_variance = sample_cell_spring(
        natoms, 1 / kbt, pressure, spring["stiffness_kcal_per_mol_a4"], length, mask,
        seed=seed_for(f"cell_spring/{key}/reference"), **spring["reference"])

    seed = seed_for(f"cell_spring/{key}")
    positions = np.random.default_rng(seed).uniform(0, length, (natoms, 3))
    write_xyz(directory / "start.xyz", positions)
    config = md_config(cfg, {**spring, "pressure_reduced": 0.0}, directory / key, model,
                       directory / "start.xyz", np.eye(3) * length, burn_in + spring["production_steps"],
                       seed, barostat=spring["barostats"][case["barostat"]], aniso=True, mask=mask)
    config["target_pressure[atm]"] = spring["pressure_atm"]
    run_cli(config, directory)
    lattices, _ = read_frames(directory / key, burn_in, every)
    entries = np.stack([lattices[:, i, j] for i, j in TRIL_ENTRIES], axis=1)

    components = {}
    passed = True
    for column, name in enumerate(COMPONENTS):
        if not mask[column]:
            # Masked entries must stay exactly at their initial value of zero.
            fixed = bool(np.all(entries[:, column] == 0.0))
            components[name] = {"fixed": fixed}
            passed &= fixed
            continue
        values = entries[:, column]
        ess = effective_samples(values) or 1.0
        mean_error = (values.mean() - reference_mean[column]) / length
        mean_se = np.sqrt(reference_variance[column] / ess) / length
        variance_error = values.var() / reference_variance[column] - 1
        ok = (abs(mean_error) <= max(spring["mean_tolerance"], 3 * mean_se)
              and abs(variance_error) <= spring["variance_tolerance"])
        passed &= ok
        components[name] = {
            "mean": float(values.mean()),
            "reference_mean": float(reference_mean[column]),
            "mean_error_over_length": float(mean_error),
            "mean_se_over_length": float(mean_se),
            "variance_relative_error": float(variance_error),
            "effective_samples": ess,
        }
    return {"components": components, "status": "pass" if passed else "fail"}


# ------------------------------------------------------------------ restart


def restart_case(cfg, label, case, models):
    system = cfg[case["system"]]
    sigma = cfg["sigma_angstrom"]
    if case["barostat"] == "berendsen":
        barostat = system["berendsen"]
    else:
        barostat = system["barostats"][case["barostat"]]
    density = system.get("density_reduced", system.get("start_density_reduced"))
    positions, cell = fcc(system["natoms"], density)
    directory = fresh_directory(WORK / "restart" / label)
    write_xyz(directory / "start.xyz", positions * sigma)

    def configure(run_dir, steps):
        config = md_config(cfg, system, run_dir / "md", models[case["system"]], directory / "start.xyz",
                           cell * sigma, steps, seed_for(f"restart/{label}"), barostat=barostat,
                           aniso=case["system"] == "crystal")
        return config, run_dir / "md"
    return common.run_restart(directory, configure, cfg["restart"]["steps"], MODULES)


# ------------------------------------------------------------------ driver


SECTIONS = ("liquid", "crystal", "ideal_gas", "cell_spring", "restart", "reference")


def plan_jobs(cfg, sections, labels):
    models = {}
    for system in ("liquid", "crystal"):
        if system in sections or "restart" in sections:
            path = WORK / "models" / f"{system}.fnx"
            build_model(cfg, cfg[system]["natoms"], path)
            models[system] = path

    def wanted(label):
        return not labels or label in labels

    jobs = []
    if "liquid" in sections:
        for key, barostat in cfg["liquid"]["barostats"].items():
            if wanted(key):
                jobs.append((f"liquid/{key}", lambda key=key, barostat=barostat: liquid_case(
                    cfg, models["liquid"], key, barostat, fresh_directory(WORK / "liquid" / key))))
        if wanted("berendsen"):
            jobs.append(("liquid/berendsen", lambda: liquid_berendsen(
                cfg, models["liquid"], fresh_directory(WORK / "liquid" / "berendsen"))))
    if "crystal" in sections:
        for key, case in cfg["crystal"]["cases"].items():
            if wanted(key):
                jobs.append((f"crystal/{key}", lambda key=key, case=case: crystal_npt(
                    cfg, models["crystal"], key, case, fresh_directory(WORK / "crystal" / key))))
        if wanted("berendsen"):
            jobs.append(("crystal/berendsen", lambda: crystal_berendsen(
                cfg, models["crystal"], fresh_directory(WORK / "crystal" / "berendsen"))))
    if "ideal_gas" in sections:
        gas_model = WORK / "models" / "ideal_gas.fnx"
        build_ideal_gas_model(cfg["ideal_gas"]["natoms"], gas_model)
        for key, case in cfg["ideal_gas"]["cases"].items():
            if wanted(key):
                jobs.append((f"ideal_gas/{key}", lambda key=key, case=case: ideal_gas_case(
                    cfg, gas_model, key, case, fresh_directory(WORK / "ideal_gas" / key))))
    if "cell_spring" in sections:
        spring_model = WORK / "models" / "cell_spring.fnx"
        build_cell_spring_model(cfg["cell_spring"], spring_model)
        for key, case in cfg["cell_spring"]["cases"].items():
            if wanted(key):
                jobs.append((f"cell_spring/{key}", lambda key=key, case=case: cell_spring_case(
                    cfg, spring_model, key, case, fresh_directory(WORK / "cell_spring" / key))))
    if "restart" in sections:
        for label, case in cfg["restart"]["cases"].items():
            if wanted(label):
                jobs.append((f"restart/{label}", lambda label=label, case=case: restart_case(
                    cfg, label, case, models)))
    if "reference" in sections:
        jobs.append(("reference/liquid_mc", lambda: mc_reference(cfg)))
    return jobs


def main():
    parser = common.argument_parser(__doc__.splitlines()[0], SECTIONS)
    parser.set_defaults(only=["liquid", "crystal", "ideal_gas", "cell_spring", "restart"])
    args = parser.parse_args()
    common.SEED = args.seed
    cfg = yaml.safe_load((args.config or HERE / "config.yaml").read_text())
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(cfg, args.only, args.methods)
    def post(results):
        return crystal_comparisons(cfg, results)
    sys.exit(common.run_suite(jobs, args.jobs, WORK / "report.json", post))


if __name__ == "__main__":
    main()
