"""Quick scientific checks for OPES, OPES-explore, metadynamics, eABF and lambda dynamics.

From the repository root::

    python validation/sampling/run.py            # every check
    python validation/sampling/run.py --only pmf_1d barrier
    python validation/sampling/run.py --only lambda --jobs 6

Every check is one short ``fennol_md`` run with a loose tolerance, designed to
catch real defects rather than to certify tight convergence:

* ``pmf_1d`` / ``pmf_2d``: free-energy reconstruction on smooth, asymmetric
  single wells that decorrelate quickly.
* ``barrier``: paired barrier-crossing screen. Plain MD should almost never
  cross an 8 kBT barrier; every biased method must cross it many times, and
  the well populations integrated from its reconstructed free energy must
  match the exact ones. OPES and explore frames are also reweighted with
  exp(beta V) from the bias energies written to ``bias.traj``.
* ``restart``: split and uninterrupted runs must leave identical dynamics and
  bias state.
* ``lambda``: lambda-ABF thermodynamic integration (with and without OPES, and
  plain lambda dynamics) against exact harmonic free energies, plus a paired
  lambda-barrier crossing screen and a restart check.

Models, configurations, trajectories and logs go to the ignored ``work/``
directory; the summary is written to ``work/report.json``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import logsumexp
import yaml

HERE = Path(__file__).resolve().parent
MODULES = HERE / "modules"
WORK = HERE / "work"
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from validation import common  # noqa: E402
from validation.common import (  # noqa: E402
    REPOSITORY, effective_samples, fresh_directory, seed_for,
)

ANALYZE_OPES = REPOSITORY / "scripts" / "analyze_opes.py"
common.use_modules(MODULES)

AXES = ("x", "y", "z")


# ---------------------------------------------------------------- references


def polynomial(coefficients, q):
    q = np.asarray(q, dtype=float)
    return sum(c * q ** (n + 1) for n, c in enumerate(coefficients))


def extended_reference(grid, coefficients, sigma):
    """eABF extended-variable PMF: F convolved with the kBT/sigma**2 restraint."""
    fine = np.linspace(grid[0] - 1.5, grid[-1] + 1.5, 6001)
    log_weight = -polynomial(coefficients, fine)
    return np.array([
        -logsumexp(log_weight - 0.5 * ((z - fine) / sigma) ** 2) for z in grid
    ])


def exact_left_population(coefficients):
    fine = np.linspace(-3.0, 3.0, 60001)
    weight = -polynomial(coefficients, fine)
    return float(np.exp(logsumexp(weight[fine < 0.0]) - logsumexp(weight)))


def lambda_schedule(lam):
    """elambda, d(elambda)/dlambda, vlambda, d(vlambda)/dlambda for bounds at 0.5."""
    lam = np.asarray(lam, dtype=float)
    elambda = np.clip(2.0 * lam - 1.0, 0.0, 1.0)
    vlambda = np.clip(2.0 * lam, 0.0, 1.0)
    return elambda, 2.0 * (lam > 0.5), vlambda, 2.0 * (lam < 0.5)


def lambda_free_energy(lam, model):
    """Exact F(lambda) in kBT, up to a constant."""
    elambda, _, vlambda, _ = lambda_schedule(lam)
    k = model["k0"] + model["ke"] * elambda + model["kv"] * vlambda
    return 1.5 * model["n_ligand"] * np.log(k)


def lambda_free_energy_gradient(lam, model):
    elambda, de, vlambda, dv = lambda_schedule(lam)
    k = model["k0"] + model["ke"] * elambda + model["kv"] * vlambda
    return 1.5 * model["n_ligand"] * (model["ke"] * de + model["kv"] * dv) / k


# ------------------------------------------------------------------- scoring


def score_profile(grid, estimate, reference, *, window, tolerance, minimum_coverage, valid=None):
    """RMSE of an additively aligned profile (kBT) where the reference is low."""
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    region = reference - reference.min() <= window
    ok = region & np.isfinite(estimate)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    coverage = float(ok.sum() / region.sum())
    if ok.sum() < 2:
        return {"status": "fail", "coverage": coverage, "reason": "no sampled bins"}
    offset = float(np.mean(reference[ok] - estimate[ok]))
    error = estimate[ok] + offset - reference[ok]
    rmse = float(np.sqrt(np.mean(error**2)))
    passed = coverage >= minimum_coverage and rmse <= tolerance
    return {
        "status": "pass" if passed else "fail",
        "rmse_kbt": rmse,
        "max_abs_error_kbt": float(np.max(np.abs(error))),
        "tolerance_kbt": tolerance,
        "coverage": coverage,
        "grid": np.asarray(grid).tolist(),
        "estimate_kbt": np.where(np.isfinite(estimate), estimate + offset, np.nan).tolist(),
        "reference_kbt": reference.tolist(),
    }


def count_passages(values, low, high):
    """Completed passages between the regions below ``low`` and above ``high``."""
    state, passages = None, 0
    for value in values:
        side = "low" if value <= low else "high" if value >= high else None
        if side is not None:
            if state is not None and side != state:
                passages += 1
            state = side
    return passages



def run_cli(config, directory, label="md"):
    common.run_cli(config, directory, MODULES, label)


# Restart files that carry each method's bias state, besides ``.dyn.restart``.
BIAS_RESTART = {"opes": ".opes.restart", "explore": ".opes.restart", "metad": ".opes.restart",
                "eabf": ".abf.restart", "eabf_metad": ".abf.restart", "lambda": ".lambdadyn.restart"}


def run_restart(directory, configure, steps, method):
    return common.run_restart(directory, configure, steps, MODULES,
                              expected=(".dyn.restart", BIAS_RESTART[method]))


def all_pass(results):
    return all(r.get("status") == "pass" for r in results if isinstance(r, dict))


# ------------------------------------------------------------------- models


def build_polynomial_model(potential, kbt, path):
    import jax
    from fennol import FENNIX

    coefficients = {axis: [float(c) for c in potential[axis]] for axis in AXES}
    model = FENNIX(
        cutoff=5.0,
        energy_unit="kcal/mol",
        modules={"polynomial": {"module_name": "SAMPLING_SEPARABLE_POLYNOMIAL", "kbt": kbt, **coefficients}},
        energy_terms=["polynomial_energy"],
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.array([18], dtype=np.int32),
            "coordinates": np.zeros((1, 3), dtype=np.float32),
            "natoms": np.array([1], dtype=np.int32),
            "batch_index": np.array([0], dtype=np.int32),
        },
    )
    model.save(path)
    loaded = FENNIX.load(path)
    point = np.array([[0.31, -0.57, 0.83]], dtype=np.float32)
    energy, forces, _ = loaded.energy_and_forces(
        species=np.array([18], dtype=np.int32), coordinates=point, unit="kcal/mol"
    )
    expected = kbt * sum(polynomial(coefficients[a], point[0, i]) for i, a in enumerate(AXES))
    expected_forces = -kbt * np.array([
        sum((n + 1) * c * point[0, i] ** n for n, c in enumerate(coefficients[a]))
        for i, a in enumerate(AXES)
    ])
    error = max(abs(float(np.asarray(energy).sum()) - expected),
                float(np.max(np.abs(np.asarray(forces)[0] - expected_forces))))
    if error > 1e-4:
        raise RuntimeError(f"polynomial model check failed (max error {error:g})")


def build_lambda_model(spec, environment_k, path):
    import jax
    import jax.numpy as jnp
    from fennol import FENNIX

    natoms = spec["n_ligand"] + 1
    module = {
        "module_name": "SAMPLING_LAMBDA_HARMONIC",
        "k0": spec["k0"], "ke": spec["ke"], "kv": spec["kv"],
        "n_ligand": spec["n_ligand"], "environment_k": environment_k,
    }
    model = FENNIX(
        cutoff=5.0,
        energy_unit="kcal/mol",
        modules={"lambda_harmonic": module},
        energy_terms=["lambda_harmonic_energy"],
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.full(natoms, 18, dtype=np.int32),
            "coordinates": np.zeros((natoms, 3), dtype=np.float32),
            "natoms": np.array([natoms], dtype=np.int32),
            "batch_index": np.zeros(natoms, dtype=np.int32),
            "alch_elambda": np.asarray(0.25, dtype=np.float32),
            "alch_vlambda": np.asarray(0.75, dtype=np.float32),
        },
    )
    model.save(path)
    loaded = FENNIX.load(path)
    coordinates = np.linspace(-0.4, 0.5, 3 * natoms).reshape(natoms, 3)
    data = loaded.preprocess(
        species=jnp.full(natoms, 18, dtype=jnp.int32),
        coordinates=jnp.asarray(coordinates),
        natoms=jnp.array([natoms], dtype=jnp.int32),
        batch_index=jnp.zeros(natoms, dtype=jnp.int32),
        alch_elambda=jnp.asarray(0.25),
        alch_vlambda=jnp.asarray(0.75),
    )
    energy, gradients, _ = loaded.get_gradient_function(
        "alch_elambda", "alch_vlambda", jit=False
    )(data)
    r2 = np.sum(coordinates**2, axis=1)
    ligand_r2 = r2[: spec["n_ligand"]].sum()
    k = spec["k0"] + 0.25 * spec["ke"] + 0.75 * spec["kv"]
    errors = [
        abs(float(np.asarray(energy).sum()) - 0.5 * (k * ligand_r2 + environment_k * r2[-1])),
        abs(float(gradients["alch_elambda"]) - 0.5 * spec["ke"] * ligand_r2),
        abs(float(gradients["alch_vlambda"]) - 0.5 * spec["kv"] * ligand_r2),
    ]
    if max(errors) > 1e-4:
        raise RuntimeError(f"lambda model check failed (errors {errors})")


# ------------------------------------------------------------ CLI configs


def write_xyz(path, positions):
    positions = np.atleast_2d(positions)
    lines = [str(len(positions)), "quick sampling check"]
    lines += [f"Ar {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in positions]
    path.write_text("\n".join(lines) + "\n")



def md_config(cfg, name, model, start, steps, seed, dt_fs=None):
    dt_fs = dt_fs or cfg["dt_fs"]
    every = cfg["output_interval_steps"]
    return {
        "model_file": str(model),
        "system_name": str(name),
        "temperature": cfg["temperature_kelvin"],
        "energy_unit": "kcal/mol",
        "random_seed": int(seed),
        "nsteps": int(steps),
        "dt[fs]": dt_fs,
        "tdump[fs]": dt_fs * steps,
        "nprint": every,
        "nsummary": 100 * every,
        "device": "cpu",
        "double_precision": False,
        "thermostat": "LGV",
        "gamma[THz]": cfg["langevin_friction_thz"],
        "traj_format": "xyz",
        "per_atom_energy": False,
        "check_temperature": False,
        "xyz_input": {"file": str(start), "indexed": False, "has_comment_line": True},
    }


def method_block(cfg, method, cvs, grid, barrier_kbt=None):
    """Sampler section of the fennol_md input for one CV method."""
    kbt = cfg["kbt_kcal_per_mol"]
    spec = cfg["methods"][method]
    if "opes" in spec:
        o = spec["opes"]
        sigma = o["sigma"] if o["sigma"] == "adaptive" else [o["sigma"]] * len(cvs)
        block = {
            "colvars": list(cvs),
            "pace": o["pace"],
            "sigma": sigma,
            "barrier": (barrier_kbt or o["barrier_kbt"]) * kbt,
            "tseg[ps]": o["tseg_ps"],
        }
        if "kernel_chunk" in o:
            block["kernel_chunk"] = o["kernel_chunk"]
        if o.get("explore"):
            block["explore"] = True
        if o.get("metadyn"):
            block.update({"metadyn": True, "hill_height": o["hill_height_kbt"] * kbt})
        return {"opes": block}
    a = spec["abf"]
    block = {
        "colvar": cvs[0],
        "sigma": a["sigma"],
        "bounds": [grid[0], grid[1]],
        "bin_width": grid[2],
        "fullsamples": a["fullsamples"],
        "tseg[ps]": a["tseg_ps"],
        "tau[fs]": a["tau_fs"],
        "gamma[THz]": a["gamma_thz"],
    }
    if "metad_height_kbt" in a:
        block.update({
            "metad_height": a["metad_height_kbt"] * kbt,
            "metad_pace": a["metad_pace"],
            "metad_sigma": a["metad_sigma"],
            "metad_barrier": a["metad_bias_factor"] * kbt,
        })
    return {"abf": block}




def colvar_config(cvs):
    return {cv: {"type": "coordinate", "atom": 1, "axis": cv} for cv in cvs}


def run_cv_method(cfg, directory, model, method, cvs, start, steps, grid, barrier_kbt=None):
    write_xyz(directory / "start.xyz", start)
    seed = seed_for(str(directory.relative_to(WORK)))
    config = md_config(cfg, directory / method, model, directory / "start.xyz", steps, seed)
    config["colvars"] = colvar_config(cvs)
    if method != "plain":
        config.update(method_block(cfg, method, cvs, grid, barrier_kbt))
    run_cli(config, directory)
    return directory / method


# ----------------------------------------------------------------- analysis


def read_colvars(name, burn_in_ps):
    data = np.loadtxt(f"{name}.colvars.traj", comments="#", ndmin=2)
    return data[data[:, 0] > burn_in_ps, 1:]


def read_columns(path, burn_in_ps):
    """Columns of a ``colvars.traj`` or ``bias.traj`` file after burn-in, keyed by header name."""
    path = Path(path)
    with path.open() as source:
        header = source.readline().lstrip("#").split()
    data = np.loadtxt(path, comments="#", ndmin=2)
    data = data[data[:, 0] > burn_in_ps]
    return {key: data[:, i] for i, key in enumerate(header)}


def read_table(path):
    return np.loadtxt(path, comments="#", ndmin=2)


def opes_free_energy(cfg, method, name, cvs, grid):
    """FES in kBT from the exported kernels through the standard analysis CLI."""
    kbt = cfg["kbt_kcal_per_mol"]
    output = Path(f"{name}.fes.dat")
    args = [sys.executable, str(ANALYZE_OPES), f"{name}.opes.kernels",
            "--temperature", str(cfg["temperature_kelvin"]),
            "--cvs", *[str(i) for i in range(len(cvs))],
            "--bounds", *[str(grid[i]) for _ in cvs for i in (0, 1)],
            "--dx", *[str(grid[2])] * len(cvs),
            "--outfile", str(output)]
    # The kernel file header tells the script whether these are OPES,
    # explore or metadynamics kernels.
    completed = subprocess.run(args, cwd=REPOSITORY, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"OPES analysis failed: {completed.stderr}")
    table = read_table(output)
    free_energy = table[:, -1] / kbt
    if len(cvs) == 1:
        return table[:, 0], free_energy
    nx = len(np.unique(table[:, 0]))
    shape = (nx, len(table) // nx)
    return (np.unique(table[:, 0]), np.unique(table[:, 1])), free_energy.reshape(shape)


def method_profile(cfg, method, name, grid):
    """1D physical free-energy estimate (grid, kBT, sampled-mask) of any CV method."""
    if "abf" in cfg["methods"][method]:
        pmf = read_table(f"{name}.czar.pmf")
        counts = read_table(f"{name}.zcount")[:, 1]
        return pmf[:, 0], pmf[:, 1] / cfg["kbt_kcal_per_mol"], counts > 0
    x, free_energy = opes_free_energy(cfg, method, name, ["x"], grid)
    return x, free_energy, None


def analyze_pmf_1d(cfg, case, method, name):
    reference_coefficients = cfg["potentials"][case["potential"]]["x"]
    burn_in_ps = case["burn_in_steps"] * cfg["dt_fs"] / 1000.0
    x_samples = read_colvars(name, burn_in_ps)[:, 0]
    settings = {
        "window": cfg["score_window_kbt"],
        "tolerance": cfg["pmf_rmse_tolerance_kbt"],
        "minimum_coverage": cfg["minimum_coverage_fraction"],
    }
    grid, estimate, sampled = method_profile(cfg, method, name, case["grid"])
    result = {
        "effective_samples": effective_samples(x_samples),
        "physical": score_profile(grid, estimate, polynomial(reference_coefficients, grid),
                                  valid=sampled, **settings),
    }
    if "abf" in cfg["methods"][method]:
        pmf = read_table(f"{name}.pmf")
        counts = read_table(f"{name}.count")[:, 1]
        sigma = cfg["methods"][method]["abf"]["sigma"]
        result["extended"] = score_profile(
            pmf[:, 0], pmf[:, 1] / cfg["kbt_kcal_per_mol"],
            extended_reference(pmf[:, 0], reference_coefficients, sigma),
            valid=counts > 0, **settings,
        )
    result["status"] = "pass" if all_pass(result.values()) else "fail"
    return result


def analyze_pmf_2d(cfg, case, method, name):
    potential = cfg["potentials"][case["potential"]]
    (xs, ys), joint = opes_free_energy(cfg, method, name, ["x", "y"], case["grid"])
    reference_x = polynomial(potential["x"], xs)
    reference_y = polynomial(potential["y"], ys)
    window = cfg["score_window_kbt"]
    coverage = cfg["minimum_coverage_fraction"]
    tolerance = cfg["pmf_rmse_tolerance_kbt"]
    result = {
        "joint": score_profile(
            None, joint.ravel(), (reference_x[:, None] + reference_y[None, :]).ravel(),
            window=window, tolerance=case["joint_rmse_tolerance_kbt"], minimum_coverage=coverage,
        ),
        "marginal_x": score_profile(xs, -logsumexp(-joint, axis=1), reference_x,
                                    window=window, tolerance=tolerance, minimum_coverage=coverage),
        "marginal_y": score_profile(ys, -logsumexp(-joint, axis=0), reference_y,
                                    window=window, tolerance=tolerance, minimum_coverage=coverage),
    }
    for key in ("grid", "estimate_kbt", "reference_kbt"):
        result["joint"].pop(key, None)
    result["status"] = "pass" if all_pass(result.values()) else "fail"
    return result


def analyze_barrier(cfg, case, method, name):
    burn_in_ps = case["burn_in_steps"] * cfg["dt_fs"] / 1000.0
    x_samples = read_colvars(name, burn_in_ps)[:, 0]
    passages = count_passages(x_samples, *case["commit"])
    result = {"passages": passages, "simulated_ps": case["production_steps"] * cfg["dt_fs"] / 1000.0}
    if method == "plain":
        result["maximum_passages"] = case["maximum_plain_passages"]
        result["status"] = "pass" if passages <= case["maximum_plain_passages"] else "fail"
        return result
    coefficients = cfg["potentials"][case["potential"]]["x"]
    grid, estimate, sampled = method_profile(cfg, method, name, case["grid"])
    usable = np.isfinite(estimate) & (sampled if sampled is not None else True)
    weights = np.where(usable, -estimate, -np.inf)
    left = float(np.exp(logsumexp(weights[grid < 0.0]) - logsumexp(weights)))
    exact = exact_left_population(coefficients)
    result.update({
        "minimum_passages": case["minimum_biased_passages"],
        "left_population": left,
        "exact_left_population": exact,
    })
    passed = (passages >= case["minimum_biased_passages"]
              and abs(left - exact) <= cfg["population_tolerance"])
    if method in ("opes", "explore"):
        # OPES biases are quasi-static: reweight each frame by exp(beta V_bias).
        # bias.traj rows match colvars.traj rows; fennol_md runs here in kcal/mol.
        x = read_columns(f"{name}.colvars.traj", burn_in_ps)["x"]
        energies = read_columns(f"{name}.bias.traj", burn_in_ps)
        bias = next(v for k, v in energies.items() if k.startswith("Eopes[")) / cfg["kbt_kcal_per_mol"]
        log_left = logsumexp(bias[x < 0.0]) - logsumexp(bias)
        reweighted = float(np.exp(log_left))
        result["reweighted_left_population"] = reweighted
        passed = passed and abs(reweighted - exact) <= cfg["population_tolerance"]
    result["status"] = "pass" if passed else "fail"
    return result




# ------------------------------------------------------------------- lambda


def lambda_config(cfg, directory, model, name, start_file, steps, model_spec, seed, *,
                  start_lambda, abf=True, opes=None):
    lam = cfg["lambda"]
    kbt = cfg["kbt_kcal_per_mol"]
    config = md_config(cfg, directory / name, model, start_file, steps, seed, lam["dt_fs"])
    block = {
        "ligand": list(range(1, model_spec["n_ligand"] + 1)),
        "lambda": start_lambda,
        "lambda_bounds": [0.0, 1.0],
        "bound_elambda": 0.5,
        "bound_vlambda": 0.5,
        "bin_width": lam["bin_width"],
        "fullsamples": lam["fullsamples"],
        "tseg[ps]": lam["tseg_ps"],
        "mass_lambda": lam["mass_lambda"],
        "gamma_lambda[THz]": lam["gamma_lambda_thz"],
        "abf": bool(abf),
    }
    if opes:
        block.update({
            "barrier_opes": opes["barrier_kbt"] * kbt,
            "pace_opes": opes["pace"],
            "sigma_opes": opes["sigma"],
        })
    config["lambdadyn"] = block
    return config


def read_lambda_series(log_path, burn_in_steps):
    values = []
    for line in Path(log_path).read_text().splitlines():
        fields = line.split()
        if fields and not line.startswith("#") and fields[0].lstrip("-").isdigit():
            if int(fields[0]) > burn_in_steps:
                values.append(float(fields[-1]))
    return np.asarray(values)


def lambda_start_file(directory, n_ligand):
    path = directory / "start.xyz"
    write_xyz(path, np.full((n_ligand + 1, 3), 0.1))
    return path


def analyze_lambda_ti(cfg, ti, name, model_spec, abf):
    kbt = cfg["kbt_kcal_per_mol"]
    counts = read_table(f"{name}.count")
    gradient = read_table(f"{name}.grad")
    centers, sampled = counts[:, 0], counts[:, 1] > 0
    estimate = gradient[:, 1] / kbt
    reference = lambda_free_energy_gradient(centers, model_spec)
    width = cfg["lambda"]["bin_width"]
    exact = float(lambda_free_energy(1.0, model_spec) - lambda_free_energy(0.0, model_spec))
    error = estimate[sampled] - reference[sampled]
    delta_f = float(np.sum(estimate) * width) if sampled.all() else None
    result = {
        "coverage": float(sampled.mean()),
        "gradient_rmse_kbt": float(np.sqrt(np.mean(error**2))) if sampled.any() else None,
        "delta_f_kbt": delta_f,
        "exact_delta_f_kbt": exact,
        "lambda_grid": centers.tolist(),
        "gradient_estimate_kbt": estimate.tolist(),
        "gradient_reference_kbt": reference.tolist(),
    }
    passed = (sampled.all()
              and result["gradient_rmse_kbt"] <= ti["gradient_rmse_tolerance_kbt"]
              and abs(delta_f - exact) <= ti["delta_f_tolerance_kbt"])
    if not abf:
        # Plain lambda dynamics must also sample the exact lambda marginal.
        histogram = -np.log(np.where(sampled, counts[:, 1], np.nan))
        result["marginal"] = score_profile(
            centers, histogram, lambda_free_energy(centers, model_spec),
            window=np.inf, tolerance=cfg["pmf_rmse_tolerance_kbt"], minimum_coverage=1.0,
        )
        passed = passed and result["marginal"]["status"] == "pass"
    result["status"] = "pass" if passed else "fail"
    return result


# -------------------------------------------------------------- scheduling


def plan_jobs(cfg, sections, methods_filter):
    """Return (key, callable) pairs; each callable runs its MD and analysis."""
    jobs = []
    models = {}
    kbt = cfg["kbt_kcal_per_mol"]

    def polynomial_model(potential):
        if potential not in models:
            path = WORK / "models" / f"{potential}.fnx"
            build_polynomial_model(cfg["potentials"][potential], kbt, path)
            models[potential] = path
        return models[potential]

    def selected(methods):
        return [m for m in methods if not methods_filter or m in methods_filter]

    for section, cvs, analyze in (("pmf_1d", ["x"], analyze_pmf_1d), ("pmf_2d", ["x", "y"], analyze_pmf_2d)):
        if section not in sections:
            continue
        case = cfg[section]
        model = polynomial_model(case["potential"])
        steps = case["burn_in_steps"] + case["production_steps"]
        for method in selected(case["methods"]):
            def job(case=case, model=model, method=method, cvs=cvs, analyze=analyze, section=section):
                directory = fresh_directory(WORK / section / method)
                name = run_cv_method(cfg, directory, model, method, cvs, case["start"], steps, case["grid"])
                return analyze(cfg, case, method, name)
            jobs.append((f"{section}/{method}", job))

    if "barrier" in sections:
        case = cfg["barrier"]
        model = polynomial_model(case["potential"])
        steps = case["burn_in_steps"] + case["production_steps"]
        for method in selected(case["methods"]):
            def job(method=method, model=model, case=case, steps=steps):
                directory = fresh_directory(WORK / "barrier" / method)
                name = run_cv_method(cfg, directory, model, method, ["x"], case["start"], steps,
                                     case["grid"], case["barrier_kbt"])
                return analyze_barrier(cfg, case, method, name)
            jobs.append((f"barrier/{method}", job))

    if "restart" in sections:
        case = cfg["restart"]
        model = polynomial_model(case["potential"])
        for method in selected(case["methods"]):
            def job(method=method, model=model, case=case):
                directory = fresh_directory(WORK / "restart" / method)
                write_xyz(directory / "start.xyz", case["start"])

                def configure(run_dir, steps):
                    config = md_config(cfg, run_dir / method, model, directory / "start.xyz", steps,
                                       seed_for(f"restart/{method}"))
                    config["colvars"] = colvar_config(["x"])
                    config.update(method_block(cfg, method, ["x"], case["grid"]))
                    return config, run_dir / method
                return run_restart(directory, configure, case["steps"], method)
            jobs.append((f"restart/{method}", job))

    if "lambda" in sections:
        lam = cfg["lambda"]
        lambda_models = {}

        def lambda_model(key):
            if key not in lambda_models:
                path = WORK / "models" / f"lambda-{key}.fnx"
                build_lambda_model(lam["models"][key], lam["environment_k"], path)
                lambda_models[key] = path
            return lambda_models[key]

        ti = lam["ti"]
        for label, spec in ti["cases"].items():
            if methods_filter and label not in methods_filter:
                continue
            model_spec = lam["models"][spec["model"]]
            model = lambda_model(spec["model"])

            def job(label=label, spec=spec, model_spec=model_spec, model=model):
                directory = fresh_directory(WORK / "lambda_ti" / label)
                start = lambda_start_file(directory, model_spec["n_ligand"])
                abf = spec.get("abf", True)
                config = lambda_config(
                    cfg, directory, model, label, start,
                    ti["burn_in_steps"] + ti["production_steps"], model_spec, seed_for(f"lambda_ti/{label}"),
                    start_lambda=ti["start_lambda"], abf=abf,
                    opes=lam["opes"] if spec.get("opes") else None,
                )
                run_cli(config, directory)
                return analyze_lambda_ti(cfg, ti, directory / label, model_spec, abf)
            jobs.append((f"lambda_ti/{label}", job))

        barrier = lam["barrier"]
        model_spec = lam["models"][barrier["model"]]
        model = lambda_model(barrier["model"])
        for label, spec in barrier["cases"].items():
            if methods_filter and label not in methods_filter:
                continue

            def job(label=label, spec=spec, model=model, model_spec=model_spec):
                directory = fresh_directory(WORK / "lambda_barrier" / label)
                start = lambda_start_file(directory, model_spec["n_ligand"])
                config = lambda_config(
                    cfg, directory, model, label, start,
                    barrier["burn_in_steps"] + barrier["production_steps"], model_spec,
                    seed_for(f"lambda_barrier/{label}"),
                    start_lambda=barrier["start_lambda"], abf=spec.get("abf", True),
                    opes=barrier["opes"] if spec.get("opes") else None,
                )
                run_cli(config, directory)
                series = read_lambda_series(directory / "md.log", barrier["burn_in_steps"])
                passages = count_passages(series, *barrier["commit"])
                plain = not spec.get("abf", True) and not spec.get("opes")
                if plain:
                    ok = passages <= barrier["maximum_plain_passages"]
                else:
                    ok = passages >= barrier["minimum_biased_passages"]
                return {"passages": passages, "status": "pass" if ok else "fail",
                        "simulated_ps": barrier["production_steps"] * lam["dt_fs"] / 1000.0}
            jobs.append((f"lambda_barrier/{label}", job))

        restart = lam["restart"]
        if not methods_filter or "restart" in methods_filter:
            restart_spec = lam["models"][restart["model"]]
            restart_model = lambda_model(restart["model"])

            def job():
                directory = fresh_directory(WORK / "lambda_restart")
                start = lambda_start_file(directory, restart_spec["n_ligand"])

                def configure(run_dir, steps):
                    config = lambda_config(
                        cfg, run_dir, restart_model, "lambda", start, steps, restart_spec,
                        seed_for("lambda_restart"),
                        start_lambda=lam["ti"]["start_lambda"],
                        opes=lam["opes"] if restart["opes"] else None,
                    )
                    return config, run_dir / "lambda"
                return run_restart(directory, configure, restart["steps"], "lambda")
            jobs.append(("lambda_restart", job))
    return jobs


SECTIONS = ("pmf_1d", "pmf_2d", "barrier", "restart", "lambda")


def main():
    parser = common.argument_parser(__doc__.splitlines()[0], SECTIONS)
    args = parser.parse_args()
    common.SEED = args.seed
    cfg = yaml.safe_load((args.config or HERE / "config.yaml").read_text())
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(cfg, args.only, args.methods)
    sys.exit(common.run_suite(jobs, args.jobs, WORK / "report.json"))


if __name__ == "__main__":
    main()
