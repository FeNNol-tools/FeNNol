import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import os
from pathlib import Path
import pickle
from functools import partial

from .utils import us, get_centroid_selector
from ..utils import Counter, read_tinker_interval
from ..utils.periodic_table import COV_RADII


def initialize_opes(
    simulation_parameters,
    dt,
    simulation_setup,
    fprec,
    rng_key,
    colvars_calculators={},
    restart_data={},
):
    opes_params = simulation_parameters["opes"]
    select_centroid = get_centroid_selector(simulation_setup)
    kT = simulation_setup["kT"]
    system_name = simulation_setup["name"]
    restart_file = system_name + ".opes.restart"

    ### COLVARS
    colvars = opes_params["colvars"]
    if isinstance(colvars, str):
        colvars = [colvars]
    colvars = [str(c).strip() for c in colvars]
    assert len(colvars) > 0, "opes/colvars must be a non-empty list of colvars"
    ncolvars = len(colvars)
    for colvar in colvars:
        assert (
            colvar in colvars_calculators
        ), f"colvar {colvar} not found in colvars_calculators"

    ### OPES PARAMETERS
    # PACE
    pace_opes = opes_params.get("pace", 100)
    assert pace_opes > 0, "pace_opes must be greater than 0"

    # SIGMA
    sigma_opes = opes_params.get("sigma", "adaptive")
    adaptive_sigma = False
    if isinstance(sigma_opes, str) and sigma_opes.strip().lower() in [
        "adaptive",
        "adapt",
    ]:
        # adaptive sigma => estimate sigma from sliding variance estimation
        sigma_opes = 1.0
        adaptive_sigma = True

    if isinstance(sigma_opes, (int, float)):
        sigma_opes = [sigma_opes] * ncolvars
    sigma_opes = np.array(sigma_opes, dtype=fprec)
    assert sigma_opes.shape == (
        ncolvars,
    ), f"opes/sigma must be a list of length {ncolvars}, got {sigma_opes.shape}"

    # SIGMA DECAY => decrease sigma with effective number of samples
    decay_sigma = opes_params.get("decay_sigma", True)
    sigma_min_rel = opes_params.get("min_sigma_fact", 1.0e-3)

    # BARRIER PARAMETER TO FIX GAMMA AND EPSILON
    barrier_opes = opes_params.get("barrier", None)
    if barrier_opes is not None:
        assert barrier_opes > kT, "opes/barrier must be greater than kT"

    # GAMMA
    gamma_opes_default = barrier_opes / kT if barrier_opes is not None else None
    gamma_opes = opes_params.get("gamma", gamma_opes_default)
    assert gamma_opes is not None, "opes/gamma or opes/barrier must be specified"
    assert (
        gamma_opes > 1
    ), f"gamma_opes={gamma_opes} must be greater than 1. Hint: increase the 'barrier' parameter."

    metadyn = opes_params.get("metadyn", False)
    if metadyn:
        hmetad = opes_params.get("hill_height", 1.0 / us.KCALPERMOL)

    # COMPUTE BIAS PREFACTOR
    explore = opes_params.get("explore", False)
    """@keyword[fennol_md] opes/explore
    Use OPES-explore: the kernels estimate the sampled well-tempered
    distribution p**(1/gamma), and the free energy is -gamma kBT log of their
    sum (scripts/analyze_opes.py applies the factor from the kernel file).
    Any error of that estimate is multiplied by gamma, so the kernel free
    energy carries a gamma-amplified finite-time bias. For an accurate final
    free energy, reweight the frames with exp(beta V) using the bias energies
    recorded in bias.traj (or in extxyz frames).
    Default: False
    """
    if explore:
        bias_prefact = (gamma_opes - 1.0) * kT
        sigma_opes = sigma_opes * gamma_opes**0.5
    else:
        bias_prefact = (1.0 - 1.0 / gamma_opes) * kT

    # EPSILON => probability regularization
    default_eps_opes = (
        math.exp(-barrier_opes / bias_prefact) if barrier_opes is not None else None
    )
    eps_opes = opes_params.get("epsilon", default_eps_opes)
    assert eps_opes is not None, "opes/epsilon or opes/barrier must be specified"

    #### DEBUG #####
    apply_bias = opes_params.get("apply_bias", True)
    if not apply_bias:
        bias_prefact = 0.0
    ################

    # shift bias to zero when no kernels are present
    Vshift = -bias_prefact * jnp.log(eps_opes)
    if metadyn:
        Vshift = 0.0  # no shift in metadynamics

    # initialize counters
    Tseg = opes_params.get("tseg", 1.0 / us.PS)
    nseg = int(Tseg / dt)
    counter = Counter(nseg)
    assert (
        nseg > pace_opes
    ), f"opes/tseg={Tseg} fs must be greater than opes/pace={pace_opes * dt} fs (dt={dt} fs)"

    # KERNEL CHUNK SIZES
    min_kernel_chunk = 1 + nseg // pace_opes
    kernel_chunk = opes_params.get("kernel_chunk", 2 * min_kernel_chunk)
    assert (
        kernel_chunk >= min_kernel_chunk
    ), f"opes/kernel_chunk must be at least {min_kernel_chunk}, got {kernel_chunk}"

    print(
        f"# OPES: gamma={gamma_opes:.3f} epsilon={eps_opes:.3e}, pace = {pace_opes}, kernel_chunk = {kernel_chunk}"
    )
    alpha_zn = opes_params.get("alpha_zn", 0.)
    assert 0.0 <= alpha_zn <= 1.0, "opes/alpha_zn must be in [0.0, 1.0]"

    ### BOUNDS
    bounds_dict = opes_params.get("bounds", None)
    use_bounds = bounds_dict is not None
    if use_bounds:
        assert isinstance(bounds_dict, dict), "opes/bounds must be a dictionary"
        for colvar in colvars:
            if colvar not in bounds_dict:
                bounds_dict[colvar] = np.array([-np.inf, np.inf])
            else:
                bounds_dict[colvar] = np.array(bounds_dict[colvar], dtype=fprec)
                assert (
                    len(bounds_dict[colvar]) == 2
                ), f"opes/bounds for {colvar} must be a list of two values [min, max], got {bounds_dict[colvar]}"
                bounds_dict[colvar] = np.array(
                    (np.min(bounds_dict[colvar]), np.max(bounds_dict[colvar])), dtype=fprec
                )
                print(
                    f"# OPES: bounds for {colvar} = {bounds_dict[colvar][0]:.3f} - {bounds_dict[colvar][1]:.3f}"
                )

    ### INITIALIZE STATE
    state = {}
    state["istep"] = 0
    state["count"] = 1
    state["nkernels"] = 0
    wk0 = 1.0  # eps_opes ** (bias_prefact / kT)
    state["wk_mean"] = wk0
    state["wk2_mean"] = wk0**2
    state["Zn"] = 1.0
    state["kernel_params"] = jnp.zeros((kernel_chunk, 1 + 2 * ncolvars), dtype=fprec)
    if adaptive_sigma:
        sigma_tau = opes_params.get("sigma_tau", 10 * pace_opes)
        state["sigma_avg"] = jnp.zeros((ncolvars,), dtype=fprec)
        state["sigma_var"] = jnp.zeros((ncolvars,), dtype=fprec)
    else:
        sigma_tau = 0

    # Restart files are written by postprocess. Preserve the complete kernel
    # state (including counters and running weight moments) when the dynamics
    # restarts and the current configuration uses a compatible
    # collective-variable layout. A fresh run ignores any stale restart file.
    do_restart = False
    if restart_data and os.path.exists(restart_file):
        with open(restart_file, "rb") as f:
            saved_state = pickle.load(f)
        expected_shape = (1 + 2 * ncolvars,)
        saved_params = np.asarray(saved_state.get("kernel_params", []))
        if saved_params.ndim == 2 and saved_params.shape[1:] == expected_shape:
            state.update(saved_state)
            state["kernel_params"] = jnp.asarray(saved_params, dtype=fprec)
            do_restart = True
            print(f"# OPES: restored state from {restart_file}")
    if restart_data and not do_restart:
        print(
            f"# OPES: warning: dynamics is restarting but {restart_file} "
            "is missing or incompatible. Starting OPES from scratch."
        )

    ### OPES FUNCTIONS
    def opes_bias_energy(cvs, state, return_d2=False):
        kernel_params = state["kernel_params"]
        h = kernel_params[:, 0]
        cvs_kernel = kernel_params[:, 1 : 1 + ncolvars]
        alphas = kernel_params[:, 1 + ncolvars :]

        d2 = jnp.sum(alphas * (cvs[None, :] - cvs_kernel) ** 2, axis=-1)

        if metadyn:
            V = jnp.sum(h * jnp.exp(-0.5 * d2))
        else:
            norm = state["Zn"] * state["count"]*(1. if explore else state["wk_mean"])
            prob = jnp.sum((h / norm) * jnp.exp(-0.5 * d2))

            V = bias_prefact * jnp.log(prob + eps_opes)

        # jax.debug.print("# OPES: d2 = {d2}, prob = {prob}, V = {V}",d2=d2, prob=prob, V=V)

        if return_d2:
            return V, d2

        return V

    def compute_sigma2_fact(neff):
        d = ncolvars
        fact = (neff * (0.25 * (d + 2))) ** (-2.0 / (d + 4))
        return fact
    
    def get_alpha_scale(state):
        if adaptive_sigma:
            sigma_var = state["sigma_var"]
            g = 1. if explore or metadyn else gamma_opes
            min_sigma2 = (sigma_min_rel * sigma_opes) ** 2
            sigma2 = jnp.maximum(sigma_var / g, min_sigma2)
        else:
            sigma2 = jnp.asarray(sigma_opes**2, dtype=fprec)

        if decay_sigma:
            neff = (
                state["count"]
                if explore
                else state["count"] * state["wk_mean"] ** 2 / state["wk2_mean"]
            )
            fact = compute_sigma2_fact(neff)
            fact = jnp.maximum(fact, sigma_min_rel**2)
            sigma2 = sigma2 * fact
            h0 = fact ** (-0.5 * ncolvars)
        else:
            fact = 1.0
            h0 = 1.0

        return 1.0 / sigma2, h0

    def combine_gaussians(h1, h2, cv1, cv2, alpha1, alpha2):
        h = h1 + h2
        w1 = h1 / h
        w2 = h2 / h
        cvs = w1 * cv1 + w2 * cv2
        sigma2_1 = 1.0 / (alpha1 + 1e-15)
        sigma2_2 = 1.0 / (alpha2 + 1e-15)
        sigma2 = (
            w1 * sigma2_1
            + w2 * sigma2_2
            + w1 * w2 * (cv1 - cv2) ** 2
        )
        return h, cvs, 1.0 / sigma2

    @jax.jit
    def add_kernel(cvs, state):
        V, d2 = opes_bias_energy(cvs, state, return_d2=True)
        count = state["count"] + 1
        new_state = {
            **state,
            "count": count,
        }
        wk = jnp.exp(V / kT)
        wkdiff = wk - state["wk_mean"] 
        wk_mean = state["wk_mean"] + wkdiff/count
        wk2diff = wk * wk - state["wk2_mean"]
        wk2_mean = state["wk2_mean"] + wk2diff/count
        new_state["wk_mean"] = wk_mean
        new_state["wk2_mean"] = wk2_mean

        if adaptive_sigma:
            # adjust for unbiased first estimation
            new_state["sigma_var"] = jax.lax.cond(
                state["count"] == 1,
                lambda x: x * (gamma_opes - 1.0),
                lambda x: x,
                state["sigma_var"],
            )

        alpha, h0 = get_alpha_scale(new_state)

        if metadyn:
            wt = jnp.exp(-V / (kT * (gamma_opes - 1.0)))
            h = jnp.array([hmetad * wt])
        else:
            h = jnp.array([h0 * (1.0 if explore else wk)], dtype=cvs.dtype)
        new_params = jnp.concatenate((h, cvs, alpha), axis=-1)

        kernel_params = state["kernel_params"]
        nkernels = state["nkernels"]
        kernel_mask = np.arange(kernel_params.shape[0]) < nkernels
        d2 = jnp.where(kernel_mask, d2, jnp.inf)

        # find closest kernel
        id2min = jnp.argmin(d2)
        d2min = d2[id2min]

        # merge or add new kernel
        def add_new_kernel(nkernels, new_params, kernel_params):
            if not use_bounds:
                new_kernel_params = kernel_params.at[nkernels].set(new_params, mode="drop")
                return nkernels + 1, new_kernel_params
            
            in_bounds = True
            for i, colvar in enumerate(colvars):
                bounds = bounds_dict[colvar]
                in_bounds = in_bounds & (bounds[0] <= cvs[i]) & (cvs[i] <= bounds[1])
            return jax.lax.cond(
                in_bounds,
                lambda n, x: (n + 1, x.at[n].set(new_params, mode="drop")),
                lambda n, x: (n, x),
                nkernels,
                kernel_params,
            )
            # new_kernel_params = kernel_params.at[nkernels].set(new_params, mode="drop")
            # return nkernels + 1, new_kernel_params

        def merge_kernel(nkernels, new_params, kernel_params):
            old_params = kernel_params[id2min]
            h1 = old_params[:1]
            h2 = new_params[:1]
            alpha1 = old_params[1 + ncolvars :]
            alpha2 = new_params[1 + ncolvars :]
            cv1 = old_params[1 : 1 + ncolvars]
            cv2 = new_params[1 : 1 + ncolvars]
            new_params = jnp.concatenate(
                combine_gaussians(h1, h2, cv1, cv2, alpha1, alpha2), axis=0
            )
            new_kernel_params = kernel_params.at[id2min].set(new_params)
            return nkernels, new_kernel_params

        nkernels, kernel_params = jax.lax.cond(
            d2min < 1.0,
            merge_kernel,
            add_new_kernel,
            nkernels,
            new_params,
            kernel_params,
        )

        # compute the new Zn value
        if metadyn:
            Zn = 1.0
        else:
            hs = kernel_params[:, 0]
            cvs = kernel_params[:, 1 : 1 + ncolvars]
            alphas = kernel_params[:, 1 + ncolvars :]

            d2s = jnp.sum(
                alphas[:, None, :] * (cvs[None, :, :] - cvs[:, None, :]) ** 2, axis=-1
            )
            norm = nkernels * count * (1.0 if explore else new_state["wk_mean"])
            active_kernel_mask = jnp.arange(kernel_params.shape[0]) < nkernels
            Zn = jax.lax.cond(
                nkernels > 0,
                lambda _: jnp.sum(
                    (hs[:, None] / norm)
                    * jnp.exp(-0.5 * d2s)
                    * active_kernel_mask[None, :]
                ),
                lambda _: state["Zn"],
                operand=None,
            )
            Zn = alpha_zn * state["Zn"] + (1. - alpha_zn) * Zn

        new_state["kernel_params"] = kernel_params
        new_state["nkernels"] = nkernels
        new_state["Zn"] = Zn

        return new_state

    @jax.jit
    def step(x, v, dynamics_state):
        state = dynamics_state["opes"]
        istep = state["istep"] + 1
        new_state = {**state, "istep": istep}

        cvs = []
        for colvar in colvars:
            cvs.append(dynamics_state["colvars"][colvar])
        cvs = jnp.array(cvs, dtype=x.dtype)

        if adaptive_sigma:
            tau = jnp.minimum(istep, sigma_tau).astype(x.dtype)
            diff = cvs - state["sigma_avg"]
            b1 = 1./tau
            m = state["sigma_avg"] + diff *b1
            new_state["sigma_avg"] = m
            diff2 = (cvs - m)*diff - state["sigma_var"]
            new_state["sigma_var"] = state["sigma_var"] + b1*diff2

        new_state = jax.lax.cond(
            (istep % pace_opes == 0) & (istep >= sigma_tau),
            add_kernel,
            lambda x, state: state,
            cvs,
            new_state,
        )

        return x, v, {**dynamics_state, "opes": new_state}

    @jax.jit
    def opes_system_bias(dynamics_state):
        x = dynamics_state["coordinates"]
        state = dynamics_state["opes"]
        cell = dynamics_state["cell"] if "cell" in dynamics_state else None


        def bias_energy(x):
            colvar_coordinates = select_centroid(x)
            cvs = []
            for colvar in colvars:
                cv = colvars_calculators[colvar](colvar_coordinates, cell)
                cvs.append(cv)
            cvs = jnp.array(cvs, dtype=x.dtype)
            return opes_bias_energy(cvs, state, return_d2=False)

        e, dedx = jax.value_and_grad(bias_energy)(x)
        de = {"coordinates": dedx}
        return e + Vshift, de

    # TRAJ FILE
    opes_traj_file = system_name + ".opes.traj"
    if not do_restart or not os.path.exists(opes_traj_file):
        with open(opes_traj_file, "w") as f:
            f.write(
                f"#{'istep':>7} {'nkernels':>8} {'neff':>15}  {'Zn':>15} {'sig_fact':>15} {'wk_mean':>15}\n"
            )

    def write_traj_file(state):
        if explore:
            neff = max(float(state["count"]), 1)
        else:
            neff = float(state["count"]) * float(state["wk_mean"]) ** 2 / float(state["wk2_mean"])
        sigma_fact = compute_sigma2_fact(neff) ** 0.5
        with open(opes_traj_file, "a") as f:
            f.write(
                f"{state['istep']:8d} {state['nkernels']:8d} {neff:15.6e} {state['Zn']:15.6e} {sigma_fact:15.6e} {state['wk_mean']:15.6e}\n"
            )

    def postprocess(state):
        counter.increment()
        if not counter.is_reset_step:
            return state

        kernel_size = state["kernel_params"].shape[0]
        nkernels = int(state["nkernels"])

        intent_merge = kernel_size - nkernels < min_kernel_chunk
        if intent_merge:
            if nkernels > kernel_size:
                print(f"# OPES: Warning: overflowed kernel buffer during last segment")
                nkernels = kernel_size
            kernel_params = np.array(state["kernel_params"])[:nkernels, :]
            kernel_mask = np.ones(nkernels, dtype=bool)
            hs = kernel_params[:, 0]
            cvs = kernel_params[:, 1 : 1 + ncolvars]
            alphas = kernel_params[:, 1 + ncolvars :]
            merged = False
            # try to merge current kernels
            for i in range(kernel_params.shape[0]):
                if not kernel_mask[i]:
                    continue
                while True:
                    cv1 = cvs[i, :]
                    d2 = np.sum(alphas * (cvs - cv1[None, :]) ** 2, axis=-1)
                    inactive = np.logical_not(kernel_mask)
                    d2[inactive] = np.inf  # exclude inactive kernels
                    d2[i] = np.inf  # exclude self

                    id2min = np.argmin(d2)
                    if d2[id2min] > 1.0:
                        break

                    merged = True
                    # merge with the closest kernel
                    h1 = hs[i]
                    h2 = hs[id2min]
                    cv2 = cvs[id2min, :]
                    alpha1 = alphas[i, :]
                    alpha2 = alphas[id2min, :]
                    h, cv, alpha = combine_gaussians(h1, h2, cv1, cv2, alpha1, alpha2)
                    # update kernel parameters
                    hs[i] = h
                    alphas[i, :] = alpha
                    cvs[i, :] = cv
                    kernel_mask[id2min] = False

            # build new kernel parameters
            if merged:
                hs = hs[kernel_mask]
                alphas = alphas[kernel_mask, :]
                cvs = cvs[kernel_mask, :]
                nkernels = hs.shape[0]
                new_kernel_params = np.concatenate((hs[:, None], cvs, alphas), axis=-1)

                # compute the new Zn value
                if metadyn:
                    Zn = 1.0
                else:
                    d2s = np.sum(
                        alphas[:, None, :] * (cvs[None, :, :] - cvs[:, None, :]) ** 2,
                        axis=-1,
                    )
                    norm = nkernels * state["count"]*float(
                        1. if explore else state["wk_mean"]
                    )
                    Zn = np.sum((hs[:, None] / norm) * np.exp(-0.5 * d2s))
                    Zn = alpha_zn * state["Zn"] + (1.-alpha_zn)* Zn
            else:
                new_kernel_params = kernel_params
                Zn = state["Zn"]

            # reallocate if needed
            reallocate = kernel_size - nkernels < min_kernel_chunk
            if reallocate:
                kernel_size = kernel_size + kernel_chunk
            kernel_params = np.zeros((kernel_size, 1 + 2 * ncolvars), dtype=fprec)
            kernel_params[:nkernels, :] = new_kernel_params

            # update state
            state["kernel_params"] = jnp.array(kernel_params, dtype=fprec)
            state["nkernels"] = nkernels
            state["Zn"] = Zn

        write_traj_file(state)

        # write kernels
        kernel_params = np.copy(state["kernel_params"])[:nkernels, :]
        if metadyn:
            # Well-tempered metadynamics: F = -gamma/(gamma-1) * V, so export
            # kernels whose sum is -F.
            norm = (gamma_opes - 1.0) / gamma_opes
        else:
            norm = state["Zn"] * state["count"]* (1. if explore else state["wk_mean"])
        kernel_params[:, 0] = kernel_params[:, 0] / norm
        # convert alpha to sigma
        kernel_params[:, 1 + ncolvars :] = 1.0 / kernel_params[:, 1 + ncolvars :] ** 0.5
        # The analysis script reads the estimator type from this line.
        mode = "metadyn" if metadyn else ("explore" if explore else "opes")
        header = f"mode={mode} gamma={gamma_opes:.10g} temperature={kT / us.K_B:.10g}\n"
        header += f"{'h':>15}"
        for colvar in colvars:
            header += f" {colvar:>15}"
        for colvar in colvars:
            st = f"sigma_{colvar}"
            header += f" {st:>15}"

        kernel_file = system_name + ".opes.kernels"
        np.savetxt(
            kernel_file,
            kernel_params,
            header=header,
            fmt="%15.6e",
        )
        print(f"# OPES: {nkernels} kernels written to", kernel_file)

        with open(restart_file, "wb") as f:
            pickle.dump(state, f)

        return state

    return step, opes_system_bias, postprocess, state
