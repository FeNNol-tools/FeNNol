import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import os
import pickle
from pathlib import Path

from .utils import us, get_centroid_selector
from ..utils import Counter, read_tinker_interval, detect_topology
from ..utils.periodic_table import COV_RADII


def initialize_lambdadyn(
    simulation_parameters, dt, simulation_setup, fprec, rng_key, restart_data={}
):
    lambdadyn_params = simulation_parameters["lambdadyn"]
    kT = simulation_setup["kT"]
    nat = len(simulation_setup["species"])
    energy_unit = simulation_setup["energy_unit"]
    energy_unit_str = simulation_setup["energy_unit_str"]
    system_name = simulation_setup["name"]

    mass_lambda = lambdadyn_params.get(
        "mass_lambda", 150000.0 / (us.KCALPERMOL * us.FS**2)
    )  # default colvars value in Tinker simulations
    gamma_lambda = lambdadyn_params.get("gamma_lambda", 1000.0 / us.THZ)

    a1 = math.exp(-gamma_lambda * dt)
    a2 = ((1 - a1 * a1) * kT / mass_lambda) ** 0.5

    alch_lambda = lambdadyn_params.get("lambda", 1.0)
    lambda_min, lambda_max = lambdadyn_params.get("lambda_bounds", (0.0, 1.0))
    lambda_min, lambda_max = min(lambda_min, lambda_max), max(lambda_min, lambda_max)
    lambda_min = np.clip(lambda_min, 0.0, 1.0)
    lambda_max = np.clip(lambda_max, 0.0, 1.0)
    alch_lambda = np.clip(alch_lambda, lambda_min, lambda_max)

    rng_key, v_key = jax.random.split(rng_key)
    alch_lambda_vel = jax.random.normal(v_key, dtype=fprec) * (kT / mass_lambda) ** 0.5
    state = {
        "istep": 0,
        "alch_lambda": jnp.asarray(alch_lambda, dtype=fprec),
        "alch_lambda_vel": alch_lambda_vel,
        "rng_key": rng_key,
        **restart_data.get("lambdadyn", {}),
    }
    print(f"# LAMBDADYN: initial lambda = {state['alch_lambda']:.4f}")

    ligand_indices_raw = lambdadyn_params.get("ligand", None)
    if ligand_indices_raw is None:
        ligand_indices_raw = simulation_parameters.get("ligand", None)
    if ligand_indices_raw is None:
        raise ValueError(
            "No ligand indices provided in lambdadyn parameters or simulation parameters"
        )
    ligand_indices = read_tinker_interval(ligand_indices_raw)
    assert np.all(ligand_indices < nat), "Ligand indices must be < nat"
    assert len(ligand_indices) > 0, "No ligand indices found"
    assert len(ligand_indices) < nat, "Error: All atoms are ligands"
    print("# LAMBDADYN: ligand_indices =", ligand_indices)

    state["alch_ligand_charge"] = lambdadyn_params.get("ligand_charge", 0.0)

    alch_group = np.zeros((nat,), dtype=np.int32)
    alch_group[ligand_indices] = 1
    state["alch_group"] = alch_group

    bound_elambda = lambdadyn_params.get("bound_elambda", 0.5)
    assert 0.0 <= bound_elambda < 1.0, "bound_elambda must be in [0, 1)"
    bound_vlambda = lambdadyn_params.get("bound_vlambda", 0.5)
    assert 0.0 < bound_vlambda <= 1.0, "bound_vlambda must be in (0, 1]"

    def split_lambda(alch_lambda):
        alchdown = alch_lambda >= 0
        alchup = alch_lambda <= 1
        alchin = alchup * alchdown
        alchbounde = alch_lambda >= bound_elambda
        alchboundv = alch_lambda <= bound_vlambda

        efact = 1.0 / (1 - bound_elambda)
        vfact = 1.0 / bound_vlambda

        elambda = alchin * alchbounde * efact * (alch_lambda - bound_elambda) + (
            1.0 - alchup
        )
        delambda = alchin * alchbounde * efact
        vlambda = alchin * vfact * alch_lambda * alchboundv + (1 - alchboundv)
        dvlambda = alchin * vfact * alchboundv
        return elambda, delambda, vlambda, dvlambda

    # Use the restored lambda on restart, not the configured initial value.
    elambda, _, vlambda, _ = split_lambda(state["alch_lambda"])
    state["alch_elambda"] = elambda
    state["alch_vlambda"] = vlambda

    #### INITIALIZE HISTOGRAMS
    width = lambdadyn_params.get("bin_width", 0.01)
    nbins = round((lambda_max - lambda_min) / width)
    assert nbins > 0, "nbins must be greater than 0"

    lambda_edges = np.linspace(lambda_min, lambda_max, nbins + 1, dtype=fprec)
    lambda_centers = 0.5 * (lambda_edges[:-1] + lambda_edges[1:])

    state["lambda_counts"] = jnp.zeros((nbins,), dtype=jnp.int32)
    state["gradients_avg"] = jnp.zeros((nbins,), dtype=fprec)

    ### INITIALIZE OPES
    barrier_opes = lambdadyn_params.get("barrier_opes", None)
    use_opes = barrier_opes is not None and barrier_opes > 0.0
    if use_opes:
        assert barrier_opes > kT, "barrier_opes must be greater than kT"
        pace_opes = lambdadyn_params.get("pace_opes", 50)
        assert pace_opes > 0, "pace_opes must be greater than 0"
        sigma_opes = lambdadyn_params.get("sigma_opes", 0.1)
        assert (
            sigma_opes > width
        ), f"sigma_opes={sigma_opes} must be greater than bin_width={width}"
        alpha_opes = 1.0 / (2 * sigma_opes**2)
        norm_opes = 1.0 / (2 * math.pi * sigma_opes**2) ** 0.5

        adaptive_opes = lambdadyn_params.get("adaptive_opes", False)

        gamma_opes = barrier_opes / kT
        eps_opes = math.exp(-gamma_opes / (gamma_opes - 1))
        print(
            f"# LAMBDADYN: using OPES with barrier = {barrier_opes*energy_unit:.2e} {energy_unit_str} (gamma={gamma_opes:.3f}) and pace = {pace_opes}"
        )
        state["opes_count"] = 0
        if adaptive_opes:
            state["opes_wk"] = 1.0
            state["opes_wk2"] = 1.0
        state["opes_prob"] = jnp.zeros((nbins,), dtype=fprec)
        state["opes_grad"] = jnp.zeros((nbins,), dtype=fprec)
        state["opes_bias"] = jnp.zeros((nbins,), dtype=fprec)
        state["opes_fbias"] = jnp.zeros((nbins,), dtype=fprec)

    ### RESTART
    restart_file = simulation_setup["name"] + ".lambdadyn.restart"
    do_restart = False
    # Only continue the histograms when the dynamics itself restarts, so a fresh
    # run in an old directory does not inherit stale counts and gradients.
    if restart_data and os.path.exists(restart_file):
        with open(restart_file, "rb") as f:
            lambdadyn_restart = pickle.load(f)

        do_restart = (
            lambdadyn_restart["centers"].shape[0] == nbins
            and np.allclose(lambdadyn_restart["centers"], lambda_centers)
            and lambdadyn_restart["ligand_indices"].shape[0] == len(ligand_indices)
            and np.all(lambdadyn_restart["ligand_indices"] == ligand_indices)
            and lambdadyn_restart["bound_elambda"] == bound_elambda
            and lambdadyn_restart["bound_vlambda"] == bound_vlambda
            and lambdadyn_restart["lambda_counts"].shape[0] == nbins
            and lambdadyn_restart["gradients_avg"].shape[0] == nbins
        )

        if do_restart:
            print("# LAMBDADYN: loading previous counts and gradients")
            state["istep"] = int(lambdadyn_restart["istep"])
            if "rng_key" in lambdadyn_restart:
                state["rng_key"] = jnp.asarray(lambdadyn_restart["rng_key"])
            state["lambda_counts"] = jnp.array(lambdadyn_restart["lambda_counts"])
            state["gradients_avg"] = jnp.array(lambdadyn_restart["gradients_avg"])
            if use_opes and "opes_prob" in lambdadyn_restart:
                state["opes_count"] = int(lambdadyn_restart["opes_count"])
                state["opes_prob"] = jnp.array(lambdadyn_restart["opes_prob"])
                state["opes_grad"] = jnp.array(lambdadyn_restart["opes_grad"])
                for key in ("opes_bias", "opes_fbias"):
                    if key in lambdadyn_restart:
                        state[key] = jnp.array(lambdadyn_restart[key])
                if adaptive_opes and "opes_wk" in lambdadyn_restart:
                    state["opes_wk"] = float(lambdadyn_restart["opes_wk"])
                    state["opes_wk2"] = float(lambdadyn_restart["opes_wk2"])
        else:
            print(
                f"# LAMBDADYN: Could not restart from {restart_file}. Starting from scratch."
            )
    elif restart_data:
        print(
            f"# LAMBDADYN: warning: dynamics is restarting but {restart_file} "
            "is missing. Starting histograms from scratch."
        )

    if not do_restart:
        with open(system_name + ".deltaG.traj", "w") as f:
            f.write("#istep     deltaG\n")

    def write_restart_file(state):
        lambdadyn_restart = {
            "istep": state["istep"],
            "rng_key": np.asarray(state["rng_key"]),
            "centers": lambda_centers,
            "ligand_indices": ligand_indices,
            "bound_elambda": bound_elambda,
            "bound_vlambda": bound_vlambda,
            "lambda_counts": np.array(state["lambda_counts"]),
            "gradients_avg": np.array(state["gradients_avg"]),
        }
        if use_opes:
            lambdadyn_restart["opes_count"] = state["opes_count"]
            lambdadyn_restart["opes_prob"] = np.array(state["opes_prob"])
            lambdadyn_restart["opes_grad"] = np.array(state["opes_grad"])
            lambdadyn_restart["opes_bias"] = np.array(state["opes_bias"])
            lambdadyn_restart["opes_fbias"] = np.array(state["opes_fbias"])
            if adaptive_opes:
                lambdadyn_restart["opes_wk"] = state["opes_wk"]
                lambdadyn_restart["opes_wk2"] = state["opes_wk2"]
        with open(restart_file, "wb") as f:
            pickle.dump(lambdadyn_restart, f)

    ### TOPOLOGY RESTRAINT
    if "topology_wall_K" in lambdadyn_params:
        wall_K = lambdadyn_params["topology_wall_K"]
        wall_r0 = lambdadyn_params.get("topology_wall_R", 0.4)
        assert wall_K > 0.0, "topology_wall_K must be greater than 0"
        assert wall_r0 > 0.0, "topology_wall_R must be greater than 0"
        topology_key = lambdadyn_params.get("topology", None)
        if topology_key is not None:
            topology_key = str(topology_key).strip()
            if topology_key.lower() == "detect":
                species = simulation_setup["species"]
                if "optimized_coordinates" in simulation_setup:
                    coordinates = simulation_setup["optimized_coordinates"]
                else:
                    coordinates = simulation_setup["initial_coordinates"]
                pbc_data = simulation_setup["pbc"]
                cell = None if pbc_data is None else pbc_data["cell"]
                topology = detect_topology(
                    species[ligand_indices], coordinates[ligand_indices], cell=cell
                )
                topology = ligand_indices[topology]
                print("# Detected ligand topology")
                np.savetxt(
                    f"{system_name}.ligand.topo",
                    topology + 1,
                    fmt="%d",
                )
            else:
                topology_key = Path(topology_key).expanduser().resolve()
                assert topology_key.exists(), f"Topology file {topology_key} not found"
                topology = np.loadtxt(str(topology_key), dtype=np.int32) - 1
                assert (
                    topology.shape[1] == 2
                ), "Topology file must have two columns (source, target)"
                print("# Topology loaded from", topology_key)
        elif simulation_setup.get("topology", None) is not None:
            topology = simulation_setup["topology"]
        else:
            raise ValueError(
                "LAMBDADYN: topology_wall_K is set, but no topology provided. "
            )
        print(
            f"# LAMBDADYN: using topology restraint with K={wall_K*energy_unit:.2e} {energy_unit_str}/A^2, r0={wall_r0:.2f} A"
        )
        rc = (COV_RADII / us.BOHR)[simulation_setup["species"]]
        bond_src, bond_dst = topology[:, 0], topology[:, 1]
        rcij = rc[bond_src] + rc[bond_dst] + wall_r0
        select_centroid = get_centroid_selector(simulation_setup)

        def topology_restraint_energy(coordinates, alch_lambda):
            coordinates = select_centroid(coordinates)
            lambda_fact = jnp.sin(alch_lambda * jnp.pi) ** 2
            r = jnp.linalg.norm(coordinates[bond_src] - coordinates[bond_dst], axis=1)
            return (
                wall_K
                * lambda_fact
                * jnp.sum(jnp.where(r > rcij, (r - rcij) ** 2, 0.0))
            )

        def lambdadyn_system_bias(dynamics_state):
            alch_lambda = dynamics_state["lambdadyn"]["alch_lambda"]
            coordinates = dynamics_state["coordinates"]
            e, (dedx, dedl) = jax.value_and_grad(
                topology_restraint_energy, argnums=(0, 1)
            )(coordinates, alch_lambda)
            return e, {"coordinates": dedx, "alch_lambda": dedl}

    else:
        lambdadyn_system_bias = None

    fullsamples = lambdadyn_params.get("fullsamples", 200)
    use_cosine_ramp = bool(lambdadyn_params.get("cosine_ramp", False))
    # abf=False keeps accumulating the TI histograms without applying the ABF
    # force, giving plain lambda dynamics (optionally with the OPES bias).
    apply_abf = bool(lambdadyn_params.get("abf", True))

    def update_bias_histograms(alch_lambda, dedlambda, state):
        counts = state["lambda_counts"]
        gradients_avg = state["gradients_avg"]
        ibin = jnp.searchsorted(lambda_edges, alch_lambda, side="right") - 1
        bin_mask = (alch_lambda >= lambda_min) & (alch_lambda <= lambda_max)
        ibin = jnp.clip(ibin, 0, nbins - 1)
        counts = counts.at[ibin].add(bin_mask.astype(counts.dtype), mode="drop")

        curr_grad = gradients_avg.at[ibin].get(mode="fill", fill_value=0.0)
        curr_count = jnp.maximum(
            counts.at[ibin].get(mode="fill", fill_value=1), 1
        )
        grad_update = (dedlambda - curr_grad) / curr_count
        gradients_avg = gradients_avg.at[ibin].add(
            jnp.where(bin_mask, grad_update, 0.0), mode="drop"
        )

        # compute the bias with a linear or cosine ease-in before fullsamples
        bias_mask = curr_count > fullsamples
        if fullsamples == 0:
            ramp = 1.0
        else:
            ramp = bias_mask + curr_count / fullsamples * (1 - bias_mask)
        if use_cosine_ramp:
            ramp = 0.5 * (1 - jnp.cos(jnp.pi * ramp))
        if apply_abf:
            fbias = jnp.where(bin_mask, (curr_grad + grad_update) * ramp, 0.0)
        else:
            fbias = jnp.zeros_like(dedlambda)

        istep = state["istep"] + 1

        new_state = {
            **state,
            "lambda_counts": counts,
            "gradients_avg": gradients_avg,
            "istep": istep,
        }

        # compute the OPES bias
        if not use_opes:
            return fbias, new_state

        # update OPES

        # if istep % pace_opes == 0:
        def update_opes(state):
            opes_count = state["opes_count"] + 1
            new_state = {**state, "opes_count": opes_count}
            if adaptive_opes:
                Vk = state["opes_bias"].at[ibin].get(mode="fill", fill_value=0.0)
                wk = jnp.exp(Vk / kT)
                wksum = state["opes_wk"] + wk
                wksum2 = state["opes_wk2"] + wk * wk
                new_state["opes_wk"] = wksum
                new_state["opes_wk2"] = wksum2
                Neff = wksum * wksum / wksum2
                ascale = (0.75 * Neff) ** (1.0 / 5.0)
                alpha = alpha_opes * ascale * ascale
                norm = norm_opes * ascale
                # jax.debug.print(
                #     "Neff={Neff:.2f}, sigma={sigma:.4f}",
                #     Neff=Neff,
                #     sigma=1./jnp.sqrt(2*alpha),
                # )
            else:
                norm = norm_opes
                alpha = alpha_opes

            dlambda = lambda_centers - alch_lambda
            def compute_kernel(dlambda):
                kernel= norm * jnp.exp(-alpha * dlambda**2)
                dkernel = -2 * alpha * dlambda * kernel
                return kernel, dkernel
            kernel, dkernel = compute_kernel(dlambda)

            # symmetrize about lambda bounds
            dlambda = lambda_centers - (2 * lambda_max - alch_lambda)
            kernel_, dkernel_ = compute_kernel(dlambda)
            kernel = kernel + kernel_
            dkernel = dkernel + dkernel_

            dlambda = lambda_centers - (2 * lambda_min - alch_lambda)
            kernel_, dkernel_ = compute_kernel(dlambda)
            kernel = kernel + kernel_
            dkernel = dkernel + dkernel_

            Pn = state["opes_prob"]
            Pn = Pn + (kernel - Pn) / opes_count
            new_state["opes_prob"] = Pn

            dPn = state["opes_grad"]
            dPn = dPn + (dkernel - dPn) / opes_count
            new_state["opes_grad"] = dPn

            # compute integral of Pn over the lambda range
            Zn = jnp.sum(Pn) * width / (lambda_max - lambda_min)
            # compute the bias
            Vopes = ((gamma_opes - 1) * kT) * jnp.log(Pn / Zn + eps_opes)
            new_state["opes_bias"] = Vopes
            # compute the force
            fopes = (-(gamma_opes - 1) * kT) * (dPn / (Pn + eps_opes * Zn))
            new_state["opes_fbias"] = fopes

            return new_state

        new_state = jax.lax.cond(
            (istep % pace_opes == 0) & bin_mask,
            update_opes,
            lambda x: x,
            new_state,
        )

        fopes = new_state["opes_fbias"]
        fbias = fbias + jnp.where(
            bin_mask, fopes.at[ibin].get(mode="fill", fill_value=0.0), 0.0
        )

        return fbias, new_state

    # initialize counters
    Tseg = lambdadyn_params.get("tseg", 1.0 / us.PS)
    nseg = int(Tseg / dt)
    counter = Counter(nseg)

    ########## INITIALIZE STEP FUNCTION
    def reflect(l, vl):
        """Reflect the lambda and its velocity if it is outside the bounds."""
        alchdown = l < lambda_min
        alchup = l > lambda_max
        alchout = jnp.logical_or(alchup, alchdown)
        vl = vl - 2 * alchout * vl
        l = l + 2 * alchup * (lambda_max - l) + 2 * alchdown * (lambda_min - l)
        return l, vl

    def step(x, v, dynamics_state):
        state = dynamics_state["lambdadyn"]
        alch_lambda = state["alch_lambda"]
        alch_lambda_vel = state["alch_lambda_vel"]

        dedelambda = dynamics_state["energy_gradients"]["alch_elambda"]
        dedvlambda = dynamics_state["energy_gradients"]["alch_vlambda"]

        # calculate the force on lambda using the chain rule
        _, delambda, _, dvlambda = split_lambda(alch_lambda)
        dedlambda = dedelambda * delambda + dedvlambda * dvlambda
        if "alch_lambda" in dynamics_state["energy_gradients"]:
            dedlambda += dynamics_state["energy_gradients"]["alch_lambda"]

        # update histograms and get bias
        bias, new_state = update_bias_histograms(alch_lambda, dedlambda, state)
        # jax.debug.print("lambda={l}, elambda={el}, vlambda={vl}, dedlambda={dl}, bias={b}", l=alch_lambda, el=state["alch_elambda"], vl=state["alch_vlambda"], dl=dedlambda,b=bias)

        # integrate the lambda velocity
        alch_lambda_vel = alch_lambda_vel + (dt / mass_lambda) * (-dedlambda + bias)

        alch_lambda = alch_lambda + (0.5 * dt) * alch_lambda_vel

        # reflect if the lambda is outside the bounds
        alch_lambda, alch_lambda_vel = reflect(alch_lambda, alch_lambda_vel)

        # langevin thermostat
        rng_key, noise_key = jax.random.split(state["rng_key"])
        noise = jax.random.normal(noise_key, dtype=alch_lambda_vel.dtype)
        alch_lambda_vel = a1 * alch_lambda_vel + a2 * noise

        # integrate the lambda position
        alch_lambda = alch_lambda + (0.5 * dt) * alch_lambda_vel

        # reflect if the lambda is outside the bounds
        alch_lambda, alch_lambda_vel = reflect(alch_lambda, alch_lambda_vel)

        alch_elambda, _, alch_vlambda, _ = split_lambda(alch_lambda)

        # update state
        new_state["rng_key"] = rng_key
        new_state["alch_lambda_vel"] = alch_lambda_vel
        new_state["alch_lambda"] = alch_lambda
        new_state["alch_elambda"] = alch_elambda
        new_state["alch_vlambda"] = alch_vlambda

        return x, v, {**dynamics_state, "lambdadyn": new_state}

    def postprocess(state):
        counter.increment()
        if not counter.is_reset_step:
            return state

        lambda_counts = np.array(state["lambda_counts"])
        gradients_avg = np.array(state["gradients_avg"])

        L = lambda_edges
        A = np.zeros_like(L)
        for i in range(1, len(L)):
            A[i] = A[i - 1] + (L[i] - L[i - 1]) * gradients_avg[i - 1]
        A = A - np.min(A)

        deltaG = A[-1] - A[0]

        np.savetxt(
            system_name + ".count",
            np.column_stack((lambda_centers, lambda_counts)),
            header="lambda    count",
        )
        np.savetxt(
            system_name + ".grad",
            np.column_stack((lambda_centers, gradients_avg * energy_unit)),
            header=f"lambda    grad[{energy_unit_str}]",
        )
        np.savetxt(
            system_name + ".pmf",
            np.column_stack((lambda_edges, A * energy_unit)),
            header=f"lambda    pmf[{energy_unit_str}]",
        )
        write_restart_file(state)
        print(f"# LAMBDADYN: data written to {system_name}.{{count,grad,pmf}} files")

        with open(system_name + ".deltaG.traj", "a") as f:
            istep = state["istep"]
            f.write(f"{istep} {deltaG*energy_unit:12.6f}\n")

        if use_opes:
            opes_prob = np.array(state["opes_prob"])
            opes_grad = np.array(state["opes_grad"])
            opes_bias = np.array(state["opes_bias"]) * energy_unit
            opes_fbias = np.array(state["opes_fbias"]) * energy_unit
            np.savetxt(
                system_name + ".opes",
                np.column_stack(
                    (lambda_centers, opes_prob, opes_grad, opes_bias, opes_fbias)
                ),
                header=f"lambda   Pn   dPn  Vn[{energy_unit_str}]  -dVn[{energy_unit_str}]",
            )

        return state

    return step, lambdadyn_system_bias, postprocess, state
