import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import os
from pathlib import Path
import pickle
from functools import partial
from scipy.integrate import cumulative_trapezoid,cumulative_simpson

from .utils import us, get_centroid_selector
from ..utils import Counter, read_tinker_interval
from ..utils.periodic_table import COV_RADII


def initialize_eabf(
    simulation_parameters,
    dt,
    simulation_setup,
    fprec,
    rng_key,
    colvars_calculators={},
    restart_data={},
):
    abf_params = simulation_parameters["abf"]
    select_centroid = get_centroid_selector(simulation_setup)
    kT = simulation_setup["kT"]
    system_name = simulation_setup["name"]

    energy_unit = simulation_setup["energy_unit"]
    energy_unit_str = simulation_setup["energy_unit_str"]

    ### COLVARS
    colvar = abf_params["colvar"]
    assert isinstance(colvar, str), "abf/colvar must be the name of a colvar"
    colvar = colvar.strip()
    assert (
        colvar in colvars_calculators
    ), f"colvar {colvar} not found in colvars_calculators"

    if "optimized_coordinates" in simulation_setup:
        coordinates = simulation_setup["optimized_coordinates"]
    else:
        coordinates = simulation_setup["initial_coordinates"]
    pbc_data = simulation_setup.get("pbc_data")
    cell = None if pbc_data is None else pbc_data["cell"]
    cv = colvars_calculators[colvar](coordinates, cell)

    ### abf PARAMETERS
    # initialize counters
    Tseg = abf_params.get("tseg", 1.0 / us.PS)
    nseg = int(Tseg / dt)
    counter = Counter(nseg)

    sigma_abf = float(abf_params["sigma"])
    assert sigma_abf > 0, "abf/sigma must be positive"
    k_abf = kT / sigma_abf**2

    tau_l = abf_params.get("tau", 100.0 / us.FS)
    assert tau_l > 0, "abf/tau must be positive"

    mass_l = k_abf*(0.5*tau_l / np.pi)**2
    # mass_l = abf_params.get(
    #     "mass", 150000.0 / (us.KCALPERMOL * us.FS**2)
    # )  # default colvars value in Tinker simulations
    # mass_l = mass_l / sigma_abf**2

    gamma_l = abf_params.get("gamma", 1000.0 / us.THZ)

    a1 = math.exp(-gamma_l * dt)
    a2 = ((1 - a1 * a1) * kT / mass_l) ** 0.5

    ### BOUNDS
    bounds = list(abf_params["bounds"])
    assert len(bounds) == 2, "abf/bounds must be a list of two values"
    lmin,lmax = float(bounds[0]), float(bounds[1])
    lmin,lmax = min(lmin,lmax), max(lmin,lmax)
    

    print(
        f"# eABF: bounds for {colvar} = {lmin:.3f} - {lmax:.3f}"
    )
    
    rng_key, v_key = jax.random.split(rng_key)
    vl = jax.random.normal(v_key, dtype=fprec) * (kT / mass_l) ** 0.5
    l = np.clip(cv, lmin, lmax)

    def compute_dedl(l,cv):
        dedl = k_abf * (l - cv)
        return dedl
    
    dedl = compute_dedl(l,cv)

    schedule_keyframes = abf_params.get("schedule_keyframes", None)
    use_keyframes = schedule_keyframes is not None
    if use_keyframes:
        schedule_keyframes = np.array(schedule_keyframes)
        assert all([s > lmin and s < lmax for s in schedule_keyframes]), "abf/schedule_keyframes must be within abf/bounds"

        schedule_intervals = abf_params["schedule_intervals"]
        if isinstance(schedule_intervals, (int,float)):
            schedule_intervals = [float(schedule_intervals)] * len(schedule_keyframes)
        schedule_intervals = np.array(schedule_intervals, dtype=float)
        assert len(schedule_intervals) == len(schedule_keyframes), "abf/schedule_intervals must have the same length as abf/schedule_keyframes"
        schedule_steps = np.cumsum((schedule_intervals / dt).astype(int))

        schedule_sigma = abf_params.get("schedule_sigma", sigma_abf)
        if isinstance(schedule_sigma, (int,float)):
            schedule_sigma = [float(schedule_sigma)] * len(schedule_keyframes)
        schedule_sigma = np.array(schedule_sigma, dtype=float)
        assert len(schedule_sigma) == len(schedule_keyframes), "abf/schedule_sigma must have the same length as abf/schedule_keyframes"
        schedule_k = kT / (2*schedule_sigma**2)

        schedule_keyframes = np.insert(schedule_keyframes,0,l)
        schedule_steps = np.insert(schedule_steps,0,0)
        schedule_k = np.insert(schedule_k,0,0.)
        schedule_keyframes = jnp.array(schedule_keyframes, dtype=fprec)
        schedule_steps = jnp.array(schedule_steps, dtype=jnp.int32)
        schedule_k = jnp.array(schedule_k, dtype=fprec)
        

        def get_restraint_force(l,istep):
            idx = jnp.searchsorted(schedule_steps, istep, side="right")
            k_restraint_down = schedule_k.at[idx-1].get()
            k_restraint_up = schedule_k.at[idx].get(mode="clip")
            l_down = schedule_keyframes.at[idx-1].get()
            l_up = schedule_keyframes.at[idx].get(mode="clip")
            step_down = schedule_steps.at[idx-1].get()
            step_up = schedule_steps.at[idx].get(mode="clip")
            interval = step_up - step_down + 1e-10
            frac = (istep - step_down) / interval
            frac = jnp.clip(frac, 0.0, 1.0)
            k_restraint = k_restraint_down + frac * (k_restraint_up - k_restraint_down)
            l_target = l_down + frac * (l_up - l_down)
            f_restraint = -2 * k_restraint * (l - l_target)
            return f_restraint



    ### INITIALIZE STATE
    state = {}
    state["istep"] = 0
    state["rng_key"] = rng_key
    state["count"] = 1
    state["l"] = jnp.array(l, dtype=fprec)
    state["vl"] = vl
    state["fl"] = -dedl
    state["fbias"] = 0.0

    #### INITIALIZE HISTOGRAMS
    width = float(abf_params["bin_width"])
    assert width > 0, "abf/bin_width must be positive"
    nbins = round((lmax - lmin) / width)
    assert nbins > 0, "nbins must be greater than 0"

    ledges = np.linspace(lmin, lmax, nbins + 1, dtype=fprec)
    lcenters = 0.5 * (ledges[:-1] + ledges[1:])

    state["l_counts"] = jnp.zeros((nbins,), dtype=jnp.int32)
    state["gradients_avg"] = jnp.zeros((nbins,), dtype=fprec)
    state["gradients_var"] = jnp.zeros((nbins,), dtype=fprec)
    state["cv_counts"] = jnp.zeros((nbins,), dtype=jnp.int32)
    state["cv_gradients_avg"] = jnp.zeros((nbins,), dtype=fprec)

    fullsamples = abf_params.get("fullsamples", 500)
    no_abf_bias = fullsamples < 0
    use_cosine_ramp = bool(abf_params.get("cosine_ramp", False))

    block_average = abf_params.get("block_average", None)
    use_block_average = block_average is not None
    if use_block_average:
        block_average = float(block_average)
        assert block_average > 0, "abf/block_average must be positive"
        nblock = int(block_average / dt)
        print(f"# eABF: using block averaging over {block_average:.3f} ps ({nblock} steps)")
        state["block_counts"] = jnp.zeros((nbins,), dtype=jnp.int32)
        state["block_gradients"] = jnp.zeros((nbins,), dtype=fprec)
        state["block_step"] = 0
        state["iblock"] = 0

    ### metadynamics
    height_metad = abf_params.get("metad_height", None)
    use_metad = height_metad is not None  
    if use_metad:
        height_metad = float(height_metad) 
        assert height_metad > 0, "abf/height_metad must be positive"

        pace_metad = int(abf_params.get("metad_pace", 100))
        assert pace_metad > 0, "abf/metad_pace must be positive"

        sigma_metad = float(abf_params.get("metad_sigma", sigma_abf))
        assert sigma_metad > 0, "abf/metad_sigma must be positive"
        alpha_metad = 1.0 / (2 * sigma_metad**2)
        
        print(f"# eABF: adding metadynamics bias with pace {pace_metad} steps, sigma {sigma_metad:.3f}, height {height_metad*us.KCALPERMOL:.3f} kcal/mol")

        barrier_metad = abf_params.get("metad_barrier", None)
        use_welltempered = barrier_metad is not None
        if use_welltempered:
            barrier_metad = float(barrier_metad)
            assert barrier_metad > 0, "abf/metad_barrier must be positive"
            gamma_metad = barrier_metad / kT
        
        switch_metad = bool(abf_params.get("metad_switch", False))
        if switch_metad:
            print(f"# eABF: switching off metadynamics bias when abf bias is fully applied")
        state["metad_count"] = 0
        state["metad_bias"] = jnp.zeros(nbins, dtype=fprec)
        state["metad_grad"] = jnp.zeros(nbins, dtype=fprec)


    ### RESTART
    start_step = 0
    restart_file = simulation_setup["name"] + ".abf.restart"
    do_restart = False
    # Only continue the eABF state when the dynamics itself restarts, so a fresh
    # run in an old directory does not inherit stale counts and gradients.
    if restart_data and os.path.exists(restart_file):
        with open(restart_file, "rb") as f:
            abf_restart = pickle.load(f)

        do_restart = (
            abf_restart["centers"].shape[0] == nbins
            and np.allclose(abf_restart["centers"], lcenters)
            and abf_restart["l_counts"].shape[0] == nbins
            and abf_restart["gradients_avg"].shape[0] == nbins
        )

        if do_restart:
            print("# eABF: loading previous counts and gradients")
            state["l_counts"] = jnp.array(abf_restart["l_counts"])
            state["gradients_avg"] = jnp.array(abf_restart["gradients_avg"])
            if "gradients_var" in abf_restart:
                state["gradients_var"] = jnp.array(abf_restart["gradients_var"])
            state["cv_counts"] = jnp.array(abf_restart["cv_counts"])
            state["cv_gradients_avg"] = jnp.array(abf_restart["cv_gradients_avg"])
            if use_block_average:
                for key in (
                    "block_counts",
                    "block_gradients",
                    "block_step",
                    "iblock",
                ):
                    if key in abf_restart:
                        state[key] = jnp.asarray(abf_restart[key])
            start_step = int(abf_restart["istep"])
            if use_metad:
                state["metad_count"] = int(abf_restart["metad_count"])
                state["metad_bias"] = jnp.array(abf_restart["metad_bias"])
                state["metad_grad"] = jnp.array(abf_restart["metad_grad"])
            state["istep"] = start_step
            for key in ("l", "vl", "rng_key", "fl", "fbias", "cv"):
                if key in abf_restart:
                    state[key] = jnp.asarray(abf_restart[key])
        else:
            print(
                f"# eABF: Could not restart from {restart_file}. Starting from scratch."
            )
    elif restart_data:
        print(
            f"# eABF: warning: dynamics is restarting but {restart_file} "
            "is missing. Starting eABF from scratch."
        )

    def write_restart_file(state):
        abf_restart = {
            "istep": int(state["istep"]),
            "centers": lcenters,
            "l_counts": np.array(state["l_counts"]),
            "gradients_avg": np.array(state["gradients_avg"]),
            "gradients_var": np.array(state["gradients_var"]),
            "cv_counts": np.array(state["cv_counts"]),
            "cv_gradients_avg": np.array(state["cv_gradients_avg"]),
            "l": np.array(state["l"]),
            "vl": np.array(state["vl"]),
            "rng_key": np.array(state["rng_key"]),
            "fl": np.array(state["fl"]),
            "fbias": np.array(state["fbias"]),
            "cv": np.array(state.get("cv", cv)),
        }
        if use_block_average:
            abf_restart["block_counts"] = np.array(state["block_counts"])
            abf_restart["block_gradients"] = np.array(state["block_gradients"])
            abf_restart["block_step"] = int(state["block_step"])
            abf_restart["iblock"] = int(state["iblock"])
        if use_metad:
            abf_restart["metad_count"] = int(state["metad_count"])
            abf_restart["metad_bias"] = np.array(state["metad_bias"])
            abf_restart["metad_grad"] = np.array(state["metad_grad"])

        with open(restart_file, "wb") as f:
            pickle.dump(abf_restart, f)
    
    

    def update_bias_histograms(cv, l, dedl, state):
        counts = state["l_counts"]
        gradients_avg = state["gradients_avg"]

        cv_counts = state["cv_counts"]
        cv_gradients_avg = state["cv_gradients_avg"]

        # STORE COUNTS FOR CV
        cv_ibin = jnp.searchsorted(ledges, cv, side="right") - 1
        cv_bin_mask = (cv >= lmin) & (cv <= lmax)
        cv_bin = jnp.clip(cv_ibin, 0, nbins - 1)
        cv_counts = cv_counts.at[cv_bin].add(cv_bin_mask.astype(cv_counts.dtype))

        curr_grad = cv_gradients_avg.at[cv_bin].get()
        curr_count = jnp.maximum(cv_counts.at[cv_bin].get(), 1)
        grad_update = (dedl - curr_grad) / curr_count
        cv_gradients_avg = cv_gradients_avg.at[cv_bin].add(jnp.where(cv_bin_mask, grad_update, 0.0))

        ## BLOCK AVERAGING
        if use_block_average:
            block_counts = state["block_counts"]
            block_gradients = state["block_gradients"]
            block_step = state["block_step"]

            block_counts = block_counts.at[cv_bin].add(cv_bin_mask.astype(block_counts.dtype))
            curr_count = jnp.maximum(block_counts.at[cv_bin].get(), 1)
            curr_grad = block_gradients.at[cv_bin].get()
            grad_update = (dedl - curr_grad) / curr_count
            block_gradients = block_gradients.at[cv_bin].add(jnp.where(cv_bin_mask, grad_update, 0.0))
            block_step += 1

        # STORE COUNTS FOR L
        ibin = jnp.searchsorted(ledges, l, side="right") - 1
        bin_mask = (l >= lmin) & (l <= lmax)
        ibin = jnp.clip(ibin, 0, nbins - 1)
        counts = counts.at[ibin].add(bin_mask.astype(counts.dtype))

        curr_grad = gradients_avg.at[ibin].get()
        curr_count = jnp.maximum(counts.at[ibin].get(), 1)
        grad_delta = (dedl - curr_grad) # / curr_count
        applied_grad_delta = grad_delta / curr_count
        gradients_avg = gradients_avg.at[ibin].add(jnp.where(bin_mask, applied_grad_delta, 0.0))

        new_grad = curr_grad + applied_grad_delta
        dgrad2 = (dedl - new_grad)*grad_delta
        curr_var = state["gradients_var"].at[ibin].get(mode="fill", fill_value=0.0)
        var_update = (dgrad2 - curr_var) / curr_count
        gradients_var = state["gradients_var"].at[ibin].add(jnp.where(bin_mask, var_update, 0.0))


        # compute the bias with a linear or cosine ease-in before fullsamples
        if no_abf_bias:
            fbias = 0.
            ramp = 0.
        else:
            fbias = curr_grad + applied_grad_delta
            bias_mask = curr_count > fullsamples
            ramp = jnp.where(fullsamples == 0, 1.0, bias_mask + curr_count / fullsamples * (1 - bias_mask))
            if use_cosine_ramp:
                ramp = 0.5 * (1 - jnp.cos(jnp.pi * ramp))
            fbias = fbias * bin_mask  * ramp

        istep = state["istep"] + 1

        if use_keyframes:
            f_restraint = get_restraint_force(l, istep)
            fbias = fbias + f_restraint

        new_state = {
            **state,
            "l_counts": counts,
            "gradients_avg": gradients_avg,
            "gradients_var": gradients_var,
            "cv_counts": cv_counts,
            "cv_gradients_avg": cv_gradients_avg,
            "istep": istep,
        }

        if use_block_average:
            new_state["block_counts"] = block_counts
            new_state["block_gradients"] = block_gradients
            new_state["block_step"] = block_step

        if not use_metad:
            return fbias, new_state
        
        # metadynamics update
        def update_metad(state):
            metad_count = state["metad_count"] + 1
            new_state = {**state, "metad_count": metad_count}

            height = height_metad
            if use_welltempered:
                height *= jnp.exp(-state["metad_bias"]/(kT * (gamma_metad - 1)))

            def compute_kernel(dl):
                kernel= height * jnp.exp(-alpha_metad * dl**2)
                dkernel = -2 * alpha_metad * dl * kernel
                return kernel, dkernel
            
            dl = lcenters - l
            kernel, dkernel = compute_kernel(dl)

            # symmetrize about bounds
            dl = lcenters - (2 * lmax - l)
            kernel_, dkernel_ = compute_kernel(dl)
            kernel = kernel + kernel_
            dkernel = dkernel + dkernel_

            dl = lcenters - (2 * lmin - l)
            kernel_, dkernel_ = compute_kernel(dl)
            kernel = kernel + kernel_
            dkernel = dkernel + dkernel_

            new_state["metad_bias"] = state["metad_bias"] + kernel
            new_state["metad_grad"] = state["metad_grad"] + dkernel

            return new_state

        new_state = jax.lax.cond(
            istep % pace_metad == 0,
            update_metad,
            lambda x: x,
            new_state,
        )

        mbias = - new_state["metad_grad"].at[ibin].get(mode="fill", fill_value=0.0)
        if switch_metad:
            mbias = mbias * (1.-ramp)
        fbias = fbias + mbias
        return fbias, new_state

    ########## INITIALIZE STEP FUNCTION
    def reflect(l, vl):
        """Reflect the lambda and its velocity if it is outside the bounds."""
        down = l < lmin
        up = l > lmax
        out = jnp.logical_or(up, down)
        vl = vl - 2 * out * vl
        l = l + 2 * up * (lmax - l) + 2 * down * (lmin - l)
        return l, vl
    
    @jax.jit
    def step(x, v, dynamics_state):
        state = dynamics_state["abf"]

        l = state["l"]
        vl = state["vl"]
        cv = dynamics_state["colvars"][colvar]

        dedl = compute_dedl(l,cv)
        # update histograms and get bias
        bias, new_state = update_bias_histograms(cv,l, dedl, state)

        vl = vl + (dt/mass_l) * (-dedl + bias)
        l = l + (0.5*dt)*vl
        l, vl = reflect(l, vl)

        rng_key,noise_key = jax.random.split(state["rng_key"])
        noise = jax.random.normal(noise_key, dtype=vl.dtype)
        vl = a1 * vl + a2 * noise

        l = l + (0.5*dt)*vl
        l, vl = reflect(l, vl)

        new_state["rng_key"] = rng_key
        new_state["l"] = l
        new_state["cv"] = cv
        new_state["vl"] = vl
        new_state["fl"] = -dedl
        new_state["fbias"] = bias

        return x, v, {**dynamics_state, "abf": new_state}
    
    @jax.jit
    def abf_system_bias(dynamics_state):
        x = dynamics_state["coordinates"]
        cell = dynamics_state["cell"] if "cell" in dynamics_state else None
        state = dynamics_state["abf"]
        l = state["l"]

        def bias_energy(x):
            colvar_coordinates = select_centroid(x)
            cv = colvars_calculators[colvar](colvar_coordinates,cell)
            e = 0.5 * k_abf * (cv - l)**2 
            return e

        e, dedx = jax.value_and_grad(bias_energy)(x)
        de = {"coordinates": dedx}
        return e, de

    # TRAJ FILE
    abf_traj_file = system_name + ".abf.traj"
    if not do_restart or not os.path.exists(abf_traj_file):
        with open(abf_traj_file, "w") as f:
            f.write(
                f"#{'time[ps]':>15} {'cv':>15} {'l':>15}  {'fl':>15} {'fbias':>15} \n"
            )

    def write_traj_file(state):
        with open(abf_traj_file, "a") as f:
            f.write(
                f"{state['istep']*dt:15.6e} {state['cv']:15.6e} {state['l']:15.6e} {state['fl']:15.6e} {state['fbias']:15.6e}\n"
            )

    def postprocess(state):
        counter.increment()
        if not counter.is_reset_step:
            return state
        
        write_traj_file(state)

        # PMF ESTIMATOR for extended variable L
        l_counts = np.array(state["l_counts"])
        gradients_avg = np.array(state["gradients_avg"])
        gradients_std = np.sqrt(np.maximum(np.array(state["gradients_var"]), 0.0))
        A = cumulative_simpson(gradients_avg, x=lcenters, initial=0)
        A = A - np.min(A)

        # CZAR estimator (unbiased PMF)
        cv_counts = np.maximum(np.array(state["cv_counts"]), 1)
        Acount = -kT*np.log(cv_counts)
        Acount -= Acount[0]
        cv_gradients_avg = np.array(state["cv_gradients_avg"])
        Aczar = cumulative_simpson(cv_gradients_avg, x=lcenters, initial=0) + Acount
        Aczar = Aczar - np.min(Aczar)

        
        np.savetxt(
            system_name + ".count",
            np.column_stack((lcenters, l_counts)),
            header="lambda    count",
        )
        np.savetxt(
            system_name + ".zcount",
            np.column_stack((lcenters, cv_counts)),
            header="cv    count",
        )
        np.savetxt(
            system_name + ".grad",
            np.column_stack((lcenters, gradients_avg * energy_unit)),
            header=f"lambda    grad[{energy_unit_str}]",
        )
        np.savetxt(
            system_name + ".gradstd",
            np.column_stack((lcenters, gradients_std * energy_unit)),
            header=f"lambda    gradstd[{energy_unit_str}]",
        )
        np.savetxt(
            system_name + ".pmf",
            np.column_stack((lcenters, A * energy_unit)),
            header=f"lambda    pmf[{energy_unit_str}]",
        )
        np.savetxt(
            system_name + ".czar.pmf",
            np.column_stack((lcenters, Aczar * energy_unit)),
            header=f"lambda    pmf[{energy_unit_str}]",
        )

        if use_metad:
            metad_grad = np.array(state["metad_grad"]) * energy_unit
            metad_bias = np.array(state["metad_bias"]) * energy_unit
            metad_bias -= np.min(metad_bias)
            np.savetxt(
                system_name + ".metad",
                np.column_stack(
                    (lcenters, metad_bias, metad_grad)
                ),
                header=f"lambda   Pn   dPn  Vn[{energy_unit_str}]  -dVn[{energy_unit_str}]",
            )

        # block averaging
        if use_block_average:
            # CZAR estimator (unbiased PMF)
            cv_counts = np.maximum(np.array(state["block_counts"]), 1)
            Acount = -kT*np.log(cv_counts)
            Acount -= Acount[0]
            cv_gradients_avg = np.array(state["block_gradients"])
            Aczar = cumulative_simpson(cv_gradients_avg, x=lcenters, initial=0) + Acount
            Aczar = Aczar - np.min(Aczar)

            iblock = int(state["iblock"])
            np.savetxt(
                system_name + f".block{iblock}.czar.pmf",
                np.column_stack((lcenters, Aczar * energy_unit)),
                header=f"lambda    pmf[{energy_unit_str}]",
            )
            np.savetxt(
                system_name + f".block{iblock}.zcount",
                np.column_stack((lcenters, cv_counts)),
                header="cv    count",
            )
            block_step = int(state["block_step"])
            if block_step >= nblock:
                print(f"# eABF: completed block averaging {iblock} (steps {state['istep'] - block_step} to {state['istep']})")
                state["block_counts"] = jnp.zeros((nbins,), dtype=jnp.int32)
                state["block_gradients"] = jnp.zeros((nbins,), dtype=fprec)
                state["block_step"] = 0
                state["iblock"] = iblock + 1

        # Save after block-boundary resets so a continuation resumes the same
        # partial block state as an uninterrupted run.
        write_restart_file(state)
        return state

    return step, abf_system_bias, postprocess, state
