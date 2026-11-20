import time
import math
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from .thermostats import get_thermostat
from .barostats import get_barostat
from .colvars import setup_colvars
from .lambdadyn import initialize_lambdadyn
from .opes import initialize_opes
from .eabf import initialize_eabf
from .spectra import initialize_ir_spectrum

from .utils import load_dynamics_restart, get_restart_file, optimize_fire2, us
from .initial import load_model, load_simulation_setup, initialize_preprocessing
from ..utils import detect_topology, find_molecules


def _validate_replica_colvars(nreplicas, colvars_definitions, nbeads=None):
    if nreplicas > 1 and nbeads is None and colvars_definitions is not None:
        raise ValueError(
            "Collective variables are not supported with independent replicas"
        )


def _validate_replica_lambdadyn(nreplicas, lambdadyn_params, nbeads=None):
    if nreplicas > 1 and nbeads is None and lambdadyn_params is not None:
        raise ValueError("lambdadyn is not supported with independent replicas")


def initialize_dynamics(simulation_parameters, fprec, rng_key):
    """Build the Simulation and its initial state.

    Returns ``(step, update_batch, simulation_setup, step_state, batch, dynamics_state)``:

    - ``simulation_setup``: fixed for the whole Simulation, whether read from the
      input or derived here (time step, thermostat name, variable cell, ...).
    - ``dynamics_state``: the Dynamics state passed through jitted functions
      (coordinates, velocities, forces, thermostat and barostat state, ...).
    - ``step_state``: mutable state that ``step`` updates outside ``jit``: step
      counter, preprocessing state, neighbor-list countdown, IR spectrum state, and
      the thermostat post-step state, whose large arrays would add overhead to
      jitted calls.
    - ``batch``: the Model input for the current coordinates.

    ``step(istep, step_state, dynamics_state, batch)`` advances the Simulation by
    one step and returns ``(step_state, dynamics_state, batch, model_output)``.
    """
    ### LOAD MODEL
    model = load_model(simulation_parameters)
    model_energy_unit = us.get_multiplier(model.energy_unit)

    ### Get the coordinates and species from the xyz file
    simulation_setup, batch = load_simulation_setup(simulation_parameters, fprec)
    simulation_setup["model_energy_unit"] = model_energy_unit
    simulation_setup["model_energy_unit_str"] = model.energy_unit

    ### FINISH BUILDING batch
    do_restart = os.path.exists(get_restart_file(simulation_setup))
    if do_restart:
        ### RESTART FROM PREVIOUS DYNAMICS
        restart_data = load_dynamics_restart(simulation_setup)
        print("# RESTARTING FROM PREVIOUS DYNAMICS")
        model.preproc_state = restart_data["preproc_state"]
        batch["coordinates"] = restart_data["coordinates"]
        if "cells" in restart_data:
            batch["cells"] = jnp.asarray(restart_data["cells"], dtype=fprec)
            batch["reciprocal_cells"] = jnp.linalg.inv(
                batch["cells"]
            )
    else:
        restart_data = {}

    ## GET FROZEN ATOMS
    nat = simulation_setup["nat"]
    nreplicas = simulation_setup["nreplicas"]
    frozen_atoms = simulation_setup["frozen_atoms"]
    if frozen_atoms is not None:
        freeze_mask = np.ones((nat, 1), dtype=bool)
        freeze_mask[frozen_atoms] = False
        freeze_mask_replicas = np.tile(freeze_mask, (nreplicas, 1))

    fix_com = simulation_parameters.get("fix_com", False)
    """@keyword[fennol_md] fix_com
    Recenter the coordinates on the centre of mass at start and after every
    step. Only positions are shifted: velocities, and hence the thermostat and
    the degree-of-freedom count, are unchanged (see remove_com). Valid only
    for translationally invariant forces.
    Default: False
    """
    if fix_com and frozen_atoms is not None:
        # Recentering would move the frozen atoms with the rest of the system.
        raise ValueError("fix_com is not compatible with frozen atoms")
    if fix_com and simulation_setup.get("thermal_com", False):
        print("# Warning: fix_com recenters positions while the centre-of-mass velocity is kept (remove_com: false)")

    ### INITIALIZE PREPROCESSING
    preproc_state, batch = initialize_preprocessing(
        simulation_parameters, model, batch, simulation_setup
    )

    minimize = simulation_parameters.get("xyz_input/minimize", False)
    """@keyword[fennol_md] xyz_input/minimize
    Perform energy minimization before dynamics.
    Default: False
    """
    if minimize and not do_restart:
        # assert (
        #     simulation_setup["nreplicas"] == 1
        # ), "Minimization is only supported for single replica systems"
        model.preproc_state = preproc_state
        convert = us.KCALPERMOL / model_energy_unit

        def energy_force_fn(coordinates):
            inputs = {**batch, "coordinates": coordinates}
            e, f, _ = model.energy_and_forces(**inputs, gpu_preprocessing=True)
            e = np.mean(np.array(e)) * convert / nat
            f = np.array(f) * convert
            if frozen_atoms is not None:
                f = (f.reshape(-1,nat,3) * freeze_mask[None,:,:]).reshape(-1,3)
            return e, f

        tol = (
            simulation_parameters.get("xyz_input/minimize_ftol", 1e-1 / us.KCALPERMOL)
            * us.KCALPERMOL
        )
        """@keyword[fennol_md] xyz_input/minimize_ftol
        Force tolerance for minimization.
        Default: 0.1 kcal/mol/Å
        """
        print(
            f"# Minimizing initial configuration with RMS force tolerance = {tol:.1e} kcal/mol/A"
        )
        batch["coordinates"], success = optimize_fire2(
            batch["coordinates"],
            energy_force_fn,
            atol=tol,
            max_disp=0.02,
        )
        if success:
            print("# Minimization successful")
        else:
            print("# Warning: Minimization failed, continuing with last configuration")
        # write the minimized coordinates as an xyz file
        from ..utils.io import write_xyz_frame

        with open(simulation_setup["name"] + ".opt.xyz", "w") as f:
            write_xyz_frame(
                f,
                simulation_setup["symbols"],
                np.array(batch["coordinates"]).reshape(-1, nat, 3)[0],
                cell=batch.get("cells", [None])[0],
            )
        print("# Minimized configuration written to", simulation_setup["name"] + ".opt.xyz")
        preproc_state = model.preproc_state
        batch = model.preprocessing.process(preproc_state, batch)
        simulation_setup["optimized_coordinates"] = np.array(
            batch["coordinates"]
        ).copy()

        ### update topology if needed
        topology_key = simulation_parameters.get("topology", None)
        if topology_key is not None and str(topology_key).lower() == "detect":
            cell = (
                simulation_setup["pbc"]["cell"] if simulation_setup["pbc"] is not None else None
            )
            topology = detect_topology(
                simulation_setup["species"], simulation_setup["optimized_coordinates"], cell=cell
            )
            topo_file = simulation_setup["name"] + ".topo"
            np.savetxt(topo_file, topology + 1, fmt="%d")
            print("# Updated detected topology after geometry optimization.")
            molecule_indices = find_molecules(nat, topology[:, 0], topology[:, 1])
            simulation_setup["topology"] = topology
            simulation_setup["molecule_indices"] = molecule_indices

    ### get dynamics parameters
    dt = simulation_parameters.get("dt")
    """@keyword[fennol_md] dt
    Integration time step. Required parameter.
    Type: float, Required
    """
    dt2 = 0.5 * dt
    mass = simulation_setup["mass"]
    densmass = simulation_setup["totmass_Da"] * (us.MOL / us.CM**3)
    dtm = jnp.asarray(dt / mass[:, None], dtype=fprec)
    ek_avg = 0.5 * nat * simulation_setup["kT"] * np.eye(3)

    nbeads = simulation_setup.get("nbeads", None)
    if nbeads is not None:
        dtm = dtm[None, :, :]
    
    if fix_com:
        massprop = simulation_setup["mass_Da"]/simulation_setup["totmass_Da"]
        print("# Reference frame fixed to center of mass")
        x = batch["coordinates"].reshape(-1, nat, 3)
        if nbeads is None:
            xcom = jnp.sum(x*massprop[None,:,None], axis=1, keepdims=True)
        else:
            xcom = jnp.sum(x*massprop[None,:,None], axis=(0,1), keepdims=True)/nbeads
        batch["coordinates"] = (x - xcom).reshape(-1, 3)

    ### INITIALIZE DYNAMICS STATE
    dynamics_state = {"coordinates": batch["coordinates"], "istep": restart_data.get("integration_step", 0)}
    simulation_setup["dt"] = dt
    simulation_setup["pimd"] = nbeads is not None
    simulation_setup["start_time_ps"] = restart_data.get("simulation_time_ps", 0.0)
    step_state = {"istep": 0, "preproc_state": preproc_state}
    forces_key = simulation_parameters.get("forces_key", None)
    """@keyword[fennol_md] forces_key
    Key for forces calculation. If not specified, forces are computed from gradient of total energy.
    """
    if forces_key is not None:
        gradient_keys = []
        print("# Using custom forces_key:", forces_key)
    else:
        gradient_keys = ["coordinates"]
    middle_updates = []
    outer_updates = []
    bias_terms = {}

    ### INITIALIZE THERMOSTAT
    thermostat_rng, rng_key = jax.random.split(rng_key)
    (
        thermostat,
        thermostat_post,
        thermostat_state,
        initial_vel,
        simulation_setup["thermostat_name"],
        ekin_instant,
    ) = get_thermostat(simulation_parameters, dt, simulation_setup, fprec, thermostat_rng,restart_data)
    do_thermostat_post = thermostat_post is not None
    if do_thermostat_post:
        thermostat_post, post_state = thermostat_post
        step_state["thermostat_post_state"] = post_state

    assert ekin_instant in ["B", "O"], "ekin_instant must be 'B' or 'O'"

    dynamics_state["thermostat"] = thermostat_state

    if "initial_velocities" in simulation_parameters:
        initial_vel_file = simulation_parameters["initial_velocities"]
        print(f"# Using initial velocities from {initial_vel_file}")
        v0 = jnp.array(np.loadtxt(initial_vel_file).reshape(-1,nat, 3).astype(fprec))
        if v0.shape[0] == 1:
            if nbeads is None:
                v0 = v0.repeat(nreplicas, axis=0)
            else:
                v0 = initial_vel.at[0].set(v0[0])
        assert v0.shape == (nreplicas, nat, 3), f"initial_velocities file must have shape (nat, 3) or (nreplicas, nat, 3), but got {v0.shape}"
        initial_vel = v0
        if nbeads is None:
            initial_vel = initial_vel.reshape(-1, 3)

    dynamics_state["vel"] = restart_data.get("vel", initial_vel).astype(fprec)

    ### PBC
    pbc_data = simulation_setup.get("pbc", None)
    if pbc_data is not None:
        ### INITIALIZE BAROSTAT
        barostat_key, rng_key = jax.random.split(rng_key)
        thermo_update_ensemble, variable_cell, barostat_state, pressure_instant = (
            get_barostat(
                thermostat,
                simulation_parameters,
                dt,
                simulation_setup,
                fprec,
                barostat_key,
                restart_data,
            )
        )
        estimate_pressure = variable_cell or pbc_data["estimate_pressure"]
        dynamics_state["barostat"] = barostat_state
        dynamics_state["cell"] = batch["cells"][0]
        if estimate_pressure:
            if forces_key is not None:
                raise ValueError("Pressure not available with custom forces_key")
            gradient_keys.append("strain")

        print("# Estimate pressure: ", estimate_pressure)
    else:
        estimate_pressure = False
        variable_cell = False

        def thermo_update_ensemble(x, v, dynamics_state):
            v, thermostat_state = thermostat(v, dynamics_state["thermostat"])
            return x, v, {**dynamics_state, "thermostat": thermostat_state}

    simulation_setup["estimate_pressure"] = estimate_pressure
    simulation_setup["variable_cell"] = variable_cell
    middle_updates.append(thermo_update_ensemble)

    if estimate_pressure:
        use_average_Pkin = simulation_parameters.get("use_average_Pkin", False)
        """@keyword[fennol_md] use_average_Pkin
        Use time-averaged kinetic energy for pressure estimation instead of instantaneous values.
        Default: False
        """
        is_qtb = simulation_setup["thermostat_name"].endswith("QTB")
        if is_qtb and use_average_Pkin:
            raise ValueError(
                "use_average_Pkin is not compatible with QTB thermostat, please set use_average_Pkin to False"
            )

    ### ENERGY ENSEMBLE
    ensemble_key = simulation_parameters.get("etot_ensemble_key", None)
    """@keyword[fennol_md] etot_ensemble_key
    Key for energy ensemble calculation. Enables computation of ensemble weights.
    Default: None
    """

    ### COLVARS
    colvars_definitions = simulation_parameters.get("colvars", None)
    """@keyword[fennol_md] colvars
    Collective variables definitions for enhanced sampling or monitoring.
    Default: None
    """
    use_colvars = colvars_definitions is not None
    if use_colvars:
        _validate_replica_colvars(nreplicas, colvars_definitions, nbeads=nbeads)
        colvars_calculators, colvars_names = setup_colvars(colvars_definitions)
        simulation_setup["colvars"] = colvars_names
    else:
        colvars_calculators = {}

    ### IR SPECTRUM
    do_ir_spectrum = simulation_parameters.get("ir_spectrum", False)
    """@keyword[fennol_md] ir_spectrum
    Calculate infrared spectrum from molecular dipole moment time series.
    Default: False
    """
    assert isinstance(do_ir_spectrum, bool), "ir_spectrum must be a boolean"
    if do_ir_spectrum:
        is_qtb = simulation_setup["thermostat_name"].endswith("QTB")
        model_ir, ir_state, save_dipole, ir_post = initialize_ir_spectrum(
            simulation_parameters, simulation_setup, fprec, dt, is_qtb
        )
        step_state["ir_spectrum"] = ir_state

    ### LAMBDADYN
    lambdadyn_params = simulation_parameters.get("lambdadyn", None)
    use_lambdadyn = lambdadyn_params is not None
    if use_lambdadyn:
        if forces_key is not None:
            raise ValueError("lambdadyn is not compatible with custom forces_key")
        assert isinstance(lambdadyn_params, dict), "lambdadyn must be a dictionary"
        _validate_replica_lambdadyn(
            nreplicas, lambdadyn_params, nbeads=nbeads
        )
        lambdadyn_key, rng_key = jax.random.split(rng_key)
        lambdadyn_step, lambdadyn_bias, lambdadyn_post, lambdadyn_state = (
            initialize_lambdadyn(
                simulation_parameters,
                dt,
                simulation_setup,
                fprec,
                lambdadyn_key,
                restart_data,
            )
        )
        outer_updates.append(lambdadyn_step)
        bias_terms["alch"] = lambdadyn_bias
        dynamics_state["lambdadyn"] = lambdadyn_state
        batch["alch_ligand_charge"] = lambdadyn_state["alch_ligand_charge"]
        batch["alch_group"] = lambdadyn_state["alch_group"]
        gradient_keys = gradient_keys + ["alch_elambda", "alch_vlambda"]

    ### OPES
    opes_params = simulation_parameters.get("opes", None)
    use_opes = opes_params is not None
    if use_opes:
        assert use_colvars, "opes requires colvars to be defined"
        assert isinstance(opes_params, dict), "opes must be a dictionary"
        opes_key, rng_key = jax.random.split(rng_key)
        opes_step, opes_bias, opes_post, opes_state = initialize_opes(
            simulation_parameters,
            dt,
            simulation_setup,
            fprec,
            opes_key,
            colvars_calculators,
            restart_data,
        )
        outer_updates.append(opes_step)
        bias_terms["opes"] = opes_bias
        dynamics_state["opes"] = opes_state
    
    ### ABF
    abf_params = simulation_parameters.get("abf", None)
    use_abf = abf_params is not None
    if use_abf:
        assert use_colvars, "abf requires colvars to be defined"
        assert isinstance(abf_params, dict), "abf must be a dictionary"
        abf_key, rng_key = jax.random.split(rng_key)
        abf_step, abf_bias, abf_post, abf_state = initialize_eabf(
            simulation_parameters,
            dt,
            simulation_setup,
            fprec,
            abf_key,
            colvars_calculators,
            restart_data,
        )
        outer_updates.append(abf_step)
        bias_terms["abf"] = abf_bias
        dynamics_state["abf"] = abf_state

    ### RESTRAINTS
    restraints_params = simulation_parameters.get("restraints", None)
    use_restraints = restraints_params is not None
    if use_restraints:
        assert isinstance(restraints_params, dict), "restraints must be a dictionary"
        from .restraints import initialize_restraints

        restraints = initialize_restraints(
            simulation_parameters, simulation_setup, fprec, colvars_calculators
        )
        for k, v in restraints.items():
            bias_terms[k] = v

    ### BUILD GRADIENT FUNCTION
    if gradient_keys == []:
        def energy_and_gradient(variables, batch):
            e, out = model._total_energy(variables, batch)
            de = {}
            if forces_key is not None:
                f = out[forces_key]
                if f.ndim == 3:
                    f=f.mean(axis=1)
                batch_index = batch["batch_index"]
                nbatch = batch["natoms"].shape[0]
                fsum = jax.ops.segment_sum(f, batch_index, num_segments=nbatch)
                fsum = fsum / batch["natoms"][:, None]
                f = f - fsum[batch_index]
                de["coordinates"] = -f
                
            return e, de, out
    else:
        if forces_key is not None:
            raise ValueError("Custom forces_key is not compatible with explicit gradients, please set forces_key to None")
        energy_and_gradient = model.get_gradient_function(
            *gradient_keys, jit=True, variables_as_input=True
        )

    ### COLLECT UPDATES
    if len(middle_updates) == 1:
        middle_update = middle_updates[0]
    else:

        def middle_update(x, v, dynamics_state):
            for update in middle_updates:
                x, v, dynamics_state = update(x, v, dynamics_state)
            return x, v, dynamics_state

    if len(outer_updates) == 1:
        outer_update = outer_updates[0]
    else:

        def outer_update(x, v, dynamics_state):
            for update in outer_updates:
                x, v, dynamics_state = update(x, v, dynamics_state)
            return x, v, dynamics_state

    ### cleanup biases
    bias_terms = {k: v for k, v in bias_terms.items() if v is not None}
    use_biases = len(bias_terms.keys()) > 0
    if use_biases:
        print("# Using bias terms in dynamics")

    ### RING POLYMER INITIALIZATION
    if nbeads is not None:
        cay_correction = simulation_parameters.get("cay_correction", True)
        """@keyword[fennol_md] cay_correction
        Use Cayley propagator for ring polymer molecular dynamics instead of standard propagation.
        Default: True
        """
        omk = simulation_setup["omk"]
        eigmat = simulation_setup["eigmat"]
        cayfact = 1.0 / (4.0 + (dt * omk[1:, None, None]) ** 2) ** 0.5
        if cay_correction:
            axx = jnp.asarray(2 * cayfact)
            axv = jnp.asarray(dt * cayfact)
            avx = jnp.asarray(-dt * cayfact * omk[1:, None, None] ** 2)
        else:
            axx = jnp.asarray(np.cos(omk[1:, None, None] * dt2))
            axv = jnp.asarray(np.sin(omk[1:, None, None] * dt2) / omk[1:, None, None])
            avx = jnp.asarray(-omk[1:, None, None] * np.sin(omk[1:, None, None] * dt2))

        # coordinates = batch["coordinates"].reshape(nbeads, -1, 3)
        # eigx = jnp.zeros_like(coordinates).at[0].set(coordinates[0])
        eigx = jnp.einsum(
            "in,i...->n...", eigmat, batch["coordinates"].reshape(nbeads, -1, 3)
        ) * (1.0 / nbeads**0.5)
        dynamics_state["coordinates"] = eigx

    ### Key to keep in input for preprocessing
    input_keys_to_keep = simulation_parameters.get("input_keys_to_keep", None)
    """@keyword[fennol_md] input_keys_to_keep
    List of additional keys from simulation_setup to keep in the input batch for preprocessing.
    Default: None
    """

    ###############################################
    ### DEFINE UPDATE FUNCTION
    @jax.jit
    def update_batch(batch, dynamics_state):
        x = dynamics_state["coordinates"]
        if nbeads is not None:
            x = jnp.einsum("in,n...->i...", eigmat, x).reshape(nbeads * nat, 3) * (
                nbeads**0.5
            )
        batch = {**batch, "coordinates": x}
        if variable_cell:
            batch["cells"] = dynamics_state["cell"][None, :, :].repeat(nreplicas, axis=0)
            batch["reciprocal_cells"] = jnp.linalg.inv(dynamics_state["cell"])[
                None, :, :
            ].repeat(nreplicas, axis=0)

        if use_lambdadyn:
            batch["alch_lambda"] = dynamics_state["lambdadyn"]["alch_lambda"]
            batch["alch_elambda"] = dynamics_state["lambdadyn"]["alch_elambda"]
            batch["alch_vlambda"] = dynamics_state["lambdadyn"]["alch_vlambda"]

        return batch

    ###############################################
    ### DEFINE INTEGRATION FUNCTIONS
    def integrate_A_half(x0, v0):
        if nbeads is None:
            if frozen_atoms is not None:
                v0 = v0 * freeze_mask_replicas
            return x0 + dt2 * v0, v0

        # update coordinates and velocities of a free ring polymer for a half time step
        eigv_c = v0[0]
        if frozen_atoms is not None:
            eigv_c = eigv_c * freeze_mask
        eigx_c = x0[0] + dt2 * eigv_c
        eigx = x0[1:] * axx + v0[1:] * axv
        eigv = x0[1:] * avx + v0[1:] * axx

        return (
            jnp.concatenate((eigx_c[None], eigx), axis=0),
            jnp.concatenate((eigv_c[None], eigv), axis=0),
        )

    @jax.jit
    def integrate(dynamics_state):
        x, v, dynamics_state = outer_update(dynamics_state["coordinates"], dynamics_state["vel"], dynamics_state)
        v = v + dtm * dynamics_state["forces"]
        x, v = integrate_A_half(x, v)
        x, v, dynamics_state = middle_update(x, v, dynamics_state)
        x, v = integrate_A_half(x, v)

        if fix_com:
            if nbeads is None:
                x = x.reshape(-1, nat, 3)
                xcom = jnp.sum(x*massprop[None,:,None], axis=1, keepdims=True)
                x = (x - xcom).reshape(-1, 3)
            else:
                xcom = jnp.sum(x[0]*massprop[:,None], axis=0, keepdims=True)
                x = x.at[0].set(x[0] - xcom)

        return {**dynamics_state, "coordinates": x, "vel": v, "istep": dynamics_state["istep"] + 1}

    ###############################################
    ### DEFINE OBSERVABLE FUNCTION
    @jax.jit
    def update_observables(dynamics_state, batch):
        ### POTENTIAL ENERGY AND FORCES
        epot, de, out = energy_and_gradient(model.variables, batch)
        out["forces"] = -de["coordinates"]
        epot = epot / model_energy_unit
        de = {k: v / model_energy_unit for k, v in de.items()}

        # Convert model gradients from physical coordinates to normal-mode
        # coordinates before adding biases. Bias terms consume `dynamics_state`, whose
        # coordinates are already in normal modes for PIMD.
        if nbeads is not None:
            de["coordinates"] = jnp.einsum(
                "in,i...->n...", eigmat, de["coordinates"].reshape(nbeads, nat, 3)
            ) * (1.0 / nbeads**0.5)

        # ADD BIASES
        if use_biases:
            ebias = {}
            for bias_name, bias_term in bias_terms.items():
                eb, deb = bias_term(dynamics_state)
                ebias[bias_name] = eb
                for k, v in deb.items():
                    de[k] = de.get(k, 0.0) + v

        forces = -de["coordinates"]

        dynamics_state = {
            **dynamics_state,
            "epot": jnp.mean(epot),
            "forces": forces,
            "energy_gradients": de,
        }
        if use_biases:
            dynamics_state["ebias"] = ebias

        ### KINETIC ENERGY
        v = dynamics_state["vel"]
        if ekin_instant == "B":
            v = v + 0.5 * dtm * forces
        if nbeads is None and frozen_atoms is not None:
            # Frozen atoms have no kinetic degrees of freedom, including during
            # the initial observable evaluation before their first A-step.
            v = v * freeze_mask_replicas
                    
        if nbeads is None:
            corr_kin = dynamics_state["thermostat"].get("corr_kin", 1.0)
            # ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
            ek = (0.5 / nreplicas / corr_kin) * jnp.sum(
                mass[:, None, None] * v[:, :, None] * v[:, None, :], axis=0
            )
        else:
            ek_c = 0.5 * jnp.sum(
                mass[:, None, None] * v[0, :, :, None] * v[0, :, None, :], axis=0
            )
            ek = ek_c - 0.5 * jnp.sum(
                dynamics_state["coordinates"][1:, :, :, None] * forces[1:, :, None, :],
                axis=(0, 1),
            )
            dynamics_state["ek_c"] = jnp.trace(ek_c)

        dynamics_state["ek"] = jnp.trace(ek)
        dynamics_state["ek_tensor"] = ek

        if estimate_pressure:
            if use_average_Pkin:
                ek = ek_avg
            elif pressure_instant != "O":
                alpha = 0.5 if pressure_instant == "B" else 1.0
                v = dynamics_state["vel"] + alpha * dtm * dynamics_state["forces"]
                if nbeads is None:
                    corr_kin = dynamics_state["thermostat"].get("corr_kin", 1.0)
                    # ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
                    if frozen_atoms is not None:
                        v = v * freeze_mask_replicas
                    ek = (0.5 / nreplicas / corr_kin) * jnp.sum(
                        mass[:, None, None] * v[:, :, None] * v[:, None, :], axis=0
                    )
                else:
                    if frozen_atoms is not None:
                        v = v.at[0].set(v[0] * freeze_mask)
                    ek_c = 0.5 * jnp.sum(
                        mass[:, None, None] * v[0, :, :, None] * v[0, :, None, :],
                        axis=0,
                    )
                    ek = ek_c - 0.5 * jnp.sum(
                        dynamics_state["coordinates"][1:, :, :, None] * forces[1:, :, None, :],
                        axis=(0, 1),
                    )

            vir = jnp.mean(de["strain"], axis=0)
            dynamics_state["virial"] = vir
            out["virial_tensor"] = vir * model_energy_unit

            volume = jnp.abs(jnp.linalg.det(dynamics_state["cell"]))
            Pres = ek * (2.0 / volume) - vir / volume
            dynamics_state["pressure_tensor"] = Pres
            dynamics_state["pressure"] = jnp.trace(Pres) * (1.0 / 3.0)
            if variable_cell:
                density = densmass / volume
                dynamics_state["density"] = density
                dynamics_state["volume"] = volume

        if ensemble_key is not None:
            kT = simulation_setup["kT"]
            dE = (
                jnp.mean(out[ensemble_key], axis=0) / model_energy_unit - dynamics_state["epot"]
            )
            dynamics_state["ensemble_weights"] = -dE / kT

        if "total_dipole" in out:
            if nbeads is None:
                dynamics_state["total_dipole"] = out["total_dipole"][0]
            else:
                dynamics_state["total_dipole"] = jnp.mean(out["total_dipole"], axis=0)

        if use_colvars:
            coords = dynamics_state["coordinates"].reshape(-1, nat, 3)[0]
            cell = dynamics_state["cell"] if "cell" in dynamics_state else None

            colvars = {}
            for colvar_name, colvar_calc in colvars_calculators.items():
                colvars[colvar_name] = colvar_calc(coords,cell)
            dynamics_state["colvars"] = colvars

        return dynamics_state, out

    ###############################################
    ### IR SPECTRUM
    if do_ir_spectrum:
        # @jax.jit
        # def update_dipole(ir_state,dynamics_state,batch):
        #     def mumodel(coords):
        #         out = model_ir._apply(model_ir.variables,{**batch,"coordinates":coords})
        #         if nbeads is None:
        #             return out["total_dipole"][0]
        #         return out["total_dipole"].sum(axis=0)
        #     dmudqmodel = jax.jacobian(mumodel)

        #     dmudq = dmudqmodel(batch["coordinates"])
        #     # print(dmudq.shape)
        #     if nbeads is None:
        #         vel = dynamics_state["vel"].reshape(-1,1,nat,3)[0]
        #         mudot = (vel*dmudq).sum(axis=(1,2))
        #     else:
        #         dmudq = dmudq.reshape(3,nbeads,nat,3)#.mean(axis=1)
        #         vel = (jnp.einsum("in,n...->i...", eigmat, dynamics_state["vel"]) *  nbeads**0.5
        #         )
        #         # vel = dynamics_state["vel"][0].reshape(1,nat,3)
        #         mudot = (vel[None,...]*dmudq).sum(axis=(1,2,3))/nbeads

        #     ir_state = save_dipole(mudot,ir_state)
        #     return ir_state
        @jax.jit
        def update_batch_ir(batch, dynamics_state):
            batch = {
                **batch,
                "coordinates": dynamics_state["coordinates"].reshape(-1, nat, 3)[0],
                "natoms": jnp.asarray([nat]),
                "batch_index": jnp.asarray([0] * nat),
                "species": jnp.asarray(simulation_setup["species"].reshape(-1, nat)[0]),
            }
            if variable_cell:
                batch["cells"] = dynamics_state["cell"][None, :, :]
                batch["reciprocal_cells"] = jnp.linalg.inv(dynamics_state["cell"])[
                    None, :, :
                ]
            return batch

        @jax.jit
        def update_dipole(ir_state, dynamics_state, batch):
            if model_ir is not None:
                out = model_ir._apply(model_ir.variables, batch)
                q = out.get("charges", jnp.zeros(nat)).reshape((-1, nat))
                dip = out.get("dipoles", jnp.zeros((nat, 3))).reshape((-1, nat, 3))
            else:
                q = dynamics_state.get("charges", jnp.zeros(nat)).reshape((-1, nat))
                dip = dynamics_state.get("dipoles", jnp.zeros((nat, 3))).reshape((-1, nat, 3))
            if nbeads is not None:
                q = jnp.mean(q, axis=0)
                dip = jnp.mean(dip, axis=0)
                vel = dynamics_state["vel"][0]
                pos = dynamics_state["coordinates"][0]
            else:
                q = q[0]
                dip = dip[0]
                vel = dynamics_state["vel"].reshape(-1, nat, 3)[0]
                pos = dynamics_state["coordinates"].reshape(-1, nat, 3)[0]

            if pbc_data is not None:
                cell_reciprocal = (
                    batch["cells"][0],
                    batch["reciprocal_cells"][0],
                )
            else:
                cell_reciprocal = None

            ir_state = save_dipole(
                q, vel, pos, dip.sum(axis=0), cell_reciprocal, ir_state
            )
            return ir_state

    ###############################################
    ### GRAPH UPDATES

    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    """@keyword[fennol_md] nblist_verbose
    Print verbose information about neighbor list updates and reallocations.
    Default: False
    """
    nblist_stride = int(simulation_parameters.get("nblist_stride", -1))
    """@keyword[fennol_md] nblist_stride
    Number of steps between full neighbor list rebuilds. Auto-calculated from skin if <= 0.
    Default: -1
    """
    nblist_warmup_time = simulation_parameters.get("nblist_warmup_time", -1.0)
    """@keyword[fennol_md] nblist_warmup_time
    Time period for neighbor list warmup before using skin updates.
    Default: -1.0
    """
    nblist_warmup = int(nblist_warmup_time / dt) if nblist_warmup_time > 0 else 0
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    """@keyword[fennol_md] nblist_skin
    Neighbor list skin distance for efficient updates (in Angstroms).
    Default: -1.0
    """
    if nblist_skin > 0:
        if nblist_stride <= 0:
            ## reference skin parameters at 300K (from Tinker-HP)
            ##   => skin of 2 A gives you 40 fs without complete rebuild
            t_ref = 40.0 / us.FS  # FS
            nblist_skin_ref = 2.0  # A
            nblist_stride = max(1,int(math.floor(nblist_skin / nblist_skin_ref * t_ref / dt)))
        print(
            f"# nblist_skin: {nblist_skin:.2f} A, nblist_stride: {nblist_stride} steps, nblist_warmup: {nblist_warmup} steps"
        )

    if nblist_skin <= 0:
        nblist_stride = 1

    step_state["nblist_countdown"] = 0
    step_state["print_skin_activation"] = nblist_warmup > 0

    def update_graphs(istep, step_state, dynamics_state, batch, force_preprocess=False):
        nblist_countdown = step_state["nblist_countdown"]
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            ### FULL NBLIST REBUILD
            step_state["nblist_countdown"] = nblist_stride - 1
            preproc_state = step_state["preproc_state"]
            batch = model.preprocessing.process(
                preproc_state, update_batch(batch, dynamics_state)
            )
            preproc_state, state_up, batch, overflow = (
                model.preprocessing.check_reallocate(preproc_state, batch)
            )
            step_state["preproc_state"] = preproc_state
            if nblist_verbose and overflow:
                print("step", istep, ", nblist overflow => reallocating nblist")
                print("size updates:", state_up)

            if do_ir_spectrum and model_ir is not None:
                batch_ir = model_ir.preprocessing.process(
                    step_state["preproc_state_ir"],
                    update_batch_ir(step_state["batch_ir"], dynamics_state),
                )
                (
                    step_state["preproc_state_ir"],
                    _,
                    step_state["batch_ir"],
                    overflow,
                ) = model_ir.preprocessing.check_reallocate(
                    step_state["preproc_state_ir"], batch_ir
                )

        else:
            ### SKIN UPDATE
            if step_state["print_skin_activation"]:
                if nblist_verbose:
                    print(
                        "step",
                        istep,
                        ", end of nblist warmup phase => activating skin updates",
                    )
                step_state["print_skin_activation"] = False

            step_state["nblist_countdown"] = nblist_countdown - 1
            batch = model.preprocessing.update_skin(
                update_batch(batch, dynamics_state)
            )
            if do_ir_spectrum and model_ir is not None:
                step_state["batch_ir"] = model_ir.preprocessing.update_skin(
                    update_batch_ir(step_state["batch_ir"], dynamics_state)
                )

        return batch, step_state

    ################################################
    ### DEFINE STEP FUNCTION
    def step(istep, step_state, dynamics_state, batch, force_preprocess=False):

        step_state = {
            **step_state,
            "istep": step_state["istep"] + 1,
        }

        ### INTEGRATE EQUATIONS OF MOTION
        dynamics_state = integrate(dynamics_state)

        ### UPDATE CONFORMATION AND GRAPHS
        batch, step_state = update_graphs(
            istep, step_state, dynamics_state, batch, force_preprocess
        )

        ## COMPUTE FORCES AND OBSERVABLES
        dynamics_state, out = update_observables(dynamics_state, batch)

        ## END OF STEP UPDATES
        if do_thermostat_post:
            dynamics_state["thermostat"], step_state["thermostat_post_state"] = thermostat_post(
                dynamics_state["thermostat"], step_state["thermostat_post_state"]
            )

        if use_lambdadyn:
            dynamics_state["lambdadyn"] = lambdadyn_post(dynamics_state["lambdadyn"])
        if use_opes:
            dynamics_state["opes"] = opes_post(dynamics_state["opes"])
        if use_abf:
            dynamics_state["abf"] = abf_post(dynamics_state["abf"])

        if do_ir_spectrum:
            ir_state = update_dipole(
                step_state["ir_spectrum"], dynamics_state, step_state["batch_ir"]
            )
            step_state["ir_spectrum"] = ir_post(ir_state)

        return step_state, dynamics_state, batch, out

    ###########################################################

    print("# Computing initial energy and forces")

    batch = update_batch(batch, dynamics_state)
    # initialize IR batch
    if do_ir_spectrum and model_ir is not None:
        step_state["preproc_state_ir"], step_state["batch_ir"] = (
            model_ir.preprocessing(
                model_ir.preproc_state,
                update_batch_ir(batch, dynamics_state),
            )
        )

    dynamics_state, out = update_observables(dynamics_state, batch)
    # The restart recomputes forces and virial on a freshly built neighbor
    # list, whose pair order changes their last bits. Continue from the saved
    # values so that a restarted run reproduces the uninterrupted one, unless
    # they differ by more than round-off (e.g. the model or a bias changed).
    for key in ("forces", "pressure_tensor"):
        if key in restart_data and key in dynamics_state:
            saved = np.asarray(restart_data[key])
            current = np.asarray(dynamics_state[key])
            scale = float(np.max(np.abs(current))) if current.size else 0.0
            if saved.shape == current.shape and np.allclose(saved, current, rtol=1e-4, atol=1e-4 * scale):
                dynamics_state[key] = jnp.asarray(saved, dtype=dynamics_state[key].dtype)
    if "pressure_tensor" in dynamics_state:
        dynamics_state["pressure"] = jnp.trace(dynamics_state["pressure_tensor"]) * (1.0 / 3.0)

    if input_keys_to_keep is not None:
        for key in input_keys_to_keep:
            batch[key] = out[key]
            

    return step, update_batch, simulation_setup, step_state, batch, dynamics_state
