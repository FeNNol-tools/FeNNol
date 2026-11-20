import numpy as np
import jax
import jax.numpy as jnp
import pickle
from ..utils import UnitSystem,read_tinker_interval
from functools import partial

# Define the unit system for fennol_md
us = UnitSystem(L="ANGSTROM", T="PS", E="KCALPERMOL")  


def get_centroid_selector(simulation_setup):
    """Return the map from dynamics coordinates to bias coordinates.

    Under PIMD the integrator stores normal-mode coordinates whose first mode
    is the centroid, so collective-variable biases and restraints act on it.
    """
    if simulation_setup.get("nbeads") is None:
        return lambda coordinates: coordinates
    return lambda coordinates: coordinates[0]


def get_restart_file(simulation_setup):
    restart_file = simulation_setup["name"] + ".dyn.restart"
    return restart_file


def load_dynamics_restart(simulation_setup):
    with open(get_restart_file(simulation_setup), "rb") as f:
        restart_data = pickle.load(f)

    assert (
        restart_data["nat"] == simulation_setup["nat"]
    ), f"Restart file does not match simulation setup 'nat'"
    assert restart_data.get("nbeads", 0) == simulation_setup.get(
        "nbeads", 0
    ), f"Restart file does not match simulation setup 'nbeads'"
    assert restart_data.get("nreplicas", 0) == simulation_setup.get(
        "nreplicas", 0
    ), f"Restart file does not match simulation setup 'nreplicas'"
    assert np.all(
        restart_data["species"] == simulation_setup["species"]
    ), f"Restart file does not match simulation setup 'species'"

    return restart_data


def save_dynamics_restart(simulation_setup, batch, step_state, dynamics_state):
    restart_data = {
        "nat": simulation_setup["nat"],
        "nbeads": simulation_setup.get("nbeads", 0),
        "nreplicas": simulation_setup.get("nreplicas", 0),
        "species": simulation_setup["species"],
        "coordinates": batch["coordinates"],
        "vel": dynamics_state["vel"],
        "preproc_state": step_state["preproc_state"],
        "simulation_time_ps": simulation_setup["start_time_ps"] + (simulation_setup["dt"]*us.PS)* step_state["istep"],
        "integration_step": dynamics_state["istep"],
    }
    if "cells" in batch:
        restart_data["cells"] = batch["cells"]

    # Integrator states include extended variables and random keys. Persist them
    # verbatim so a restarted stochastic/extended-system trajectory can continue.
    for key in ("thermostat", "barostat"):
        if key in dynamics_state:
            restart_data[key] = dynamics_state[key]
    if "thermostat_post_state" in step_state:
        restart_data["thermostat_post_state"] = step_state["thermostat_post_state"]
    for key in ("forces", "pressure_tensor"):
        if key in dynamics_state:
            restart_data[key] = dynamics_state[key]

    if "lambdadyn" in dynamics_state:
        restart_data["lambdadyn"] = {
            "alch_lambda": dynamics_state["lambdadyn"]["alch_lambda"],
            "alch_lambda_vel": dynamics_state["lambdadyn"]["alch_lambda_vel"],
        }

    restart_file = get_restart_file(simulation_setup)
    with open(restart_file, "wb") as f:
        pickle.dump(restart_data, f)




def _merge_wrap_groups(wrap_groups,verbose=True):
    """ Merge intersecting groups
    If two groups share any atoms, they should be treated as a single group
    This ensures consistent wrapping behavior for connected molecular fragments
    """

    wrap_groups = {k: set([int(i) for i in v]) for k, v in wrap_groups.items()}
    wrap_groups_keys = list(wrap_groups.keys())
    merged = True
    while merged:
        merged = False
        # Check all pairs of groups for intersections
        for i in range(len(wrap_groups_keys)):
            if i >= len(wrap_groups_keys):
                break
            i_key = wrap_groups_keys[i]
            if i_key not in wrap_groups:
                continue
            w1 = wrap_groups[i_key]
            for j in range(i + 1, len(wrap_groups_keys)):
                if j >= len(wrap_groups_keys):
                    break
                j_key = wrap_groups_keys[j]
                if j_key not in wrap_groups:
                    continue
                w2 = wrap_groups[j_key]
                # If groups i and j have common atoms, merge j into i
                if w1.intersection(w2):
                    wrap_groups[i_key] = w1.union(w2)
                    del wrap_groups[j_key]
                    wrap_groups_keys.remove(j_key)
                    if verbose:
                        print(f"# Wrap groups: Merged wrap group {j_key} into {i_key} because they share atoms {w1.intersection(w2)}")
                    merged = True
                    break
            if merged:
                break  # Restart the merging process to handle transitive merges
    return {k:sorted(v) for k,v in wrap_groups.items()}

def build_wrapbox(simulation_setup,wrap_groups_def=None,jit=True):
    """
    Build a function to wrap atomic coordinates into the periodic box.
    
    Args:
        simulation_setup: Dictionary containing the fixed Simulation setup including 'nat' (number of atoms)
        wrap_groups_def: Optional dictionary defining groups of atoms to wrap together.
                        If None, each atom is wrapped independently.
                        If provided, atoms in the same group will be wrapped together 
                        based on their center of mass.
        jit: Whether to JIT-compile the wrapping function
    
    Returns:
        wrapbox function that wraps coordinates into the periodic box
    """

    use_topo = "molecule_indices" in simulation_setup and simulation_setup["molecule_indices"] is not None
    if wrap_groups_def is None and not use_topo:
        # Simple case: wrap each atom independently into the unit cell
        def wrapbox(x, cell, reciprocal_cell):
            q = x @ reciprocal_cell
            q = q - jnp.floor(q)
            return q @ cell
        
        return jax.jit(wrapbox) if jit else wrapbox
    
    wrap_groups_def = {} if wrap_groups_def is None else wrap_groups_def
    # Initialize data structures for wrap groups
    wrap_groups = {}
    nat = simulation_setup["nat"]
    # wrap_index maps each atom to its group (0 = no group, >0 = group index)
    assert isinstance(
        wrap_groups_def, dict
    ), "wrap_groups must be a dictionary"
    
    # Parse wrap group definitions and assign atoms to groups
    for i, (k, v) in enumerate(wrap_groups_def.items()):
        if str(k).startswith('__'):
            raise ValueError(f"Wrap group names cannot start with '__' (reserved for internal use): {k}")
        idxs = read_tinker_interval(v)
        wrap_groups[k] = idxs


    if use_topo:
        print("# Wrap groups: wrapping based on initial molecular topology")
        molecule_indices = simulation_setup["molecule_indices"]
        # extend each wrap group to include all atoms of the same molecule
        for k in list(wrap_groups.keys()):
            idxs = wrap_groups[k]
            mol_idxs = set(molecule_indices[idxs])
            all_idxs = np.where(np.isin(molecule_indices, list(mol_idxs)))[0]
            wrap_groups[k] = np.unique(np.concatenate([idxs, all_idxs])).astype(np.int32)
            if len(all_idxs)>len(idxs):
                print(f"# Wrap groups: Extended wrap group {k} from {len(idxs)} to {len(all_idxs)} atoms to include entire molecules.")

    wrap_groups = _merge_wrap_groups(wrap_groups)
    print(f"# Wrap groups: {wrap_groups}")
    
    # build wrap_index and nat_groups after merging
    wrap_index = np.zeros(nat, dtype=np.int32)
    nat_groups = []
    for i, k in enumerate(wrap_groups.keys()):
        idxs = wrap_groups[k]
        wrap_index[idxs] = i + 1
        nat_groups.append(len(idxs))
    
    if use_topo:
        other_indices = []
        ngroups = len(nat_groups)
        mol_group = molecule_indices + ngroups + 1
        wrap_index = wrap_index + mol_group*(wrap_index==0)
        unique_groups, counts_groups = np.unique(wrap_index, return_counts=True)
        max_group = int(np.max(wrap_index))
        group_map = np.zeros(max_group+1,dtype=np.int32)
        new_idx = 0
        nat_groups = []
        for i, old_idx in enumerate(unique_groups):
            if counts_groups[i] == 1:
                group_map[old_idx] = 0
            else:
                new_idx += 1
                group_map[old_idx] = new_idx
                nat_groups.append(counts_groups[i])
        wrap_index = group_map[wrap_index]

    # Identify atoms that are not in any wrap group
    # These atoms will be wrapped individually, while grouped atoms wrap together
    other_indices = np.where(wrap_index==0)[0]
    # Group 0 contains all ungrouped atoms, followed by the defined groups
    nat_groups = np.array([len(other_indices)] + nat_groups, dtype=np.int32)
    nat_groups = np.maximum(nat_groups,1)
    ngroups = len(nat_groups)
    shift_mask = wrap_index[:,None] > 0
    
    def wrapbox(x, cell, reciprocal_cell):
        # Convert Cartesian coordinates to fractional coordinates
        q = x @ reciprocal_cell
        
        # Wrap ungrouped atoms individually into the unit cell
        if len(other_indices)>0:
            q = q.at[other_indices].add(-jnp.floor(q[other_indices]))
        
        # Wrap grouped atoms together based on their center of mass
        if ngroups>1:
            # Calculate center of mass for each group in fractional coordinates
            com_groups = jax.ops.segment_sum(q, wrap_index, ngroups)/nat_groups[:,None]
            # Determine the shift needed to bring each group's COM into the unit cell
            shift_groups = -jnp.floor(com_groups)
            # Apply the group-based shift to each atom in its respective group
            shift = shift_groups[wrap_index]
            # Only apply shifts to grouped atoms (wrap_index > 0)
            q = q + shift*shift_mask
        
        # Convert fractional coordinates back to Cartesian coordinates
        return q @ cell
    
    return jax.jit(wrapbox) if jit else wrapbox


def test_pressure_fd(simulation_setup, batch, model, verbose=True):
    model_energy_unit = us.get_multiplier(model.energy_unit)
    nat = simulation_setup["nat"]
    volume = simulation_setup["pbc"]["volume"]
    coordinates = batch["coordinates"]
    cell = batch["cells"][0]
    ek = 1.5 * nat * simulation_setup["kT"]
    Pkin = (2 * us.KBAR / 3.) * ek / volume
    e, f, vir_t, _ = model._energy_and_forces_and_virial(model.variables, batch)
    KBAR = us.KBAR / model_energy_unit
    Pvir = -(np.trace(vir_t[0]) * KBAR) / (3.0 * volume)
    vstep = volume * 0.000001
    scalep = ((volume + vstep) / volume) ** (1.0 / 3.0)
    cellp = cell * scalep
    reciprocal_cell = np.linalg.inv(cellp)
    sysp = model.preprocess(
        **{
            **batch,
            "coordinates": coordinates * scalep,
            "cells": cellp[None, :, :],
            "reciprocal_cells": reciprocal_cell[None, :, :],
        }
    )
    ep, _ = model._total_energy(model.variables, sysp)
    scalem = ((volume - vstep) / volume) ** (1.0 / 3.0)
    cellm = cell * scalem
    reciprocal_cell = np.linalg.inv(cellm)
    sysm = model.preprocess(
        **{
            **batch,
            "coordinates": coordinates * scalem,
            "cells": cellm[None, :, :],
            "reciprocal_cells": reciprocal_cell[None, :, :],
        }
    )
    em, _ = model._total_energy(model.variables, sysm)
    Pvir_fd = -(ep[0] * KBAR - em[0] * KBAR) / (2.0 * vstep)
    if verbose:
        print(
            f"# Initial pressure: {Pkin+Pvir:.3f} (virial); {Pkin+Pvir_fd:.3f} (finite difference) ; Pkin: {Pkin:.3f} ; Pvir: {Pvir:.3f} ; Pvir_fd: {Pvir_fd:.3f}"
        )
    return Pkin, Pvir, Pvir_fd


def optimize_fire2(
    x0,
    ef_func,
    atol=1e-4,
    dt=0.002,
    logoutput=True,
    Nmax=10000,
    keep_every=-1,
    max_disp=None,
):
    """Fast inertial relaxation engine (FIRE)
    adapted from https://github.com/elvissoares/PyFIRE
    """
    # global variables
    alpha0 = 0.1
    Ndelay = 5
    finc = 1.1
    fdec = 0.5
    fa = 0.99
    Nnegmax = Nmax // 5

    error = 10 * atol
    dtmax = dt * 10.0
    dtmin = 0.02 * dt
    alpha = alpha0
    Npos = 0
    Nneg = 0

    nat = x0.shape[0]
    x = x0.copy()
    e, F = ef_func(x)
    V = -0.5 * dt * F  # initial velocity
    P = 0.0
    maxF = np.max(np.abs(F))
    rmsF = np.sqrt(3 * np.mean(F**2))
    error = rmsF
    if error < atol:
        if keep_every > 0:
            return x, True, [x.copy()]
        return x, True

    if logoutput:
        print(
            f"{'#Step':>10} {'Energy':>15} {'RMS Force':>15} {'MAX Force':>15} {'dt':>15} {'Power':>15}"
        )
    if logoutput:
        print(f"{0:10d} {e:15.5f} {rmsF:15.5f} {maxF:15.5f} {dt:15.5f} {0:15.5f}")

    if keep_every > 0:
        x_keep = [x.copy()]

    if max_disp is not None:
        assert max_disp > 0, "max_disp must be positive"

    success = False
    for i in range(Nmax):

        V = V + dt * F
        V = (1 - alpha) * V + alpha * F * np.linalg.norm(V) / np.linalg.norm(F)
        if max_disp is not None:
            V = np.clip(V, -max_disp / dt, max_disp / dt)
        x = x + dt * V
        e, F = ef_func(x)

        maxF = np.max(np.abs(F))
        rmsF = np.sqrt(3 * np.mean(F**2))
        error = rmsF

        if logoutput:
            print(f"{i+1:10d} {e:15.5f} {rmsF:15.5f} {maxF:15.5f} {dt:15.5f} {P:15.5f}")

        if error <= atol:
            success = True
            break

        P = (F * V).sum()  # dissipated power

        if P > 0:
            Npos = Npos + 1
            Nneg = 0
            if Npos > Ndelay:
                dt = min(dt * finc, dtmax)
                alpha = alpha * fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax:
                break
            if i >= Ndelay:
                dt = max(dt * fdec, dtmin)
                alpha = alpha0
            x = x - 0.5 * dt * V
            V = np.zeros(x.shape)

        if keep_every > 0 and (i + 1) % keep_every == 0:
            x_keep.append(x.copy())

    if keep_every > 0:
        x_keep.append(x.copy())
        return x, success, x_keep
    return x, success


def align_vectors(f, fref, apply_shift=True,return_rotation_matrix=True,return_rmsd=False):
    """https://pubmed.ncbi.nlm.nih.gov/15376254/"""

    if apply_shift:
        f = f - jnp.mean(f, axis=0, keepdims=True)
        fref = fref - jnp.mean(fref, axis=0, keepdims=True)
    gyr = jnp.sum(f * f) + jnp.sum(fref * fref)

    R = jnp.sum(f[:, :, None] * fref[:, None, :], axis=0)
    scal = jnp.trace(R)

    M11 = scal
    M22 = 2 * R[0, 0] - scal
    M33 = 2 * R[1, 1] - scal
    M44 = 2 * R[2, 2] - scal

    M12 = R[1, 2] - R[2, 1]
    M13 = R[2, 0] - R[0, 2]
    M14 = R[0, 1] - R[1, 0]

    M23 = R[0, 1] + R[1, 0]
    M24 = R[2, 0] + R[0, 2]
    M34 = R[1, 2] + R[2, 1]

    M = jnp.asarray(
        [M11, M12, M13, M14, M12, M22, M23, M24, M13, M23, M33, M34, M14, M24, M34, M44]
    ).reshape(4, 4)

    M = gyr * np.eye(4) - 2 * M

    l, vecs = jnp.linalg.eigh(M)
    if not return_rotation_matrix:
        if not return_rmsd:
            # Return only the eigenvalues
            return l,vecs
        return (l[0]/f.shape[0])**0.5
    
    q = vecs[0]
    # R = Rotation.from_quat(q).as_matrix()
    q2 = q * q
    D11 = q2[:, 0] + q2[:, 1] - q2[:, 2] - q2[:, 3]
    D12 = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    D13 = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])

    D21 = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    D22 = q2[:, 0] - q2[:, 1] + q2[:, 2] - q2[:, 3]
    D23 = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])

    D31 = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    D32 = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    D33 = q2[:, 0] - q2[:, 1] - q2[:, 2] + q2[:, 3]

    D = jnp.stack([D11, D12, D13, D21, D22, D23, D31, D32, D33], axis=-1).reshape(3, 3)

    if return_rmsd:
        return (l[0]/f.shape[0])**0.5, D

    return D
