import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import os
import pickle
from pathlib import Path

from ..utils.activations import safe_sqrt
from .utils import us, get_centroid_selector
from ..utils.io import last_xyz_frame
from ..utils import Counter, read_tinker_interval, detect_topology, read_species_list
from ..utils.periodic_table import COV_RADII, PERIODIC_TABLE_REV_IDX


def initialize_restraints(
    simulation_parameters, simulation_setup, fprec, colvars_calculators
):
    restraints_params = simulation_parameters.get("restraints", None)
    if restraints_params is None:
        return {}

    restraints = {}
    for bname, bdef in restraints_params.items():
        assert isinstance(
            bdef, dict
        ), f"restraints {bname} definition must be a dictionary"

        restraint_type = str(bdef.get("type", "colvar")).strip().lower()

        if restraint_type == "topology":
            restraints[bname] = get_topology_bond_restraints(
                bname, bdef, simulation_setup, fprec
            )
            continue
        elif restraint_type == "colvar":
            restraints[bname] = build_wall_restraint(
                bname,
                bdef,
                colvars_calculators,
                fprec,
                simulation_setup,
            )
        elif restraint_type == "atoms":
            restraints[bname] = build_atoms_restraints(bname, bdef, simulation_setup, fprec)
        else:
            raise ValueError(
                f"Unknown restraint type '{restraint_type}' for restraint '{bname}'."
            )
    return restraints


def __check_list_of_floats(x, name, length, fprec):
    if isinstance(x, (int, float)):
        x = [float(x)] * length
    assert (
        isinstance(x, (list, tuple)) and len(x) == length
    ), f"{name} must be a number or a list of length {length}"
    return np.array(x, dtype=fprec)

def __get_piecewise_scheduler(restraint_param_def, name, ncolvars, fprec):
    if not isinstance(restraint_param_def, dict):
        x = __check_list_of_floats(
            restraint_param_def, name, ncolvars, fprec
        )
        return lambda step: x
    
    anchor_steps = np.array([int(s) for s in restraint_param_def.keys()], dtype=np.int32)
    anchor_values = np.stack(
        [__check_list_of_floats(restraint_param_def[s], name, ncolvars, fprec) for s in restraint_param_def.keys()],
        dtype=fprec,
    )

    argsort = np.argsort(anchor_steps)
    anchor_steps = anchor_steps[argsort]
    anchor_values = anchor_values[argsort]
    assert np.all(anchor_steps >= 0), f"All anchor steps in {name} must be non-negative"
    if anchor_steps[0] != 0:
        anchor_steps = np.concatenate(([0], anchor_steps))
        anchor_values = np.concatenate((anchor_values[:1], anchor_values))

    interval_sizes = np.maximum(anchor_steps[1:] - anchor_steps[:-1],1)
    anchor_diffs = anchor_values[1:] - anchor_values[:-1]

    print(f"# RESTRAINT PARAMETER SCHEDULER: {name} will be scheduled with piecewise linear interpolation between the following anchor points:")
    for s, v in zip(anchor_steps, anchor_values):
        print(f"#    step {s}: {v}")

    def piecewise_scheduler(step):
        ds = step - anchor_steps[:-1]
        indicator = jnp.logical_and(ds >= 0, step < anchor_steps[1:])
        pct = ds / interval_sizes
        interp_vals = anchor_values[:-1] + pct[:, None] * anchor_diffs
        return jnp.sum(interp_vals * indicator[:, None], axis=0) + anchor_values[-1] * (step >= anchor_steps[-1])
        
    return piecewise_scheduler





def build_wall_restraint(
    restraint_name,
    restraint_def,
    colvars_calculators,
    fprec,
    simulation_setup={},
):
    ## GET COLVARS
    colvars = restraint_def.get("colvars", None)
    if isinstance(colvars, str):
        colvars = [colvars]
    colvars = [str(c).strip() for c in colvars]
    assert (
        len(colvars) > 0
    ), f"restraints/{restraint_name}/colvars must be a non-empty list of colvars"
    ncolvars = len(colvars)
    for colvar in colvars:
        assert (
            colvar in colvars_calculators
        ), f"colvar {colvar} not found in colvars_calculators"
    select_centroid = get_centroid_selector(simulation_setup)

    ## GET WALLS PARAMETERS
    restraint_center = restraint_def.get("center", None)
    if restraint_center is not None:
        restraint_upper_wall = restraint_lower_wall = restraint_center
    else:
        restraint_upper_wall = restraint_def.get("upper_wall", None)
        restraint_lower_wall = restraint_def.get("lower_wall", None)
        assert (
            restraint_upper_wall is not None or restraint_lower_wall is not None
        ), f"restraints/{restraint_name} must have either center or upper_wall/lower_wall defined"
    if restraint_upper_wall is not None:
        restraint_upper_wall = __get_piecewise_scheduler(
            restraint_upper_wall,
            f"restraints/{restraint_name} center or upper_wall",
            ncolvars,
            fprec,
        )
    if restraint_lower_wall is not None:
        restraint_lower_wall = __get_piecewise_scheduler(
            restraint_lower_wall,
            f"restraints/{restraint_name}/lower_wall",
            ncolvars,
            fprec,
        )

    ## GET FORCE CONSTANTS
    restraint_k = restraint_def.get("force_constant", None)
    if restraint_k is not None:
        restraint_k_upper = restraint_k_lower = restraint_k
    else:
        restraint_k_upper = restraint_def.get("force_constant_upper", None)
        restraint_k_lower = restraint_def.get("force_constant_lower", None)
        assert (
            restraint_k_upper is not None or restraint_k_lower is not None
        ), f"restraints/{restraint_name} must have either force_constant or force_constant_upper/force_constant_lower defined"

    if restraint_k_upper is not None:
        restraint_k_upper = __get_piecewise_scheduler(
            restraint_k_upper,
            f"restraints/{restraint_name}/force_constant(_upper)",
            ncolvars,
            fprec,
        )
        # assert np.all(
        #     restraint_k_upper > 0.0
        # ), f"restraints/{restraint_name}/force_constant(_upper) must be > 0"
    if restraint_k_lower is not None:
        restraint_k_lower = __get_piecewise_scheduler(
            restraint_k_lower,
            f"restraints/{restraint_name}/force_constant_lower",
            ncolvars,
            fprec,
        )
        # assert np.all(
        #     restraint_k_lower > 0.0
        # ), f"restraints/{restraint_name}/force_constant_lower must be > 0"

    lambda_schedule_str = restraint_def.get("lambda_schedule", None)
    use_lambda_schedule = lambda_schedule_str is not None
    if use_lambda_schedule:
        lambda_schedule = _get_lambda_schedule(lambda_schedule_str)
        print(
            f"# RESTRAINTS: using lambda schedule for restraint '{restraint_name}' => `{lambda_schedule_str}`"
        )

    ## BUILD restraint FUNCTION
    def walls_restraint_energy(coordinates, alch_lambda=None, cell=None, step=0):
        colvar_coordinates = select_centroid(coordinates)
        if use_lambda_schedule and alch_lambda is not None:
            scale = 0.5 * lambda_schedule(alch_lambda)
        else:
            scale = 0.5
        colvar_values = jnp.array(
            [
                colvars_calculators[colvar](colvar_coordinates, cell)
                for colvar in colvars
            ],
            dtype=fprec,
        )
        # energy = (
        #     (colvar_values > restraint_upper_wall)
        #     * (restraint_k_upper * (colvar_values - restraint_upper_wall) ** 2)
        #     + (colvar_values < restraint_lower_wall)
        #     * (restraint_k_lower * (colvar_values - restraint_lower_wall) ** 2)
        # ).sum()
        energy = 0
        if restraint_upper_wall is not None:
            _upper_wall = restraint_upper_wall(step)
            energy += jnp.sum(
                (colvar_values > _upper_wall)
                * (restraint_k_upper(step) * (colvar_values - _upper_wall) ** 2)
            )
        if restraint_lower_wall is not None:
            _lower_wall = restraint_lower_wall(step)
            energy += jnp.sum(
                (colvar_values < _lower_wall)
                * (restraint_k_lower(step) * (colvar_values - _lower_wall) ** 2)
            )
        return scale * energy

    def walls_system_restraint(dynamics_state):
        x = dynamics_state["coordinates"]
        cell = dynamics_state["cell"] if "cell" in dynamics_state else None
        istep = dynamics_state.get("istep", 0)
        if "lambdadyn" in dynamics_state:
            alch_lambda = dynamics_state["lambdadyn"]["alch_lambda"]
            e, (dedx, dedl) = jax.value_and_grad(
                walls_restraint_energy, argnums=(0, 1)
            )(x, alch_lambda, cell=cell, step=istep)
            return e, {"coordinates": dedx, "alch_lambda": dedl}

        e, dedx = jax.value_and_grad(walls_restraint_energy)(x,cell=cell, step=istep)
        return e, {"coordinates": dedx}

    print(f"# RESTRAINTS: using wall restraint on colvars:{', '.join(colvars)}")
    return walls_system_restraint


def get_topology_bond_restraints(restraint_name, restraint_def, simulation_setup, fprec):

    # PIMD also sets nreplicas (to nbeads); only independent replicas are
    # rejected, since PIMD restraints act on the centroid.
    use_pimd = simulation_setup.get("nbeads") is not None
    if not use_pimd and simulation_setup.get("nreplicas", 1) > 1:
        raise ValueError(
            f"restraints/{restraint_name}: topology restraints are not supported "
            "with multiple replicas"
        )
    select_centroid = get_centroid_selector(simulation_setup)

    # GET WALL PARAMETERS
    wall_K = restraint_def.get("force_constant", 400.0 / us.KCALPERMOL * us.ANGSTROM**2)
    restraint_upper_wall = restraint_def.get("upper_wall", 0.4 / us.ANGSTROM)
    assert (
        wall_K > 0.0
    ), f"restraints/{restraint_name}: force_constant must be greater than 0"
    assert restraint_upper_wall > 0.0, "upper_wall must be greater than 0"

    # GET TOPOLOGY
    topology_key = restraint_def.get("topology", None)
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
            topology = detect_topology(species, coordinates, cell=cell)
            print("# Detected topology for bond restraints")
            system_name = simulation_setup["name"]
            np.savetxt(
                f"{system_name}.restraint.topo",
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
            "RESTRAINTS: topology restraint is set, but no topology provided. "
        )
    energy_unit = simulation_setup["energy_unit"]
    energy_unit_str = simulation_setup["energy_unit_str"]

    rc = (COV_RADII / us.BOHR)[simulation_setup["species"]]
    bond_src, bond_dst = topology[:, 0], topology[:, 1]
    rcij = rc[bond_src] + rc[bond_dst] + restraint_upper_wall

    lambda_schedule_str = restraint_def.get("lambda_schedule", None)
    use_lambda_schedule = lambda_schedule_str is not None
    if use_lambda_schedule:
        lambda_schedule = _get_lambda_schedule(lambda_schedule_str)
        print(
            f"# RESTRAINTS: using lambda schedule for restraint '{restraint_name}' => `{lambda_schedule_str}`"
        )

    def topology_system_restraint(dynamics_state):
        inputs = {"coordinates": dynamics_state["coordinates"]}
        use_lambda = "lambdadyn" in dynamics_state and use_lambda_schedule
        if use_lambda:
            inputs["alch_lambda"] = dynamics_state["lambdadyn"]["alch_lambda"]
        use_pbc = "cell" in dynamics_state and dynamics_state["cell"] is not None
        if use_pbc:
            inputs["strain"] = jnp.eye(3, dtype=fprec)

        def topology_restraint_energy(inputs):
            coordinates = select_centroid(inputs["coordinates"])
            if use_lambda:
                alch_lambda = inputs["alch_lambda"]
                lambda_fact = lambda_schedule(alch_lambda)
            else:
                lambda_fact = 1.0
            if use_pbc:
                scaling = inputs["strain"]
                coordinates = coordinates @ scaling
                cell = dynamics_state["cell"] @ scaling
                reciprocal_cell = jnp.linalg.inv(cell)

            v = coordinates[bond_dst] - coordinates[bond_src]
            if use_pbc:
                v = v - jnp.round(v @ reciprocal_cell) @ cell
            r = safe_sqrt(jnp.sum(v**2, axis=1))
            return (
                0.5
                * wall_K
                * lambda_fact
                * jnp.sum(jnp.where(r > rcij, (r - rcij) ** 2, 0.0))
            )

        e, de = jax.value_and_grad(topology_restraint_energy)(inputs)
        return e, de

    print(
        f"# RESTRAINTS: using topology restraint with K={wall_K*energy_unit:.2e} {energy_unit_str}/A^2, r0={restraint_upper_wall:.2f} A"
    )
    return topology_system_restraint


def build_atoms_restraints(restraint_name, restraint_def, simulation_setup, fprec):
    # GET LIST OF ATOMS
    nat = simulation_setup["nat"]
    use_pimd = simulation_setup.get("nbeads") is not None
    select_centroid = get_centroid_selector(simulation_setup)
    species = simulation_setup["species"]
    if not use_pimd:
        species = species[:nat]
    atoms = read_species_list(
        species, restraint_def.get("atoms_species", None)
    )

    if "atoms" in restraint_def:
        atoms = np.concatenate((atoms, read_tinker_interval(restraint_def["atoms"])))

    if len(atoms) == 0:
        raise ValueError(
            f"restraints/{restraint_name} did not match any atom in the system"
        )

    # GET REFERENCE COORDINATES
    ref_coords = restraint_def.get("ref_coords", None)
    if ref_coords is None:
        if "optimized_coordinates" in simulation_setup:
            coordinates = simulation_setup["optimized_coordinates"]
        else:
            coordinates = simulation_setup["initial_coordinates"]
        ref_coords = coordinates[atoms]
    elif isinstance(ref_coords, str):
        xyzfile = Path(ref_coords).expanduser().resolve()
        assert (
            xyzfile.is_file()
        ), f"Reference coordinates file '{xyzfile}' does not exist"
        has_comment_line = restraint_def.get("has_comment_line", True)
        symbols, ref_coords, comment = last_xyz_frame(
            xyzfile, has_comment_line=has_comment_line
        )
        if ref_coords.shape[0] != atoms.shape[0]:
            ref_coords = ref_coords[atoms]
    else:
        ref_coords = np.array(ref_coords, dtype=np.float32)
        assert (
            ref_coords.ndim == 2
            and ref_coords.shape[0] == len(atoms)
            and ref_coords.shape[1] == 3
        ), f"Reference coordinates must be a 2D array with shape [n_atoms, 3], got shape {ref_coords.shape}"

    # Independent replicas are contiguous nat-sized blocks. Apply the same
    # restraint target to each replica so they remain equivalent. In PIMD the
    # first block is the centroid normal mode, so retaining only that block
    # defines a symmetric centroid restraint rather than a single-bead force.
    nblocks = simulation_setup.get("nreplicas", 1) if not use_pimd else 1
    if nblocks > 1:
        atoms = np.concatenate([atoms + iblock * nat for iblock in range(nblocks)])
        ref_coords = np.tile(ref_coords, (nblocks, 1))

    use_pbc = simulation_setup.get("pbc", None) is not None
    if use_pbc:
        reciprocal_cell = simulation_setup["pbc"]["reciprocal_cell"]
        ref_coords = ref_coords @ reciprocal_cell

    # GET FORCE CONSTANT
    restraint_k = restraint_def.get(
        "force_constant", 50.0 / us.KCALPERMOL * us.ANGSTROM**2
    )
    assert restraint_k > 0, f"restraints/{restraint_name}/force_constant must be > 0"
    print(
        f"# RESTRAINTS: using atoms restraint on {len(atoms)} atoms with K={restraint_k*simulation_setup['energy_unit']:.2e} {simulation_setup['energy_unit_str']}/A^2"
    )

    lambda_schedule_str = restraint_def.get("lambda_schedule", None)
    use_lambda_schedule = lambda_schedule_str is not None
    if use_lambda_schedule:
        lambda_schedule = _get_lambda_schedule(lambda_schedule_str)
        print(
            f"# RESTRAINTS: using lambda schedule for restraint '{restraint_name}' => `{lambda_schedule_str}`"
        )

    def atoms_system_restraint(dynamics_state):
        inputs = {"coordinates": dynamics_state["coordinates"]}
        use_lambda = "lambdadyn" in dynamics_state and use_lambda_schedule
        if use_lambda:
            inputs["alch_lambda"] = dynamics_state["lambdadyn"]["alch_lambda"]
        if use_pbc:
            inputs["strain"] = jnp.eye(3, dtype=fprec)

        def atoms_restraint_energy(inputs):
            coordinates = inputs["coordinates"]
            if use_lambda:
                alch_lambda = inputs["alch_lambda"]
                lambda_fact = lambda_schedule(alch_lambda)
            else:
                lambda_fact = 1.0
            restraint_coordinates = select_centroid(coordinates)
            if use_pbc:
                scaling = inputs["strain"]
                restraint_coordinates = restraint_coordinates @ scaling
                cell = dynamics_state["cell"] @ scaling
                reciprocal_cell = jnp.linalg.inv(cell)

                v = restraint_coordinates[atoms] @ reciprocal_cell - ref_coords
                v = (v - jnp.round(v)) @ cell
            else:
                v = restraint_coordinates[atoms] - ref_coords
            return 0.5 * restraint_k * lambda_fact * jnp.sum(v**2)

        e, de = jax.value_and_grad(atoms_restraint_energy)(inputs)
        return e, de

    return atoms_system_restraint


def _get_lambda_schedule(lambda_schedule_str):
    assert isinstance(
        lambda_schedule_str, str
    ), f"lambda_schedule must be a string (try using quotation marks)"
    lambda_schedule_str = lambda_schedule_str.strip()
    assert ":" in lambda_schedule_str, f"lambda_schedule must be of the form 'l: f(l)'"
    fn_args = lambda_schedule_str.split(":", 1)[0]
    fn_args = fn_args.strip().split(",")
    assert len(fn_args) == 1, f"lambda_schedule must be of the form 'l: f(l)'"

    lambda_schedule = eval(
        "lambda " + lambda_schedule_str,
        {
            "__builtins__": None,
            **jax.nn.__dict__,
            **jax.numpy.__dict__,
            **jax.__dict__,
        },
    )
    try:
        test_lambda = jnp.array(0.5)
        lambda_schedule(test_lambda)
    except Exception as e:
        raise ValueError(f"Error evaluating lambda_schedule `{lambda_schedule_str}`. Error: {e}")
    return lambda_schedule
