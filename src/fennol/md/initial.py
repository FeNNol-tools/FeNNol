from pathlib import Path

import numpy as np
import jax.numpy as jnp

from flax.core import freeze, unfreeze

from ..utils.io import last_xyz_frame


from ..models import FENNIX

from ..utils.periodic_table import PERIODIC_TABLE,PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES,get_atomic_number
from .utils import us
from .thermostats import MOMENTUM_CONSERVING_THERMOSTATS
from ..utils import detect_topology,parse_cell,cell_is_triangular,tril_cell,read_tinker_interval,read_species_list,find_molecules,read_tinker_topology, convert_sparse_to_dense_bonds


def load_model(simulation_parameters):
    model_file = simulation_parameters.get("model_file")
    """@keyword[fennol_md] model_file
    Path to the machine learning model file (.fnx format). Required parameter.
    Type: str, Required
    """
    model_file = Path(str(model_file).strip()).expanduser().resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        graph_config = simulation_parameters.get("graph_config", {})
        """@keyword[fennol_md] graph_config
        Advanced graph configuration for model initialization.
        Default: {}
        """
        model = FENNIX.load(model_file, graph_config=graph_config)  # \
        print(f"# model_file: {model_file}")

    if "energy_terms" in simulation_parameters:
        energy_terms = simulation_parameters["energy_terms"]
        if isinstance(energy_terms, str):
            energy_terms = energy_terms.split()
        model.set_energy_terms(energy_terms)
        print("# energy terms:", model.energy_terms)

    return model


def load_simulation_setup(simulation_parameters, fprec):

    additional_keys = simulation_parameters.get("additional_keys", {})
    """@keyword[fennol_md] additional_keys
    Additional custom keys for model input.
    Default: {}
    """
    if len(additional_keys) >= 1:
        print("# additional keys:",additional_keys)

    ## LOAD SYSTEM CONFORMATION FROM FILES
    system_name = str(simulation_parameters.get("system_name", "system")).strip()
    """@keyword[fennol_md] system_name
    Name prefix for output files. If not specified, uses the xyz filename stem.
    Default: "system"
    """
    indexed = simulation_parameters.get("xyz_input/indexed", False)
    """@keyword[fennol_md] xyz_input/indexed
    Whether first column contains atom indices (Tinker format).
    Default: False
    """
    has_comment_line = simulation_parameters.get("xyz_input/has_comment_line", True)
    """@keyword[fennol_md] xyz_input/has_comment_line
    Whether file contains comment lines.
    Default: True
    """
    xyzfile = Path(simulation_parameters.get("xyz_input/file", system_name + ".xyz")).expanduser().resolve()
    """@keyword[fennol_md] xyz_input/file
    Path to xyz/arc coordinate file. Required parameter.
    Type: str, Required
    """
    if not xyzfile.exists():
        raise FileNotFoundError(f"xyz file {xyzfile} not found")
    system_name = str(simulation_parameters.get("system_name", xyzfile.stem)).strip()
    symbols, coordinates, _ = last_xyz_frame(
        xyzfile, indexed=indexed, has_comment_line=has_comment_line
    )
    coordinates = coordinates.astype(fprec)
    species = np.array([get_atomic_number(s) for s in symbols], dtype=np.int32)
    nat = species.shape[0]
    nfree = nat
    species_set, species_counts = np.unique(species, return_counts=True)
    system_composition = "_".join(
        [f"{PERIODIC_TABLE[s]}{c if c>1 else''}" for s, c in zip(species_set, species_counts)]
    )
    print("# system composition: ", system_composition)

    ## GET FROZEN ATOMS
    frozen_atoms_species = read_species_list(species, simulation_parameters.get("frozen_atoms_species", None))
    """@keyword[fennol_md] frozen_atoms_species
    List of frozen atoms by species (element symbols or atomic numbers).
    Default: None
    """

    frozen_atoms = read_tinker_interval(simulation_parameters.get("frozen_atoms", None))
    """@keyword[fennol_md] frozen_atoms
    List or interval of frozen atoms (1-based indices).
    Default: None
    """
    frozen_atoms = np.unique(np.concatenate((frozen_atoms, frozen_atoms_species)))
    if len(frozen_atoms) == 0:
        frozen_atoms = None
    else:
        assert np.all(
            (frozen_atoms >= 0) & (frozen_atoms < nat)
        ), "Frozen atoms indices out of range"
        print("# frozen atoms: ", frozen_atoms + 1)
        nfree -= len(frozen_atoms)

    ## GET MASS
    mass_Da = np.array(ATOMIC_MASSES, dtype=fprec)[species]
    deuterate = simulation_parameters.get("deuterate", False)
    """@keyword[fennol_md] deuterate
    Replace hydrogen masses with deuterium masses.
    Default: False
    """
    if deuterate:
        print("# Replacing all hydrogens with deuteriums")
        mass_Da[species == 1] *= 2.0

    totmass_Da = mass_Da.sum()

    mass = mass_Da.copy()
    hmr = simulation_parameters.get("hmr", 0)
    """@keyword[fennol_md] hmr
    Hydrogen mass repartitioning factor. 0 = no repartitioning.
    Default: 0
    """
    if hmr > 0:
        print(f"# Adding {hmr} Da to H masses and repartitioning on others for total mass conservation.")
        Hmask = species == 1
        added_mass = hmr * Hmask.sum()
        mass[Hmask] += hmr
        wmass = mass[~Hmask]
        mass[~Hmask] -= added_mass * wmass/ wmass.sum()

        assert np.isclose(mass.sum(), totmass_Da), "Mass conservation failed"

    # convert to internal units
    mass = mass / us.DA

    ### GET TEMPERATURE
    temperature = np.clip(simulation_parameters.get("temperature", 300.0), 1.0e-6, None)
    """@keyword[fennol_md] temperature
    Target temperature in Kelvin.
    Default: 300.0
    """
    kT = us.K_B * temperature 

    ### GET TOTAL CHARGE
    total_charge = simulation_parameters.get("total_charge", None)
    """@keyword[fennol_md] total_charge
    Total system charge for charged systems.
    Default: None (interpreted as 0)
    """
    if total_charge is None:
        total_charge = 0
    else:
        total_charge = int(total_charge)
        print("# total charge: ", total_charge,"e")

    ### ENERGY UNIT
    energy_unit_str = simulation_parameters.get("energy_unit", "kcal/mol")
    """@keyword[fennol_md] energy_unit
    Energy unit for output. Common options: 'kcal/mol', 'eV', 'Ha', 'kJ/mol'.
    Default: "kcal/mol"
    """
    energy_unit = us.get_multiplier(energy_unit_str)

    ## SYSTEM DATA
    simulation_setup = {
        "name": system_name,
        "nat": nat,
        "nfree": nfree,
        "frozen_atoms": frozen_atoms,
        "symbols": symbols,
        "species": species,
        "mass": mass,
        "mass_Da": mass_Da,
        "totmass_Da": totmass_Da,
        "temperature": temperature,
        "kT": kT,
        "total_charge": total_charge,
        "energy_unit": energy_unit,
        "energy_unit_str": energy_unit_str,
    }
    input_flags = simulation_parameters.get("model_flags", [])
    """@keyword[fennol_md] model_flags
    Additional flags to pass to the model.
    Default: []
    """
    flags = {f:None for f in input_flags}
    if len(flags) > 0:
        print("# model flags:", list(flags.keys()))

    ### Set boundary conditions
    cell = simulation_parameters.get("cell", None)
    """@keyword[fennol_md] cell
    Unit cell vectors. Required for PBC. It is a sequence of floats:
    - 9 floats: components of cell vectors [ax, ay, az, bx, by, bz, cx, cy, cz]
    - 6 floats: lengths and angles [a, b, c, alpha, beta, gamma]
    - 3 floats: lengths of cell vectors [a, b, c] (orthorhombic)
    - 1 float: length of cell vectors (cubic cell)
    Lengths are in Angstroms, angles in degrees.
    Default: None
    """
    if cell is not None:
        cell = parse_cell(cell).astype(fprec)
        rotate_cell = not cell_is_triangular(cell)
        if rotate_cell:
            print("# Warning: provided cell is not lower triangular. Rotating to canonical cell orientation.")
            cell, cell_rotation = tril_cell(cell)
        # cell = np.array(cell, dtype=fprec).reshape(3, 3)
        reciprocal_cell = np.linalg.inv(cell)
        volume = np.abs(np.linalg.det(cell))
        print("# cell matrix:")
        for l in cell:
            print("# ", l)
        # print(cell)
        dens = (totmass_Da/volume) * (us.MOL/us.CM**3)
        print("# density: ", dens.item(), " g/cm^3")
        minimum_image = simulation_parameters.get("minimum_image", True)
        """@keyword[fennol_md] minimum_image
        Use minimum image convention for neighbor lists in periodic systems.
        Default: True
        """
        estimate_pressure = simulation_parameters.get("estimate_pressure", False)
        """@keyword[fennol_md] estimate_pressure
        Calculate and print pressure during simulation.
        Default: False
        """
        print("# minimum_image: ", minimum_image)

        crystal_input = simulation_parameters.get("xyz_input/crystal", False)
        """@keyword[fennol_md] xyz_input/crystal
        Use crystal coordinates.
        Default: False
        """
        if crystal_input:
            coordinates = coordinates @ cell
        
        if rotate_cell:
            coordinates = coordinates @ cell_rotation

        pbc_data = {
            "cell": cell,
            "reciprocal_cell": reciprocal_cell,
            "volume": volume,
            "minimum_image": minimum_image,
            "estimate_pressure": estimate_pressure,
        }
        if minimum_image:
            flags["minimum_image"] = None
    else:
        pbc_data = None
    simulation_setup["pbc"] = pbc_data
    simulation_setup["initial_coordinates"] = coordinates.copy()

    ### TOPOLOGY
    topology_key = simulation_parameters.get("topology", None)
    """@keyword[fennol_md] topology
    Topology specification for molecular systems. Use "detect" for automatic detection.
    Default: None
    """
    if topology_key is not None:
        topology_key = str(topology_key).strip()
        if topology_key.lower() == "detect":
            topology = detect_topology(species,coordinates,cell=cell)
            np.savetxt(system_name +".topo", topology+1, fmt="%d")
            print("# Detected topology saved to", system_name + ".topo")
        elif topology_key.lower() == "from_xyz":
            _,topology = read_tinker_topology(xyzfile, has_comment_line=has_comment_line)
            print("# Topology loaded from", xyzfile)
        else:
            topology_key = Path(topology_key).expanduser().resolve()
            assert topology_key.exists(), f"Topology file {topology_key} not found"
            topology = np.loadtxt(str(topology_key), dtype=np.int32)-1
            assert topology.shape[1] == 2, "Topology file must have two columns (source, target)"
            print("# Topology loaded from", topology_key)
        molecule_indices = find_molecules(nat, topology[:,0], topology[:,1])

        if "bonds_atidx" in additional_keys.keys():
            # Convert dense topology to sparse bond index matrix
            # Convert also to 0-based indexing
            bonds_atidx_key = additional_keys["bonds_atidx"]["key"]
            size = additional_keys["bonds_atidx"].get("size", 6)
            bonds_atidx = convert_sparse_to_dense_bonds(nat, topology[:,0], topology[:,1], size)
            additional_keys[bonds_atidx_key] = bonds_atidx

    else:
        topology = None
        molecule_indices = None
    
    simulation_setup["topology"] = topology
    simulation_setup["molecule_indices"] = molecule_indices

    ### Formal charges
    if "formal_charges" in additional_keys.keys():
        formal_charges_key = additional_keys["formal_charges"]["key"]
        formal_charges_file = Path(additional_keys["formal_charges"]["file"]).expanduser().resolve()
        assert formal_charges_file.exists(), f"Formal charges file {formal_charges_file} not found"
        formal_charges = jnp.zeros((nat,), dtype=fprec)
        formal_charges_data = np.loadtxt(str(formal_charges_file), dtype=fprec)
        formal_charges_index = formal_charges_data[:,0].astype(np.int32)-1
        formal_charges_values = formal_charges_data[:,1]
        formal_charges = formal_charges.at[formal_charges_index].set(formal_charges_values)
        additional_keys[formal_charges_key] = formal_charges

        print("# formal charges loaded from", formal_charges_file)

    ### PIMD
    nbeads = simulation_parameters.get("nbeads", None)
    """@keyword[fennol_md] nbeads
    Number of beads for Path Integral MD.
    Default: None
    """
    nreplicas = simulation_parameters.get("nreplicas", None)
    """@keyword[fennol_md] nreplicas
    Number of replicas for independent replica simulations.
    Default: None
    """
    if nbeads is not None:
        nbeads = int(nbeads)
        print("# nbeads: ", nbeads)
        simulation_setup["nbeads"] = nbeads
        coordinates = np.repeat(coordinates[None, :, :], nbeads, axis=0).reshape(-1, 3)
        species = np.repeat(species[None, :], nbeads, axis=0).reshape(-1)
        bead_index = np.arange(nbeads, dtype=np.int32).repeat(nat)
        natoms = np.array([nat] * nbeads, dtype=np.int32)

        eigmat = np.zeros((nbeads, nbeads))
        for i in range(nbeads - 1):
            eigmat[i, i] = 2.0
            eigmat[i, i + 1] = -1.0
            eigmat[i + 1, i] = -1.0
        eigmat[-1, -1] = 2.0
        eigmat[0, -1] = -1.0
        eigmat[-1, 0] = -1.0
        omk, eigmat = np.linalg.eigh(eigmat)
        omk[0] = 0.0
        omk = (nbeads * kT / us.HBAR) * omk**0.5
        for i in range(nbeads):
            if eigmat[i, 0] < 0:
                eigmat[i] *= -1.0
        eigmat = jnp.asarray(eigmat, dtype=fprec)
        simulation_setup["omk"] = omk
        simulation_setup["eigmat"] = eigmat
        simulation_setup["nreplicas"] = nbeads
        nreplicas = nbeads
    elif nreplicas is not None:
        nreplicas = int(nreplicas)
        print("# nreplicas: ", nreplicas)
        simulation_setup["nreplicas"] = nreplicas
        simulation_setup["mass"] = np.repeat(mass[None, :], nreplicas, axis=0).reshape(-1)
        simulation_setup["species"] = np.repeat(species[None, :], nreplicas, axis=0).reshape(
            -1
        )
        coordinates = np.repeat(coordinates[None, :, :], nreplicas, axis=0).reshape(
            -1, 3
        )
        species = np.repeat(species[None, :], nreplicas, axis=0).reshape(-1)
        bead_index = np.arange(nreplicas, dtype=np.int32).repeat(nat)
        natoms = np.array([nat] * nreplicas, dtype=np.int32)
    else:
        simulation_setup["nreplicas"] = 1
        bead_index = np.array([0] * nat, dtype=np.int32)
        natoms = np.array([nat], dtype=np.int32)

    ### DEGREES OF FREEDOM
    # Thermostats that rescale or leave velocities untouched conserve the total
    # momentum when the forces sum to zero. The centre of mass is then not a
    # thermal degree of freedom: its velocity is removed every step, which also
    # absorbs round-off, and 3N-3 degrees of freedom are counted.
    thermostat_name = str(simulation_parameters.get("thermostat", "LGV")).upper()
    remove_com = bool(simulation_parameters.get("remove_com", True))
    """@keyword[fennol_md] remove_com
    For momentum-conserving thermostats (NVE, BUSSI, NOSE): remove the
    centre-of-mass velocity at start and every step, and count 3N-3 degrees
    of freedom. Set to false when the forces do not conserve total momentum
    (external fields, absolute position restraints, biases on Cartesian
    coordinates, a single particle): the centre-of-mass velocity is then
    initialized thermally, left to the dynamics, and 3N degrees of freedom
    are counted. Ignored for Langevin-type thermostats, PIMD and with frozen
    atoms, which always count the centre of mass of the free atoms.
    Default: True
    """
    momentum_conserving = thermostat_name in MOMENTUM_CONSERVING_THERMOSTATS and nbeads is None
    if momentum_conserving and remove_com and frozen_atoms is not None:
        print("# frozen atoms break momentum conservation: centre-of-mass velocity not removed")
        remove_com = False
    remove_com = remove_com and momentum_conserving
    if remove_com and nfree < 2:
        raise ValueError("remove_com leaves no degree of freedom for a single atom; set remove_com: false")
    simulation_setup["remove_com_velocity"] = remove_com
    # Without removal, a momentum-conserving thermostat keeps a thermal centre-of-mass velocity.
    simulation_setup["thermal_com"] = momentum_conserving and not remove_com
    simulation_setup["ndof"] = 3 * nfree - (3 if remove_com else 0)
    if remove_com:
        print("# centre-of-mass velocity removed every step; degrees of freedom:", simulation_setup["ndof"])

    batch = {
        "species": species,
        "coordinates": coordinates,
        "batch_index": bead_index,
        "natoms": natoms,
        "total_charge": total_charge,
    }
    if cell is not None:
        cell = cell[None, :, :]
        reciprocal_cell = reciprocal_cell[None, :, :]
        if nreplicas is not None:
            cell = np.repeat(cell, nreplicas, axis=0)
            reciprocal_cell = np.repeat(reciprocal_cell, nreplicas, axis=0)
        batch["cells"] = cell
        batch["reciprocal_cells"] = reciprocal_cell
    
    for key, value in additional_keys.items():
        batch[key] = value
    
    batch["flags"] = flags

    return simulation_setup, batch


def initialize_preprocessing(simulation_parameters, model, batch, simulation_setup):
    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    """@keyword[fennol_md] nblist_verbose
    Print detailed neighbor list information.
    Default: False
    """
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    """@keyword[fennol_md] nblist_skin
    Neighbor list skin distance in Angstroms.
    Default: -1.0 (no skin)
    """

    ### CONFIGURE PREPROCESSING
    preproc_state = unfreeze(model.preproc_state)
    layer_state = []
    for st in preproc_state["layers_state"]:
        stnew = unfreeze(st)
        if nblist_skin > 0:
            stnew["nblist_skin"] = nblist_skin
        if "nblist_mult_size" in simulation_parameters:
            stnew["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]
            """@keyword[fennol_md] nblist_mult_size
            Multiplier for neighbor list size.
            Default: None
            """
        if "nblist_add_neigh" in simulation_parameters:
            stnew["add_neigh"] = simulation_parameters["nblist_add_neigh"]
            """@keyword[fennol_md] nblist_add_neigh
            Additional neighbors to include in lists.
            Default: None
            """
        layer_state.append(freeze(stnew))
    preproc_state["layers_state"] = layer_state
    preproc_state = freeze(preproc_state)

    ## initial preprocessing
    preproc_state = preproc_state.copy({"check_input": True})
    preproc_state, batch = model.preprocessing(preproc_state, batch)

    preproc_state = preproc_state.copy({"check_input": False})

    if nblist_verbose:
        graphs_keys = list(model._graphs_properties.keys())
        print("# graphs_keys: ", graphs_keys)
        print("# nblist state:", preproc_state)

    ### print model
    if simulation_parameters.get("print_model", False):
        """@keyword[fennol_md] print_model
        Print detailed model information at startup.
        Default: False
        """
        print(model.summarize(example_data=batch))

    return preproc_state, batch



