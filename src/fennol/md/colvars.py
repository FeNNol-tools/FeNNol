import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from pathlib import Path

from ..utils import read_tinker_interval
from ..utils.io import last_xyz_frame
from .utils import align_vectors


def get_coms(coordinates, *groups, cell=None):
    """
    Calculate the barycenter for each group of atoms in the coordinates.
    Each group is a list of atom indices.
    """
    coms = []
    for group in groups:
        if isinstance(group, int) or group.ndim == 0:
            com = coordinates[group]
        elif len(group) == 1:
            com = coordinates[group[0]]
        else:
            group_coordinates = coordinates[group]
            if cell is not None:
                anchor = group_coordinates[0]
                group_coordinates = anchor + minimage(group_coordinates - anchor, cell)
            com = jnp.mean(group_coordinates, axis=0)
        coms.append(com)
    if len(coms) == 1:
        return coms[0]
    if len(coms) == 0:
        raise ValueError("No groups provided to get_coms")
    return tuple(coms)

def minimage(v, cell=None):
    if cell is None:
        return v
    cell_inv = jnp.linalg.inv(cell)
    v_frac = jnp.dot(v, cell_inv)
    v_frac = v_frac - jnp.round(v_frac)
    return jnp.dot(v_frac, cell)


def build_colvar_coordinate(colvar_name, colvar_def, global_colvars={}):
    """Return one Cartesian component of a selected atom as a collective variable.

    Atom indices follow the 1-based convention used by the other atom-based
    collective variables. ``axis`` may be ``x``, ``y``, ``z`` or an integer
    component index.
    """
    atom = int(colvar_def["atom"]) - 1
    axis = colvar_def.get("axis", "x")
    if isinstance(axis, str):
        try:
            axis = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        except KeyError as error:
            raise ValueError(
                f"Colvar {colvar_name} axis must be x, y, z, 0, 1, or 2"
            ) from error
    axis = int(axis)
    if atom < 0:
        raise ValueError(f"Colvar {colvar_name} atom index must be greater than 0")
    if axis not in (0, 1, 2):
        raise ValueError(f"Colvar {colvar_name} axis must be x, y, z, 0, 1, or 2")

    def colvar_coordinate(coordinates, cell=None):
        if atom >= coordinates.shape[0]:
            raise ValueError(
                f"Colvar {colvar_name} atom index {atom + 1} exceeds "
                f"the {coordinates.shape[0]} available atoms"
            )
        return coordinates[atom, axis]

    return colvar_coordinate


def build_colvar_distance(colvar_name, colvar_def, global_colvars={}):
    atoms = colvar_def.get("atoms", None)
    if atoms is not None:
        atoms = np.array(colvar_def["atoms"], dtype=int) - 1
        assert (
            atoms.ndim == 1 and atoms.shape[0] == 2
        ), f"Colvar {colvar_name} atoms must be a list of shape [2], got shape {atoms.shape}"
        assert np.all(atoms >= 0), f"atom indices must be > 0"
        atom1 = atoms[0]
        atom2 = atoms[1]
    else:
        atom1 = read_tinker_interval(colvar_def["group1"])
        atom2 = read_tinker_interval(colvar_def["group2"])

    def colvar_distance(coordinates,cell=None):
        assert np.all(atom1 < coordinates.shape[0]) and np.all(
            atom2 < coordinates.shape[0]
        ), f"atom indices must be < {coordinates.shape[0]}"

        v1, v2 = get_coms(coordinates, atom1, atom2, cell=cell)
        return jnp.linalg.norm(minimage(v1 - v2, cell))

    return colvar_distance

def build_colvar_coordnum(colvar_name, colvar_def, global_colvars={}):
    atom1 = read_tinker_interval(colvar_def["group1"])
    atom2 = read_tinker_interval(colvar_def["group2"])
    cutoff = colvar_def.get("cutoff",4.0)
    expnumer = colvar_def.get("expnumer",6.0)
    expdenom = colvar_def.get("expdenom",12.0)

    def colvar_coordnum(coordinates,cell=None):
        assert np.all(atom1 < coordinates.shape[0]) and np.all(
            atom2 < coordinates.shape[0]
        ), f"atom indices must be < {coordinates.shape[0]}"

        coord_group1 = coordinates[atom1]
        coord_group2 = coordinates[atom2]
        v12 = minimage(coord_group1[:,None,:]-coord_group2[None,:,:],cell)
        d12 = jnp.linalg.norm(v12,axis=-1)/cutoff
        near_cutoff = jnp.abs(d12 - 1.0) < 1e-4
        # The switching ratio has a removable 0/0 at d=1. Use its first
        # order Taylor expansion in a small neighborhood to retain both the
        # analytic value and derivative there; avoid evaluating 0/0 in the
        # unused branch so autodiff stays finite.
        d_safe = jnp.where(near_cutoff, 0.5, d12)
        regular = (1.0 - d_safe**expnumer) / (1.0 - d_safe**expdenom)
        limit = expnumer / expdenom
        near = limit * (1.0 + 0.5 * (expnumer - expdenom) * (d12 - 1.0))
        return jnp.sum(jnp.where(near_cutoff, near, regular))

    return colvar_coordnum


def build_colvar_angle(colvar_name, colvar_def, global_colvars={}):
    atoms = colvar_def.get("atoms", None)
    if atoms is not None:
        atoms = np.array(colvar_def["atoms"], dtype=int) - 1
        assert np.all(atoms >= 0), f"atom indices must be > 0"
        assert (
            atoms.ndim == 1 and atoms.shape[0] == 3
        ), f"Colvar {colvar_name} atoms must be a list of shape [3], got shape {atoms.shape}"
        atom1 = atoms[0]
        atom2 = atoms[1]
        atom3 = atoms[2]
    else:
        atom1 = read_tinker_interval(colvar_def["group1"])
        atom2 = read_tinker_interval(colvar_def["group2"])
        atom3 = read_tinker_interval(colvar_def["group3"])
    use_radians = colvar_def.get("use_radians", False)
    fact = 1.0 if use_radians else 180.0 / np.pi

    def colvar_angle(coordinates,cell=None):
        assert (
            np.all(atom1 < coordinates.shape[0])
            and np.all(atom2 < coordinates.shape[0])
            and np.all(atom3 < coordinates.shape[0])
        ), f"atom indices must be < {coordinates.shape[0]}"
        v1, v2, v3 = get_coms(coordinates, atom1, atom2, atom3, cell=cell)
        v21 = minimage(v1 - v2,cell)
        v23 = minimage(v3 - v2,cell)
        v21 = v21 / jnp.linalg.norm(v21)
        v23 = v23 / jnp.linalg.norm(v23)
        return jnp.arccos(jnp.dot(v21, v23)) * fact

    return colvar_angle


def build_colvar_dihedral(colvar_name, colvar_def, global_colvars={}):
    atoms = colvar_def.get("atoms", None)
    if atoms is not None:
        atoms = np.array(colvar_def["atoms"], dtype=int) - 1
        assert np.all(atoms >= 0), f"atom indices must be > 0"
        assert (
            atoms.ndim == 1 and atoms.shape[0] == 4
        ), f"Colvar {colvar_name} atoms must be a list of shape [4], got shape {atoms.shape}"
        atom1 = atoms[0]
        atom2 = atoms[1]
        atom3 = atoms[2]
        atom4 = atoms[3]
    else:
        atom1 = read_tinker_interval(colvar_def["group1"])
        atom2 = read_tinker_interval(colvar_def["group2"])
        atom3 = read_tinker_interval(colvar_def["group3"])
        atom4 = read_tinker_interval(colvar_def["group4"])
    use_radians = colvar_def.get("use_radians", False)
    fact = 1.0 if use_radians else 180.0 / np.pi

    def colvar_dihedral(coordinates,cell=None):
        assert (
            np.all(atom1 < coordinates.shape[0])
            and np.all(atom2 < coordinates.shape[0])
            and np.all(atom3 < coordinates.shape[0])
            and np.all(atom4 < coordinates.shape[0])
        ), f"atom indices must be < {coordinates.shape[0]}"
        v1, v2, v3, v4 = get_coms(
            coordinates, atom1, atom2, atom3, atom4, cell=cell
        )
        v12 = minimage(v2 - v1, cell)
        v23 = minimage(v3 - v2, cell)
        v34 = minimage(v4 - v3, cell)
        n1 = jnp.cross(v12, v23)
        n2 = jnp.cross(v23, v34)
        n1 = n1 / jnp.linalg.norm(n1)
        n2 = n2 / jnp.linalg.norm(n2)
        cos_phi = (n1 * n2).sum()
        sin_phi = jnp.dot(jnp.cross(n1, n2), v23 / jnp.linalg.norm(v23))
        phi = jnp.arctan2(sin_phi, cos_phi)
        return phi * fact

    return colvar_dihedral


def build_colvar_maxdcom(colvar_name, colvar_def, global_colvars={}):
    atoms = read_tinker_interval(colvar_def["atoms"])

    def colvar_maxdcom(coordinates, cell=None):
        sel_coord = coordinates[atoms]
        com = jnp.mean(sel_coord, axis=0, keepdims=True)
        return jnp.max(jnp.linalg.norm(sel_coord - com, axis=1))

    return colvar_maxdcom


def build_gyration_radius(colvar_name, colvar_def, global_colvars={}):
    atoms = read_tinker_interval(colvar_def["atoms"])

    def colvar_gyration(coordinates, cell=None):
        coords = coordinates[atoms]
        coords -= jnp.mean(coords, axis=0, keepdims=True)
        return jnp.sqrt(jnp.sum(coords**2) / coords.shape[0])

    return colvar_gyration


def build_colvar_rmsd(colvar_name, colvar_def, global_colvars={}):
    atoms = read_tinker_interval(colvar_def["atoms"])
    ref_coords = colvar_def["ref_coords"]
    if isinstance(ref_coords, str):
        xyzfile = Path(ref_coords).expanduser().resolve()
        assert (
            xyzfile.is_file()
        ), f"Reference coordinates file '{xyzfile}' does not exist"
        has_comment_line = colvar_def.get("has_comment_line", True)
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
    ref_coords -= jnp.mean(
        ref_coords, axis=0, keepdims=True
    )  # Center the reference coordinates

    def colvar_rmsd(coordinates, cell=None):
        coords = coordinates[atoms]
        coords = coords - jnp.mean(
            coords, axis=0, keepdims=True
        )  # Center the coordinates
        return align_vectors(
            coords,
            ref_coords,
            apply_shift=False,
            return_rotation_matrix=False,
            return_rmsd=True,
        )

    return colvar_rmsd


__RAW_COLVAR = {
    "coordinate": build_colvar_coordinate,
    "distance": build_colvar_distance,
    "angle": build_colvar_angle,
    "dihedral": build_colvar_dihedral,
    "torsion": build_colvar_dihedral,
    "maxdcom": build_colvar_maxdcom,
    "rmsd": build_colvar_rmsd,
    "gyration_radius": build_gyration_radius,
    "rgyr": build_gyration_radius,
    "coordnum": build_colvar_coordnum,
}


def build_colvar_function(colvar_name, colvar_def, global_colvars={}):
    func_def = colvar_def["lambda"]
    assert isinstance(
        func_def, str
    ), f"'lambda' field of colvar '{colvar_name}' must be a string (try using quotation marks)"
    arguments = [s.strip() for s in func_def.split(":")[0].split(",")]
    _cvs = []
    for cv_name in arguments:
        if cv_name in colvar_def:
            cv_def = colvar_def[cv_name]
        elif cv_name in global_colvars:
            cv_def = global_colvars[cv_name]
        else:
            raise ValueError(
                f"Colvar '{colvar_name}' references colvar '{cv_name}' which is not defined in the colvars section"
            )
        cv_type = str(cv_def.get("type", "distance")).lower()
        assert (
            cv_type in __RAW_COLVAR
        ), f"Unknown colvar type '{cv_type}' for colvar '{colvar_name}/{cv_name}'. Available colvars are {list(__RAW_COLVAR.keys())}"
        _cvs.append(__RAW_COLVAR[cv_type](cv_name, cv_def))

    func = eval(
        "lambda " + func_def,
        {
            "__builtins__": None,
            **jax.nn.__dict__,
            **jax.numpy.__dict__,
            **jax.__dict__,
        },
    )

    def colvar_function(coordinates, cell=None):
        cv_values = []
        for cv in _cvs:
            cv_values.append(cv(coordinates, cell))
        return func(*cv_values)

    return colvar_function


__BUILD_COLVAR = {
    **__RAW_COLVAR,
    "function": build_colvar_function,
}


def setup_colvars(colvars_definitions):
    colvars = {}
    for colvar_name, colvar_def in colvars_definitions.items():
        colvar_type = str(colvar_def.get("type", "distance")).lower()
        assert (
            colvar_type in __BUILD_COLVAR
        ), f"Unknown colvar type '{colvar_type}' for colvar '{colvar_name}'. Available colvars are {list(__BUILD_COLVAR.keys())}"
        colvars[colvar_name] = __BUILD_COLVAR[colvar_type](
            colvar_name, colvar_def, colvars_definitions
        )

    return colvars, list(colvars.keys())
