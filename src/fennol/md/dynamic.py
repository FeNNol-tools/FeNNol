import sys, io
import argparse
import time
from pathlib import Path

import numpy as np
from collections import defaultdict
import jax
import jax.numpy as jnp
import pickle
import yaml

from ..utils.io import (
    write_arc_frame,
    write_xyz_frame,
    write_extxyz_frame,
    human_time_duration,
)
from .utils import build_wrapbox, save_dynamics_restart, us
from ..utils import minmaxone, read_tinker_interval
from ..utils.input_parser import parse_input, convert_dict_units, InputFile
from .integrate import initialize_dynamics
from ..utils.device import configure_device, resolve_device


def main():
    """
    Main entry point for the fennol_md command-line interface.

    Parses command-line arguments and runs a molecular dynamics simulation
    based on the provided parameter file.

    Command-line Usage:
        fennol_md input.fnl
        fennol_md config.yaml
        fennol_md input.fnl --device cuda:1

    Returns:
        int: Exit code (0 for success)
    """
    # os.environ["OMP_NUM_THREADS"] = "1"
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )

    ### Read the parameter file
    parser = argparse.ArgumentParser(prog="fennol_md")
    parser.add_argument("param_file", type=Path, help="Parameter file")
    parser.add_argument(
        "--device",
        type=str,
        help="device to use, overrides FENNOL_DEVICE and the parameter file",
    )
    args = parser.parse_args()
    param_file = args.param_file.expanduser().resolve()

    return config_and_run_dynamic(param_file, device=args.device)


def config_and_run_dynamic(param_file: Path, device: str | None = None):
    """
    Configure and run a molecular dynamics simulation.

    This function loads simulation parameters from a configuration file,
    sets up the computation device and precision, and runs the MD simulation.

    Parameters:
        param_file (Path): Path to the parameter file (.fnl, .yaml, or .yml)
        device (str, optional): Device overriding FENNOL_DEVICE and the
            parameter file

    Returns:
        int: Exit code (0 for success)

    Raises:
        FileNotFoundError: If the parameter file doesn't exist
        ValueError: If the parameter file format is unsupported

    Supported file formats:
        - .fnl: FeNNol native format
        - .yaml/.yml: YAML format

    Internal units are specified by UnitSystem(L="ANGSTROM", T="PS", E="KCALPERMOL")

    Unit conversion in the parameter file:
        - Units specified in brackets: dt[fs] = 0.5
        - All specified units converted to internal units
        - non-specified units are assumed to be in internal units
    """

    if not param_file.exists() and not param_file.is_file():
        raise FileNotFoundError(f"Parameter file {param_file} not found")

    if param_file.suffix in [".yaml", ".yml"]:
        with open(param_file, "r") as f:
            simulation_parameters = convert_dict_units(yaml.safe_load(f), us=us)
            simulation_parameters = InputFile(**simulation_parameters)
    elif param_file.suffix == ".fnl":
        simulation_parameters = parse_input(param_file, us=us)
    else:
        raise ValueError(
            f"Unknown parameter file format '{param_file.suffix}'. Supported formats are '.yaml', '.yml' and '.fnl'"
        )

    ### Set the device
    device = resolve_device(
        cli=device, input_file=simulation_parameters.get("device")
    )
    """@keyword[fennol_md] device
    Computation device. Options: 'cpu', 'gpu', 'cuda', 'cuda:N', 'gpu:N' where N is
    the index among the GPUs exposed by CUDA_VISIBLE_DEVICES.
    Overridden by the FENNOL_DEVICE environment variable, itself overridden by
    the --device command-line argument.
    Default: 'cpu'
    """
    jax_device = configure_device(device)

    ### Set the precision
    enable_x64 = simulation_parameters.get("double_precision", False)
    """@keyword[fennol_md] double_precision
    Enable double precision (64-bit) calculations. Default is single precision (32-bit).
    Default: False
    """
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"

    matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()
    """@keyword[fennol_md] matmul_prec
    Matrix multiplication precision. Options: 'default', 'high', 'highest'.
    Default: "highest"
    """
    assert matmul_precision in [
        "default",
        "high",
        "highest",
    ], "matmul_prec must be one of 'default','high','highest'"
    if matmul_precision != "highest":
        print(f"# Setting matmul precision to '{matmul_precision}'")
    if matmul_precision == "default" and fprec == "float32":
        print(
            "# Warning: default matmul precision involves float16 operations which may lead to large numerical errors on energy and pressure estimations ! It is recommended to set matmul_prec to 'high' or 'highest'."
        )
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    # with jax.default_device(_device):
    return dynamic(simulation_parameters, jax_device, fprec)


def dynamic(simulation_parameters, device, fprec):
    """
    Execute the molecular dynamics simulation loop.

    This function performs the main MD simulation using the initialized
    system, integrator, and thermostats/barostats. It handles trajectory
    output, property monitoring, and restart file generation.

    Parameters:
        simulation_parameters: Parsed simulation parameters
        device (jax.Device): Computation device selected by configure_device
        fprec (str): Floating point precision ('float32' or 'float64')

    Returns:
        int: Exit code (0 for success)
    """
    tstart_dyn = time.time()

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    """@keyword[fennol_md] random_seed
    Random seed for reproducible simulations. If not specified, a random seed is generated.
    Default: Random integer between 0 and 2^32-1
    """
    print(f"# random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)

    ## INITIALIZE INTEGRATOR AND SYSTEM
    rng_key, subkey = jax.random.split(rng_key)
    step, update_batch, simulation_setup, step_state, batch, dynamics_state = (
        initialize_dynamics(simulation_parameters, fprec, subkey)
    )

    nat = simulation_setup["nat"]
    ndof = simulation_setup.get("ndof", 3 * simulation_setup["nfree"])
    dt = simulation_setup["dt"]
    ## get number of steps
    nsteps = int(simulation_parameters.get("nsteps"))
    """@keyword[fennol_md] nsteps
    Total number of MD steps to perform. Required parameter.
    Type: int, Required
    """
    start_time_ps = simulation_setup.get("start_time_ps", 0.0)

    ### Set I/O parameters
    Tdump = simulation_parameters.get("tdump", 1.0 / us.PS)
    """@keyword[fennol_md] tdump
    Time interval between trajectory frames.
    Default: 1.0 ps
    """
    ndump = max(1, int(Tdump / dt))
    system_name = simulation_setup["name"]
    estimate_pressure = simulation_setup["estimate_pressure"]

    @jax.jit
    def check_nan(dynamics_state):
        return jnp.any(jnp.isnan(dynamics_state["vel"])) | jnp.any(
            jnp.isnan(dynamics_state["coordinates"])
        )

    if simulation_setup["pbc"] is not None:
        cell = dynamics_state["cell"]
        reciprocal_cell = np.linalg.inv(cell)
        do_wrap_box = simulation_parameters.get("wrap_box", False)
        """@keyword[fennol_md] wrap_box
        Wrap coordinates into primary unit cell.
        Default: False
        """
        if do_wrap_box:
            wrap_groups_def = simulation_parameters.get("wrap_groups", None)
            """@keyword[fennol_md] wrap_groups
            Specific atom groups to wrap independently. Dictionary mapping group names to atom indices.
            Default: None
            """
            wrapbox = build_wrapbox(simulation_setup,wrap_groups_def)

    else:
        cell = None
        reciprocal_cell = None
        do_wrap_box = False

    ### Energy units and print initial energy
    model_energy_unit = simulation_setup["model_energy_unit"]
    model_energy_unit_str = simulation_setup["model_energy_unit_str"]
    per_atom_energy = simulation_parameters.get("per_atom_energy", True)
    """@keyword[fennol_md] per_atom_energy
    Print energies per atom instead of total energies.
    Default: True
    """
    energy_unit_str = simulation_setup["energy_unit_str"]
    energy_unit = simulation_setup["energy_unit"]
    print("# Energy unit: ", energy_unit_str)
    atom_energy_unit = energy_unit
    atom_energy_unit_str = energy_unit_str
    if per_atom_energy:
        atom_energy_unit /= nat
        atom_energy_unit_str = f"{energy_unit_str}/atom"
        print("# Printing Energy per atom")
    print(
        f"# Initial potential energy: {dynamics_state['epot']*atom_energy_unit}; kinetic energy: {dynamics_state['ek']*atom_energy_unit}"
    )
    f = dynamics_state["forces"]
    minmaxone(jnp.abs(f * energy_unit), "# forces min/max/rms:")

    ## printing options
    nprint = int(simulation_parameters.get("nprint", 10))
    """@keyword[fennol_md] nprint
    Number of steps between energy/property printing.
    Default: 10
    """
    assert nprint > 0, "nprint must be > 0"
    nsummary = simulation_parameters.get("nsummary", 100 * nprint)
    """@keyword[fennol_md] nsummary
    Number of steps between summary statistics.
    Default: 100 * nprint
    """
    assert nsummary > nprint, "nsummary must be > nprint"

    save_keys = simulation_parameters.get("save_keys", [])
    """@keyword[fennol_md] save_keys
    Additional model output keys to save to trajectory.
    Default: []
    """
    if save_keys:
        print(f"# Saving keys: {save_keys}")
        fkeys = open(f"{system_name}.traj.pkl", "wb+")
    else:
        fkeys = None

    ### initialize colvars
    use_colvars = "colvars" in simulation_setup
    if use_colvars:
        print(f"# Colvars: {simulation_setup['colvars']}")
        colvars_names = simulation_setup["colvars"]
        # open colvars file and print header
        fcolvars = open(f"{system_name}.colvars.traj", "a")
        fcolvars.write("#time[ps] ")
        fcolvars.write(" ".join(colvars_names))
        fcolvars.write("\n")
        fcolvars.flush()

    # Instantaneous bias energies, for reweighting: written to bias.traj at the
    # same stride as colvars.traj, and as properties of extxyz frames.
    bias_names = list(dynamics_state["ebias"].keys()) if "ebias" in dynamics_state else []
    write_bias_file = bool(bias_names)
    if write_bias_file:
        fbias = open(f"{system_name}.bias.traj", "a")
        fbias.write("#time[ps] " + " ".join(f"E{b}[{energy_unit_str}]" for b in bias_names) + "\n")
        fbias.flush()

    ### Print header
    include_thermostat_energy = "thermostat_energy" in dynamics_state["thermostat"]
    use_lambdadyn = "lambdadyn" in dynamics_state
    use_biases = "ebias" in dynamics_state
    thermostat_name = simulation_setup["thermostat_name"]
    pimd = simulation_setup["pimd"]
    variable_cell = simulation_setup["variable_cell"]
    nbeads = simulation_setup.get("nbeads", 1)
    dyn_name = "PIMD" if pimd else "MD"
    print("#" * 84)
    print(
        f"# Running {nsteps:_} steps of {thermostat_name} {dyn_name} simulation on {device}"
    )
    header = f"#{'Step':>9} {'Time[ps]':>10} {'Etot':>10} {'Epot':>10} {'Ekin':>10} {'Temp[K]':>10}"
    if pimd:
        header += f" {'Temp_c[K]':>10}"
    if include_thermostat_energy:
        header += f" {'Etherm':>10}"
    if estimate_pressure:
        print_aniso_pressure = simulation_parameters.get("print_aniso_pressure", False)
        """@keyword[fennol_md] print_aniso_pressure
        Print anisotropic pressure tensor components.
        Default: False
        """
        pressure_unit_str = simulation_parameters.get("pressure_unit", "atm")
        """@keyword[fennol_md] pressure_unit
        Pressure unit for output. Options: 'atm', 'bar', 'Pa', 'GPa'.
        Default: "atm"
        """
        pressure_unit = us.get_multiplier(pressure_unit_str)
        p_str = f"  Press[{pressure_unit_str}]"
        header += f" {p_str:>10}"
    if variable_cell:
        header += f" {'Density':>10}"
    if use_lambdadyn:
        header += f" {'Lambda':>10}"
    print(header)

    ### Open trajectory file
    traj_format = simulation_parameters.get("traj_format", "arc").lower()
    """@keyword[fennol_md] traj_format
    Trajectory file format. Options: 'arc' (Tinker), 'xyz' (standard), 'extxyz' (extended).
    Only extxyz frames carry properties: time, potential energy and the
    instantaneous energy of every bias term (E<name>), which is what is
    needed to reweight observables computed from the frames.
    Default: "arc"
    """
    if traj_format == "xyz":
        traj_ext = ".traj.xyz"
        write_frame = write_xyz_frame
    elif traj_format == "extxyz":
        traj_ext = ".traj.extxyz"
        write_frame = write_extxyz_frame
    elif traj_format == "arc":
        traj_ext = ".arc"
        write_frame = write_arc_frame
    else:
        raise ValueError(
            f"Unknown trajectory format '{traj_format}'. Supported formats are 'arc' and 'xyz'"
        )

    write_all_beads = simulation_parameters.get("write_all_beads", False) and pimd
    """@keyword[fennol_md] write_all_beads
    Write all PIMD beads to separate trajectory files.
    Default: False
    """

    if write_all_beads:
        fout = [
            open(f"{system_name}_bead{i+1:03d}" + traj_ext, "a") for i in range(nbeads)
        ]
    else:
        fout = open(system_name + traj_ext, "a")

    ensemble_key = simulation_parameters.get("etot_ensemble_key", None)
    """@keyword[fennol_md] etot_ensemble_key
    Key for ensemble weighting in enhanced sampling.
    Default: None
    """
    if ensemble_key is not None:
        fens = open(f"{system_name}.ensemble_weights.traj", "a")

    write_centroid = simulation_parameters.get("write_centroid", False) and pimd
    """@keyword[fennol_md] write_centroid
    Write PIMD centroid coordinates to separate file.
    Default: False
    """
    if write_centroid:
        fcentroid = open(f"{system_name}_centroid" + traj_ext, "a")

    ### initialize proprerty trajectories
    properties_traj = defaultdict(list)

    ### initialize counters and timers
    t0 = time.time()
    t0dump = t0
    istep = 0
    t0full = time.time()
    force_preprocess = False
    temper_mean = 0.0
    print_count = 0
    check_temper = simulation_parameters.get("check_temperature", True)
    """@keyword[fennol_md] check_temperature
    Check temperature stability during the simulation. If temperature deviates too much from the mean, an error is raised.
    Default: True
    """

    for istep in range(1, nsteps + 1):

        ### update the system
        step_state, dynamics_state, batch, model_output = step(
            istep, step_state, dynamics_state, batch, force_preprocess
        )
        force_preprocess = False

        ### print properties
        if istep % nprint == 0:
            t1 = time.time()
            tperstep = (t1 - t0) / nprint
            t0 = t1
            nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6

            ek = dynamics_state["ek"]
            epot = dynamics_state["epot"]
            etot = ek + epot
            temper = temper_c = 2 * ek / ndof * us.KELVIN

            th_state = dynamics_state["thermostat"]
            if include_thermostat_energy:
                etherm = th_state["thermostat_energy"]
                etot = etot + etherm

            properties_traj[f"Etot[{atom_energy_unit_str}]"].append(
                etot * atom_energy_unit
            )
            properties_traj[f"Epot[{atom_energy_unit_str}]"].append(
                epot * atom_energy_unit
            )
            properties_traj[f"Ekin[{atom_energy_unit_str}]"].append(
                ek * atom_energy_unit
            )
            properties_traj["Temper[Kelvin]"].append(temper)
            if pimd:
                ek_c = dynamics_state["ek_c"]
                temper_c = 2 * ek_c / ndof * us.KELVIN
                properties_traj["Temper_c[Kelvin]"].append(temper_c)

            ### construct line of properties
            simulated_time_ps = start_time_ps + istep * dt * us.PS
            line = f"{istep:10.6g} {simulated_time_ps:10.3f} {etot*atom_energy_unit:#10.4f}  {epot*atom_energy_unit:#10.4f} {ek*atom_energy_unit:#10.4f} {temper:10.2f}"
            if pimd:
                line += f" {temper_c:10.2f}"
            if include_thermostat_energy:
                line += f" {etherm*atom_energy_unit:#10.4f}"
                properties_traj[f"Etherm[{atom_energy_unit_str}]"].append(
                    etherm * atom_energy_unit
                )
            if estimate_pressure:
                pres = dynamics_state["pressure"] * pressure_unit
                properties_traj[f"Pressure[{pressure_unit_str}]"].append(pres)
                if print_aniso_pressure:
                    pres_tensor = dynamics_state["pressure_tensor"] * pressure_unit
                    pres_tensor = 0.5 * (pres_tensor + pres_tensor.T)
                    properties_traj[f"Pressure_xx[{pressure_unit_str}]"].append(
                        pres_tensor[0, 0]
                    )
                    properties_traj[f"Pressure_yy[{pressure_unit_str}]"].append(
                        pres_tensor[1, 1]
                    )
                    properties_traj[f"Pressure_zz[{pressure_unit_str}]"].append(
                        pres_tensor[2, 2]
                    )
                    properties_traj[f"Pressure_xy[{pressure_unit_str}]"].append(
                        pres_tensor[0, 1]
                    )
                    properties_traj[f"Pressure_xz[{pressure_unit_str}]"].append(
                        pres_tensor[0, 2]
                    )
                    properties_traj[f"Pressure_yz[{pressure_unit_str}]"].append(
                        pres_tensor[1, 2]
                    )
                line += f" {pres:10.3f}"
            if variable_cell:
                density = dynamics_state["density"]
                properties_traj["Density[g/cm^3]"].append(density)
                if print_aniso_pressure:
                    cell = dynamics_state["cell"]
                    properties_traj[f"Cell_Ax[Angstrom]"].append(cell[0, 0])
                    properties_traj[f"Cell_Ay[Angstrom]"].append(cell[0, 1])
                    properties_traj[f"Cell_Az[Angstrom]"].append(cell[0, 2])
                    properties_traj[f"Cell_Bx[Angstrom]"].append(cell[1, 0])
                    properties_traj[f"Cell_By[Angstrom]"].append(cell[1, 1])
                    properties_traj[f"Cell_Bz[Angstrom]"].append(cell[1, 2])
                    properties_traj[f"Cell_Cx[Angstrom]"].append(cell[2, 0])
                    properties_traj[f"Cell_Cy[Angstrom]"].append(cell[2, 1])
                    properties_traj[f"Cell_Cz[Angstrom]"].append(cell[2, 2])
                line += f" {density:10.4f}"
                if "piston_temperature" in dynamics_state["barostat"]:
                    piston_temperature = dynamics_state["barostat"]["piston_temperature"]
                    properties_traj["T_piston[Kelvin]"].append(piston_temperature)

            if use_lambdadyn:
                alch_lambda = float(dynamics_state["lambdadyn"]["alch_lambda"])
                line += f" {alch_lambda:10.4f}"
            if use_biases:
                for bias_name, bias_energy in dynamics_state["ebias"].items():
                    properties_traj[f"E{bias_name}[{energy_unit_str}]"].append(
                        bias_energy * energy_unit
                    )
            print(line)

            if check_temper:
                print_count += 1
                temper_mean += (temper_c - temper_mean) / print_count
                if (
                    print_count > 10
                    and temper_c - temper_mean > 10 * temper_mean / (0.5 * ndof) ** 0.5
                ):
                    print(
                        f"# Temperature {temper_c:.2f} K deviates too much from the mean {temper_mean:.2f} K. Dynamics may have diverged. Exiting"
                    )
                    raise ValueError(f"Dynamics diverged.")

            ### save colvars and bias energies
            if use_colvars:
                colvars = dynamics_state["colvars"]
                fcolvars.write(f"{simulated_time_ps:.6f} ")
                fcolvars.write(" ".join([f"{colvars[k]:.8e}" for k in colvars_names]))
                fcolvars.write("\n")
                fcolvars.flush()
            if write_bias_file:
                fbias.write(f"{simulated_time_ps:.6f} ")
                fbias.write(" ".join(f"{float(dynamics_state['ebias'][b]) * energy_unit:.8e}" for b in bias_names))
                fbias.write("\n")
                fbias.flush()

            if save_keys:
                data = {
                    k: (
                        np.asarray(model_output[k])
                        if isinstance(model_output[k], jnp.ndarray)
                        else model_output[k]
                    )
                    for k in save_keys
                }
                data["properties"] = {
                    k: float(v) for k, v in zip(header.split()[1:], line.split())
                }
                data["properties"]["properties_energy_unit"] = (
                    atom_energy_unit,
                    atom_energy_unit_str,
                )
                data["properties"]["model_energy_unit"] = (
                    model_energy_unit,
                    model_energy_unit_str,
                )

                pickle.dump(data, fkeys)

        ### save frame
        if istep % ndump == 0:
            line = "# Write XYZ frame"
            if variable_cell:
                cell = np.array(dynamics_state["cell"])
                reciprocal_cell = np.linalg.inv(cell)
            if do_wrap_box:
                if pimd:
                    centroid = wrapbox(dynamics_state["coordinates"][0], cell, reciprocal_cell)
                    dynamics_state["coordinates"] = dynamics_state["coordinates"].at[0].set(centroid)
                else:
                    dynamics_state["coordinates"] = wrapbox(
                        dynamics_state["coordinates"], cell, reciprocal_cell
                    )
                batch = update_batch(batch, dynamics_state)
                line += " (atoms have been wrapped into the box)"
                force_preprocess = True
            print(line)

            save_dynamics_restart(simulation_setup, batch, step_state, dynamics_state)

            properties = {
                "energy": float(dynamics_state["epot"]) * energy_unit,
                "Time_ps": start_time_ps + istep * dt * us.PS,
                "energy_unit": energy_unit_str,
            }
            for bias_name in bias_names:
                properties[f"E{bias_name}"] = float(dynamics_state["ebias"][bias_name]) * energy_unit

            if write_all_beads:
                coords = np.asarray(batch["coordinates"].reshape(-1, nat, 3))
                for i, fb in enumerate(fout):
                    write_frame(
                        fb,
                        simulation_setup["symbols"],
                        coords[i],
                        cell=cell,
                        properties=properties,
                        forces=None,  # np.asarray(dynamics_state["forces"].reshape(nbeads, nat, 3)[0]) * energy_unit,
                    )
            else:
                write_frame(
                    fout,
                    simulation_setup["symbols"],
                    np.asarray(batch["coordinates"].reshape(-1, nat, 3)[0]),
                    cell=cell,
                    properties=properties,
                    forces=None,  # np.asarray(dynamics_state["forces"].reshape(nbeads, nat, 3)[0]) * energy_unit,
                )
            if write_centroid:
                centroid = np.asarray(dynamics_state["coordinates"][0])
                write_frame(
                    fcentroid,
                    simulation_setup["symbols"],
                    centroid,
                    cell=cell,
                    properties=properties,
                    forces=np.asarray(dynamics_state["forces"].reshape(nbeads, nat, 3)[0])
                    * energy_unit,
                )
            if ensemble_key is not None:
                weights = " ".join(
                    [f"{w:.6f}" for w in dynamics_state["ensemble_weights"].tolist()]
                )
                fens.write(f"{weights}\n")
                fens.flush()

        ### summary over last nsummary steps
        if istep % (nsummary) == 0:
            if check_nan(dynamics_state):
                raise ValueError(f"dynamics crashed at step {istep}.")
            tfull = time.time() - t0full
            t0full = time.time()
            tperstep = tfull / (nsummary)
            nsperday = (24 * 60 * 60 / tperstep) * dt * us.NS
            elapsed_time = time.time() - tstart_dyn
            estimated_remaining_time = tperstep * (nsteps - istep)
            estimated_total_time = elapsed_time + estimated_remaining_time

            print("#" * 50)
            print(f"# Step {istep:_} of {nsteps:_}  ({istep/nsteps*100:.5g} %)")
            print(f"# Simulated time      : {istep * dt*us.PS:.3f} ps")
            print(f"# Tot. Simu. time     : {start_time_ps + istep * dt*us.PS:.3f} ps")
            print(f"# Tot. elapsed time   : {human_time_duration(elapsed_time)}")
            print(
                f"# Est. total duration   : {human_time_duration(estimated_total_time)}"
            )
            print(
                f"# Est. remaining time : {human_time_duration(estimated_remaining_time)}"
            )
            print(f"# Time for {nsummary:_} steps : {human_time_duration(tfull)}")

            corr_kin = dynamics_state["thermostat"].get("corr_kin", None)
            if corr_kin is not None:
                print(f"# QTB kin. correction : {100*(corr_kin-1.):.2f} %")
            print(f"# Averages over last {nsummary:_} steps :")
            for k, v in properties_traj.items():
                if len(properties_traj[k]) == 0:
                    continue
                mu = np.mean(properties_traj[k])
                sig = np.std(properties_traj[k])
                ksplit = k.split("[")
                name = ksplit[0].strip()
                unit = ksplit[1].replace("]", "").strip() if len(ksplit) > 1 else ""
                print(f"#   {name:10} : {mu: #10.5g}   +/- {sig: #9.3g}  {unit}")

            print(f"# Perf.: {nsperday:.2f} ns/day  ( {1.0 / tperstep:.2f} step/s )")
            print("#" * 50)
            if istep < nsteps:
                print(header)
            ## reset property trajectories
            properties_traj = defaultdict(list)

    print(f"# Run done in {human_time_duration(time.time()-tstart_dyn)}")
    ### close trajectory file
    if isinstance(fout, list):
        for bead_file in fout:
            bead_file.close()
    else:
        fout.close()
    if ensemble_key is not None:
        fens.close()
    if use_colvars:
        fcolvars.close()
    if write_bias_file:
        fbias.close()
    if fkeys is not None:
        fkeys.close()
    if write_centroid:
        fcentroid.close()

    return 0


if __name__ == "__main__":
    main()
