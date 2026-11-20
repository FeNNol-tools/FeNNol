import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax
from enum import Enum

from .utils import us


def get_barostat(
    thermostat,
    simulation_parameters,
    dt,
    simulation_setup,
    fprec,
    rng_key=None,
    restart_data={},
):
    state = {}

    barostat_name = str(simulation_parameters.get("barostat", "NONE")).upper()
    """@keyword[fennol_md] barostat
    Type of barostat for pressure control (NONE, LGV, LANGEVIN).
    Default: NONE
    """

    kT = simulation_setup.get("kT", None)
    assert kT is not None, "kT must be specified for NPT/NPH simulations"
    target_pressure = simulation_parameters.get("target_pressure")
    """@keyword[fennol_md] target_pressure
    Target pressure for NPT ensemble simulations.
    Required for barostat != NONE
    """
    if barostat_name != "NONE":
        assert (
            target_pressure is not None
        ), "target_pressure must be specified for NPT/NPH simulations"

    nbeads = simulation_setup.get("nbeads", None)
    variable_cell = True
    default_pressure_instant = "O"

    anisotropic = simulation_parameters.get("aniso_barostat", False)
    """@keyword[fennol_md] aniso_barostat
    Use anisotropic barostat allowing independent cell parameter scaling.
    Default: False
    """

    isotropic = not anisotropic
    if anisotropic:
        aniso_mask = simulation_parameters.get(
            "aniso_mask", [True, True, True, True, True, True]
        )
        """@keyword[fennol_md] aniso_mask
        Mask for anisotropic barostat degrees of freedom [xx, yy, zz, xy, xz, yz].
        Default: [True, True, True, True, True, True]
        """
        assert len(aniso_mask) == 6, "aniso_mask must have 6 elements"
        aniso_mask = np.array(aniso_mask, dtype=bool).astype(np.int32)
        ndof_piston = np.sum(aniso_mask)
        # xx   yy   zz    xy   xz   yz
        aniso_mask = np.array(
            [
                [aniso_mask[0], aniso_mask[3], aniso_mask[4]],
                [aniso_mask[3], aniso_mask[1], aniso_mask[5]],
                [aniso_mask[4], aniso_mask[5], aniso_mask[2]],
            ],
            dtype=np.int32,
        )
    else:
        ndof_piston = 1.0


    pbc_data = simulation_setup["pbc"]
    start_barostat = simulation_parameters.get("start_barostat", 0.0)
    """@keyword[fennol_md] start_barostat
    Time delay before starting barostat pressure coupling.
    Default: 0.0
    """
    start_time = restart_data.get("simulation_time_ps", 0.0) / us.PS
    start_barostat = max(0.0, start_barostat - start_time)
    istart_barostat = int(round(start_barostat / dt))
    if istart_barostat > 0 and barostat_name not in ["NONE"]:
        print(
            f"# BAROSTAT will start at {start_barostat*us.PS:.3f} ps ({istart_barostat} steps)"
        )
    else:
        istart_barostat = 0
    
    # Isotropic barostats scale the cell exponentially; anisotropic ones apply
    # first-order lower-triangular deformations. Both sample the cell
    # uniformly (in V, or in the lower-triangular entries).
    if simulation_parameters.get("barostat_first_order", anisotropic) != anisotropic:
        raise ValueError(
            "barostat_first_order was removed: isotropic barostats always use "
            "exponential scaling and anisotropic barostats first-order scaling"
        )
    if anisotropic:
        # First-order updates apply the symmetric strain eps as the
        # lower-triangular deformation diag(eps) + 2*tril(eps, -1), which
        # equals I + eps up to a rotation and keeps the cell lower-triangular.
        tril_strain = np.tril(2.0 * np.ones((3, 3)), -1) + np.eye(3)
        # The anisotropic barostats sample the lower-triangular cell entries
        # uniformly. Off-diagonal entry h_ij (i > j) moves with the noise of
        # h_jj, so the stochastic barostats need a drift of kBT/V on P_jj per
        # active entry below the diagonal in column j. Without it, the cell
        # entries are sampled with measure 1/(h_xx**2 h_yy).
        shear_measure = np.diag(np.tril(aniso_mask, -1).sum(axis=0)).astype(float)

    if barostat_name in ["LGV", "LANGEVIN"]:
        assert rng_key is not None, "rng_key must be provided for LANGEVIN barostat"
        gamma = simulation_parameters.get("gamma_piston", 20.0 / us.THZ)
        """@keyword[fennol_md] gamma_piston
        Piston friction coefficient for Langevin barostat.
        Default: 20.0 ps^-1
        """
        tau_piston = simulation_parameters.get("tau_piston", 0.5 / us.PS)
        """@keyword[fennol_md] tau_piston
        Piston time constant for barostat coupling.
        Default: 0.5 ps
        """
        nat = simulation_setup["nat"]
        ndof = simulation_setup.get("ndof", 3 * simulation_setup["nfree"])
        masspiston = 3 * nat * kT * tau_piston**2
        print(
            f"# LANGEVIN barostat with piston mass={masspiston*us.get_multiplier('KCALPERMOL*PS^{2}'):.1e} kcal/mol.ps^2"
        )
        a1 = math.exp(-gamma * dt)
        a2 = ((1 - a1 * a1) * kT / masspiston) ** 0.5

        rng_key, v_key = jax.random.split(rng_key)
        if anisotropic:
            extvol = pbc_data["cell"]
            vextvol = (
                jax.random.normal(v_key, (3, 3), dtype=extvol.dtype)
                * (kT / masspiston) ** 0.5
            )
            vextvol = 0.5 * (vextvol + vextvol.T)

            
        else:
            extvol = jnp.asarray(pbc_data["volume"])
            vextvol = (
                jax.random.normal(v_key, (1,), dtype=extvol.dtype)
                * (kT / masspiston) ** 0.5
            )

        state["extvol"] = extvol
        state["vextvol"] = vextvol
        state["rng_key"] = rng_key
        state["istep"] = 0

        def barostat(x, vel, dynamics_state):
            if nbeads is not None:
                x, eigx = x[0], x[1:]
                vel, eigv = vel[0], vel[1:]
            barostat_state = dynamics_state["barostat"]
            extvol = barostat_state["extvol"]
            vextvol = barostat_state["vextvol"]
            cell = dynamics_state["cell"]
            volume = jnp.abs(jnp.linalg.det(cell))

            istep = barostat_state["istep"] + 1
            dt_bar = dt * (istep >= istart_barostat)

            # apply B
            # pV = 2 * (dynamics_state["ek_tensor"] + jnp.trace(dynamics_state["ek_tensor"])*jnp.eye(3)/(3*x.shape[0])) - dynamics_state["virial"]
            ek = dynamics_state["ek_c"] if nbeads is not None else dynamics_state["ek"]
            Pres = (
                dynamics_state["pressure_tensor"]
                + ek * jnp.array(np.eye(3) * (2 / ndof)) / volume
            )
            if isotropic:
                dPres = jnp.trace(Pres) - 3 * target_pressure
            else:
                dPres = (
                    0.5 * (Pres + Pres.T)
                    + jnp.array(kT * shear_measure) / volume
                    - jnp.array(target_pressure * np.eye(3))
                )

            vextvol = vextvol + ((dt_bar / masspiston) * volume) * dPres

            # apply A
            if isotropic:
                vdt2 = 0.5 * dt_bar * vextvol
                scalev = jnp.exp(-vdt2 * (1 + 3.0 / ndof))
                scale1 = jnp.exp(vdt2)
                vel = vel * scalev
            else:
                vextvol = aniso_mask * vextvol
                vdt2 = (0.5 * dt_bar) * vextvol * tril_strain
                lcorr = jnp.trace(vdt2) / ndof * jnp.eye(3)
                scalev = jnp.eye(3) - vdt2 - lcorr
                scale1 = jnp.eye(3) + vdt2
                vel = vel @ scalev

            # apply O
            if nbeads is not None:
                eigv, thermostat_state = thermostat(
                    jnp.concatenate((vel[None], eigv), axis=0), dynamics_state["thermostat"]
                )
                vel, eigv = eigv[0], eigv[1:]
            else:
                vel, thermostat_state = thermostat(vel, dynamics_state["thermostat"])
            rng_key, noise_key = jax.random.split(barostat_state["rng_key"])

            if isotropic:
                noise = jax.random.normal(noise_key, (1,), dtype=vextvol.dtype)
            else:
                noise = jax.random.normal(noise_key, (3, 3), dtype=vextvol.dtype)
                noise = 0.5 * (noise + noise.T)

            vextvol = a1 * vextvol + a2 * noise

            # apply A
            if isotropic:
                vdt2 = 0.5 * dt_bar * vextvol
                scalev = jnp.exp(-vdt2 * (1 + 3.0 / ndof))
                scale2 = jnp.exp(vdt2)
                vel = vel * scalev
                x = x * (scale1 * scale2)
                extvol = extvol * (scale1 * scale2) ** 3
                cell = cell * (scale1 * scale2)
            else:
                vextvol = aniso_mask * vextvol
                vdt2 = (0.5 * dt_bar) * vextvol * tril_strain
                lcorr = jnp.trace(vdt2) / ndof * jnp.eye(3)
                scalev = jnp.eye(3) - vdt2 - lcorr
                scale = scale1 @ (jnp.eye(3) + vdt2)
                vel = vel @ scalev
                x = x @ scale
                extvol = extvol @ scale
                cell = extvol


            if nbeads is not None:
                x = jnp.concatenate((x[None], eigx), axis=0)
                vel = jnp.concatenate((vel[None], eigv), axis=0)

            piston_temperature = (us.KELVIN * masspiston / ndof_piston) * jnp.sum(
                vextvol**2
            )
            barostat_state = {
                **barostat_state,
                "istep": istep,
                "rng_key": rng_key,
                "vextvol": vextvol,
                "extvol": extvol,
                "piston_temperature": piston_temperature,
            }
            return (
                x,
                vel,
                {
                    **dynamics_state,
                    "barostat": barostat_state,
                    "cell": cell,
                    "thermostat": thermostat_state,
                },
            )
        
    elif barostat_name in ["BERENDSEN","BUSSI","C-RESCALE","CRESCALE"]:
        default_pressure_instant = "A"

        tau_piston = simulation_parameters.get("tau_piston", 0.5 / us.PS)
        """@keyword[fennol_md] tau_piston
        Piston time constant for barostat coupling.
        Default: 0.5 ps
        """
        compress = simulation_parameters.get("compressibility", 4.e-5 * us.ATM)
        """@keyword[fennol_md] tau_piston
        System compressibility for Berendsen barostat.
        Default: 4.e-5 atm^-1 (~ water at ambient conditions)
        """
        scale_vel = simulation_parameters.get("barostat_scale_velocities", True)
        nat = simulation_setup["nat"]
        nfree = simulation_setup["nfree"]

        alpha = compress/tau_piston
        if anisotropic:
            compress_scale = simulation_parameters.get("aniso_compressibility_scale", [1]*6)
            compress_scale = np.array(compress_scale, dtype=float)
            if compress_scale.size == 1:
                compress_scale = compress_scale * np.ones(6)
            elif compress_scale.shape[0] == 2:
                compress_scale = np.array(
                    [compress_scale[0], compress_scale[0], compress_scale[0], compress_scale[1], compress_scale[1], compress_scale[1]],
                    dtype=float,
                )
            elif compress_scale.shape[0] == 3:
                compress_scale = np.array(
                    [compress_scale[0], compress_scale[1], compress_scale[2], (compress_scale[0]*compress_scale[1])**0.5, (compress_scale[0]*compress_scale[2])**0.5, (compress_scale[1]*compress_scale[2])**0.5],
                    dtype=float,
                )
            assert compress_scale.size == 6, "aniso_compressibility_scale must have 1, 2, 3 or 6 elements"

            compress_scale = np.array(
                [
                    [compress_scale[0], compress_scale[3], compress_scale[4]],
                    [compress_scale[3], compress_scale[1], compress_scale[5]],
                    [compress_scale[4], compress_scale[5], compress_scale[2]],
                ],
                dtype=float,
            )
            if "aniso_compressibility_scale" in simulation_parameters:
                print("# Barostat compressibility scale:")
                for l in compress_scale:
                    print("# ", l)
            alpha = alpha * compress_scale

        print(f"# {barostat_name} barostat with tau={tau_piston*us.PS:.1e} ps")

        state["istep"] = 0
        stochastic = barostat_name != "BERENDSEN"
        if stochastic:
            assert rng_key is not None, f"rng_key must be provided for {barostat_name} barostat"
            state["rng_key"] = rng_key


        def barostat(x, vel, dynamics_state):
            if nbeads is not None:
                x, eigx = x[0], x[1:]
                vel, eigv = vel[0], vel[1:]

            barostat_state = dynamics_state["barostat"]
            cell = dynamics_state["cell"]

            istep = barostat_state["istep"] + 1
            dt_bar = dt * (istep >= istart_barostat)
            barostat_state = {
                **barostat_state,
                "istep": istep,
            }

            Pres = dynamics_state["pressure_tensor"]
            if isotropic:
                dPres = (1./3.)*jnp.trace(Pres) -  target_pressure
            else:
                dPres = 0.5 * (Pres + Pres.T) - jnp.array(target_pressure * np.eye(3), dtype=Pres.dtype)

            # Internal pressure above target expands the cell; for isotropic
            # scaling this is one third of the log-volume increment.
            exponent = dt_bar * alpha * dPres / 3.0
            if stochastic:
                volume = jnp.abs(jnp.linalg.det(cell))
                barostat_state["rng_key"], noise_key = jax.random.split(barostat_state["rng_key"])
                sigma = jnp.sqrt((2*kT*alpha/3.)*dt_bar/volume)
                if isotropic:
                    noise = jax.random.normal(noise_key, dtype=exponent.dtype)/(3**0.5)
                else:
                    noise = jax.random.normal(noise_key,shape=(3,3), dtype=exponent.dtype)
                    noise = 0.5 * (noise + noise.T)
                    # The first-order update h(1 + eps) also needs kBT/V on
                    # every diagonal entry, which the exponential isotropic
                    # update and the Langevin kinetic term already provide;
                    # without it the cell samples V**(N-1).
                    exponent = exponent + (dt_bar * alpha / 3.0) * jnp.array(
                        kT * (np.eye(3) + shear_measure), dtype=exponent.dtype
                    ) / volume
                exponent = exponent + sigma*noise

            if anisotropic:
                # Apply the same degrees-of-freedom mask to the deterministic
                # pressure response and stochastic cell noise. Without this,
                # C-rescale can move cell components that the user fixed with
                # ``aniso_mask`` even though Berendsen and Langevin respect it.
                exponent = aniso_mask * exponent
            
            if isotropic:
                scale = jnp.exp(exponent)
                x = x * scale
                cell = cell * scale
                if scale_vel:
                    vel = vel / scale
            else:
                scale = jnp.eye(3) + tril_strain*exponent
                x = x @ scale
                cell = cell @ scale
                if scale_vel:
                    scale = jnp.eye(3) - tril_strain*exponent
                    vel = vel @ scale
            
            if nbeads is not None:
                x = jnp.concatenate((x[None], eigx), axis=0)
                vel = jnp.concatenate((vel[None], eigv), axis=0)
                
            vel, thermostat_state = thermostat(vel, dynamics_state["thermostat"])

            
            return (
                x,
                vel,
                {
                    **dynamics_state,
                    "barostat": barostat_state,
                    "cell": cell,
                    "thermostat": thermostat_state,
                },
            )


    elif barostat_name in ["NONE"]:
        variable_cell = False

        def barostat(x, vel, dynamics_state):
            vel, thermostat_state = thermostat(vel, dynamics_state["thermostat"])
            return x, vel, {**dynamics_state, "thermostat": thermostat_state}

    else:
        raise ValueError(f"Unknown barostat {barostat_name}")

    pressure_instant = str(simulation_parameters.get("pressure_instant", default_pressure_instant)).upper()
    """@keyword[fennol_md] pressure_instant
    choice of where in the BAOAB step to estimate instantaneous pressure from:
    - 'A': before thermostat step
    - 'O': after thermostat step
    - 'B': end of step
    Default: O
    """
    assert pressure_instant in ["A","O","B"], "pressure_instant must be A, O or B"

    # Restore the barostat variables after constructing the configured state.
    if "barostat" in restart_data:
        state = {**state, **restart_data["barostat"]}
    

    return barostat, variable_cell, state, pressure_instant

