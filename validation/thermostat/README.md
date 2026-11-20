# Thermostat checks

The model is 512 Ar atoms in non-interacting harmonic wells, with stiffness
(1, 2, 4) kcal/mol/Å² along x, y and z (`modules/harmonic_lattice.py`).
Every atom is an independent sample, so one 100 ps run measures
⟨k_a q_a²⟩/kBT on each axis to about 0.5%.

| Check | Pass condition |
| --- | --- |
| `sampling/LGV`, `sampling/FFLGV` | ⟨k q²⟩/kBT on each axis and the kinetic temperature within 2% |
| `sampling/LGV_dt60` | Same, at 60 fs (ω·dt = 0.39): BAOAB configurational sampling of a harmonic system must stay exact |
| `lj_nvt/{lgv,bussi,nose}` | 32-atom LJ argon liquid at fixed volume: mean kinetic energy equal to (ndof/2)kBT within max(1.5%, 3 SE), with ndof = 3N for Langevin and 3N−3 for the momentum-conserving thermostats; reported temperature within 1.5% |
| `lj_nvt/{bussi,nose}_vs_lgv` | Mean potential energy and pressure equal to Langevin's within 4 combined standard errors |
| `restart/{lgv,fflgv,bussi,nose}` | Split and uninterrupted runs leave bit-identical restart states |

Global thermostats (Bussi, Nosé–Hoover) cannot redistribute energy between
independent oscillators, because they only rescale all velocities together.
That is why they are checked on the interacting LJ liquid instead. The
kinetic-energy check uses the physical count of degrees of freedom, not the
one the code computes. Counting 3N instead of 3N−3 makes these thermostats run
3.2% hot here, which this check fails at about 6σ.
