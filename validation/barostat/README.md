# Barostat checks on Lennard-Jones argon

The model is force-shifted LJ argon: ε = 0.238 kcal/mol, σ = 3.4 Å, cutoff
3σ, no tail correction. It uses the built-in `LENNARD_JONES_FS` module with
explicit periodic images (`minimum_image: false`).

## Liquid: 32 atoms, T* = 1.0, P* = 0.8

The reference is an independent NPT Monte Carlo sampler (`npt_mc.py`). It
uses its own NumPy/Numba lattice sum, which is checked against the NIST
force-shifted configuration in `tests/`. Three chains of 65,000 sweeps give
V = 43.92 ± 0.11 σ³ and Var(V) = 6.53 ± 0.31 σ⁶. Regenerate them with
`--only reference`; this takes several minutes.

| Check | Pass condition |
| --- | --- |
| `liquid/crescale`, `liquid/langevin` | Volume mean within 1.5% and variance within 25% of MC; Langevin piston temperature within 10% of T |
| `liquid/berendsen` | Relaxes a box expanded by 20% to within 3% of the MC volume |

## Crystal: 108-atom FCC, T* = 0.5, P* = 3, float64

There is no Monte Carlo reference for the flexible cell. Instead, the
barostats are checked against cubic symmetry and against each other.

| Check | Pass condition |
| --- | --- |
| `crystal/{flexible,isotropic,orthorhombic}_*` | Mean cell metric G = h hᵀ stays cubic: equal diagonal and zero shear within 1% of the diagonal. Langevin piston temperature within 10% of T. |
| `crystal/crescale_vs_langevin` | Mean metric within 0.5%; variances of the six metric components within 35% |
| `crystal/isotropic_vs_flexible_volume` | Mean volumes within 0.5% |
| `crystal/orthorhombic_crescale` | Masked shear components stay exactly zero while the diagonal fluctuates |
| `crystal/berendsen` | A sheared and strained cell (5% from cubic) relaxes to a cubic mean metric within 1% |
| `crystal/berendsen_relaxation` | The relaxed Berendsen metric matches the C-rescale mean within 1% (when `flexible_crescale` runs too) |

## Ideal gas: 8 non-interacting atoms (exact reference)

A zero-energy model (`modules/zero_energy.py`) has an exactly known NPT
volume distribution: p(V) ∝ V^(N_f/3) e^(−PV/kBT), a Gamma distribution of
shape N_f/3 + 1. N_f counts the kinetic degrees of freedom: 3N with a
Langevin thermostat, 3N−3 with a momentum-conserving one.

The pressure here is purely kinetic. That isolates the parts the LJ checks
cannot resolve: the degree-of-freedom (MTK) terms, the V^N measure and the
kinetic pressure estimator. A wrong power of V shifts ⟨V⟩ by about 1/(N+1),
which is 11%. Counting 3N instead of 3N−3 under Bussi fails at about 10σ.

| Check | Pass condition |
| --- | --- |
| `ideal_gas/{crescale,langevin}` | ⟨V⟩ within max(3%, 3 SE) and Var(V) within 15% of the Gamma(N+1) values |
| `ideal_gas/{crescale,langevin}_bussi` | Same, with the Bussi thermostat and shape N |
| `ideal_gas/{crescale,langevin}_dt20` | Same as the Langevin-thermostat cases, at a 4× timestep (20 fs) |

## Cell spring: ideal gas with an anisotropic exact reference

The same 8-atom ideal gas sits in a cell held by a rotation-invariant spring,
E(h) = k/2 Σ_ij (G_ij − L0² δ_ij)² with G = h hᵀ (`modules/cell_spring.py`).
The anisotropic barostats keep h lower-triangular and sample its six entries
with a uniform measure, so the exact distribution is

p(h) ∝ V^N exp(−(PV + E(h))/kBT).

`npt_mc.sample_cell_spring` draws it with vectorized Metropolis chains over
the six entries, in a few seconds. The spring gives about 10% fluctuations of
the diagonal and 14% of the shear entries.

| Check | Pass condition |
| --- | --- |
| `cell_spring/orthorhombic_{crescale,langevin}` | Diagonal entries: mean within max(0.5% of L0, 3 SE), variance within 15% of MC. Masked shear stays exactly zero. |
| `cell_spring/flexible_{crescale,langevin}` | Same for all six entries |

The check separates the measure conventions. Weighting by 1/(h_xx² h_yy) or
by h_xx² h_yy instead of uniformly shifts the diagonal means by about 2%. A
factor of 2 in the shear noise halves the shear variances.

## Restart

Split and uninterrupted runs must leave bit-identical restart states, and
both must write `.dyn.restart`, which holds the barostat state. Both runs dump
(and wrap atoms into the box) at the split point, so they follow the same
sequence of operations. This is
checked for C-rescale, Langevin and Berendsen on the liquid, and for C-rescale
and Langevin on the crystal.

## Compressibility setting

The C-rescale and Berendsen `compressibility` sets only the relaxation rate,
not the ensemble. It is set close to the physical value (5×10⁻⁴ atm⁻¹ for the
liquid, 5×10⁻⁵ atm⁻¹ for the crystal). An underestimate of 100× makes volume
relaxation 100× slower.

## Known blind spots

Errors in the virial itself are covered by the deterministic explicit-image
tests in `tests/`. The isotropic degree-of-freedom terms, the measure and the
timestep dependence are covered by the ideal gas.

The flexible-cell measure and the shear fluctuations are covered by the cell
spring.
