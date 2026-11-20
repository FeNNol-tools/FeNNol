# Quick sampling checks

One shared suite for OPES, OPES-explore, metadynamics, eABF (with and without
metadynamics) and lambda dynamics. Every check is a single short `fennol_md`
run on an analytical FENNIX model with a loose tolerance. The goal is to
catch real defects quickly, not to certify tight convergence.

```bash
python validation/sampling/run.py                  # all 28 checks, ~3 min with --jobs 11
python validation/sampling/run.py --only pmf_1d barrier --methods metad
python validation/sampling/run.py --seed 1         # independent seed set
```

If a check fails, rerun it with another `--seed` before investigating. A real
defect fails on every seed. Results are written to `work/report.json`, and
generated files stay under the ignored `work/` directory.

| Section | What it checks | Pass condition |
| --- | --- | --- |
| `pmf_1d` | PMF reconstruction on a tilted quartic single well. Exported OPES kernels go through `scripts/analyze_opes.py`. eABF is checked through both CZAR and the extended-variable PMF. | Aligned RMSE ≤ 0.3 kBT where F ≤ 4 kBT; coverage ≥ 90% |
| `pmf_2d` | OPES, explore and metaD on a separable 2D quartic surface | Marginals ≤ 0.3 kBT; joint surface ≤ 0.6 kBT |
| `barrier` | 8 kBT tilted double well, plain MD paired with each biased method | Plain ≤ 3 passages; biased ≥ 10; left-well population within 0.1 of exact, both integrated from the reconstructed free energy and, for OPES and explore, reweighted from the trajectory with exp(βV) using the bias energies of `bias.traj` |
| `restart` | Split vs uninterrupted run, every restart file compared | Bit-identical dynamics, RNG and bias state; `.dyn.restart` and the method's bias file (`.opes.restart`, `.abf.restart` or `.lambdadyn.restart`) must exist |
| `lambda` | λ-ABF TI for electrostatic, vdW and combined schedules, with OPES, and plain λ-dynamics (`abf: false`); λ barrier screen; restart | dF/dλ RMSE ≤ 0.5 kBT; ΔF within 0.2 kBT of exact; plain λ marginal ≤ 0.3 kBT; barrier passages as above |

## Model design

All CV models use one Ar-mass atom whose CV is its x (and y) coordinate.
Every axis is confined by a separable polynomial (`modules/polynomial.py`).

The single wells are asymmetric. This way, sign, offset or Gaussian-shape
errors in the kernel reconstruction cannot cancel. The 5 fs timestep is about
1/60 of the stiffest oscillation period, so runs decorrelate within picoseconds.

The lambda model (`modules/lambda_harmonic.py`) puts `n_ligand` atoms in a
harmonic well with stiffness k(λ) = k0 + ke·elambda + kv·vlambda. The exact
free energy is F(λ) = 1.5·n_ligand·kBT·log k(λ). The barrier variant raises
k to 10 at λ = 0.5, giving a 13.8 kBT barrier. A lower barrier is not enough:
the end-state basins are narrow in λ, so plain λ-dynamics crosses a 6.9 kBT
barrier about 15 times in 300 ps.

## Defects found by this suite

- Metadynamics kernel export divided heights by γ/(γ−1) instead of multiplying
  by it. Reconstructed free energies were scaled by ((γ−1)/γ)².
- `analyze_opes.py --metadyn` read kernel heights as Hartree, while
  `fennol_md` writes them in kcal/mol.
- λ-dynamics restarts computed the electrostatic and vdW couplings from the
  configured initial λ instead of the restored one, so the first step after a
  restart used the wrong forces.
