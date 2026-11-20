# Quick scientific validation

Three suites run short `fennol_md` simulations on analytical or
Lennard-Jones models and score the outputs against exact or independent
references. The checks are meant to catch real defects quickly: a wrong
estimator, a missing bias, a broken restart or a wrong ensemble. They do not
certify tight convergence, so their tolerances are loose.

| Suite | Covers | Checks | Wall time (`--jobs 12`) |
| --- | --- | --- | --- |
| [`sampling`](sampling/README.md) | OPES, OPES-explore, metadynamics, eABF ± metaD, λ-ABF ± OPES, plain λ-dynamics | 28 | ~3 min |
| [`thermostat`](thermostat/README.md) | Langevin samplers on independent harmonic wells; Bussi and Nosé–Hoover on an LJ liquid; restarts of all thermostats | 12 | ~1.5 min |
| [`barostat`](barostat/README.md) | C-rescale, Langevin and Berendsen barostats (exact ideal gas, ideal gas in a cell spring, isotropic liquid, flexible crystal) | 26 | ~9 min |

Run them from the repository root:

```bash
python validation/sampling/run.py --jobs 12
python validation/thermostat/run.py
python validation/barostat/run.py --jobs 12
```

Every suite accepts these options:

- `--only <sections>` runs a subset of sections.
- `--methods <labels>` runs a subset of cases.
- `--seed N` runs an independent seed set. If a check fails, rerun it with
  another seed before investigating: a real defect fails on every seed.
- `--config` uses an alternative configuration file.

Each suite writes `work/report.json`. All generated files stay under the
ignored `work/` directories. The shared runner lives in `common.py`. It
builds every `fennol_md` input, runs the jobs concurrently, and compares split
and uninterrupted restarts file by file.
