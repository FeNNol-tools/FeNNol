# Changelog

All notable changes to `FeNNol` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

FeNNol uses calendar versioning. Releases use `YYYY.MM`. Additional releases
within the same month use `YYYY.MM.N`, where `N` is an incrementing revision
number. Backward-incompatible changes are documented explicitly under
**Breaking changes**.

## Unreleased

### Added

- Added `PAIR_FORCES` for learned direct pair forces and per-system virials, and optional force-switched direct-force/virial outputs for ZBL and NLH repulsion with alchemical and owned-atom corrections.
- Added **OPES enhanced sampling** to `fennol_md`, with support for multiple collective variables, adaptive kernel widths, kernel merging, bounds, OPES-explore and metadynamics modes, and restartable simulations.
- Added `scripts/analyze_opes.py` to reconstruct one- and two-dimensional free-energy surfaces from OPES, OPES-explore and metadynamics kernel files. Kernel files record the estimator, bias factor and temperature, which the script reads.
- Added bias-energy output for reweighting: runs with bias terms (OPES, metadynamics, ABF, lambda dynamics, restraints) write `<name>.bias.traj` with the instantaneous energy of every bias term (`E<name>[unit]` columns), at the same stride as `colvars.traj`, and `extxyz` trajectory frames carry these energies as `E<name>` properties, so that biased frames can be reweighted. `colvars.traj` now writes the time with six decimals.
- Added **extended-system adaptive biasing force (eABF)** sampling, including mean-force estimation, scheduled restraints, block averaging, optional metadynamics and restart support.
- Added **lambda dynamics** for alchemical simulations, including separate electrostatic and van der Waals coupling, free-energy-gradient accumulation, lambda-ABF acceleration (which `abf: false` switches off for plain lambda dynamics), optional OPES acceleration and topology restraints. Lambda dynamics does not support multiple independent replicas.
- Added a general **MD restraint framework** supporting collective-variable walls, atom-position restraints and topology/bond restraints, with time-dependent and lambda-dependent schedules.
- Added new collective variables including **coordination number**, **RMSD**, **radius of gyration** and a single Cartesian **coordinate**.
- Added atom-group definitions and minimum-image periodic-boundary handling for distance, angle, dihedral and composite collective variables.
- Added **Berendsen** and **Bussi/C-rescale** barostats, together with anisotropic compressibility controls.
- Added the `LENNARD_JONES_FS` force-shifted Lennard-Jones model module.
- Added quick scientific validation suites under `validation/` for enhanced sampling, thermostats and barostats, checked against exact or independent references.
- Added support for freezing atoms during MD by atom index or chemical species. Frozen atoms break momentum conservation, so the centre-of-mass velocity is not removed and `fix_com` is refused.
- Added support for non-conservative force outputs and for preserving additional model input keys during MD.
- Added support for using models with graph-based inputs during MD.
- Added molecule-aware periodic wrapping based on molecular topology.
- Added virial calculation and preprocessing-state import/export to `fennol_analyze`.
- Added a `--device` argument to `fennol_md`. All entry points now choose the device with the same precedence (command line > `FENNOL_DEVICE` > input file > `cpu`) and print where the device comes from and which values it overrides.
- Added a `--total_charge` (`-q`) option to `scripts/minimize.py` for charged systems.
- Added Parquet training-data support, mixed periodic/non-periodic data, preprocessing-state reuse and HTTP batch streaming.
- Added MLflow training integration, with optional `train` and `mlflow` dependency groups for PyTorch and MLflow respectively.
- Added reproducible stochastic learning-rate schedules through `scheduler_seed`.
- Added an L2-norm training loss and support for additional Optax optimizers.
- Added `softabs`, `softsat` and `TrainableSoftSat` activation functions.
- Added the `ApplyFunction` model module for applying configurable functions.
- Added configurable normalization modes to `LayerNorm`, including RMS normalization, layer normalization, dynamic tanh/erf transformations and optional learned affine parameters.
- Added a `keepdims` option to the `SumAxis` model module.
- Extended `ChargeHypothesis` with optional reference charges, weight scaling and additional controls for alchemical charge decoupling.
- Added optional corrected Na–O parameters to the NLH repulsion model.
- Added cutoff-based radial-basis scaling and cell masking for graph preprocessing.
- Added `fennol.__version__`.

### Breaking changes

- FeNNol now requires Python 3.12 or newer, with updated minimum versions for several dependencies.
- HIPNN angular coefficients now use a corrected parameter layout. Existing HIPNN checkpoints using the previous layout are not compatible and must be retrained.
- `LayerNorm` now defaults to RMS normalization with learned affine weights. Its configuration replaces `epsilon` with `eps` and removes the fixed `scale` and `shift` fields. Existing configurations and checkpoints using this module require migration for the changed behavior and parameter layout.
- Training configuration validation is now strict: unknown keys under `training` raise a validation error instead of being silently discarded.
- Collective variables with multiple independent replicas (`nreplicas > 1`, excluding PIMD) are now rejected instead of implicitly operating on a single replica.
- Device selection (`fennol_md`, `fennol_analyze`, training, `scripts/minimize.py`) no longer overwrites `CUDA_VISIBLE_DEVICES`. The index in `gpu:N`/`cuda:N` now counts among the GPUs that `CUDA_VISIBLE_DEVICES` exposes, so it follows the allocation of job schedulers; when `CUDA_VISIBLE_DEVICES` is set, `cuda:N` no longer means physical GPU `N`. JAX now opens a context and allocates memory only on the selected GPU. This requires `jax>=0.4.26`. The training configuration no longer exposes the `target_backend` and `cuda_visible_device` fields.

### Changed

- Refactored training into separate configuration, orchestration, trainer and callback components, with reusable setup for scripted training workflows.
- The default RaSTER close-range covalent switching interval changed from `0.5–0.6` to `0.2–0.4` times the covalent radii.
- The Langevin barostat default piston time constant changed from `0.2 ps` to `0.5 ps`.
- The anisotropic Langevin barostat now applies first-order lower-triangular cell updates, instead of exponential updates followed by a rotation of the whole system back to a lower-triangular cell. It samples the lower-triangular cell entries uniformly.
- MD input handling is more robust for decorated atom labels and topology information stored in Tinker XYZ files.
- The PyPI publishing workflow now runs the test suite before building distributions, and publishes only on `v*` release tags.
- Builds that are not on a release tag are now versioned `<last release>.post1.devN` instead of a guessed future version, such as `2026.6.29.post1.dev240` instead of `2026.6.30.dev240`.
- Overlapping `wrap_groups` entries are now merged into a single group, with a printed message, instead of raising an error.

### Fixed

- Fixed dihedral collective-variable values depending on bond lengths.
- Fixed metric-driven learning-rate floors.
- Fixed MD restart handling to restore variable simulation cells, thermostat/barostat state, forces and pressure, so that a restarted run reproduces the uninterrupted one.
- Fixed `fennol_md` shutdown when `write_all_beads` is enabled.
- Fixed trajectory dump intervals shorter than the integration timestep.
- Fixed the simulated-annealing thermostat schedule.
- Fixed Bussi and Nosé–Hoover thermostats running hot in small systems: NVE, Bussi and Nosé–Hoover dynamics now remove the centre-of-mass velocity and count 3N−3 kinetic degrees of freedom instead of 3N (without frozen atoms or PIMD), which also changes the reported temperature (3% at 32 atoms). The Langevin barostat uses the same count. Set `remove_com: false` when the forces do not conserve total momentum (external fields, absolute position restraints, biases on Cartesian coordinates): the centre-of-mass velocity is then thermal and 3N degrees of freedom are counted.
- Fixed neighbor-list updates with explicit periodic images (`minimum_image: false`).
- Fixed QTB restart handling.
- Fixed ASE calculator handling of PBC-only changes and stale preprocessing state.
- Fixed correctness issues in model components including CAIMAN, PAINN, SchNet, HierarchicalNet, GatedPerceptron, oxidation-state encodings, ZLoRA, FENNIX energy-term handling, `FullTensorProduct` and QEq.
- Fixed training reference-mask handling for `model_ref` targets.
- Fixed command-line/config precedence for training `model_file`.
- Fixed training-loss averaging across batches.
- Fixed trainable switching-function exponents becoming negative.
- Fixed duplicate input keys caused by case-insensitive initialization.
- Fixed home-directory expansion in model-inspection and topology-detection paths.
- Fixed `last_xyz_frame` returning the wrong frame.
- Fixed malformed paths for empty `.fnl` sections.
- Fixed spherical-harmonic generation for `lmax=0`.
- Fixed `compute_rdf` handling of charged species labels and same-species pair normalization.
- Fixed the documentation workflow to use a Python version supported by FeNNol.

## [2026.6.29] - 2026-06-29

Earlier release; changes were not recorded in this changelog.

[2026.6.29]: https://github.com/FeNNol-tools/FeNNol/tree/v2026.06.29
