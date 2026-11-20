
[![PyPI - Version](https://img.shields.io/pypi/v/FeNNol?link=https%3A%2F%2Fpypi.org%2Fproject%2FFeNNol%2F)](https://pypi.org/project/FeNNol/)
[![DOI:10.1063/5.0217688](https://zenodo.org/badge/DOI/10.1063/5.0217688.svg)](https://doi.org/10.1063/5.0217688) 

## FeNNol: Force-field-enhanced Neural Networks optimized library
**FeNNol** is a library for building, training and running neural network potentials for molecular simulations. It is based on the JAX library and is designed to be fast and flexible.

FeNNol's documentation is available [here](https://fennol-tools.github.io/FeNNol/) and the article describing the library at https://doi.org/10.1063/5.0217688

Active Learning tutorial in this [Colab notebook](https://colab.research.google.com/drive/1Z3G_jVSF60_nbDdJwbgyLdJBHTYuQ5nL?usp=sharing)

### Table of Contents

- [Pre-trained models](#pre-trained-models)
- [Installation](#installation)
    - [From PyPI](#from-pypi)
    - [Latest version from Github repo](#latest-version-from-github-repo)
    - [Optional dependencies](#optional-dependencies)
- [Selecting a device](#selecting-a-device)
- [Examples](#examples)
- [Citation](#citation)
- [License](#license)

## Pre-trained models
Pre-trained models can be downloaded from the [Pre-trained Models Collection](https://github.com/FeNNol-tools/FeNNol-PMC).
Available models include:
- [FeNNix-Bio1](https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/FENNIX-BIO1)
- [ANI1x](https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/ANI/ANI1X), [ANI1ccx](https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/ANI/ANI1CCX), [ANI2x](https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/ANI/ANI2X)
- [MACE-OFF23](https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/MACE-OFF23)
- [MACE-MP](https://github.com/FeNNol-tools/FeNNol-PMC/tree/main/MACE-MP)

After downloading the model (for example [`fennix-bio1S.fnx`](https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/fennix-bio1S.fnx)), you can load it and start predicting energies and forces with:
```python
import fennol
import numpy as np

model = fennol.load("fennix-bio1S.fnx")
species = np.array([8, 1, 1])
coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

energy, forces, output_dict = model.energy_and_forces(species=species, coordinates=coordinates, unit="ev")
print("Energy:", energy,"ev")
print("Forces:", forces,"ev/Å")
```

Alternatively, you can use the provided ASE calculator and run a short molecular dynamics simulation:
```python
from fennol.ase import FENNIXCalculator
from ase import Atoms
from ase.md.verlet import VelocityVerlet    
import ase.units as units

atoms = Atoms(symbols=["O", "H", "H"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
calculator = FENNIXCalculator(model="fennix-bio1S.fnx", gpu_preprocessing=True)
atoms.calc = calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
print("Energy:", energy,"ev")
print("Forces:", forces,"ev/Å")

dyn = VelocityVerlet(atoms, timestep=1.0*units.fs) 
print("Starting MD simulation. ")
for step in range(10):
    dyn.run(10)
    print(f"Step {10*(step+1)}, Potential energy: {atoms.get_potential_energy()} ev")
```

**For better performance, we recommend using `fennol_md`, FeNNol's native MD engine.** (see the [`examples/md`](https://github.com/fennol-tools/FeNNol/tree/main/examples/md) directory for instructions and example input files)



## Installation

Use Python 3.12 or newer. The dependency minimums are declared in
`pyproject.toml`; install into a fresh virtual environment when upgrading.

### From PyPI
```bash
# CPU version
pip install fennol

# GPU version
pip install "fennol[cuda]"

# Training (includes PyTorch dataloaders)
pip install "fennol[train]"

# Training with optional MLflow tracking
pip install "fennol[train,mlflow]"
```

### Latest version from Github repo
You can start with a fresh environment, for example using venv:
```bash
python -m venv fennol
source fennol/bin/activate
```

The first step is to install jax (see details at: https://jax.readthedocs.io/en/latest/installation.html). For example, to install the latest version using pip:
```bash
# CPU version
pip install -U jax

# GPU version
pip install -U "jax[cuda12]"
```

Then, you can clone the repo and install FeNNol using pip:
```bash
git clone https://github.com/FeNNol-tools/FeNNol.git
cd FeNNol
pip install .
```

For a development checkout, `uv sync` installs the training extra and test
dependencies by default. Use `uv sync --no-group training` to omit PyTorch.

### Optional dependencies
- Some modules require e3nn-jax (https://github.com/e3nn/e3nn-jax) which can be installed with:
```bash
pip install --upgrade e3nn-jax
```
- The training extra installs PyTorch. To choose its CPU-only build explicitly:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- For the Deep-HP interface, cffi and pycuda are required:
```bash
pip install cffi pycuda
```

## Selecting a device
FeNNol runs on the CPU unless a device is selected. `fennol_md`, `fennol_analyze` and `fennol_train` take the device from, by decreasing priority:
1. the `--device` argument (`fennol_md` and `fennol_analyze`),
2. the `FENNOL_DEVICE` environment variable,
3. the `device` keyword of the input or configuration file,
4. `cpu` by default.

```bash
fennol_md input.fnl --device cuda:1
FENNOL_DEVICE=gpu fennol_md input.fnl
```

The accepted values are `cpu`, `gpu` or `cuda` (the first visible GPU) and `gpu:N` or `cuda:N`. `N` counts among the GPUs exposed by `CUDA_VISIBLE_DEVICES`: on a cluster, `cuda:0` is the first GPU that the scheduler allocated to the job. FeNNol only opens the selected GPU, on which JAX preallocates 75% of the memory by default; see the [JAX documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) to change this.

When FeNNol is used as a library, for example through the ASE calculator, JAX opens every visible GPU by default. To select a single device, call `configure_device` before any JAX computation, including loading a model:
```python
from fennol import configure_device
from fennol.ase import FENNIXCalculator

configure_device("cuda:0")
calc = FENNIXCalculator("model.fnx")
```

## Examples
To learn how to train a FeNNol model, you can check the examples in the [`examples/training`](https://github.com/fennol-tools/FeNNol/tree/main/examples/training) directory. The `README.md` file in that directory contains instructions on how to train a model on the aspirin revMD17 dataset.

To learn how to run molecular dynamics simulations with FeNNol models, you can check the examples in the [`examples/md`](https://github.com/fennol-tools/FeNNol/tree/main/examples/md) directory. The `README.md` file in that directory contains instructions on how to run simulations with the provided ANI-2x model.



## Citation

Please cite this paper if you use the library.
```
T. Plé, O. Adjoua, L. Lagardère and J-P. Piquemal. FeNNol: an Efficient and Flexible Library for Building Force-field-enhanced Neural Network Potentials. J. Chem. Phys. 161, 042502 (2024)
```

```
@article{ple2024fennol,
    author = {Plé, Thomas and Adjoua, Olivier and Lagardère, Louis and Piquemal, Jean-Philip},
    title = {FeNNol: An efficient and flexible library for building force-field-enhanced neural network potentials},
    journal = {The Journal of Chemical Physics},
    volume = {161},
    number = {4},
    pages = {042502},
    year = {2024},
    month = {07},
    doi = {10.1063/5.0217688},
    url = {https://doi.org/10.1063/5.0217688},
}

```

## License

This project is licensed under the terms of the GNU LGPLv3 license. See [LICENSE](https://github.com/fennol-tools/FeNNol/blob/main/LICENSE) for additional details.
