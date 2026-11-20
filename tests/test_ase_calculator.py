"""ASE calculator interface."""

from types import SimpleNamespace

import numpy as np
from ase import Atoms
from fennol.ase import FENNIXCalculator


def test_ase_calculator_removes_pbc_after_pbc_only_change_with_gpu_preprocessing():
    calculator = FENNIXCalculator.__new__(FENNIXCalculator)
    calculator.model = SimpleNamespace(preprocess=lambda **inputs: inputs)
    calculator._raw_inputs = None
    calculator._fennol_inputs = None
    calculator.dtype = "float32"
    calculator.gpu_preprocessing = True
    calculator.verbose = False

    atoms = Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4, pbc=True
    )
    calculator.preprocess(atoms)
    assert "cells" in calculator._raw_inputs

    atoms.set_pbc(False)
    result = calculator.preprocess(atoms, system_changes=["pbc"])

    assert "cells" not in calculator._raw_inputs
    assert "reciprocal_cells" not in calculator._raw_inputs
    assert "cells" not in result
    assert "reciprocal_cells" not in result
