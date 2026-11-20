"""Numerical checks at the public analysis boundary (`fennol.analyze`)."""

import multiprocessing as mp
import pickle
import subprocess
import sys

import jax
import numpy as np
import pytest
from fennol import FENNIX
from fennol.analyze import fennix_analyzer


def test_analysis_output_worker_under_spawn(tmp_path):
    from fennol.analyze import _queue_reader

    context = mp.get_context("spawn")
    queue = context.Queue()
    output = tmp_path / "frames.pkl"
    process = context.Process(target=_queue_reader, args=(queue, output, "pkl"))
    process.start()
    queue.put({"forces": np.array([[1.0, 2.0, 3.0]])})
    queue.put(None)
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join()
    assert process.exitcode == 0
    with output.open("rb") as handle:
        frame = pickle.load(handle)
    np.testing.assert_array_equal(frame["forces"], [[1.0, 2.0, 3.0]])


@pytest.fixture
def model():
    return FENNIX(
        cutoff=5.0,
        modules={"energy": {
            "module_name": "APPLY_FUNCTION", "argument_keys": "coordinates",
            "function_str": "x: sum(x*x, axis=-1)",
        }},
        energy_terms=["energy"], rng_key=jax.random.PRNGKey(0),
    )


@pytest.mark.parametrize("output_keys", [["forces"], ["forces", "virial"]])
def test_analyzer_returns_requested_forces_with_or_without_virial(tmp_path, output_keys, model):
    xyz = tmp_path / "pair.xyz"
    xyz.write_text("2\n\nH 1 2 3\nH -1 0 0.5\n")
    batches = list(fennix_analyzer(xyz, model, output_keys))
    frame = batches[0]
    np.testing.assert_allclose(frame["forces"], [[-2., -4., -6.], [2., 0., -1.]])
    if "virial" in output_keys:
        np.testing.assert_allclose(frame["virial"], [[4., 4., 5.], [4., 8., 12.], [5., 12., 18.5]])


def test_analyze_cli_reuses_exported_preprocessing_state(tmp_path, model):
    model_file = tmp_path / "quadratic.fnx"
    model.save(model_file)
    xyz = tmp_path / "pair.xyz"
    xyz.write_text("2\n\nH 1 2 3\nH -1 0 0.5\n")
    state_file = tmp_path / "preprocessing.pkl"
    outputs = [tmp_path / "first.pkl", tmp_path / "reused.pkl"]
    for output, option in zip(outputs, ["--dump_preproc_state", "--read_preproc_state"]):
        result = subprocess.run(
            [sys.executable, "-m", "fennol.analyze", str(xyz), str(model_file),
             "--device", "cpu", "--output_keys", "forces", "virial",
             "-o", str(output), option, str(state_file)],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    with outputs[0].open("rb") as handle:
        first = pickle.load(handle)
    with outputs[1].open("rb") as handle:
        reused = pickle.load(handle)
    np.testing.assert_allclose(first["forces"], reused["forces"])
    np.testing.assert_allclose(first["virial"], reused["virial"])
