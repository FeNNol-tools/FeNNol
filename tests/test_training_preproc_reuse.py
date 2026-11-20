import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import jax
import numpy as np
import pytest
import yaml
from flax.core import unfreeze

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.io import load_model
from fennol.training.training import main


def test_tiny_training_run_reuses_imported_preprocessing_state(tmp_path, mocker):
    with open("tests/fixtures/ani.yaml") as config_file:
        config = yaml.safe_load(config_file)

    model = load_model(config["model"], rng_key=jax.random.PRNGKey(0))
    model_file = tmp_path / "initial.fnx"
    model.save(str(model_file))
    imported_state = unfreeze(model.preproc_state)
    imported_state["check_input"] = True
    imported_state["layers_state"] = tuple(
        unfreeze(layer_state) for layer_state in imported_state["layers_state"]
    )
    imported_state["layers_state"][0]["nblist_mult_size"] = 1.7
    preproc_file = tmp_path / "preprocessing.pkl"
    with open(preproc_file, "wb") as output_file:
        pickle.dump({"preproc_state": imported_state}, output_file)

    row = {
        "species": np.array([1, 1, 1]),
        "coordinates": np.array(
            [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [0.0, 0.74, 0.0]]
        ),
        "formation_energy": np.array(0.1),
    }
    dataset_file = tmp_path / "dataset.pkl"
    with open(dataset_file, "wb") as output_file:
        pickle.dump({"training": [row], "validation": [row]}, output_file)

    config["device"] = "cpu"
    config["output_directory"] = str(tmp_path / "run")
    config["model_file"] = str(model_file)
    config["training"].update(
        {
            "dspath": str(dataset_file),
            "load_preproc_state": str(preproc_file),
            "batch_size": 1,
            "batch_size_val": 1,
            "max_epochs": 1,
            "nbatch_per_epoch": 1,
            "nbatch_per_validation": 1,
            "schedule_type": "constant",
            "ema_decay": [],
            "loss": {
                "energy": {
                    "key": "total_energy",
                    "ref": "formation_energy",
                    "weight": 1.0,
                }
            },
        }
    )
    config["rng_seed"] = 1
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as output_file:
        yaml.safe_dump(config, output_file)

    dummy_stdout = MagicMock()
    dummy_stdout.close.return_value = None
    dummy_stdout.fileno.return_value = 1
    mocker.patch("fennol.training.training.open", return_value=dummy_stdout)
    mocker.patch("fennol.training.training.io.TextIOWrapper", return_value=dummy_stdout)
    mocker.patch.object(sys.stdout, "fileno", return_value=1, create=True)
    mocker.patch("fennol.training.io.TeeLogger.bind_stdout")
    mocker.patch("fennol.training.io.TeeLogger.unbind_stdout")
    mocker.patch.object(sys, "argv", ["fennol_train", str(config_file)])
    original_stdout = sys.stdout
    try:
        main()
    finally:
        sys.stdout = original_stdout

    with open(Path(config["output_directory"]) / "restart_checkpoint.pkl", "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert checkpoint["preproc_state"]["check_input"] is True
    assert float(checkpoint["preproc_state"]["layers_state"][0]["nblist_mult_size"]) == 1.7
