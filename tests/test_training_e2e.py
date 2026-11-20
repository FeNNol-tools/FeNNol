import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.training import main

from .utils import parse_metric_traj_file, symmetric_absolute_relative_delta


def _load_ani_config():
    """Load the test's fixed copy of the ANI example configuration."""
    with open("tests/fixtures/ani.yaml") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def test_training_e2e_ani(tmp_path, mocker):
    """Launch ANI GNN  example end to end in pytest context."""
    output_dir = tmp_path

    # Mock sys.stdout to test trough the sys.stdout manipulation.
    original_stdout = sys.stdout
    _mock_stdout_manipulation(mocker)

    # Load ANI config
    config_data = _load_ani_config()

    # Config modification for tests
    config_data["rng_seed"] = 21
    config_data["device"] = "cpu"
    config_data["output_directory"] = str(output_dir)
    config_data["training"]["dspath"] = "examples/training/rmd17_aspirin_01.pkl"
    config_data["training"]["batch_size"] = 10
    config_data["training"]["max_epochs"] = 2
    config_data["training"]["nbatch_per_epoch"] = 3
    config_data["training"]["nbatch_per_validation"] = 3
    config_data["mlflow"] = {}
    config_data["mlflow"]["tracking_uri"] = (tmp_path / "mlruns").as_uri()

    # Dump ANI config in a tmp file.
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config_data,
            f,
            sort_keys=False,  # essential not to mess steps order
        )

    # replace main argument config to our config
    test_args = ["fennol_train", str(config_path)]
    mocker.patch.object(sys, "argv", test_args)

    try:
        main()
    finally:
        sys.stdout = original_stdout

    # Check model has been saved
    assert (output_dir / "initial_model.fnx").exists()
    assert (output_dir / "final_model.fnx").exists()
    assert (output_dir / "metrics.traj").exists()

    # Check loss decrease in training
    first_epochs_cut_off = 5 # check only early epochs
    actual_metrics_traj = parse_metric_traj_file(output_dir / "metrics.traj")
    print(f"Trajectory: {actual_metrics_traj}")
    for key, values in actual_metrics_traj.items():
        if key in ["loss", "mae_tot", "rmse_tot"]:
            delta_loss = np.array(values[:first_epochs_cut_off][1:]) - np.array(
                values[:first_epochs_cut_off][:-1])
            assert np.all(delta_loss <  0)

    # Check values of the metrics trajectory file
    expected_data_count = int(
        config_data["training"]["batch_size"]
        * config_data["training"]["max_epochs"]
        * config_data["training"]["nbatch_per_epoch"]
    )
    expected_final_values = {
        "data_count": expected_data_count,
        "lr": 0.00019,
        "mae_e": 400,
        "mae_f": 41,
        "loss": 0.026,
        "best_metric": 0.107,
    }
    # Compare actual metrics to reference metrics.
    for key, final_value in expected_final_values.items():
        actual_value = actual_metrics_traj[key][-1]
        relative_delta = symmetric_absolute_relative_delta(
            float(final_value), actual_value
        )
        assert relative_delta < .15
        # np.testing.assert_almost_equal(float(final_value), actual_value)


def test_training_e2e_ani_with_metric_smoothing(tmp_path, mocker):
    """Launch ANI GNN  example end to end in pytest context."""
    output_dir = tmp_path

    # Mock sys.stdout to test trough the sys.stdout manipulation.
    original_stdout = sys.stdout
    _mock_stdout_manipulation(mocker)

    # Load ANI config
    config_data = _load_ani_config()

    # Config modification for tests
    config_data["rng_seed"] = 21
    config_data["device"] = "cpu"
    config_data["output_directory"] = str(output_dir)
    config_data["training"]["dspath"] = "examples/training/rmd17_aspirin_01.pkl"
    config_data["training"]["batch_size"] = 10
    config_data["training"]["max_epochs"] = 2
    config_data["training"]["nbatch_per_epoch"] = 3
    # *** Here we add metric smoothing ***
    config_data["training"]["metrics_ema_decay"] = .9
    config_data["mlflow"] = {}
    config_data["mlflow"]["tracking_uri"] = (tmp_path / "mlruns").as_uri()

    # Dump ANI config in a tmp file.
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config_data,
            f,
            sort_keys=False,  # essential not to mess steps order
        )

    # replace main argument config to our config
    test_args = ["fennol_train", str(config_path)]
    mocker.patch.object(sys, "argv", test_args)

    try:
        main()
    finally:
        sys.stdout = original_stdout

    # Check model has been saved
    assert (output_dir / "initial_model.fnx").exists()
    assert (output_dir / "final_model.fnx").exists()
    assert (output_dir / "metrics.traj").exists()

    # Check loss decrease in training
    first_epochs_cut_off = 5 # check only early epochs
    actual_metrics_traj = parse_metric_traj_file(output_dir / "metrics.traj")
    print(f"Trajectory: {actual_metrics_traj}")
    for key, values in actual_metrics_traj.items():
        if key in ["loss", "mae_tot", "rmse_tot"]:
            delta_loss = np.array(values[:first_epochs_cut_off][1:]) - np.array(
                values[:first_epochs_cut_off][:-1])
            assert np.all(delta_loss <  0)

    # Check values of the metrics trajectory file
    expected_data_count = int(
        config_data["training"]["batch_size"]
        * config_data["training"]["max_epochs"]
        * config_data["training"]["nbatch_per_epoch"]
    )
    expected_final_values = {
        "data_count": expected_data_count,
        "lr": 0.00019,
        "mae_e": 400,
        "mae_f": 41,
        "loss": 0.026,
        "best_metric": 0.107,
        "rmse_tot_smooth": 0.139,
        "mae_tot_smooth": 0.085,
    }
    # Compare actual metrics to reference metrics.
    for key, final_value in expected_final_values.items():
        actual_value = actual_metrics_traj[key][-1]
        relative_delta = symmetric_absolute_relative_delta(
            float(final_value), actual_value
        )
        assert relative_delta < .15
        # np.testing.assert_almost_equal(float(final_value), actual_value)

def test_restart_training_e2e_ani(tmp_path, mocker):
    """Launch ANI GNN  example in pytest context."""
    output_dir = tmp_path

    # Mock sys.stdout to test trough the sys.stdout manipulation.
    original_stdout = sys.stdout
    _mock_stdout_manipulation(mocker)

    # Load ANI config
    config_data = _load_ani_config()

    # Config modification for tests
    config_data["rng_seed"] = 42
    config_data["device"] = "cpu"
    config_data["output_directory"] = str(output_dir)
    config_data["training"]["dspath"] = "examples/training/rmd17_aspirin_01.pkl"
    config_data["training"]["batch_size"] = 10  # orig :100
    config_data["training"]["max_epochs"] = 2  # orig :1000
    config_data["training"]["nbatch_per_epoch"] = 2  # orig :10
    config_data["mlflow"] = {}
    config_data["mlflow"]["tracking_uri"] = (tmp_path / "mlruns").as_uri()

    # Dump ANI config in a tmp file.
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config_data,
            f,
            sort_keys=False,  # essential not to mess steps order
        )

    # remove stdout binding for it not to mess with pytest
    mocker.patch("fennol.training.io.TeeLogger.bind_stdout")
    mocker.patch("fennol.training.io.TeeLogger.unbind_stdout")
    # replace main argument config to our config
    test_args = ["fennol_train", str(config_path)]
    mocker.patch.object(sys, "argv", test_args)

    # launch training
    try:
        main()
    finally:
        sys.stdout = original_stdout

    # Check model has been saved
    assert (output_dir / "initial_model.fnx").exists()
    assert (output_dir / "latest_model.fnx").exists()  # model to reload

    # Restart training
    path_restart_config = output_dir / "config.yaml"
    with open(path_restart_config) as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    # Update the number of epochs
    config_data["training"]["max_epochs"] = 4
    with open(path_restart_config, "w") as f:
        yaml.dump(
            config_data,
            f,
            sort_keys=False,
        )
    test_args = ["fennol_train", str(output_dir)]
    mocker.patch.object(sys, "argv", test_args)
    main()

    assert (output_dir / "final_model.fnx").exists()
    assert (output_dir / "metrics.traj").exists()
    # Check values of the metrics trajectory file
    actual_metrics_traj = parse_metric_traj_file(output_dir / "metrics.traj")
    # First run epoch 1, 2, second run 3,4.
    assert actual_metrics_traj["epoch"] == [1, 2, 3, 4]


def _mock_stdout_manipulation(mocker):
    dummy_stdout = MagicMock()
    dummy_stdout.close.return_value = None
    dummy_stdout.fileno.return_value = 1
    module_path = "fennol.training.training"
    mocker.patch(f"{module_path}.open", return_value=dummy_stdout)
    mocker.patch(f"{module_path}.io.TextIOWrapper", return_value=dummy_stdout)
    mocker.patch.object(sys.stdout, "fileno", return_value=1, create=True)
    mocker.patch("fennol.training.io.TeeLogger.bind_stdout")
    mocker.patch("fennol.training.io.TeeLogger.unbind_stdout")
