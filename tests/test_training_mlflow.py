"""MLflow logging during training."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.config import RunConfig


def test_training_setup_works_without_optional_mlflow(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class MissingMlflow(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlflow" or fullname.startswith("mlflow."):
                    raise ModuleNotFoundError("blocked optional dependency", name=fullname)

        sys.meta_path.insert(0, MissingMlflow())
        from fennol.training.config import RunConfig
        from fennol.training.training import setup_run_context

        config = RunConfig(output_directory=SYS_ARG, mlflow={"logging_enabled": False})
        context = setup_run_context(config)
        context.logger.unbind_stdout()
        context.logger.close()
        """
    ).replace("SYS_ARG", repr(str(tmp_path)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_enabled_mlflow_without_dependency_has_install_guidance():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class MissingMlflow(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlflow" or fullname.startswith("mlflow."):
                    raise ModuleNotFoundError("blocked optional dependency", name=fullname)

        sys.meta_path.insert(0, MissingMlflow())
        from fennol.training.config import RunConfig
        from fennol.training.training import setup_run_context

        try:
            setup_run_context(RunConfig(mlflow={"logging_enabled": True}))
        except RuntimeError as exc:
            assert "Install fennol[train,mlflow]" in str(exc)
        else:
            raise AssertionError("Expected a helpful error for missing MLflow")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_enabled_mlflow_records_metrics_and_finishes_local_run(tmp_path):
    pytest.importorskip("mlflow")
    from fennol.training.mlflow import MlFlowLogger
    from mlflow.tracking import MlflowClient

    tracking_uri = (tmp_path / "mlruns").as_uri()
    config = RunConfig(
        output_directory=tmp_path,
        mlflow={
            "logging_enabled": True,
            "tracking_uri": tracking_uri,
            "experiment_name": "mlflow-training-smoke",
            "run_name": "local-lifecycle",
            "log_params": False,
            "log_artifacts": [],
        },
    )
    logger = MlFlowLogger(config)
    with logger.start_run():
        run_id = logger.run.info.run_id
        logger.log_epoch_metrics({"rmse_tot": 1.25}, epoch=3, split="val")
        logger.log_restart(1, step=4)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    rmse_history = client.get_metric_history(run_id, "val_rmse_tot")
    restore_history = client.get_metric_history(run_id, "divergence_restores")
    assert run.info.status == "FINISHED"
    assert [(item.step, item.value) for item in rmse_history] == [(3, 1.25)]
    assert [(item.step, item.value) for item in restore_history] == [(4, 1.0)]


def test_smoothed_mlflow_metrics_use_current_epoch_number():
    pytest.importorskip("torch", reason="training requires PyTorch")
    from fennol.training.callbacks import MLflowHandler

    class Logger:
        def __init__(self):
            self.metrics = []

        def log_epoch_metrics(self, *args, **kwargs):
            pass

        def log_metric(self, key, value, step, split):
            self.metrics.append((key, value, step, split))

    logger = Logger()
    handler = MLflowHandler(logger)
    trainer = SimpleNamespace(
        metrics={"train_loss": 1.0, "lr": 0.1, "rmse_tot": 2.0, "mae_tot": 1.0},
        loss_definition={"energy": {"weight": 1.0}},
        config=SimpleNamespace(smoothen_metrics=True),
        get_scaled_metrics=lambda: {
            "rmse_energy": 2.0,
            "mae_energy": 1.0,
            "rmse_energy_smooth": 1.5,
            "mae_energy_smooth": 0.75,
            "rmse_tot_smooth": 1.5,
            "mae_tot_smooth": 0.75,
        },
    )
    state = SimpleNamespace(epoch=7)

    handler.on_epoch_end(trainer, state)

    smooth_metrics = [entry for entry in logger.metrics if entry[0].endswith("_smooth")]
    assert smooth_metrics
    assert all(step == state.epoch for _, _, step, _ in smooth_metrics)
