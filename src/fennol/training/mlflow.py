"""Tools for mlflow integration."""

from __future__ import annotations

import glob
import hashlib
import os
import tempfile
from functools import wraps
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow import MlflowClient
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.dataset import Dataset as MlFlowDataset

from ..models.fennix import FENNIX
from ..training.config import RunConfig

DEFAULT_MAPPING = {
    "MLFLOW_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "MLFLOW_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_SESSION_TOKEN": "AWS_SESSION_TOKEN",
    "MLFLOW_REGION": "AWS_REGION",
    "MLFLOW_DEFAULT_REGION": "AWS_DEFAULT_REGION",
    "MLFLOW_PROFILE": "AWS_PROFILE",
    "MLFLOW_SHARED_CREDENTIALS_FILE": "AWS_SHARED_CREDENTIALS_FILE",
    "MLFLOW_CONFIG_FILE": "AWS_CONFIG_FILE",
    "MLFLOW_S3_ADDRESSING_STYLE": "AWS_S3_ADDRESSING_STYLE",
    "MLFLOW_S3_SIGNATURE_VERSION": "AWS_S3_SIGNATURE_VERSION",
    "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_TRACKING_TOKEN": "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
}


def _collect_overrides(
    mapping: dict[str, str] = DEFAULT_MAPPING,
) -> dict[str, str]:
    """
    Inplace values to map configuration to AWS env. var.

    Retrieve values of defined env variable and output a mapping
    correspond to AWS env. var. and target values.

    Arg:
        mapping (dict[str, str]): MLflow env. var. name to
            AWS infra env. var. name.
    """
    overrides = {}
    for src, dst in mapping.items():
        if src in os.environ and os.environ[src] != "":
            overrides[dst] = os.environ[src]
    return overrides


@contextmanager
def use_mlflow_env_override(mapping=DEFAULT_MAPPING):
    """Temporarily overlay env so MLflow/boto3 use the MLFLOW_AWS_* overrides.

    This lets you scope credentials/region/profile to MLflow operations while
    leaving global AWS_* untouched. Use this exactly around mlflow.log_* calls.
    """
    overrides = _collect_overrides(mapping)
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def get_hash(f_path, mode="md5"):
    """Generate a stable hash for a file or directory path."""
    h = hashlib.new(mode)
    if os.path.isdir(f_path):
        h.update(os.path.abspath(f_path).encode("utf-8"))
        return h.hexdigest()
    with open(f_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a dict."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):  # Convert lists to strings for mlflow params
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


class FennixPyfunc(mlflow.pyfunc.PythonModel):
    """
    Lightweight pyfunc wrapper that loads a saved FENNIX model artifact.

    Exposes a predict() for MLflow. Keeps heavy state out of the pickled object.
    """

    def __init__(
        self,
        *,
        use_atom_padding: bool = False,
        graph_config: dict[str, Any] | None = None,
    ):
        self.use_atom_padding = use_atom_padding
        self.graph_config = graph_config or {}

    def load_context(self, context):
        """Load the FENNIX model from the artifact file."""
        model_file = context.artifacts["fennix_model_file"]
        self.fennix = FENNIX.load(
            model_file,
            use_atom_padding=self.use_atom_padding,
            graph_config=self.graph_config,
        )

    def predict(
        self,
        context: Any,
        model_input: list[dict[str, list[float]]],
    ):
        """
        Accept either a dict-like batch or a pandas DataFrame with keys.

        Keys: species (N,), coordinates (N,3), batch_index (N,), natoms (B,)
        Returns total energy per system as a numpy array.
        """
        if isinstance(model_input, list):
            if not model_input:
                return np.asarray([])
            if isinstance(model_input[0], dict):
                if len(model_input) == 1:
                    model_input = model_input[0]
                else:
                    model_input = {
                        k: [row[k] for row in model_input]
                        for k in model_input[0]
                    }
        # Allow DataFrame or dict-like
        if hasattr(model_input, "to_dict"):
            model_input = model_input.to_dict(orient="list")
        if isinstance(model_input, dict):
            model_input = {
                k: np.atleast_1d(np.asarray(model_input[k]))
                for k in model_input
            }
        e, _out = self.fennix.total_energy(**model_input)
        return np.asarray(e)


def _log_and_register(
    *,
    fennix: FENNIX,
    model_name: str | None,
    model_artifacts: dict,
    example_batch: dict | None,
    alias: str | None,
    graph_config: dict | None,
    use_atom_padding: bool,
):
    log_kwargs = {
        "name": "model",
        "python_model": FennixPyfunc(
            use_atom_padding=use_atom_padding,
            graph_config=graph_config,
        ),
        "artifacts": model_artifacts,
        "pip_requirements": [],
    }
    if model_name:
        log_kwargs["registered_model_name"] = model_name
    info = mlflow.pyfunc.log_model(**log_kwargs)
    if model_name:
        print(f"Logged model: {info.model_uri} (registered as '{model_name}')")
    else:
        print(f"Logged model: {info.model_uri}")

    if alias and model_name:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        version = max((int(v.version) for v in versions), default=None)
        if version is None:
            raise RuntimeError(
                f"No versions found to set alias '{alias}' for '{model_name}'."
            )
        client.set_registered_model_alias(model_name, alias, version)
        print(f"Alias '{alias}' → {model_name} v{version}")


# --- Helper Classes ---


class DisabledMixin:
    """Disable mixin."""

    def __init__(self, disabled: bool = True) -> None:
        self._disabled = disabled

    @property
    def disabled(self) -> bool:
        """Wether or not the class is disabled."""
        return self._disabled

    @disabled.setter
    def disabled(self, val: bool) -> None:
        self._disabled = val


def skip_if_disabled(f):
    """Scope MLflow credentials and keep logging failures out of training."""

    @wraps(f)
    def wrap(self: DisabledMixin, *nargs, **kwargs):
        if self.disabled:
            return None
        try:
            with use_mlflow_env_override():
                return f(self, *nargs, **kwargs)
        except Exception as exc:
            if self.config.mlflow.strict:
                raise
            print(f"MLflow operation {f.__name__} failed; continuing: {exc}")
            return None

    return wrap


class FennolDataset(MlFlowDataset):
    """Fennol dataset class enabling mlflow tracking.

    This class does not actually contains the data. It only exposes
    information for versioning the input dataset. Better implementation
    should make the Fennol dataset classes inherit from MlFlowDataset.
    """

    def __init__(
        self,
        dspath: str,
        dataset: Any,
    ):
        name = Path(dspath).name
        source = CodeDatasetSource({})
        digest = get_hash(dspath)
        if isinstance(dataset, dict):
            dataset_dict = dataset
        elif dataset is None:
            dataset_dict = {}
        else:
            dataset_dict = {"training": dataset}
        profile = {}
        for key, value in dataset_dict.items():
            try:
                num_rows = len(value)
            except TypeError:
                continue
            profile[f"num_rows_{key}"] = num_rows
        self._profile = profile
        self._schema = dict(profile)

        super().__init__(source, name, digest)

    @property
    def profile(self) -> Any | None:
        """FennolDataset profile property."""
        return self._profile

    @property
    def schema(self) -> Any | None:
        """FennolDataset schema property."""
        return self._schema


class MlFlowLogger(DisabledMixin):
    """MlFlow Logger class."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        is_disabled = not self.config.mlflow.logging_enabled
        super().__init__(disabled=is_disabled)

        if self.disabled:
            print("Mlflow disabled, did not find configuration.")
            return

        self._output_directory = str(config.output_directory)
        self._config_path = config.path_config_file
        self.tracking_uri = self.config.mlflow.tracking_uri
        self.experiment_name = self.config.mlflow.experiment_name
        self.current_run_name = self.config.mlflow.run_name
        self.registered_model_name = self.config.mlflow.registered_model_name
        self.model_alias = self.config.mlflow.model_alias
        self.registry_uri = self.config.mlflow.registry_uri
        self._tags = None
        if not isinstance(self.config.mlflow.tags, str):
            self._tags = self.config.mlflow.tags

        self.current_run = None
        self._metric_prefix = ""

        with use_mlflow_env_override():
            mlflow.enable_system_metrics_logging()

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            else:
                print(
                    "Warning: No MLflow Tracking URI set. "
                    "Logs will be saved locally."
                )
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)
            mlflow.set_experiment(self.experiment_name)

    # PROPERTIES ---------------------------------------------------------------

    @property
    def output_directory(self) -> str | None:
        """Registered output directory."""
        return self._output_directory

    @output_directory.setter
    def output_directory(self, path: str) -> None:
        self._output_directory = path

    @property
    def tags(self) -> dict[str, str]:
        """MLflow tags to add context to a run."""
        if self._tags is None:
            return {}
        return self._tags

    @tags.setter
    def tags(self, tags_value: dict[str, str]) -> None:
        self._tags = tags_value

    @property
    def metric_prefix(self) -> str:
        """Current metric prefix."""
        return self._metric_prefix

    @metric_prefix.setter
    def metric_prefix(self, _prefix: str) -> None:
        self._metric_prefix = _prefix

    @property
    def config_path(self) -> Path | None:
        """Registered config path."""
        return self._config_path

    @config_path.setter
    def config_path(self, _path: str | Path) -> None:
        if isinstance(_path, str):
            _path = Path(_path)
        self._config_path = _path

    # METHODS ------------------------------------------------------------------

    @skip_if_disabled
    def set_stage(self, stage_name: str, split: str | None = None):
        """Set the current stage and optionally the split in MLflow tags."""
        mlflow.set_tag("current_stage", stage_name)
        if split:
            mlflow.set_tag("current_split", split)

    # LOGGING METHODS ----------------------------------------------------------

    @skip_if_disabled
    def log_restart(self, count: int, **kwargs):
        """Log restart metrics."""
        mlflow.log_metric(
            f"{self.metric_prefix}divergence_restores", count, **kwargs
        )

    @skip_if_disabled
    def log_epoch_metrics(
        self, metrics: dict, epoch: int = 1, split: str | None = None
    ):
        """Log metrics at epoch granularity, optionally namespaced by split."""
        if not self.config.mlflow.log_epoch_metrics:
            return

        base = self.metric_prefix
        ns = f"{split}_" if split else ""
        print(
            f"Logging epoch metrics to MLflow "
            f"(epoch {epoch}, split={split}): {list(metrics.keys())}"
        )
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(f"{base}{ns}{k}", v, step=epoch)
            else:
                print(
                    f"Warning: metric '{base}{ns}{k}' = None "
                    f"(epoch {epoch})"
                )

    @skip_if_disabled
    def log_metric(
        self,
        metric_name: str,
        metric_value: Any,
        split: str | None = None,
        **kwargs,
    ):
        """Log a single metric to MLflow."""
        ns = f"{split}_" if split else ""
        mlflow.log_metric(
            f"{self.metric_prefix}{ns}{metric_name}", metric_value, **kwargs
        )

    @skip_if_disabled
    def log_artifacts(self):
        """Log provided artifacts."""
        if self.output_directory is None:
            return

        artifact_patterns = self.config.mlflow.log_artifacts
        if not artifact_patterns:
            return
        print("Logging artifacts to MLFlow...")

        for pattern in artifact_patterns:
            full_pattern = os.path.join(self.output_directory, pattern)
            for local_artifact_path in glob.glob(full_pattern):
                if not os.path.exists(local_artifact_path):
                    continue

                artifact_folder = Path("generic")
                artifact_name = Path(local_artifact_path).name
                if artifact_name.endswith(".fnx"):
                    artifact_folder = Path("models")
                elif artifact_name.endswith(".html"):
                    artifact_folder = Path("results")

                try:
                    mlflow.log_artifact(
                        local_artifact_path,
                        artifact_path=str(artifact_folder),
                    )
                    print(f"Logged artifact: {local_artifact_path}")
                except Exception as e:
                    if self.config.mlflow.strict:
                        raise
                    print(f"Failed to log artifact {local_artifact_path}: {e}")

    @skip_if_disabled
    def log_config(self):
        """Log the config file."""
        if self.config_path is None:
            return

        if os.path.exists(self.config_path):
            mlflow.log_artifact(
                self.config_path.as_posix(), artifact_path="config_used.yaml"
            )
            print(f"Logged main config: {self.config_path.as_posix()}")

    @skip_if_disabled
    def log_dataset(self, dspath: str, dataset: dict):
        """Log the dataset info."""
        try:
            mlflow_dataset = FennolDataset(dspath, dataset)
            mlflow.log_input(mlflow_dataset, context="training")
        except Exception as e:
            if self.config.mlflow.strict:
                raise
            print(f"Could not log dataset to mlflow: {e}")

    @skip_if_disabled
    def log_model(
        self,
        *,
        fennix: FENNIX,
        model_name: str | None = None,
        source_model_file: str | None = None,
        example_batch: dict | None = None,
        alias: str | None = None,
        graph_config: dict | None = None,
        use_atom_padding: bool = False,
        ensure_run: bool = False,
    ):
        """
        Log a FENNIX model as a PyFunc, register it to the Model Registry.

        Optionally set a stable alias (e.g., 'staging' or 'prod').

        If you already loaded from a file, pass it as source_model_file to
        avoid resaving.
        """
        model_name = model_name or self.registered_model_name
        alias = self.model_alias if alias is None else alias

        ctx = (
            self
            if ensure_run and mlflow.active_run() is None
            else nullcontext()
        )
        with ctx:
            if source_model_file is None:
                with tempfile.TemporaryDirectory() as td:
                    tmp_path = os.path.join(td, "fennix.msgpack")
                    fennix.save(tmp_path)
                    model_artifacts = {"fennix_model_file": tmp_path}

                    _log_and_register(
                        fennix=fennix,
                        model_name=model_name,
                        model_artifacts=model_artifacts,
                        example_batch=example_batch,
                        alias=alias,
                        graph_config=graph_config,
                        use_atom_padding=use_atom_padding,
                    )
            else:
                model_artifacts = {"fennix_model_file": source_model_file}
                _log_and_register(
                    fennix=fennix,
                    model_name=model_name,
                    model_artifacts=model_artifacts,
                    example_batch=example_batch,
                    alias=alias,
                    graph_config=graph_config,
                    use_atom_padding=use_atom_padding,
                )

    # CONTEXT RELATED METHODS --------------------------------------------------

    def start_run(self):
        """Start mlflow run logging."""
        return self

    @skip_if_disabled
    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.current_run_name)
        print(f"MLFlow Run ID: {self.run.info.run_id}")

        if getattr(self, "tags", None):
            mlflow.set_tags(self.tags)
            print(f"Logged tags {self.tags} to MLFlow.")

        if self.config.mlflow.log_params:
            _params = flatten_dict(self.config.model_dump(mode='json'))
            for key, value in _params.items():
                if isinstance(value, str) and len(value) > 250:
                    _params[key] = value[:247] + "..."
            mlflow.log_params(_params)
            print("Logged parameters to MLFlow.")
        self.log_artifacts()
        return self

    @skip_if_disabled
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.log_artifacts()
        finally:
            self.run.__exit__(exc_type, exc_val, exc_tb)
