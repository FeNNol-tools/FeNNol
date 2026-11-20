"""Configuration classes.

To modify a configuration value using ennvironment variables, you should
rely on pydantic objects hierarchy. For instance, to access
run_config.training.init_lr use FENNOL_TRAINING__INIT_LR.
You can also use aliases such as MLFLOW_EXPERIMENT_NAME when provided.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
    YamlConfigSettingsSource,
)

from ..utils.device import normalize_device

DEFAULT_CONFIG_PATH = "./config.yaml"


class TrainingConfig(BaseModel):
    """Configuration for a specific training stage."""

    # validate fields at assignemnts.
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    print_stages_params: bool = False
    print_model: bool = False
    max_epochs: int = 2000
    nbatch_per_epoch: int = 200
    nbatch_per_validation: int = 20
    batch_size: int = 16
    batch_size_val: int | None = None
    metric_best: str = Field(default="rmse_tot")
    end_event: tuple[str, float] | None = None  # stopping criterion
    metrics_ema_decay: float = -1  # metric smoothing beta

    # Preprocessing
    gpu_preprocessing: bool = False
    nblist_mult_size: float | None = None
    nblist_add_neigh: int | None = None
    nblist_add_atoms: int | None = None
    load_preproc_state: Path | None = None

    # Optimization
    optimizer_config: dict[str, Any] = Field(default_factory=dict[str, Any])
    optimizer: str = "adabelief"
    init_lr: float = 0.001
    lr: float = 0.002
    final_lr: float = 1e-06
    schedule_type: str = "cosine_onecycle"
    schedule_metrics: str = "rmse_tot"
    onecycle_epochs: int | None = None
    peak_epoch: float | None = None
    linear_warmup: bool = False
    restore_scaling: float = Field(default=1.0, gt=0.0, le=1.0)
    max_restore_count: int = 5
    ema_decay: list[float] = Field(default_factory=list)
    ema_power: bool = False
    ema_validate: int = 0  # Index of EMA to use for validation
    gradient_clipping: float = -1.
    clipping_range: str = "global"
    adaptive_gradient_clipping: float = -1.
    weight_decay: float = Field(default=0.0, ge=0.0)
    decay_targets: list[str] = Field(default_factory=lambda: [""])
    regularize_init_weights: float = Field(default=0.0, ge=0.0)
    use_grokfast: bool = False
    alpha_grokfast: float = 0.9
    l_grokfast: float = 1.0

    # Parameters
    default_status: Literal["trainable", "frozen"] = Field(default="trainable")
    frozen: list[str] = Field(default=[])
    trainable: list[str] = Field(default=[])
    zero_nans: bool = False

    # Data
    dspath: Path | str = Field(default=Path("dataset.pkl"), validate_default=True)
    rename_refs: dict[str, str] = Field(default_factory=dict)
    pbc_training: bool = False
    load_mode: Literal["inmemory", "on_demand"] = Field(default="on_demand")
    load_chunk_size: int = 4096
    minimum_image: bool = False
    additional_input_keys: list[str] = Field(default_factory=list)
    random_rotation: bool = False
    rotated_keys: dict[str, Any] = Field(default_factory=dict)
    flow_matching: bool = False
    length_nopbc: float = 1000.0
    noise_sigma: dict[str, float] | None = None
    padder_add_sys: int = 0
    shuffle_dataset: bool = True
    batch_prefetch: int = 2

    # Scheduler options consumed by optimizer construction.
    scheduler: str | None = None
    scheduler_parameters: dict[str, Any] = Field(default_factory=dict)
    patience: int = Field(default=10, ge=1)
    lr_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience_thr: float = Field(default=0.0, ge=0.0)
    stochastic_scheduler: bool = False
    scheduler_seed: int = 0

    # Loss
    default_loss_type: str = "mse"
    loss: dict[str, Any] = Field(default_factory=dict)
    energy_terms: list[str] = Field(default_factory=list)

    # Reference
    model_ref: str | None = None  # Path to reference model file
    ref_parameters: list[str] = Field(default_factory=list)  # Params to copy
    coordinates_ref_key: str | None = None

    # Model files
    keep_all_bests: bool = False
    save_model_at_epochs: list[int] = Field(default_factory=list)
    save_model_every_epoch: int = 0
    save_model_schedule: list[int] = Field(default_factory=list)

    @computed_field
    @property
    def metrics_beta(self) -> float:
        """Metric smoothing beta."""
        return self.metrics_ema_decay

    @computed_field
    @property
    def smoothen_metrics(self) -> bool:
        """Whether or no smooth metrics."""
        return self.metrics_beta < 1.0 and self.metrics_beta > 0.0

    @computed_field
    @property
    def compute_ref_coords(self) -> bool:
        """Float precision."""
        return self.coordinates_ref_key is not None

    @field_validator("ema_decay", mode="before")
    @classmethod
    def normalize_ema_decay(cls, value: Any) -> list[float]:
        """Standardizes float to List[float] and checks decays."""
        if isinstance(value, (float, int, str)):
            value = [float(value)]
        normalized = []
        for v in value:
            decay = float(v)
            if not (0.0 <= decay < 1.0):
                raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
            normalized.append(decay)
        return value

    @field_validator("dspath", mode="before")
    @classmethod
    def preserve_http_dspath(cls, value: Any) -> Any:
        """Keep remote dataset URLs as strings instead of normalizing them as paths."""
        if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
            return value
        return Path(value)

    @field_validator("dspath", mode="after")
    @classmethod
    def check_dspath_is_valid(cls, value: Path | str) -> Path | str:
        """Check dataset is a dir or a supported file."""
        if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
            return value
        if not value.exists():
            print(f"Config dataset not found at: {value}")
        if value.is_dir():
            return value
        elif value.suffix:
            allowed_extensions = ['.db', '.h5', '.pkl', '.pickle', '.parquet']
            actual_extension = value.suffix
            if actual_extension not in allowed_extensions:
                raise ValueError(
                    f"Unknown dataset format {actual_extension}.\n"
                    f"Supported formats: {allowed_extensions}")
        return value

    @field_validator("end_event", mode="after")
    @classmethod
    def parse_end_event(cls, value: Any) -> tuple[str, float] | None:
        """Check dataset file exists."""
        if value is None:
            return value
        if (len(value) != 2) or (not isinstance(value[1], (float, int))):
            raise ValueError(
                "When given, end_event should be a tuple (metric, threshold) "
                f"but was {value}"
            )
        return value


class MlflowConfig(BaseSettings):
    """Configuration for MLflow logging.

    MLflow is disabled by default. Install its optional dependencies with
    ``fennol[train,mlflow]`` (or ``uv sync --extra train --extra mlflow``)
    before setting ``logging_enabled`` to true.

    Use AliasChoices to works with standard MLflow environment variables.
    For instance, using env. var we can set run_name using MLFLOW_RUN_NAME
    (highest priority) or standard FENNOL_MLFLOW__RUN_NAME.
    """

    # --- STANDARD MLFLOW VARIABLES ---
    tracking_uri: str | None = Field(
        default=None,  # MLflow create ./mlruns if not provided.
        validation_alias=AliasChoices("tracking_uri", "MLFLOW_TRACKING_URI"),
    )
    experiment_name: str = Field(
        default="Default",  # if not defined, go to the default group.
        validation_alias=AliasChoices(
            "experiment_name", "MLFLOW_EXPERIMENT_NAME"
        ),
    )
    run_name: str | None = Field(
        default=None,  # rely on MLflow to generate unique name.
        validation_alias=AliasChoices("run_name", "MLFLOW_RUN_NAME"),
    )
    tags: dict[str, str] | str = Field(
        default_factory=dict,
        validation_alias=AliasChoices("tags", "MLFLOW_TAGS"),
    )

    # --- MODEL REGISTRY ---
    registered_model_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "registered_model_name", "MLFLOW_REGISTERED_MODEL_NAME"
        ),
    )
    model_alias: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model_alias", "MLFLOW_MODEL_ALIAS"),
    )
    registry_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("registry_uri", "MLFLOW_REGISTRY_URI"),
    )

    # --- CUSTOM FENNOL CONTROLS ---
    logging_enabled: bool = False  # Master switch
    strict: bool = False  # Raise errors from MLflow logging operations.
    log_params: bool = True  # log training configuraiton
    log_step_metrics: bool = True
    log_epoch_metrics: bool = True
    log_artifacts: list[str] = Field(default_factory=list)  # Safer for local.

    @field_validator("log_artifacts", mode="before")
    @classmethod
    def parse_log_artifacts(cls, v: Any) -> Any:
        """Convert log_artifacts from str to list."""
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, str):
            return [x.strip() for x in str(v).split(",") if x.strip()]
        return v

    @field_validator("run_name")
    @classmethod
    def resolve_run_name(cls, v: str | None) -> str | None:
        """Inject timestamp if {now} is present in the run name."""
        if v and "{now}" in v:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            return v.replace("{now}", timestamp)
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: str | dict) -> dict[str, str]:
        """Parse tags from JSON or comma-separated strings."""
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                # example: '{"env": "prod", "team": "data"}'
                return json.loads(v)
            except json.JSONDecodeError:
                # example: 'env=prod,team=data'
                parsed_tags = {}
                for item in v.split(","):
                    if "=" in item:
                        key, val = item.split("=", 1)
                        parsed_tags[key.strip()] = val.strip()
                return parsed_tags
        return v

    @model_validator(mode="after")
    def check_logging_setup(self) -> MlflowConfig:
        """Ensure consistency between URI and enabled status."""
        if self.logging_enabled and self.tracking_uri is None:
            print(
                "Warning: No MLflow Tracking URI set. "
                "Logs will be saved locally to ./mlruns"
            )
            pass

        if not self.logging_enabled:
            self.log_params = False
            self.log_step_metrics = False
            self.log_epoch_metrics = False
            self.log_artifacts = []

        return self


class RunConfig(BaseSettings):
    """Top-level settings. Parse and prioritize config sources.

    Precedence (highest → lowest):

    1. Environment variables (prefix ``FENNOL_``, nested with ``__``)
    2. YAML file (``FENNOL_CONFIG_FILE`` or ``configs/default.yaml``)
    3. Init kwargs
    4. Secrets (unused here)
    """

    # Infra
    output_directory: Path = Path(".")
    path_config_file: Path = Path(DEFAULT_CONFIG_PATH)
    model_file: str | None = None
    is_restart: bool = False
    double_precision: bool = False
    matmul_prec: Literal["default", "high", "highest"] = "highest"
    rng_seed: int = np.random.randint(0, 2**32 - 1)
    device: str = "cpu"
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    # Domain
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: dict[str, Any] | None = Field(default_factory=dict)
    stages: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def fprec(self) -> Literal["float32", "float64"]:
        """Float precision."""
        return "float64" if self.double_precision else "float32"

    @field_validator("device", mode="before")
    @classmethod
    def normalize_device(cls, device_value: Any) -> str:
        """Normalize device string and validate it."""
        if not isinstance(device_value, str):
            return device_value
        return normalize_device(device_value)

    @field_validator("output_directory", mode="before")
    @classmethod
    def resolve_output_directory_path(cls, value: str | Path) -> Path:
        """Explicit and make output dir a resolved path."""
        if isinstance(value, str) and "{now}" in value:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            value = value.replace("{now}", timestamp)

        path = Path(value).resolve()

        return path

    @model_validator(mode="before")
    @classmethod
    def normalize_model_source_format(cls, data: dict) -> dict:
        """Allow model key to have either file path or model definition.

        If 'model' is a string path, move it to 'model_file'
        and clear the definition.
        """
        if "model" in data and isinstance(data["model"], str):
            model_file = data["model"]
            if "model_file" not in data or data["model_file"] is None:
                data["model_file"] = model_file
            data["model"] = None

        return data

    model_config = SettingsConfigDict(
        env_prefix="FENNOL_",
        env_nested_delimiter="__",
        extra="ignore",  # "forbid"
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define the order of configuration sources.

        **highest priority**
        - Init kwargs
        - Environment Variables
        - configuration file
        - .env file
        **lowest priority**
        """
        file_source = None
        config_path = init_settings.init_kwargs.get("path_config_file")  # type: ignore
        if config_path is None:
            config_path = Path(DEFAULT_CONFIG_PATH)
        else:
            config_path = Path(config_path)
        if config_path.exists() and config_path.is_file():
            if config_path.suffix in (".yaml", ".yml"):
                file_source = YamlConfigSettingsSource(
                    settings_cls, config_path
                )
            elif config_path.suffix == ".json":
                file_source = JsonConfigSettingsSource(
                    settings_cls, config_path
                )
            elif config_path.suffix == ".toml":
                file_source = TomlConfigSettingsSource(
                    settings_cls, config_path
                )
            else:
                supported_formats = [".yaml", ".yml", ".json", ".toml"]
                if "path_config_file" in init_settings.init_kwargs:  # type: ignore
                    supported_formats = [".yaml", ".yml", ".json", ".toml"]
                    raise ValueError(
                        "Unknown config file extension {path.suffix}. "
                        f"Supported formats: {supported_formats}"
                    )

        # 3. Define Priority (Highest -> Lowest)
        sources = []
        sources.append(init_settings)
        sources.append(env_settings)
        if file_source:
            sources.append(file_source)
        sources.append(dotenv_settings)

        return tuple(sources)

    @classmethod
    def from_cli_args(
        cls, config_file_arg: str, model_file_arg: str | None = None
    ) -> RunConfig:
        """Initialize RunConfig from CLI arguments.

        Determines if the run is a new one or a restart based on whether
        the input argument is a file or a directory.

        Args:
            config_file_arg (str): CLI arg that hold the path to config file
                or output directory .
            model_file_arg (str | None): Optional override for the model file.

        Returns:
            RunConfig: new instance of RunConfig

        Raises:
            FileNotFoundError: no config found in restart mode.
        """
        input_path = Path(config_file_arg).resolve()

        # Input is a directory: restart mode
        if input_path.is_dir():
            config_file = input_path / "config.yaml"
            if not config_file.exists():
                error_msg = f"Cannot restart: {config_file} does not exist."
                raise FileNotFoundError(error_msg)
            init_values = dict(
                is_restart=True,
                output_directory=input_path,
                path_config_file=config_file,
            )
        # Input is a file: new run mode
        else:
            init_values = dict(
                is_restart=False,
                path_config_file=input_path,
            )

        # Only pass the override when given, so the config file value is kept.
        if model_file_arg is not None:
            init_values["model_file"] = model_file_arg
        return cls(**init_values)
