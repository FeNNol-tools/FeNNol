"""Training orchestration.

- Stages execution.
- Training environment instantiation.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from flax.core import freeze, unfreeze

from ..models import FENNIX
from ..utils import deep_update
from .callbacks import (
    Callback,
    Checkpointer,
    DivergenceChecker,
    MetricTracker,
    MLflowHandler,
    ThresholdStopper,
    TrainingLogger,
)
from .config import RunConfig, TrainingConfig
from .constants import (
    INITIAL_MODEL,
    LATEST_MODEL,
)
from .core import (
    get_evaluation_fn,
    get_loss_definition,
    get_train_step_function,
    get_validation_function,
)
from .io import (
    TeeLogger,
    copy_parameters,
    load_dataset,
    load_model,
)
from .mlflow_interface import MlFlowLogger
from .optimizers import get_lr_schedule, get_optimizer, multi_ema
from .trainer import CoreFunctions, Trainer, TrainState
from .utils import cast_variables_precision, load_restart_checkpoint


class TrainingOrchestrator:
    """Orchestrate training.

    Responsible for:
    - Handling restarts (checkpoint loading).
    - Resolving and iterating through training stages.
    - Building the 'TrainingEnv' for each stage.
    - Instantiating and running the 'Trainer'.

    Args:
        run_context (RunContext): runtime configuration of the run.
        _explicit_stages (bool): private state variable that hold whether user
            provided stages or not.
    """

    single_stage_key = "single_stage_training"

    def __init__(self, run_context: RunContext):
        self._run_context = run_context
        self._explicit_stages: bool = True

    @property
    def run_context(self) -> RunContext:
        """Access to the runtime context."""
        return self._run_context

    @property
    def config(self) -> RunConfig:
        """Access the global configuration."""
        return self._run_context.config

    def run(
        self,
    ) -> None:
        """Execute training."""
        # Load restart checkpoint if needed.
        if self.config.is_restart:
            restart_checkpoint = load_restart_checkpoint(
                self.config.output_directory
            )
        else:
            restart_checkpoint = None

        # Stages definition.
        self.stage_training_params = self.config.training.model_dump(
            exclude={"metrics_beta", "smoothen_metrics", "compute_ref_coords"}
        )
        stages = self._resolve_stages(restart_checkpoint)

        # Iterate through stages.
        final_env = None
        final_checkpointer = None
        with self.run_context.mlflow_logger.start_run():
            for stage_name, stage_params in stages.items():
                stage_idx = self._stage_indices[stage_name]
                print(f"\n### STAGE {stage_idx}: {stage_name} ###")
                stage_training_config = self._build_stage_config(
                    stage_params, stage_idx, stage_name
                )

                # Training initialization
                metrics_state = (
                    restart_checkpoint.get("metrics_state")
                    if restart_checkpoint
                    else {}
                )
                env = self._setup_training_environment(
                    stage_training_config,
                    stage_idx,
                    stage_name,
                    metrics_state,
                )
                final_env = env

                # Initiate TrainState
                if restart_checkpoint is not None:
                    train_state = TrainState.init_from_checkpoint(
                        restart_checkpoint
                    )
                    print("Restored training state.")
                else:
                    _, subkey = jax.random.split(self.run_context.rng_key)
                    ema_state = env.ema.init(env.model.variables)
                    variables_ema, _ = env.ema.update(
                        env.model.variables, ema_state
                    )
                    opt_state = env.optimizer.init(env.model.variables)
                    train_state = TrainState.create(
                        variables=env.model.variables,
                        variables_ema=variables_ema,
                        ema_state=ema_state,
                        opt_state=opt_state,
                        sch_state=env.sch_state,
                        preproc_state=env.model.preproc_state,
                        rng_key=subkey,
                    )

                # Create trainer
                trainer = Trainer(
                    env.model,
                    stage_training_config,
                    env.core_functions,
                    env.callbacks,
                    env.loss_definition,
                    # TODO remove next args. (delegate lr update to optax)
                    env.schedule,
                    env.index_lr,
                    env.schedule_metrics,
                    env.is_sch_adaptative,
                )
                checkpointer = trainer.get_callback(Checkpointer)
                if checkpointer is not None:
                    if restart_checkpoint is not None:
                        if "best_metric" in restart_checkpoint.get(
                            "training", {}
                        ):
                            checkpointer.best_metric_val = (
                                train_state.best_metric
                            )
                        checkpointer.best_model_file = restart_checkpoint.get(
                            "best_model_file", checkpointer.best_model_file
                        )
                # Run training
                train_state = trainer.fit(
                    train_state,
                    env.training_iterator,
                    env.validation_iterator,
                )
                restart_checkpoint = None
                # Can be improve so next stage just start from previous state.
                if checkpointer is not None:
                    final_checkpointer = checkpointer
                    self.config.model_file = str(
                        self.config.output_directory
                        / checkpointer.final_model_file
                    )

            # Only publish a model after every stage completed successfully.
            if final_env is not None and final_checkpointer is not None:
                checkpoint_name = (
                    final_checkpointer.best_model_file
                    if (self.config.output_directory
                        / final_checkpointer.best_model_file).exists()
                    else final_checkpointer.final_model_file
                )
                checkpoint_path = (
                    self.config.output_directory / checkpoint_name
                )
                self.run_context.mlflow_logger.log_model(
                    fennix=final_env.model,
                    source_model_file=str(checkpoint_path),
                    use_atom_padding=final_env.model.use_atom_padding,
                )

    def _resolve_stages(
        self, restart_checkpoint: dict[str, Any] | None
    ) -> dict:
        """Determine which stages to run based on config and restart state."""
        if not self.config.stages:
            stages = {self.single_stage_key: {}}
            self._explicit_stages = False
        else:
            stages = self.config.stages
        original_indices = {
            name: index for index, name in enumerate(stages, start=1)
        }
        self._stage_indices = original_indices

        if not restart_checkpoint:
            return stages

        saved_stage_idx = restart_checkpoint.get(
            "stage_idx", restart_checkpoint.get("stage", 1)
        )
        # Filter stages
        filtered_stages = {}
        for i, (k, v) in enumerate(stages.items(), start=1):
            if i >= saved_stage_idx:
                filtered_stages[k] = v
            else:
                print(f"Skipping stage {i} (already completed)")
                self.stage_training_params = deep_update(
                    self.stage_training_params, v
                )

        self._stage_indices = {
            name: original_indices[name] for name in filtered_stages
        }
        return filtered_stages

    def _build_stage_config(
        self, stage_params: dict, stage_idx: int, stage_name: str
    ) -> TrainingConfig:
        """Merge global training params with stage-specific overrides."""
        stage_training_params = deep_update(
            self.stage_training_params, stage_params
        )
        self.stage_training_params = stage_training_params

        if stage_idx > 1:
            # remove stopping criteria from previous stage
            stage_training_params.pop("end_event", None)

        try:
            stage_training_config = TrainingConfig(**stage_training_params)
            if stage_training_config.print_stages_params:
                print(
                    f"Stage {stage_idx} Config:\n"
                    f"{
                        json.dumps(
                            stage_training_config.model_dump(mode='json'),
                            indent=2,
                        )
                    }"
                )
            return stage_training_config
        except Exception as e:
            raise ValueError(f"Configuration Error in stage {stage_name}: {e}")

    def _setup_training_environment(
        self,
        stage_training_config: TrainingConfig,
        stage_idx: int,
        stage_name: str,
        metrics_state: dict[str, Any] | None = None,
    ) -> TrainingEnv:
        """Build the training environment.

        1. Loading the model (and optionally reference model).
        2. Setting up loss functions.
        3. Loading datasets.
        4. creating Optimizers, Schedules, and EMA.
        5. Compiling Core Functions (Train/Val steps).
        6. Instantiating Callbacks.

        Args:
            stage_training_config (TrainingConfig): stage configuration.
            stage_idx (int): base 1 index of the stage to check whether or not
                we need to save initial model.
            stage_name (str): stage name for logging.
            metrics_state (str):  Optional metrics state from a restart
                checkpoint to resume updating. Defaults to None

        Returns:
            TrainingEnv: complete training env (see dataclass).

        """
        # 1. load model
        rng_key, model_key = jax.random.split(self.run_context.rng_key)
        self.run_context.rng_key = rng_key
        model_file = (
            str(self.config.output_directory / LATEST_MODEL)
            if self.config.is_restart
            else self.config.model_file
        )

        model = load_model(
            self.config.model,
            model_file,
            rng_key=model_key,
            print_model=stage_training_config.print_model,
        )
        if not self.config.is_restart and (stage_idx == 1):
            model.save(str(self.config.output_directory / INITIAL_MODEL))

        model.variables = cast_variables_precision(
            model.variables, self.config.fprec
        )
        if len(stage_training_config.energy_terms) > 0:
            model.set_energy_terms(
                stage_training_config.energy_terms, jit=False
            )
        print("energy terms:", model.energy_terms)

        # 2. optionally load ref model
        model_ref = None
        if stage_training_config.model_ref:
            print("Reference model:", stage_training_config.model_ref)
            model_ref = load_model(model_file=stage_training_config.model_ref)
            if stage_training_config.ref_parameters:
                print(
                    "Copying reference parameters: "
                    f"{stage_training_config.ref_parameters}"
                )
                model.variables = copy_parameters(
                    variables=model.variables,
                    variables_ref=model_ref.variables,
                    params=stage_training_config.ref_parameters,
                )

        # 3. set up loss function
        loss_definition, used_keys, ref_keys = get_loss_definition(
            stage_training_config,
            model_energy_unit=model.energy_unit,
        )
        if stage_training_config.compute_ref_coords:
            print(
                "Reference coordinates:",
                stage_training_config.coordinates_ref_key,
            )

        # 4. build datasets
        training_iterator, validation_iterator = load_dataset(
            dspath=str(stage_training_config.dspath),
            batch_size=stage_training_config.batch_size,
            batch_size_val=stage_training_config.batch_size_val,
            training_config=stage_training_config,
            infinite_iterator=True,
            atom_padding=True,
            ref_keys=ref_keys,
            split_data_inputs=True,
            np_rng=self.run_context.np_rng,
            add_flags=["training"],
            fprec=self.config.fprec,
            rename_refs=stage_training_config.rename_refs,
            log_dataset=self.run_context.mlflow_logger.log_dataset,
            load_mode=stage_training_config.load_mode,
            load_chunk_size=stage_training_config.load_chunk_size
        )

        # 5. schedule, optimizer and ema
        schedule, sch_state, schedule_metrics, adaptive_scheduler = (
            get_lr_schedule(stage_training_config)
        )
        optimizer, index_lr = get_optimizer(
            stage_training_config,
            model.variables,
            schedule(sch_state)[0],
        )
        ema = multi_ema(
            decays=stage_training_config.ema_decay,
            power=stage_training_config.ema_power,
        )

        # 6. Core functions
        evaluate = get_evaluation_fn(
            model, used_keys, stage_training_config.pbc_training
        )
        train_step = get_train_step_function(
            loss_definition=loss_definition,
            model=model,
            model_ref=model_ref,
            compute_ref_coords=stage_training_config.compute_ref_coords,
            evaluate=evaluate,
            optimizer=optimizer,
            ema=ema,
        )
        validation_step = get_validation_function(
            loss_definition=loss_definition,
            model=model,
            model_ref=model_ref,
            compute_ref_coords=stage_training_config.compute_ref_coords,
            evaluate=evaluate,
            return_targets=False,
        )
        core_functions = CoreFunctions(
            train_step=train_step,
            validation_step=validation_step,
            init_opt=optimizer.init,
            init_ema=ema.init,
        )

        # 7. preprocessor
        model = configure_preprocessing(model, stage_training_config)
        if stage_training_config.load_preproc_state is not None:
            with open(stage_training_config.load_preproc_state, "rb") as f:
                loaded_state = pickle.load(f)["preproc_state"]
            model.preproc_state = freeze(loaded_state)
            print(
                "Loaded preprocessing state from",
                stage_training_config.load_preproc_state,
            )

        # 9. create Trainer callbacks
        stage_prefix = "" if stage_name == self.single_stage_key else stage_name

        checkpointer = Checkpointer.init_from_config(
            output_dir=self.config.output_directory,
            config=stage_training_config,
            stage_idx=stage_idx,
            stage_prefix=stage_prefix,
        )

        logger = TrainingLogger(
            output_dir=self.config.output_directory,
            writing_mode="a" if self.config.is_restart else "w",
            stage_prefix=stage_prefix,
        )

        divergence_checker = DivergenceChecker()

        metrics_tracker = MetricTracker(
            stage_training_config.metrics_beta,
            initial_metrics_state=metrics_state,
        )

        if self._explicit_stages:
            self.run_context.mlflow_logger.set_stage(stage_name)
            self.run_context.mlflow_logger.metric_prefix = (
                f"stage_{stage_name}_" if stage_name else ""
            )
        mlflow_handler = MLflowHandler(
            logger=self.run_context.mlflow_logger,
        )

        callbacks = [  # Order matters.
            metrics_tracker,
            divergence_checker,
            mlflow_handler,
            checkpointer,
            logger,
        ]
        if stage_training_config.end_event:
            metric_name, threshold = stage_training_config.end_event
            stopper = ThresholdStopper(
                metric_to_monitor=metric_name, threshold=threshold, mode="min"
            )
            callbacks.append(stopper)

        training_environment = TrainingEnv(
            model=model,
            core_functions=core_functions,
            optimizer=optimizer,
            index_lr=index_lr,
            ema=ema,
            schedule=schedule,
            sch_state=sch_state,
            training_iterator=training_iterator,
            validation_iterator=validation_iterator,
            loss_definition=loss_definition,
            callbacks=callbacks,
            schedule_metrics=schedule_metrics,
            is_sch_adaptative=adaptive_scheduler,
        )

        return training_environment


def configure_preprocessing(model: FENNIX, config: TrainingConfig) -> FENNIX:
    """Update the model's preprocessing state based on configuration.

    Args:
        model (FENNIX): initial model.
        config (TrainingConfig): training configuration.

    Returns:
        FENNIX: model with prepocessor updated.
    """
    if config.gpu_preprocessing:
        print("GPU preprocessing activated.")

    preproc_state = unfreeze(model.preproc_state)
    new_layers = []

    for layer_st in preproc_state["layers_state"]:
        st = unfreeze(layer_st)
        if config.nblist_mult_size is not None:
            st["nblist_mult_size"] = config.nblist_mult_size
        if config.nblist_add_neigh is not None:
            st["add_neigh"] = config.nblist_add_neigh
        if config.nblist_add_atoms is not None:
            st["add_atoms"] = config.nblist_add_atoms

        new_layers.append(freeze(st))

    preproc_state["layers_state"] = tuple(new_layers)
    preproc_state["check_input"] = False

    model.preproc_state = freeze(preproc_state)  # type: ignore

    return model


# *** DATA CLASSES ***


@dataclass(frozen=True)
class TrainingEnv:
    """Container for all static objects needed to run a training stage.

    Args:
        model (Any): The model instance.
        core_functions (CoreFunctions): Compiled JAX functions.
        optimizer (Any): Optax optimizer instance.
        index_lr (int): Index of the learning rate parameter in the optimizer.
        ema (Any): EMA handler.
        schedule (Callable): Learning rate schedule function.
        sch_state (Any): Scheduler state.
        training_iterator (Iterator): Data iterator for training.
        validation_iterator (Iterator): Data iterator for validation.
        loss_definition (dict[str, Any]): Dictionary defining loss weights
            and types.
        callbacks (list[Callback]): List of callbacks to execute.
        schedule_metrics (str): Metric name used for adaptive scheduling.
        is_sch_adaptative (bool): Whether the scheduler is adaptive (Plateau).
    """

    model: Any
    core_functions: CoreFunctions
    optimizer: Any
    index_lr: int
    ema: Any
    schedule: Callable
    sch_state: Any
    training_iterator: Iterator
    validation_iterator: Iterator
    loss_definition: dict[str, Any]
    callbacks: list[Callback]
    schedule_metrics: str
    is_sch_adaptative: bool


@dataclass
class RunContext:
    """Holds the runtime state of the application.

    Args:
        config (RunConfig): Global configuration object.
        rng_key (jax.Array): Current JAX PRNG key.
        np_rng (np.random.Generator): Numpy random generator.
        jax_device (Any): The JAX device being used (TPU/GPU/CPU).
        logger (TeeLogger): Logger instance for stdout capturing.
        mlflow_logger (MlFlowLogger): Logger to MLflow.
    """

    config: RunConfig
    rng_key: jax.Array
    np_rng: np.random.Generator
    jax_device: Any  # jax._src.lib.xla_client.Device
    logger: TeeLogger
    mlflow_logger: MlFlowLogger
