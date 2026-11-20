"""Loss definitions, training step and validation, including reference-Model targets."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import optax
import pytest

pytest.importorskip("torch", reason="training requires PyTorch")

from fennol.training.config import TrainingConfig
from fennol.training.core import (
    get_loss_definition,
    get_train_step_function,
    get_validation_function,
)


def test_l2norm_loss_updates_vector_predictions():
    model = SimpleNamespace(variables={"prediction": jnp.array([[3.0, 4.0, 0.0]])})
    optimizer = optax.sgd(0.1)
    loss_definition, _, _ = get_loss_definition(
        TrainingConfig(
            loss={
                "forces": {
                    "key": "prediction",
                    "ref": "forces",
                    "type": "l2norm",
                    "weight": 1.0,
                }
            }
        )
    )

    def evaluate(_model, variables, inputs):
        return {"prediction": variables["prediction"], "batch_index": inputs["batch_index"]}

    train_step = get_train_step_function(
        loss_definition,
        model,
        evaluate,
        optimizer,
        jit=False,
    )
    inputs = {
        "true_sys": jnp.array([True]),
        "true_atoms": jnp.array([True]),
        "natoms": jnp.array([1]),
        "batch_index": jnp.array([0]),
    }
    state = {
        "variables": model.variables,
        "opt_state": optimizer.init(model.variables),
        "epoch": 0,
        "step": 0,
    }

    loss, updated, _ = train_step({"forces": jnp.zeros((1, 3))}, inputs, state)

    assert float(loss) == pytest.approx(5.0 / 3.0)
    assert not np.array_equal(
        np.asarray(updated["variables"]["prediction"]), [[3.0, 4.0, 0.0]]
    )


def test_model_ref_atom_mask_excludes_unlabeled_atoms_from_validation():
    loss_definition = {
        "energy": {
            "key": "prediction",
            "ref": "model_ref/energy",
            "mult": 1.0,
            "type": "mse",
        }
    }
    model = SimpleNamespace(variables={})
    model_ref = SimpleNamespace(variables={})

    def evaluate(which_model, _variables, inputs):
        if which_model is model_ref:
            return {
                "energy": jnp.array([1.0, 100.0]),
                "energy_mask": jnp.array([True, False]),
            }
        return {
            "prediction": jnp.array([3.0, 10.0]),
            "batch_index": inputs["batch_index"],
        }

    validate = get_validation_function(
        loss_definition,
        model,
        evaluate,
        model_ref=model_ref,
        jit=False,
    )
    inputs = {
        "true_sys": jnp.array([True]),
        "true_atoms": jnp.array([True, True]),
        "natoms": jnp.array([2]),
        "batch_index": jnp.array([0, 0]),
    }

    rmses, maes, _ = validate({}, inputs, {})

    assert float(rmses["energy"]) == pytest.approx(2.0)
    assert float(maes["energy"]) == pytest.approx(2.0)


def test_loss_definition_consumes_training_config_model():
    config = TrainingConfig(
        loss={"energy": {"key": "total_energy", "weight": 1.0}}
    )

    loss_definition, used_keys, _ = get_loss_definition(config)

    assert loss_definition["energy"]["key"] == "total_energy"
    assert used_keys == ["total_energy"]


def test_model_ref_validation_uses_reference_model_mask():
    pytest.importorskip("torch", reason="training requires PyTorch")
    from fennol.training.core import get_validation_function

    loss_definition = {
        "energy": {
            "key": "prediction",
            "ref": "model_ref/energy",
            "mult": 1.0,
            "type": "mse",
        }
    }
    model = SimpleNamespace(variables={})
    reference_model = SimpleNamespace(variables={})

    def evaluate(which_model, variables, inputs):
        if which_model is reference_model:
            return {
                "energy": jnp.array([10.0]),
                "energy_mask": jnp.array([False]),
            }
        return {
            "prediction": jnp.array([2.0]),
            "batch_index": inputs["batch_index"],
        }

    validate = get_validation_function(
        loss_definition,
        model,
        evaluate,
        model_ref=reference_model,
        jit=False,
    )
    inputs = {
        "true_sys": jnp.array([True]),
        "true_atoms": jnp.array([True]),
        "natoms": jnp.array([1]),
        "batch_index": jnp.array([0]),
    }

    rmses, maes, _ = validate({}, inputs, {})

    assert float(rmses["energy"]) == 0.0
    assert float(maes["energy"]) == 0.0


def test_model_ref_training_uses_reference_model_mask():
    pytest.importorskip("torch", reason="training requires PyTorch")
    from fennol.training.core import get_train_step_function

    loss_definition = {
        "energy": {
            "key": "prediction",
            "ref": "model_ref/energy",
            "mult": 1.0,
            "type": "mse",
            "weight": 1.0,
        }
    }
    model = SimpleNamespace(variables={"prediction": jnp.array([2.0])})
    reference_model = SimpleNamespace(variables={})

    def evaluate(which_model, variables, inputs):
        if which_model is reference_model:
            return {
                "energy": jnp.array([10.0]),
                "energy_mask": jnp.array([False]),
            }
        return {
            "prediction": variables["prediction"],
            "batch_index": inputs["batch_index"],
        }

    optimizer = optax.sgd(0.1)
    training_step = get_train_step_function(
        loss_definition,
        model,
        evaluate,
        optimizer,
        model_ref=reference_model,
        jit=False,
    )
    variables = model.variables
    training_state = {
        "variables": variables,
        "opt_state": optimizer.init(variables),
        "epoch": 0,
        "step": 0,
    }
    inputs = {
        "true_sys": jnp.array([True]),
        "true_atoms": jnp.array([True]),
        "natoms": jnp.array([1]),
        "batch_index": jnp.array([0]),
    }

    loss, updated_state, _ = training_step({}, inputs, training_state)

    assert float(loss) == 0.0
    np.testing.assert_allclose(updated_state["variables"]["prediction"], [2.0])
