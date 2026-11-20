"""Model Modules: embeddings, networks, activations and FENNIX energy terms."""

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fennol.models.embeddings.caiman import CaimanEmbedding
from fennol.models.embeddings.hipnn import HIPNNEmbedding
from fennol.models.embeddings.painn import PAINNEmbedding
from fennol.models.embeddings.schnet import SchNetEmbedding
from fennol.models.fennix import FENNIX
from fennol.models.misc.e3 import FullTensorProduct
from fennol.models.misc.encodings import RadialBasis, SpeciesEncoding
from fennol.models.misc.misc import ApplyFunction
from fennol.models.misc.nets import GatedPerceptron, HierarchicalNet, ZLoRANet
from fennol.utils.activations import TrainableSoftSat, softabs, softsat
from fennol.utils.periodic_table import OXIDATION_STATES
from flax.errors import ScopeParamNotFoundError


def test_apply_function_accepts_single_string_key():
    module = ApplyFunction(function_str="x: x + 1", argument_keys="energy")
    inputs = {"energy": jnp.array([2.0])}

    variables = module.init(jax.random.PRNGKey(0), inputs)
    output = module.apply(variables, inputs)

    np.testing.assert_allclose(output[module.name], [3.0])


def test_hierarchical_net_can_initialize_and_run():
    module = HierarchicalNet(neurons=[3, 1], squeeze=True)
    inputs = jnp.ones((2, 4))

    variables = module.init(jax.random.PRNGKey(0), inputs)
    output = module.apply(variables, inputs)

    assert output.shape == (2,)


def test_gated_perceptron_can_squeeze_single_channel_output():
    module = GatedPerceptron(dim=1, squeeze=True)
    inputs = jnp.ones((2, 4))

    variables = module.init(jax.random.PRNGKey(0), inputs)
    output = module.apply(variables, inputs)

    assert output.shape == (2,)


def test_full_tensor_product_honors_zero_output_limit():
    module = FullTensorProduct(lmax1=1, lmax2=1, lmax_out=0)
    x = jnp.ones((1, 4))
    variables = module.init(jax.random.PRNGKey(0), x, x)

    result, irreps = module.apply(variables, x, x)

    assert result.shape == (1, len(irreps))
    assert all(l == 0 for l, _ in irreps)


def test_zlora_has_nonzero_adapter_gradient_at_initialization():
    module = ZLoRANet(neurons=[3], ranks=[2], zmax=1)
    species = jnp.array([1])
    inputs = (species, jnp.ones((1, 2)))
    variables = module.init(jax.random.PRNGKey(0), inputs)

    def loss(params):
        return module.apply({"params": params}, inputs).sum()

    grads = jax.grad(loss)(variables["params"])

    assert jnp.linalg.norm(grads["B_1"]) > 0


def _hipnn_module_and_inputs():
    inputs = {
        "species": jnp.array([1, 1]),
        "graph": {
            "edge_src": jnp.array([0, 1]),
            "edge_dst": jnp.array([1, 0]),
            "distances": jnp.array([1.0, 1.0]),
            "switch": jnp.ones((2,)),
            "vec": jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        },
    }
    module = HIPNNEmbedding(
        _graphs_properties={"graph": {"cutoff": 5.0}},
        dim=4,
        n_onsite=0,
        nlayers=1,
        lmax=2,
        species_encoding={"encoding": "one_hot", "zmax": 1},
        radial_basis={"dim": 3},
        keep_all_layers=False,
    )
    return module, inputs


def test_hipnn_uses_one_coefficient_per_angular_order():
    module, inputs = _hipnn_module_and_inputs()
    variables = module.init(jax.random.PRNGKey(0), inputs)

    def loss(params):
        output = module.apply({"params": params}, inputs)["embedding"]
        return jnp.sum(output**2)

    gradients = jax.grad(loss)(variables["params"])

    # l=1 and l=2 each own one coefficient and both contribute to the output.
    assert gradients["tl_0"].shape == (2,)
    assert jnp.all(gradients["tl_0"] != 0)


def test_hipnn_rejects_checkpoints_with_legacy_angular_coefficients():
    module, inputs = _hipnn_module_and_inputs()
    params = dict(module.init(jax.random.PRNGKey(0), inputs)["params"])
    params["ts_0"] = jnp.ones((2, 3))
    del params["tl_0"]

    with pytest.raises(ScopeParamNotFoundError, match="tl_0"):
        module.apply({"params": params}, inputs)


def _two_atom_graph_inputs():
    return {
        "species": jnp.array([1, 8]),
        "coordinates": jnp.zeros((2, 3)),
        "graph": {
            "edge_src": jnp.array([0, 1]),
            "edge_dst": jnp.array([1, 0]),
            "distances": jnp.array([1.2, 1.2]),
            "switch": jnp.ones((2,)),
            "vec": jnp.array([[1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]]),
        },
    }


def test_painn_runs_with_its_default_channel_setting():
    inputs = _two_atom_graph_inputs()
    module = PAINNEmbedding(
        _graphs_properties={"graph": {"cutoff": 5.0}},
        dim=4,
        nlayers=1,
        message_hidden=[],
        update_hidden=[],
        species_encoding={"encoding": "one_hot", "zmax": 8},
        radial_basis={"basis": "gaussian", "dim": 4},
    )

    variables = module.init(jax.random.PRNGKey(0), inputs)
    output = module.apply(variables, inputs)

    assert output["embedding"].shape == (2, 4)


def test_schnet_runs_with_its_default_activation():
    inputs = _two_atom_graph_inputs()
    module = SchNetEmbedding(
        _graphs_properties={"graph": {"cutoff": 5.0}},
        dim=4,
        nlayers=1,
        conv_hidden=[],
        species_encoding={"encoding": "one_hot", "zmax": 8},
        radial_basis={"basis": "gaussian", "dim": 4},
    )

    variables = module.init(jax.random.PRNGKey(1), inputs)
    output = module.apply(variables, inputs)

    assert output["embedding"].shape == (2, 4)


def test_oxidation_encoding_is_stored_at_the_atomic_number_row():
    atomic_number = 8
    states = sorted(set(sum(OXIDATION_STATES, [])))
    state_to_column = {state: i for i, state in enumerate(states)}
    expected = np.zeros((len(states),), dtype=np.float32)
    for state in OXIDATION_STATES[atomic_number]:
        expected[state_to_column[state]] = 1.0

    module = SpeciesEncoding(encoding="oxidation", zmax=10)
    species = jnp.array([atomic_number])
    variables = module.init(jax.random.PRNGKey(2), species)
    encoding = module.apply(variables, species)

    np.testing.assert_array_equal(encoding[0], expected)


def test_caiman_uses_destination_half_of_message_projection():
    inputs = _two_atom_graph_inputs()
    module = CaimanEmbedding(
        _graphs_properties={"graph": {"cutoff": 5.0}},
        dim=4,
        nchannels=2,
        nchannels_density=2,
        nlayers=1,
        lmax=0,
        latent_hidden=[],
        message_passing=True,
        species_encoding={"encoding": "one_hot", "zmax": 8},
        radial_basis={"basis": "gaussian", "dim": 4},
    )
    variables = module.init(jax.random.PRNGKey(3), inputs)

    def summed_embedding(params):
        return module.apply({"params": params}, inputs)["embedding"].sum()

    gradients = jax.grad(summed_embedding)(variables["params"])
    destination_gradient = gradients["message_linear_0"]["kernel"][:, 2:]

    assert jnp.linalg.norm(destination_gradient) > 0


def test_fennix_accepts_a_single_string_energy_term():
    model = FENNIX(
        cutoff=5.0,
        modules=OrderedDict(
            [
                (
                    "energy",
                    {
                        "module_name": "APPLY_FUNCTION",
                        "function_str": "x: x",
                        "argument_keys": "energy_input",
                    },
                )
            ]
        ),
        energy_terms="energy",
        rng_key=jax.random.PRNGKey(4),
        example_data={
            "species": np.array([1]),
            "coordinates": np.array([[0.0, 0.0, 0.0]]),
            "natoms": np.array([1]),
            "batch_index": np.array([0]),
            "energy_input": np.array([2.5]),
        },
    )

    energy, _ = model.total_energy(
        species=np.array([1]),
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        natoms=np.array([1]),
        batch_index=np.array([0]),
        energy_input=np.array([2.5]),
    )

    np.testing.assert_allclose(energy, [2.5])


def test_soft_functions_and_trainable_parameters_have_finite_derivatives():
    x = jnp.array([-np.sqrt(3.), 0., np.sqrt(3.)])
    np.testing.assert_allclose(softabs(x), [1., 0., 1.], atol=2e-7)
    np.testing.assert_allclose(softsat(x), [0., 0., (1. + np.tanh(1.)) / 2.], atol=2e-7)
    layer = TrainableSoftSat()
    values = jnp.array([[0.3, 1.2], [0.7, 2.0]])
    variables = layer.init(jax.random.PRNGKey(2), values)
    gradients = jax.grad(lambda p: layer.apply({"params": p}, values).sum())(variables["params"])
    for gradient in jax.tree_util.tree_leaves(gradients):
        assert np.isfinite(gradient).all()
        assert np.any(np.asarray(gradient) != 0.)


def test_radial_neural_basis_depends_on_fraction_of_cutoff():
    small = RadialBasis(end=2., basis="nn", dim=3, extra_params={"scale_by_cutoff": True})
    large = RadialBasis(end=4., basis="nn", dim=3, extra_params={"scale_by_cutoff": True})
    distances = jnp.array([0.2, 0.5, 1.0])
    variables = small.init(jax.random.PRNGKey(3), distances)
    # Keeping r/cutoff fixed should leave every basis channel unchanged.
    np.testing.assert_allclose(
        small.apply(variables, distances), large.apply(variables, distances * 2.), atol=1e-7,
    )


def test_layer_norm_modes_match_reference_and_expose_affine_parameters():
    from fennol.models.misc.misc import LayerNorm

    x = jnp.array([[1.0, 2.0, 4.0], [-2.0, 0.0, 3.0]])
    for kind in ("rmsnorm", "layernorm"):
        module = LayerNorm(kind=kind, eps=1e-5)
        variables = module.init(jax.random.PRNGKey(11), x)
        actual = module.apply(variables, x)
        centered = x - jnp.mean(x, axis=-1, keepdims=True) if kind == "layernorm" else x
        expected = centered / jnp.sqrt(
            1e-5 + jnp.mean(centered**2, axis=-1, keepdims=True)
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(variables["params"]["weight"], jnp.ones((3,)))
        if kind == "layernorm":
            np.testing.assert_allclose(variables["params"]["bias"], jnp.zeros((3,)))
        else:
            assert "bias" not in variables["params"]


def test_fennix_energy_force_and_virial_match_quadratic_reference():
    from collections import OrderedDict

    from fennol.models.fennix import FENNIX

    coordinates = np.array([[1.0, 2.0, 0.0], [-1.0, 0.5, 2.0]], dtype=np.float32)
    model = FENNIX(
        cutoff=5.0,
        modules=OrderedDict(
            [
                (
                    "energy",
                    {
                        "module_name": "APPLY_FUNCTION",
                        "function_str": "x: (x**2).sum(axis=-1)",
                        "argument_keys": "coordinates",
                    },
                )
            ]
        ),
        energy_terms="energy",
        rng_key=jax.random.PRNGKey(0),
        example_data={
            "species": np.array([1, 1]),
            "coordinates": coordinates,
            "natoms": np.array([2]),
            "batch_index": np.array([0, 0]),
        },
    )
    energy, forces, virial, _ = model.energy_and_forces_and_virial(
        species=np.array([1, 1]),
        coordinates=coordinates,
        natoms=np.array([2]),
        batch_index=np.array([0, 0]),
    )
    np.testing.assert_allclose(energy, [10.25], atol=1e-6)
    np.testing.assert_allclose(forces, -2.0 * coordinates, atol=1e-6)
    np.testing.assert_allclose(
        virial[0], 2.0 * coordinates.T @ coordinates, atol=1e-6
    )
