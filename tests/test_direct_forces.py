"""Numerical contracts for direct pair forces and strain-convention virials."""

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fennol.models import FENNIX


def pair_data():
    """Two unlike atoms with a non-axis-aligned separation."""
    return {
        "species": np.array([1, 8], dtype=np.int32),
        "coordinates": np.array(
            [[0.2, 0.3, 0.1], [1.1, 0.7, 0.4]], dtype=np.float32
        ),
        "natoms": np.array([2], dtype=np.int32),
        "batch_index": np.zeros(2, dtype=np.int32),
    }


def repulsion_model(
    kind,
    data,
    *,
    energy_unit="Ha",
    preprocessing=None,
    graph_config=None,
    **options,
):
    return FENNIX(
        cutoff=3.0,
        modules=OrderedDict(
            repulsion={
                "module_name": kind,
                "trainable": False,
                "direct_forces_key": "direct",
                **options,
            }
        ),
        energy_terms=["repulsion"],
        energy_unit=energy_unit,
        preprocessing=OrderedDict() if preprocessing is None else preprocessing,
        graph_config={} if graph_config is None else graph_config,
        example_data=data,
        rng_key=jax.random.PRNGKey(0),
    )


def force_switched_reference(
    kind,
    data,
    *,
    switch_type="cosine",
    switch_start=0.0,
    energy_unit="Ha",
    trainable=False,
    variables=None,
):
    """Unswitched energy derivatives times an independently evaluated envelope.

    Reference systems are isolated dimers, optionally in orthorhombic cells.
    Hard switching removes the envelope's derivative from the autodiff oracle
    while retaining every alchemical distance transformation in the energy.
    """
    reference = repulsion_model(
        kind,
        data,
        direct_forces_key=None,
        energy_unit=energy_unit,
        trainable=trainable,
        graph_config={"switch_params": {"switch_type": "hard"}},
    )
    _, forces, virial, _ = reference.energy_and_forces_and_virial(
        variables=variables,
        **data,
    )
    envelopes = []
    offset = 0
    for system, count in enumerate(data["natoms"]):
        assert count == 2
        delta = data["coordinates"][offset + 1] - data["coordinates"][offset]
        if "cells" in data:
            cells = data["cells"]
            cell = cells[system if len(cells) > 1 else 0]
            delta = delta - np.round(delta @ np.linalg.inv(cell)) @ cell
        distance = np.linalg.norm(delta)
        x = (distance - 3.0 * switch_start) / (3.0 * (1.0 - switch_start))
        if x <= 0.0:
            envelope = 1.0
        elif x >= 1.0:
            envelope = 0.0
        elif switch_type == "cosine":
            envelope = 0.5 * (1.0 + np.cos(np.pi * x))
        elif switch_type == "polynomial":
            envelope = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
        elif switch_type == "exponential":
            envelope = np.exp(-(x**2) / (1.0 - x**2))
        elif switch_type == "hard":
            envelope = 1.0
        else:
            raise ValueError(switch_type)
        envelopes.append(envelope)
        offset += count
    envelopes = np.asarray(envelopes)
    return (
        np.asarray(forces) * envelopes[data["batch_index"], None],
        np.asarray(virial) * envelopes[:, None, None],
    )


@pytest.mark.parametrize("alchemical", [False, True])
def test_graph_processing_does_not_produce_switch_derivatives(alchemical):
    data = pair_data()
    if alchemical:
        data.update(
            alch_group=np.array([0, 1], dtype=np.int32),
            alch_elambda=0.4,
            alch_vlambda=0.6,
        )
    model = repulsion_model(
        "REPULSION_NLH",
        data,
        direct_forces_key=None,
        preprocessing=OrderedDict(
            short={
                "module_name": "GRAPH_FILTER",
                "graph_key": "short",
                "parent_graph": "graph",
                "cutoff": 2.0,
            }
        ),
    )
    out = model(**data)
    for key in ["graph", "short"]:
        assert "switch_derivative" not in out[key]
        assert "switch_raw_derivative" not in out[key]


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
def test_repulsion_direct_outputs_use_force_switching(kind):
    data = pair_data()
    model = repulsion_model(kind, data)
    _, energy_forces, _, out = model.energy_and_forces_and_virial(**data)
    forces, virial = force_switched_reference(kind, data)
    np.testing.assert_allclose(out["direct"], forces, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(
        out["direct_virial_tensor"], virial, rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(out["direct"].sum(axis=0), 0.0, atol=1e-7)
    assert not np.allclose(out["direct"], energy_forces)


def test_zbl_force_switching_has_analytical_periodic_force_and_virial():
    from fennol.utils.atomic_units import au

    data = {
        **pair_data(),
        "species": np.array([1, 1], dtype=np.int32),
        "coordinates": np.array(
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32
        ),
        "cells": np.eye(3, dtype=np.float32)[None] * 8.0,
    }
    # These fixed coefficients give the unswitched pair energy exp(-2r)/r
    # in atomic electrostatic units. At r=1, -U'=3 exp(-2), and s=3/4.
    model = repulsion_model(
        "REPULSION_ZBL",
        data,
        d=1.0,
        p=0.0,
        alphas=(1.0, 1.0, 1.0, 1.0),
        cs=(1.0, 1.0, 1.0, 1.0),
    )
    out = model(**data)
    amplitude = 2.25 * np.exp(-2.0) * au.ANG
    np.testing.assert_allclose(
        out["direct"], [[amplitude, 0, 0], [-amplitude, 0, 0]], atol=1e-7
    )
    expected = np.zeros((1, 3, 3))
    expected[0, 0, 0] = -amplitude
    np.testing.assert_allclose(out["direct_virial_tensor"], expected, atol=1e-7)


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
@pytest.mark.parametrize(
    "softcore",
    [
        {},
        {"alch_softcore_e": 0.8},
        {"alch_ascale_rep": 2.0},
        {"alch_softcore_rep": 0.5},
    ],
)
def test_alchemical_repulsion_direct_outputs_match_force_switched_reference(
    kind, softcore
):
    data = {
        **pair_data(),
        "alch_group": np.array([0, 1], dtype=np.int32),
        "alch_elambda": 0.4,
        "alch_vlambda": 0.6,
        **softcore,
    }
    model = repulsion_model(kind, data)
    out = model(**data)
    forces, virial = force_switched_reference(kind, data)
    np.testing.assert_allclose(out["direct"], forces, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(
        out["direct_virial_tensor"], virial, rtol=2e-5, atol=1e-7
    )


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
@pytest.mark.parametrize(
    "switch_type", ["cosine", "polynomial", "exponential", "hard"]
)
def test_filtered_alchemical_repulsion_uses_raw_parent_distances(
    kind, switch_type
):
    data = {
        **pair_data(),
        "alch_group": np.array([0, 1], dtype=np.int32),
        "alch_elambda": 0.4,
        "alch_vlambda": 0.6,
    }
    switch_params = {"switch_type": switch_type}
    full = repulsion_model(
        kind, data, graph_config={"switch_params": switch_params}
    )
    filtered = repulsion_model(
        kind,
        data,
        graph_key="short",
        preprocessing=OrderedDict(
            short={
                "module_name": "GRAPH_FILTER",
                "graph_key": "short",
                "parent_graph": "graph",
                "cutoff": 3.0,
                "switch_params": switch_params,
            }
        ),
    )
    expected = full(**data)
    out = filtered(**data)
    forces, virial = force_switched_reference(
        kind, data, switch_type=switch_type
    )
    np.testing.assert_allclose(
        out["direct"], expected["direct"], rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(out["direct"], forces, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(
        out["direct_virial_tensor"], virial, rtol=2e-5, atol=1e-7
    )


def pair_force_model(data, **options):
    return FENNIX(
        cutoff=2.0,
        modules=OrderedDict(
            pair={
                "module_name": "PAIR_FORCES",
                "embedding_key": "embedding",
                "radial_basis": {"dim": 1},
                "intermediate_dim": 1,
                "output_key": "direct",
                **options,
            }
        ),
        example_data=data,
        rng_key=jax.random.PRNGKey(0),
    )


def test_pair_forces_match_analytical_periodic_pair_and_serialize(tmp_path):
    data = {
        **pair_data(),
        "coordinates": np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32
        ),
        "cells": np.eye(3, dtype=np.float32)[None] * 4.0,
        "embedding": np.array([[1.0], [2.0]], dtype=np.float32),
    }
    model = pair_force_model(data, virial_key="pair_virial")
    # A checkpoint with unit projection kernels gives Q_i K_j = 2.
    variables = jax.tree.map(jnp.ones_like, model.variables)
    out = model(variables=variables, **data)
    # r_01 = (-1,0,0), bessel(r)=2/pi, cosine switch=1/2.
    # Two directed edges each contribute (-2/pi,0,0) on atom 0.
    np.testing.assert_allclose(
        out["direct"], [[-4 / np.pi, 0, 0], [4 / np.pi, 0, 0]], atol=1e-6
    )
    expected_virial = np.zeros((1, 3, 3))
    expected_virial[0, 0, 0] = 4 / np.pi
    np.testing.assert_allclose(out["pair_virial"], expected_virial, atol=1e-6)
    path = tmp_path / "pair.fnx"
    model.save(path)
    restored = FENNIX.load(path)(**data)
    original = model(**data)
    np.testing.assert_allclose(
        restored["direct"], original["direct"], atol=1e-7
    )
    np.testing.assert_allclose(
        restored["pair_virial"], original["pair_virial"], atol=1e-7
    )


@pytest.mark.parametrize("vectors", [False, True])
def test_pair_forces_are_translation_invariant_and_rotation_covariant(vectors):
    data = {
        **pair_data(),
        "embedding": np.array([[1.0, 0.5], [0.2, 0.8]], dtype=np.float32),
    }
    options = {}
    if vectors:
        data["tensors"] = np.array(
            [[[0.0, 0.1, 0.8, -0.2]], [[0.0, -0.4, 0.1, 0.5]]], dtype=np.float32
        )
        options["tensors_key"] = "tensors"
    model = pair_force_model(data, **options)
    original = model(**data)
    # A proper rotation that exercises off-diagonal tensor components.
    rotation = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    transformed = {**data, "coordinates": data["coordinates"] @ rotation + 3.0}
    if vectors:
        transformed["tensors"] = data["tensors"].copy()
        transformed["tensors"][:, :, 1:4] = (
            data["tensors"][:, :, 1:4] @ rotation
        )
    out = model(**transformed)
    np.testing.assert_allclose(
        out["direct"], original["direct"] @ rotation, rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(out["direct"].sum(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        out["direct_virial_tensor"],
        rotation.T @ original["direct_virial_tensor"] @ rotation,
        rtol=2e-5,
        atol=1e-7,
    )
    separation = data["coordinates"][1] - data["coordinates"][0]
    np.testing.assert_allclose(
        original["direct_virial_tensor"][0],
        np.outer(separation, original["direct"][0]),
        rtol=2e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "kind", ["PAIR_FORCES", "REPULSION_NLH", "REPULSION_ZBL"]
)
def test_owned_partitions_reproduce_full_forces_and_virials(kind):
    data = {
        **pair_data(),
        "embedding": np.array([[1.0], [0.5]], dtype=np.float32),
    }
    model = (
        pair_force_model(data)
        if kind == "PAIR_FORCES"
        else repulsion_model(kind, data)
    )
    full = model(**data)
    first = model(**{**data, "true_atoms": np.array([True, False])})
    # The second partition stores atom 1 first and atom 0 as a ghost.
    second_data = {
        **data,
        "coordinates": data["coordinates"][::-1].copy(),
        "species": data["species"][::-1].copy(),
        "embedding": data["embedding"][::-1].copy(),
        "true_atoms": np.array([True, False]),
    }
    second = model(**second_data)
    if kind != "PAIR_FORCES":
        # Symmetric repulsion computes each owner's complete force locally;
        # it must not depend on reverse communication of ghost contributions.
        np.testing.assert_allclose(
            first["direct"][0], full["direct"][0], rtol=2e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            second["direct"][0], full["direct"][1], rtol=2e-5, atol=1e-7
        )
        np.testing.assert_allclose(first["direct"][1], 0.0, atol=1e-7)
        np.testing.assert_allclose(second["direct"][1], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        first["direct"] + second["direct"][::-1],
        full["direct"],
        rtol=2e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        first["direct_virial_tensor"] + second["direct_virial_tensor"],
        full["direct_virial_tensor"],
        rtol=2e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
@pytest.mark.parametrize(
    "switch_type", ["cosine", "polynomial", "exponential", "hard"]
)
def test_repulsion_batched_periodic_units_and_switches(kind, switch_type):
    from fennol.utils.atomic_units import au

    data = {
        "coordinates": np.array(
            [[0, 0, 0], [7, 0.4, 0], [0, 0, 0], [0.5, 0.8, 0.2]],
            dtype=np.float32,
        ),
        "species": np.array([1, 8, 6, 1], dtype=np.int32),
        "natoms": np.array([2, 2], dtype=np.int32),
        "batch_index": np.array([0, 0, 1, 1], dtype=np.int32),
        "cells": np.tile(np.eye(3, dtype=np.float32)[None] * 8.0, (2, 1, 1)),
    }
    graph_config = {
        "switch_params": {"switch_type": switch_type, "switch_start": 0.1}
    }
    model = repulsion_model(
        kind, data, energy_unit="kcal/mol", graph_config=graph_config
    )
    out = model(**data)
    forces, virial = force_switched_reference(
        kind,
        data,
        energy_unit="kcal/mol",
        switch_type=switch_type,
        switch_start=0.1,
    )
    assert out["direct"].shape == (4, 3)
    assert out["direct_virial_tensor"].shape == (2, 3, 3)
    np.testing.assert_allclose(out["direct"], forces, rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(
        out["direct_virial_tensor"], virial, rtol=3e-5, atol=1e-5
    )
    hartree = repulsion_model(kind, data, graph_config=graph_config)(**data)
    factor = au.get_multiplier("kcal/mol")
    np.testing.assert_allclose(
        out["direct"], hartree["direct"] * factor, rtol=3e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out["direct_virial_tensor"],
        hartree["direct_virial_tensor"] * factor,
        rtol=3e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
def test_trainable_repulsion_and_switch_remain_differentiable(kind):
    data = pair_data()
    model = repulsion_model(
        kind,
        data,
        trainable=True,
        graph_config={
            "switch_params": {"switch_type": "cosine", "trainable": True}
        },
    )
    out = model(**data)
    forces, virial = force_switched_reference(
        kind,
        data,
        trainable=True,
        variables=model.variables,
    )
    np.testing.assert_allclose(out["direct"], forces, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(
        out["direct_virial_tensor"], virial, rtol=3e-5, atol=1e-7
    )
    inputs = model.preprocess(**data)
    gradients = jax.grad(
        lambda variables: jnp.sum(
            model._apply(variables, inputs)["direct"] ** 2
        )
    )(model.variables)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))


@pytest.mark.parametrize(
    "kind", ["PAIR_FORCES", "REPULSION_NLH", "REPULSION_ZBL"]
)
def test_direct_force_model_runs_in_existing_md_interface(kind, tmp_path):
    from fennol.md.integrate import initialize_dynamics
    from fennol.md.utils import us

    data = pair_data()
    if kind == "PAIR_FORCES":
        model = FENNIX(
            cutoff=3.0,
            modules=OrderedDict(
                embedding={"module_name": "SPECIES_ENCODING", "dim": 2},
                direct={
                    "module_name": "PAIR_FORCES",
                    "embedding_key": "embedding",
                    "radial_basis": {"dim": 2},
                    "intermediate_dim": 2,
                },
            ),
            example_data=data,
            rng_key=jax.random.PRNGKey(0),
        )
    else:
        model = repulsion_model(kind, data)
    model_file = tmp_path / "model.fnx"
    model.save(model_file)
    xyz = tmp_path / "pair.xyz"
    xyz.write_text("2\n\nH 0.2 0.3 0.1\nO 1.1 0.7 0.4\n")
    velocities = tmp_path / "velocities.dat"
    np.savetxt(velocities, np.zeros((2, 3)))
    parameters = {
        "model_file": str(model_file),
        "xyz_input/file": str(xyz),
        "system_name": str(tmp_path / "pair"),
        "thermostat": "NVE",
        "dt": 0.001,
        "initial_velocities": str(velocities),
        "forces_key": "direct",
    }
    step, _, _, state, conf, system = initialize_dynamics(
        parameters, "float32", jax.random.PRNGKey(1)
    )
    expected = model(**data)["direct"] / us.get_multiplier(model.energy_unit)
    np.testing.assert_allclose(system["forces"], expected, rtol=3e-5, atol=1e-6)
    for istep in range(3):
        state, system, conf, out = step(istep, state, system, conf)
        assert np.isfinite(system["coordinates"]).all()
        assert np.isfinite(out["direct_virial_tensor"]).all()
        np.testing.assert_allclose(system["forces"].sum(axis=0), 0.0, atol=1e-5)
    assert not np.array_equal(system["coordinates"], data["coordinates"])
    with pytest.raises(
        ValueError, match="Pressure not available with custom forces_key"
    ):
        initialize_dynamics(
            {**parameters, "cell": [8.0], "estimate_pressure": True},
            "float32",
            jax.random.PRNGKey(1),
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"intermediate_dim": 0}, "intermediate_dim must be positive"),
        ({"tensors_key": "embedding"}, "tensors must have shape"),
    ],
)
def test_pair_forces_reject_invalid_configurations(options, message):
    data = {**pair_data(), "embedding": np.ones((2, 1), dtype=np.float32)}
    with pytest.raises(ValueError, match=message):
        pair_force_model(data, **options)


def test_pair_forces_batching_and_isolated_system_have_independent_outputs():
    data = {
        "species": np.array([1, 8, 1, 8, 1], dtype=np.int32),
        "coordinates": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
            dtype=np.float32,
        ),
        "embedding": np.array([[1], [2], [1], [2], [1]], dtype=np.float32),
        "natoms": np.array([2, 2, 1], dtype=np.int32),
        "batch_index": np.array([0, 0, 1, 1, 2], dtype=np.int32),
    }
    model = pair_force_model(data)
    out = model(variables=jax.tree.map(jnp.ones_like, model.variables), **data)
    amplitude = 4 / np.pi
    np.testing.assert_allclose(
        out["direct"],
        [
            [amplitude, 0, 0],
            [-amplitude, 0, 0],
            [0, amplitude, 0],
            [0, -amplitude, 0],
            [0, 0, 0],
        ],
        atol=1e-6,
    )
    expected_virial = np.zeros((3, 3, 3))
    expected_virial[0, 0, 0] = amplitude
    expected_virial[1, 1, 1] = amplitude
    np.testing.assert_allclose(
        out["direct_virial_tensor"], expected_virial, atol=1e-6
    )


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
@pytest.mark.parametrize("lambda_v", [0.0, 1.0])
def test_alchemical_endpoints_preserve_same_group_repulsion(kind, lambda_v):
    data = {
        **pair_data(),
        "alch_group": np.array([1, 1], dtype=np.int32),
        "alch_elambda": lambda_v,
        "alch_vlambda": lambda_v,
        "alch_softcore_rep": 0.5,
    }
    out = repulsion_model(kind, data)(**data)
    normal_data = pair_data()
    expected = repulsion_model(kind, normal_data)(**normal_data)
    np.testing.assert_allclose(
        out["direct"], expected["direct"], rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        out["direct_virial_tensor"],
        expected["direct_virial_tensor"],
        rtol=2e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("kind", ["REPULSION_NLH", "REPULSION_ZBL"])
def test_repulsion_energy_only_configuration_and_checkpoint_are_preserved(
    kind, tmp_path
):
    data = pair_data()
    legacy = repulsion_model(kind, data, direct_forces_key=None)
    direct = repulsion_model(kind, data)
    original = legacy(**data)
    assert "direct" not in original
    np.testing.assert_allclose(
        original["repulsion"], direct(**data)["repulsion"], atol=1e-7
    )
    path = tmp_path / "repulsion.fnx"
    legacy.save(path)
    restored = FENNIX.load(path)
    _, expected_forces, expected_virial, _ = (
        legacy.energy_and_forces_and_virial(**data)
    )
    _, forces, virial, _ = restored.energy_and_forces_and_virial(**data)
    np.testing.assert_allclose(forces, expected_forces, atol=1e-7)
    np.testing.assert_allclose(virial, expected_virial, atol=1e-7)
