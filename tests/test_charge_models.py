"""Charge models: charge equilibration and charge hypothesis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fennol.models.physics.electrostatics import QeqD4


def test_qeq_noninteracting_initial_guess_obeys_total_charge(monkeypatch):
    inputs = {
        "species": jnp.array([1, 8]),
        "natoms": jnp.array([2]),
        "batch_index": jnp.array([0, 0]),
        "total_charge": jnp.array([1.0]),
        "graph": {
            "edge_src": jnp.array([], dtype=jnp.int32),
            "edge_dst": jnp.array([], dtype=jnp.int32),
            "distances": jnp.array([], dtype=jnp.float32),
            "switch": jnp.array([], dtype=jnp.float32),
        },
    }
    module = QeqD4(non_interacting_guess=True)

    # Hold the iterative solver at its initial guess so the test exercises the
    # charge-constrained guess itself, independently of solver convergence.
    monkeypatch.setattr(
        "jax.scipy.sparse.linalg.gmres",
        lambda _matvec, _rhs, x0=None, **_kwargs: (x0, jnp.array(0)),
    )
    variables = module.init(jax.random.PRNGKey(4), inputs)
    output = module.apply(variables, inputs)

    assert float(jnp.sum(output["charges"])) == pytest.approx(1.0, abs=1e-6)


def test_charge_hypothesis_conserves_global_and_alchemical_charge():
    from fennol.models.embeddings.charge_embeddings import ChargeHypothesis

    inputs = {
        "embedding": jnp.array([[0.2, 1.0], [0.3, -0.2], [1.0, 0.1]]),
        "coordinates": jnp.zeros((3, 3)),
        "batch_index": jnp.zeros((3,), dtype=jnp.int32),
        "natoms": jnp.array([3]),
        "total_charge": jnp.array([1.0]),
        "alch_group": jnp.array([0, 1, 1]),
        "alch_ligand_charge": jnp.array([-0.2]),
        "alch_elambda": jnp.array([0.0]),
        "weight_scale": jnp.array([1.0, 0.5, 2.0]),
        "q_reference": jnp.array([0.1, 0.1, -0.2]),
    }
    module = ChargeHypothesis(
        embedding_key="embedding",
        output_key="charges_out",
        ncharges=2,
        ref_charges_key="q_reference",
        wscale_key="weight_scale",
    )
    variables = module.init(jax.random.PRNGKey(1), inputs)

    independent = module.apply(variables, inputs)["charges_out"]
    np.testing.assert_allclose(independent.sum(axis=0), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(independent[1:].sum(axis=0), [-0.2, -0.2], atol=1e-6)

    coupled = module.apply(
        variables, {**inputs, "alch_elambda": jnp.array([1.0])}
    )["charges_out"]
    np.testing.assert_allclose(coupled.sum(axis=0), [1.0, 1.0], atol=1e-6)


def test_qeq_solution_conserves_charge_and_is_stationary_on_constraint_plane():
    from fennol.models.physics.electrostatics import QeqD4

    inputs = {
        "species": jnp.array([1, 8]),
        "coordinates": jnp.zeros((2, 3)),
        "natoms": jnp.array([2]),
        "batch_index": jnp.array([0, 0]),
        "total_charge": jnp.array([1.0]),
        "graph": {
            "edge_src": jnp.array([], dtype=jnp.int32),
            "edge_dst": jnp.array([], dtype=jnp.int32),
            "distances": jnp.array([], dtype=jnp.float32),
            "switch": jnp.array([], dtype=jnp.float32),
        },
    }
    module = QeqD4(non_interacting_guess=True, energy_key="qeq_energy")
    variables = module.init(jax.random.PRNGKey(4), inputs)
    output = module.apply(variables, inputs)
    charges = output["charges"]
    np.testing.assert_allclose(charges.sum(), 1.0, atol=1e-5)

    def energy_for_charges(candidate):
        return module.apply(
            variables, {**inputs, "charges": candidate}
        )["qeq_energy"].sum()

    charge_gradient = jax.grad(energy_for_charges)(charges)
    # At a constrained minimum, all atomic energy derivatives share the same
    # Lagrange multiplier; moving charge between atoms has zero first derivative.
    assert float(jnp.ptp(charge_gradient)) < 1e-5
