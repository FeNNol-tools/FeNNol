"""Alchemical harmonic wells with independent electrostatic and vdW couplings."""

import jax.numpy as jnp
from flax import linen as nn


class LambdaHarmonic(nn.Module):
    """Ligand atoms feel 0.5*k(lambda)*r**2 with k = k0 + ke*elambda + kv*vlambda.

    The first ``n_ligand`` atoms form the alchemical group; the remaining atoms
    are confined by a lambda-independent harmonic well.
    """

    FID = "SAMPLING_LAMBDA_HARMONIC"

    k0: float = 1.0
    ke: float = 0.0
    kv: float = 0.0
    n_ligand: int = 1
    environment_k: float = 1.0

    @nn.compact
    def __call__(self, inputs):
        coordinates = inputs["coordinates"]
        stiffness = (
            self.k0
            + self.ke * inputs["alch_elambda"]
            + self.kv * inputs["alch_vlambda"]
        )
        ligand = jnp.arange(coordinates.shape[0]) < self.n_ligand
        k = jnp.where(ligand, stiffness, self.environment_k)
        energy = 0.5 * k * jnp.sum(coordinates**2, axis=-1)
        return {**inputs, "lambda_harmonic_energy": energy}
