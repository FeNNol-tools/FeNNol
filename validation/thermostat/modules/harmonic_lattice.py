"""Independent anisotropic harmonic wells on a widely spaced lattice."""

import jax.numpy as jnp
from flax import linen as nn


class HarmonicLattice(nn.Module):
    """Atom i feels 0.5 * sum_a k_a (q_ia - c_ia)**2 around lattice site c_i.

    Sites are ``spacing`` apart on a ``per_side``**3 grid, so with a short
    model cutoff no two atoms interact: every atom is an independent sample.
    """

    FID = "VALIDATION_HARMONIC_LATTICE"

    stiffness: tuple = (1.0, 2.0, 4.0)
    spacing: float = 10.0
    per_side: int = 8

    @nn.compact
    def __call__(self, inputs):
        coordinates = inputs["coordinates"]
        index = jnp.arange(coordinates.shape[0])
        n = self.per_side
        sites = self.spacing * jnp.stack([index % n, (index // n) % n, index // (n * n)], axis=-1)
        k = jnp.asarray(self.stiffness, dtype=coordinates.dtype)
        energy = 0.5 * jnp.sum(k * (coordinates - sites) ** 2, axis=-1)
        return {**inputs, "harmonic_energy": energy}
