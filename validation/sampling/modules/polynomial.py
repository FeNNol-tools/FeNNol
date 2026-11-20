"""Separable polynomial potential for one particle, used by every CV method."""

import jax.numpy as jnp
from flax import linen as nn


class SeparablePolynomial(nn.Module):
    """U = kBT * sum_axis sum_n c[axis][n-1] * q_axis**n, with n starting at 1.

    Coefficients are given per Cartesian axis in kBT/Angstrom**n.
    """

    FID = "SAMPLING_SEPARABLE_POLYNOMIAL"

    kbt: float
    x: tuple = (0.0, 0.5)
    y: tuple = (0.0, 0.5)
    z: tuple = (0.0, 0.5)

    @nn.compact
    def __call__(self, inputs):
        coordinates = inputs["coordinates"]
        energy = jnp.zeros(coordinates.shape[:-1], dtype=coordinates.dtype)
        for axis, coefficients in enumerate((self.x, self.y, self.z)):
            q = coordinates[..., axis]
            for power, coefficient in enumerate(coefficients, start=1):
                energy = energy + coefficient * q**power
        return {**inputs, "polynomial_energy": self.kbt * energy}
