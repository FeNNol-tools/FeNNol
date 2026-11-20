"""Ideal gas in a cell held by a rotation-invariant spring on the cell metric."""

from flax import linen as nn
import jax.numpy as jnp


class CellSpring(nn.Module):
    """E(h) = k/2 sum_ij (G_ij - L0**2 delta_ij)**2 with G = h h^T.

    The energy depends on the cell only, so the atoms form an ideal gas and the
    exact NPT distribution of the cell is p(h) ~ V**N exp(-(PV + E(h))/kBT),
    up to the cell measure. The strain virial supplies the spring stress.
    """

    FID = "VALIDATION_CELL_SPRING"
    stiffness: float = 1.5e-3  # kcal/mol/A**4
    length: float = 10.0  # A

    @nn.compact
    def __call__(self, inputs):
        coordinates = inputs["coordinates"]
        cells = inputs["cells"]
        metric = jnp.einsum("bij,bkj->bik", cells, cells) - self.length**2 * jnp.eye(3)
        energy = 0.5 * self.stiffness * jnp.sum(metric**2, axis=(1, 2))
        natoms = inputs["natoms"][inputs["batch_index"]]
        per_atom = energy[inputs["batch_index"]] / natoms + 0.0 * coordinates[:, 0]
        return {**inputs, "cell_spring_energy": per_atom}
