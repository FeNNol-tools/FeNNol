"""Zero potential energy: an ideal gas with exactly known NPT statistics."""

from flax import linen as nn


class ZeroEnergy(nn.Module):
    """Non-interacting atoms; the pressure is purely kinetic."""

    FID = "VALIDATION_ZERO_ENERGY"

    @nn.compact
    def __call__(self, inputs):
        coordinates = inputs["coordinates"]
        return {**inputs, "zero_energy": 0.0 * coordinates[:, 0]}
