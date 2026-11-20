"""Analytical Lennard-Jones interaction for scientific validation."""

from typing import ClassVar, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...utils.atomic_units import au


def force_shifted_lj(distance, epsilon, sigma, cutoff):
    """Return force-shifted LJ energy and radial derivative.

    The unshifted 12-6 potential and its derivative are both shifted to zero at
    ``cutoff``. Values at and beyond the cutoff are exactly zero.
    """
    distance = jnp.asarray(distance)
    epsilon = jnp.asarray(epsilon, dtype=distance.dtype)
    sigma = jnp.asarray(sigma, dtype=distance.dtype)
    cutoff = jnp.asarray(cutoff, dtype=distance.dtype)

    # Avoid singular intermediate values on masked/padded zero-length edges.
    safe_distance = jnp.maximum(distance, jnp.finfo(distance.dtype).tiny ** 0.25)
    ratio6 = (sigma / safe_distance) ** 6
    pair_energy = 4.0 * epsilon * (ratio6**2 - ratio6)
    pair_derivative = 24.0 * epsilon / safe_distance * (ratio6 - 2.0 * ratio6**2)

    ratio6_cutoff = (sigma / cutoff) ** 6
    energy_cutoff = 4.0 * epsilon * (ratio6_cutoff**2 - ratio6_cutoff)
    derivative_cutoff = (
        24.0 * epsilon / cutoff * (ratio6_cutoff - 2.0 * ratio6_cutoff**2)
    )
    shifted_energy = pair_energy - energy_cutoff - (safe_distance - cutoff) * derivative_cutoff
    shifted_derivative = pair_derivative - derivative_cutoff
    inside = distance < cutoff
    return (
        jnp.where(inside, shifted_energy, 0.0),
        jnp.where(inside, shifted_derivative, 0.0),
    )


class LennardJonesForceShifted(nn.Module):
    """Pairwise force-shifted LJ energy over a FENNIX neighbor graph.

    ``cutoff`` must match the graph cutoff. No tail correction or additional
    switching function is applied. Directed edges are counted with a factor of
    one half, including nonzero self-image edges.

    FID: LENNARD_JONES_FS
    """

    _graphs_properties: Dict
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: Optional[float] = None
    graph_key: str = "graph"
    energy_key: Optional[str] = None
    _energy_unit: str = "Ha"

    FID: ClassVar[str] = "LENNARD_JONES_FS"

    @nn.compact
    def __call__(self, inputs):
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        cutoff = self.cutoff if self.cutoff is not None else graph["cutoff"]
        energy, _ = force_shifted_lj(
            distances, self.epsilon, self.sigma, cutoff
        )
        energy = energy * graph["edge_mask"]
        atomic_energy = 0.5 * jax.ops.segment_sum(
            energy, graph["edge_src"], num_segments=inputs["species"].shape[0]
        )
        output_key = self.energy_key if self.energy_key is not None else self.name
        return {
            **inputs,
            output_key: atomic_energy * au.get_multiplier(self._energy_unit),
        }
