"""Learned, generally nonconservative pair forces."""

from typing import Any, ClassVar, Optional

import flax.linen as nn
import jax.numpy as jnp

from ...utils.activations import activation_from_str
from ..misc.encodings import RadialBasis
from ._direct_forces import sum_pair_forces


class PairForces(nn.Module):
    """Predict forces directly from scalar embeddings and optional vector features.

    FID: PAIR_FORCES

    Forces use model energy units per Angstrom. The learned coefficients must
    be trained in those units; no energy gradient or energy output is implied.
    Each directed edge contributes a force on its source and the opposite
    force on its destination. The output virial is sum(r_ij outer f_i), using
    FeNNol's row-vector strain convention, in model energy units. It need not
    be symmetric when vector features produce noncentral forces.
    """

    _graphs_properties: dict
    radial_basis: dict[str, Any]
    embedding_key: str
    tensors_key: Optional[str] = None
    """Optional (atoms, channels, spherical components) tensor; 1:4 are vectors."""
    graph_key: str = "graph"
    intermediate_dim: int = 16
    activation: str = "identity"
    output_key: Optional[str] = None
    """Force key; defaults to the module name."""
    virial_key: Optional[str] = None
    """Virial key; defaults to <output_key>_virial_tensor."""

    FID: ClassVar[str] = "PAIR_FORCES"

    @nn.compact
    def __call__(self, inputs: dict) -> dict:
        if self.intermediate_dim < 1:
            raise ValueError("PairForces intermediate_dim must be positive")
        properties = self._graphs_properties[self.graph_key]
        if not properties.get("directed", True):
            raise ValueError("PairForces requires a directed graph")
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        src, dst = graph["edge_src"], graph["edge_dst"]
        radial_basis = jnp.asarray(
            RadialBasis(
                **{
                    "end": properties["cutoff"],
                    **self.radial_basis,
                    "name": "RadialBasis",
                }
            )(distances)
        )

        vector_channels = 0
        vectors = None
        if self.tensors_key is not None:
            tensors = inputs[self.tensors_key]
            if tensors.ndim != 3 or tensors.shape[-1] < 4:
                raise ValueError(
                    "PairForces tensors must have shape (atoms, channels, >=4)"
                )
            vectors = tensors[:, :, 1:4]
            vector_channels = vectors.shape[1]

        dim = radial_basis.shape[-1] + vector_channels
        embedding = inputs[self.embedding_key]
        # Keep projection parameter names/shapes compatible with the source branch.
        neurons = dim * self.intermediate_dim
        q = nn.Dense(neurons, use_bias=False)(embedding).reshape(
            -1, dim, self.intermediate_dim
        )
        k = nn.Dense(neurons, use_bias=False)(embedding).reshape(
            -1, dim, self.intermediate_dim
        )
        qi = q.at[src].get(mode="fill", fill_value=0.0)
        kj = k.at[dst].get(mode="fill", fill_value=0.0)
        coefficients = activation_from_str(self.activation)(
            (qi * kj).sum(axis=-1) / self.intermediate_dim**0.5
        )
        direction = graph["vec"] / distances[:, None]
        if vectors is not None:
            vi = vectors.at[src].get(mode="fill", fill_value=0.0)
            direction = direction + (
                vi * coefficients[:, :vector_channels, None]
            ).sum(axis=1)
            coefficients = coefficients[:, vector_channels:]
        pair_forces = (
            (coefficients * radial_basis).sum(axis=-1) * graph["switch"]
        )[:, None] * direction
        forces, virial = sum_pair_forces(inputs, graph, pair_forces)
        output_key = (
            self.output_key
            if self.output_key is not None
            else (self.name or "pair_forces")
        )
        virial_key = (
            self.virial_key
            if self.virial_key is not None
            else output_key + "_virial_tensor"
        )
        return {**inputs, output_key: forces, virial_key: virial}
