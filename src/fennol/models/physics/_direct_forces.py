"""Reduction of directed pair contributions using FeNNol's strain convention."""

import jax
import jax.numpy as jnp


def sum_pair_forces(
    inputs: dict,
    graph: dict,
    pair_forces: jax.Array,
    *,
    symmetric: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Reduce directed pair forces, exploiting symmetry where available.

    For asymmetric learned contributions, scatter to both endpoints; ghost
    reactions must be communicated back to their owners. For symmetric radial
    interactions, each row is a full source force: reverse edges supply the
    opposite force, so only the source reduction is needed. Their directed
    virial contributions carry a half factor to count each pair once.
    Only edges leaving owned atoms contribute on a partition in either case.
    The virial is sum(r_ij outer f_i), in force units times Angstrom. Its
    tensor orientation matches dE/dS for row-vector deformations x -> x @ S.
    """
    src, dst = graph["edge_src"], graph["edge_dst"]
    nat = inputs["species"].shape[0]
    valid = (src < nat) & (dst < nat) & (src >= 0) & (dst >= 0)
    if "true_atoms" in inputs:
        valid = valid & inputs["true_atoms"].at[src].get(
            mode="fill", fill_value=False
        )
    pair_forces = jnp.where(valid[:, None], pair_forces, 0.0)
    forces = jax.ops.segment_sum(pair_forces, src, nat)
    if not symmetric:
        forces = forces - jax.ops.segment_sum(pair_forces, dst, nat)
    batch = inputs["batch_index"].at[src].get(mode="fill", fill_value=0)
    virial = jax.ops.segment_sum(
        graph["vec"][:, :, None] * pair_forces[:, None, :],
        batch,
        inputs["natoms"].shape[0],
    )
    if symmetric:
        virial = 0.5 * virial
    return forces, virial


def repulsion_outputs(
    inputs,
    graph,
    radial_derivative,
    distances,
    switch,
    output_key,
    energy_factor,
):
    """Switch radial forces, retaining alchemical distance chain factors.

    This intentionally omits U(r) s'(r): direct outputs use force switching,
    while the reported energies retain their existing potential switching.
    """
    beta = 1.0
    if "alch_group" in inputs:
        if "beta_alch" not in graph:
            raise ValueError(
                "Alchemical direct repulsion forces require graph beta_alch"
            )
        src, dst = graph["edge_src"], graph["edge_dst"]
        same_group = inputs["alch_group"][src] == inputs["alch_group"][dst]
        beta = jnp.where(same_group, 1.0, graph["beta_alch"])
    # d(r_eff)/d(r_raw) = beta * r_raw / r_eff, including the
    # optional extra repulsion softcore. The switch only scales the force.
    coefficient = radial_derivative * switch * beta / distances
    # The energy derivative includes the directed half-pair factor. Restore
    # the full source force for the single reduction of symmetric repulsion.
    pair_forces = (2.0 * energy_factor * coefficient)[:, None] * graph["vec"]
    forces, virial = sum_pair_forces(inputs, graph, pair_forces, symmetric=True)
    return {output_key: forces, output_key + "_virial_tensor": virial}
