"""Graph construction in Preprocessing layers."""

import jax
import jax.numpy as jnp
import numpy as np


def test_mixed_pbc_numpy_graph_matches_jax_processor_vectors():
    from fennol.models.preprocessing import GraphGenerator, GraphProcessor

    cell = np.eye(3, dtype=np.float32) * 10.0
    inputs = {
        "coordinates": np.array(
            [[0.1, 0.0, 0.0], [9.8, 0.0, 0.0], [20.0, 0.0, 0.0], [20.5, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "species": np.ones((4,), dtype=np.int32),
        "batch_index": np.array([0, 0, 1, 1], dtype=np.int32),
        "natoms": np.array([2, 2], dtype=np.int32),
        "cells": np.stack([cell, cell]),
        "reciprocal_cells": np.stack([np.linalg.inv(cell), np.linalg.inv(cell)]),
        "cells_mask": np.array([True, False]),
        "flags": {"minimum_image": None},
    }
    generator = GraphGenerator(cutoff=1.0)
    state, neighbor_inputs = generator(generator.init(), inputs)
    processor = GraphProcessor(cutoff=1.0)
    device_inputs = jax.tree_util.tree_map(jnp.asarray, neighbor_inputs)
    processed = processor.apply({}, device_inputs)
    graph = processed["graph"]
    valid = np.asarray(graph["edge_mask"])
    vectors = np.asarray(graph["vec"])[valid]
    sources = np.asarray(graph["edge_src"])[valid]
    destinations = np.asarray(graph["edge_dst"])[valid]

    np.testing.assert_allclose(
        np.sort(np.linalg.norm(vectors, axis=1)), [0.3, 0.3, 0.5, 0.5], atol=1e-5
    )
    numpy_lengths = np.sqrt(np.asarray(neighbor_inputs["graph"]["d12"]))
    numpy_valid = np.asarray(neighbor_inputs["graph"]["edge_src"]) < 4
    np.testing.assert_allclose(
        np.sort(numpy_lengths[numpy_valid]), np.sort(np.linalg.norm(vectors, axis=1)), atol=1e-5
    )
    # The periodic pair crosses the x boundary; the masked nonperiodic pair does not.
    assert any(np.isclose(vectors[:, 0], -0.3))
    assert any(np.isclose(vectors[:, 0], 0.5))
