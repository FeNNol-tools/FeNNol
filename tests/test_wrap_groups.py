"""Periodic wrapping with wrap groups (build_wrapbox and _merge_wrap_groups)."""

import numpy as np

from fennol.md.utils import _merge_wrap_groups, build_wrapbox

CELL = np.eye(3) * 10.0


def _wrap(wrap_groups_def, coordinates):
    # Jitted, as in fennol_md.
    wrapbox = build_wrapbox({"nat": len(coordinates)}, wrap_groups_def)
    return np.asarray(wrapbox(np.asarray(coordinates), CELL, np.linalg.inv(CELL)))


def test_overlapping_groups_merge_transitively_and_disjoint_groups_stay_apart():
    merged = _merge_wrap_groups(
        {"a": [0, 1], "b": [1, 2], "c": [2, 3], "d": [5, 6]}, verbose=False
    )
    assert merged == {"a": [0, 1, 2, 3], "d": [5, 6]}


def test_group_moves_as_a_whole_and_other_atoms_wrap_individually():
    # Atoms 1-2 form a group straddling the x boundary, centred outside the box:
    # the whole group shifts by one cell, so it is not split across the boundary.
    coordinates = [[9.8, 1.0, 1.0], [10.4, 1.0, 1.0], [12.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]
    wrapped = _wrap({"mol": [1, 2]}, coordinates)
    np.testing.assert_allclose(wrapped[:, 0], [-0.2, 0.4, 2.0, 9.0], atol=1e-5)


def test_overlapping_groups_are_wrapped_as_one_group():
    coordinates = [[9.6, 1.0, 1.0], [10.2, 1.0, 1.0], [10.8, 1.0, 1.0], [5.0, 1.0, 1.0]]
    merged = _wrap({"a": [1, 2], "b": [2, 3]}, coordinates)
    # The merged group (atoms 1-3) is centred at x=10.2 and shifts by one cell.
    np.testing.assert_allclose(merged[:, 0], [-0.4, 0.2, 0.8, 5.0], atol=1e-5)
