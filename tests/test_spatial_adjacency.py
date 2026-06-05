from __future__ import annotations

import numpy as np

from neweds.analysis.spatial_adjacency import (
    adjacency_edges,
    neighbor_offsets,
    region_grow,
    valid_neighbors,
)


def test_neighbor_offsets_match_volume_connectivity_sizes() -> None:
    assert len(neighbor_offsets(6)) == 6
    assert len(neighbor_offsets(18)) == 18
    assert len(neighbor_offsets(26)) == 26


def test_valid_neighbors_are_constrained_by_mask() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    mask[1, 1, 2] = True
    mask[1, 2, 1] = True
    mask[2, 2, 2] = True

    neighbors = valid_neighbors((1, 1, 1), mask, connectivity=6)

    assert set(neighbors) == {(1, 1, 2), (1, 2, 1)}


def test_adjacency_edges_do_not_cross_mask_gaps() -> None:
    mask = np.zeros((1, 3, 1), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 2, 0] = True

    assert adjacency_edges(mask, connectivity=6) == []


def test_region_grow_stops_at_mask_boundary() -> None:
    mask = np.zeros((1, 5, 1), dtype=bool)
    mask[0, 0:2, 0] = True
    mask[0, 3:5, 0] = True

    grown = region_grow((0, 0, 0), mask, connectivity=6)

    assert int(np.sum(grown)) == 2
    assert bool(grown[0, 0, 0]) is True
    assert bool(grown[0, 1, 0]) is True
    assert bool(grown[0, 3, 0]) is False
