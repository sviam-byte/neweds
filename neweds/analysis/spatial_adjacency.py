"""Volume-space voxel adjacency constrained by a mask.

This module intentionally implements only Euclidean grid adjacency in a 3D
volume. It is useful for conservative mask-constrained region growing, but it
must not be interpreted as adjacency along the folded cortical sheet.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np


def neighbor_offsets(connectivity: int = 6) -> list[tuple[int, int, int]]:
    """Return 3D neighbor offsets for 6-, 18-, or 26-neighborhoods."""
    conn = int(connectivity)
    if conn not in {6, 18, 26}:
        raise ValueError("connectivity must be one of 6, 18, or 26")

    offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if (
                    (conn == 6 and manhattan == 1)
                    or (conn == 18 and manhattan <= 2)
                    or conn == 26
                ):
                    offsets.append((dx, dy, dz))
    return offsets


def _as_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError("mask must be a 3D volume")
    return np.isfinite(arr) & arr.astype(bool)


def _inside(coord: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    x, y, z = coord
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]


def valid_neighbors(
    coord: tuple[int, int, int],
    mask: np.ndarray,
    *,
    connectivity: int = 6,
) -> list[tuple[int, int, int]]:
    """Return valid volume neighbors of ``coord`` that remain inside ``mask``."""
    valid = _as_mask(mask)
    seed = tuple(int(v) for v in coord)
    if not _inside(seed, valid.shape) or not bool(valid[seed]):
        return []
    out: list[tuple[int, int, int]] = []
    for dx, dy, dz in neighbor_offsets(connectivity):
        nxt = (seed[0] + dx, seed[1] + dy, seed[2] + dz)
        if _inside(nxt, valid.shape) and bool(valid[nxt]):
            out.append(nxt)
    return out


def adjacency_edges(
    mask: np.ndarray,
    *,
    connectivity: int = 6,
    as_coords: bool = False,
) -> list[tuple[int | tuple[int, int, int], int | tuple[int, int, int]]]:
    """Return undirected adjacency edges between valid voxels in a 3D mask."""
    valid = _as_mask(mask)
    offsets = neighbor_offsets(connectivity)
    shape = valid.shape
    edges: list[tuple[int | tuple[int, int, int], int | tuple[int, int, int]]] = []
    for coord_arr in np.argwhere(valid):
        coord = tuple(int(v) for v in coord_arr)
        src_flat = int(np.ravel_multi_index(coord, shape))
        for dx, dy, dz in offsets:
            nxt = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
            if not _inside(nxt, shape) or not bool(valid[nxt]):
                continue
            dst_flat = int(np.ravel_multi_index(nxt, shape))
            if dst_flat <= src_flat:
                continue
            edges.append((coord, nxt) if as_coords else (src_flat, dst_flat))
    return edges


def region_grow(
    seed: tuple[int, int, int],
    mask: np.ndarray,
    *,
    connectivity: int = 6,
    max_voxels: int | None = None,
    allowed: Iterable[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Grow a connected component from ``seed`` while staying inside ``mask``.

    ``allowed`` can further restrict the region, for example to an atlas parcel
    or candidate bin. The returned array is a boolean mask with the same shape as
    ``mask``.
    """
    valid = _as_mask(mask)
    seed = tuple(int(v) for v in seed)
    out = np.zeros(valid.shape, dtype=bool)
    if not _inside(seed, valid.shape) or not bool(valid[seed]):
        return out

    allowed_set = None if allowed is None else {tuple(int(v) for v in c) for c in allowed}
    if allowed_set is not None and seed not in allowed_set:
        return out

    limit = int(max_voxels) if max_voxels is not None and int(max_voxels) > 0 else None
    queue: deque[tuple[int, int, int]] = deque([seed])
    seen = {seed}
    offsets = neighbor_offsets(connectivity)

    while queue:
        coord = queue.popleft()
        out[coord] = True
        if limit is not None and int(np.sum(out)) >= limit:
            break
        for dx, dy, dz in offsets:
            nxt = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
            if nxt in seen or not _inside(nxt, valid.shape) or not bool(valid[nxt]):
                continue
            if allowed_set is not None and nxt not in allowed_set:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return out


__all__ = ["adjacency_edges", "neighbor_offsets", "region_grow", "valid_neighbors"]
