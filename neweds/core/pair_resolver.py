"""Логика разбора пар каналов для расчёта connectivity.

Что умеет: разбирать пользовательский текст с парами, строить пары соседей
по координатам, автоматически выбирать режим пар в зависимости от размера
матрицы и держать защиту от OOM.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def parse_pairs_text(text: str, columns: pd.Index, n_cols: int) -> list[tuple[int, int]]:
    """Parse user pairs like: a-b; a->b; 0-1; 0->1."""
    text = (text or "").strip()
    if not text:
        return []
    raw_items: list[str] = []
    for token in text.replace("\u2014", "-").replace(",", ";").split(";"):
        tt = token.strip()
        if tt:
            raw_items.append(tt)
    col_to_idx = {str(c): i for i, c in enumerate(columns)}
    pairs_out: list[tuple[int, int]] = []
    for it in raw_items:
        if "->" in it:
            a, b = [x.strip() for x in it.split("->", 1)]
        elif "-" in it:
            a, b = [x.strip() for x in it.split("-", 1)]
        else:
            continue

        def _to_idx(x: str) -> int | None:
            if x.isdigit():
                ix = int(x)
                return ix if 0 <= ix < n_cols else None
            return col_to_idx.get(x)

        ia, ib = _to_idx(a), _to_idx(b)
        if ia is None or ib is None or ia == ib:
            continue
        pairs_out.append((int(ia), int(ib)))
    return pairs_out


def build_neighbor_pairs(
    coords_df: pd.DataFrame | None,
    columns: pd.Index,
    kind: str = "26",
    radius: int = 1,
) -> list[tuple[int, int]]:
    """Spatial neighborhood pairs from coords_df (voxel_id/x/y/z)."""
    if coords_df is None or coords_df.empty:
        return []
    col_to_idx = {str(c): i for i, c in enumerate(columns)}
    coord_to_idx: dict[tuple[int, int, int], int] = {}
    for _, r in coords_df.iterrows():
        vid = str(r.get("voxel_id"))
        if vid not in col_to_idx:
            continue
        try:
            x, y, z = int(r.get("x")), int(r.get("y")), int(r.get("z"))
        except Exception:
            continue
        coord_to_idx[(x, y, z)] = int(col_to_idx[vid])

    if not coord_to_idx:
        return []
    kind = str(kind or "26")
    radius = int(max(1, radius))
    offsets: list[tuple[int, int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == dy == dz == 0:
                    continue
                if kind == "6":
                    if abs(dx) + abs(dy) + abs(dz) == 1:
                        offsets.append((dx, dy, dz))
                else:
                    if max(abs(dx), abs(dy), abs(dz)) <= radius:
                        offsets.append((dx, dy, dz))

    seen: set[tuple[int, int]] = set()
    pairs_out: list[tuple[int, int]] = []
    for (x, y, z), i in coord_to_idx.items():
        for dx, dy, dz in offsets:
            j = coord_to_idx.get((x + dx, y + dy, z + dz))
            if j is None or j == i:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs_out.append(key)
    return pairs_out


def resolve_pairs(
    n_cols: int,
    columns: pd.Index,
    coords_df: pd.DataFrame | None,
    pair_mode: str,
    auto_thr: int = 500,
    **kwargs,
) -> tuple[list[tuple[int, int]] | None, str, dict]:
    """Resolve which pairs to compute. Returns (pairs_idx, resolved_mode, meta)."""
    meta: dict = {}

    if pair_mode == "auto":
        has_coords = coords_df is not None and not getattr(coords_df, "empty", True)
        if n_cols >= auto_thr and has_coords:
            pair_mode = "neighbors"
        elif n_cols >= auto_thr:
            pair_mode = "random"
            logging.info("[auto pair_mode] N=%d >= %d, no coords -> random", n_cols, auto_thr)
        else:
            pair_mode = "full"

    pairs_idx: list[tuple[int, int]] | None = None
    if pair_mode == "pairs":
        pairs_idx = parse_pairs_text(str(kwargs.get("pairs_text") or ""), columns, n_cols)
    elif pair_mode == "neighbors":
        pairs_idx = build_neighbor_pairs(
            coords_df,
            columns,
            str(kwargs.get("neighbor_kind") or "26"),
            int(kwargs.get("neighbor_radius") or 1),
        )
        if not pairs_idx and n_cols >= auto_thr:
            logging.info("[pair_mode] neighbors empty, fallback -> random for N=%d", n_cols)
            pair_mode = "random"
    elif pair_mode == "random":
        pass  # handled below

    if pair_mode == "random":
        max_pairs = int(max(1, kwargs.get("max_pairs") or 50000))
        rng = np.random.default_rng(12345)
        m = min(max_pairs, max(1, n_cols * 5))
        pairs_set: set[tuple[int, int]] = set()
        while len(pairs_set) < m:
            i, j = int(rng.integers(0, n_cols)), int(rng.integers(0, n_cols))
            if i != j:
                pairs_set.add((min(i, j), max(i, j)))
        pairs_idx = list(pairs_set)

    # OOM guard
    if pairs_idx is None:
        mat_gb = (n_cols * n_cols * 8) / (1024**3)
        if mat_gb > 8.0:
            logging.error(
                "[memory] Matrix %dx%d=%.1fGB>8GB, forcing random.", n_cols, n_cols, mat_gb
            )
            rng = np.random.default_rng(12345)
            mp = min(n_cols * 5, 500_000)
            ps: set[tuple[int, int]] = set()
            bi = rng.integers(0, n_cols, size=mp * 3)
            bj = rng.integers(0, n_cols, size=mp * 3)
            for ii, jj in zip(bi, bj):
                if ii != jj:
                    ps.add((int(min(ii, jj)), int(max(ii, jj))))
                if len(ps) >= mp:
                    break
            pairs_idx = list(ps)
            pair_mode = "random (OOM guard)"

    if pairs_idx is not None:
        meta["pair_mode"] = str(pair_mode)
        meta["pairs_count"] = int(len(pairs_idx))
        meta["pairs_explain"] = "Матрица считается упрощённо: только по выбранным парам."

    return pairs_idx, pair_mode, meta
