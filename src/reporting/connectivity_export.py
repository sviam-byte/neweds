#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Утилиты экспорта матриц связности в машиночитаемом формате."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class ExportPolicy:
    """Политика экспорта плотных/разреженных представлений матриц."""

    dense_n_limit: int = 2000
    topk_per_node: int = 50
    min_abs_weight: float = 0.0
    dense_csv: bool = True
    include_names_in_edges: bool = True


def _safe_variant(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))


def save_nodes_csv(data_dir: str, names: list[str], coords: dict[str, tuple[Any, Any, Any]] | None = None) -> str:
    """Сохраняет список узлов и опциональные xyz-координаты."""
    rows = []
    for i, nm in enumerate(names):
        row = {"id": i, "name": nm}
        if coords and nm in coords:
            x, y, z = coords[nm]
            row.update({"x": x, "y": y, "z": z})
        rows.append(row)
    path = os.path.join(data_dir, "nodes.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_connectivity_matrix(
    data_dir: str,
    name_prefix: str,
    variant: str,
    matrix: np.ndarray,
    names: list[str],
    policy: ExportPolicy,
    extra_meta: dict | None = None,
) -> dict:
    """Экспортирует одну матрицу: edge-list, sparse и (опционально) dense.

    Реализация избегает построения огромного Python-списка словарей для больших матриц.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    n = int(arr.shape[0])
    if len(names) != n:
        names = [f"v{i:04d}" for i in range(n)]
    if n == 0:
        raise ValueError("matrix must be non-empty")
    var = _safe_variant(variant)

    work = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(work, 0.0)

    topk = int(getattr(policy, "topk_per_node", 0) or 0)
    min_abs = float(getattr(policy, "min_abs_weight", 0.0) or 0.0)

    if 0 < topk < n:
        idx = np.argpartition(np.abs(work), -topk, axis=1)[:, -topk:]
        src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
        dst = idx.reshape(-1).astype(np.int64, copy=False)
        weights = work[src, dst]
    else:
        src, dst = np.nonzero(work)
        weights = work[src, dst]

    mask = src != dst
    if min_abs > 0.0:
        mask &= np.abs(weights) >= min_abs
    src = src[mask]
    dst = dst[mask]
    weights = weights[mask]

    if src.size:
        edges_df = pd.DataFrame({
            "src": src.astype(np.int64, copy=False),
            "dst": dst.astype(np.int64, copy=False),
            "weight": weights.astype(float, copy=False),
        })
        if bool(getattr(policy, "include_names_in_edges", True)):
            names_arr = np.asarray(names, dtype=object)
            edges_df["src_name"] = names_arr[src]
            edges_df["dst_name"] = names_arr[dst]
            edges_df = edges_df[["src", "dst", "src_name", "dst_name", "weight"]]
    else:
        cols = ["src", "dst", "weight"]
        if bool(getattr(policy, "include_names_in_edges", True)):
            cols = ["src", "dst", "src_name", "dst_name", "weight"]
        edges_df = pd.DataFrame(columns=cols)

    edges_path = os.path.join(data_dir, f"{name_prefix}_{var}_edges.csv.gz")
    with gzip.open(edges_path, "wt", encoding="utf-8") as f:
        edges_df.to_csv(f, index=False)

    if src.size == 0:
        sp = sparse.csr_matrix((n, n), dtype=float)
    else:
        sp = sparse.coo_matrix((weights, (src, dst)), shape=(n, n)).tocsr()
    sparse_path = os.path.join(data_dir, f"{name_prefix}_{var}_sparse.npz")
    sparse.save_npz(sparse_path, sp)

    dense_file = None
    dense_csv_file = None
    if n <= int(policy.dense_n_limit):
        dense_file = f"{name_prefix}_{var}_dense.npy"
        np.save(os.path.join(data_dir, dense_file), arr)
        if bool(getattr(policy, "dense_csv", True)):
            dense_csv_file = f"{name_prefix}_{var}_dense.csv"
            pd.DataFrame(arr, index=names, columns=names).to_csv(os.path.join(data_dir, dense_csv_file), index=True)

    meta = {
        "variant": variant,
        "n": n,
        "edges_file": os.path.basename(edges_path),
        "sparse_file": os.path.basename(sparse_path),
        "dense_file": dense_file,
        "dense_csv_file": dense_csv_file,
        "edges_count": int(sp.nnz),
        "topk_per_node": topk,
        "min_abs_weight": min_abs,
        "names_embedded": bool(getattr(policy, "include_names_in_edges", True)),
    }
    if extra_meta is not None:
        meta["extra_meta"] = extra_meta
    meta_path = os.path.join(data_dir, f"{name_prefix}_{var}_meta.json")
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def save_manifest(data_dir: str, name_prefix: str, manifest: dict) -> str:
    """Сохраняет JSON-манифест набора экспортированных данных."""
    path = os.path.join(data_dir, f"{name_prefix}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
