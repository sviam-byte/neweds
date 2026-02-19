#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Утилиты экспорта матриц связности в машиночитаемом формате."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
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
    """Экспортирует одну матрицу: edge-list, sparse и (опционально) dense."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    n = int(arr.shape[0])
    var = _safe_variant(variant)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = arr[i]
        if policy.topk_per_node > 0 and policy.topk_per_node < n:
            idx = np.argpartition(np.abs(row), -policy.topk_per_node)[-policy.topk_per_node:]
        else:
            idx = np.arange(n)
        idx = idx[idx != i]
        for j in idx:
            w = float(arr[i, j])
            if abs(w) < float(policy.min_abs_weight):
                continue
            rows.append({"src": i, "dst": int(j), "src_name": names[i], "dst_name": names[int(j)], "weight": w})

    edges_df = pd.DataFrame(rows)
    edges_path = os.path.join(data_dir, f"{name_prefix}_{var}_edges.csv.gz")
    with gzip.open(edges_path, "wt", encoding="utf-8") as f:
        edges_df.to_csv(f, index=False)

    if edges_df.empty:
        sp = sparse.csr_matrix((n, n), dtype=float)
    else:
        sp = sparse.coo_matrix((edges_df["weight"].to_numpy(), (edges_df["src"].to_numpy(), edges_df["dst"].to_numpy())), shape=(n, n)).tocsr()
    sparse_path = os.path.join(data_dir, f"{name_prefix}_{var}_sparse.npz")
    sparse.save_npz(sparse_path, sp)

    dense_file = None
    if n <= int(policy.dense_n_limit):
        dense_file = f"{name_prefix}_{var}_dense.npy"
        np.save(os.path.join(data_dir, dense_file), arr)

    meta = {
        "variant": variant,
        "n": n,
        "edges_file": os.path.basename(edges_path),
        "sparse_file": os.path.basename(sparse_path),
        "dense_file": dense_file,
        "edges_count": int(sp.nnz),
    }
    if extra_meta is not None:
        meta["extra_meta"] = extra_meta
    return meta


def save_manifest(data_dir: str, name_prefix: str, manifest: dict) -> str:
    """Сохраняет JSON-манифест набора экспортированных данных."""
    path = os.path.join(data_dir, f"{name_prefix}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
