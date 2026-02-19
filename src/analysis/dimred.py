"""Утилиты уменьшения размерности по множеству рядов (ось колонок DataFrame)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DimRedResult:
    """Результат уменьшения размерности.

    Attributes:
        reduced: DataFrame уменьшенных рядов (T x K).
        mapping: Таблица соответствия source->target с весом.
        meta: Метаданные выполнения метода.
    """

    reduced: pd.DataFrame
    mapping: pd.DataFrame
    meta: dict


def _variance_select(data: pd.DataFrame, target_n: int) -> DimRedResult:
    vars_ = data.var(axis=0, numeric_only=True).fillna(0.0)
    keep = list(vars_.sort_values(ascending=False).head(int(target_n)).index)
    reduced = data[keep].copy()
    mapping = pd.DataFrame({"source": keep, "target": keep, "weight": 1.0})
    return DimRedResult(reduced=reduced, mapping=mapping, meta={"method": "variance", "k": len(keep)})


def _random_select(data: pd.DataFrame, target_n: int, seed: int) -> DimRedResult:
    rng = np.random.default_rng(int(seed))
    cols = np.array(list(data.columns), dtype=object)
    if target_n >= len(cols):
        keep = list(cols)
    else:
        keep = list(rng.choice(cols, size=int(target_n), replace=False))
    reduced = data[keep].copy()
    mapping = pd.DataFrame({"source": keep, "target": keep, "weight": 1.0})
    return DimRedResult(reduced=reduced, mapping=mapping, meta={"method": "random", "k": len(keep), "seed": int(seed)})


def _spatial_bin(data: pd.DataFrame, coords_df: pd.DataFrame | None, target_n: int, bin_size: int) -> DimRedResult:
    n_features = int(data.shape[1])
    if coords_df is None or coords_df.empty:
        return _variance_select(data, target_n or n_features)

    cdf = coords_df.copy()
    cols_low = {c.lower(): c for c in cdf.columns}
    name_col = cols_low.get("name") or cols_low.get("node")
    if not name_col or not all(k in cols_low for k in ("x", "y", "z")):
        return _variance_select(data, target_n or n_features)

    xcol, ycol, zcol = cols_low["x"], cols_low["y"], cols_low["z"]
    cdf["__bin_x"] = np.floor(pd.to_numeric(cdf[xcol], errors="coerce") / max(1, int(bin_size))).astype("Int64")
    cdf["__bin_y"] = np.floor(pd.to_numeric(cdf[ycol], errors="coerce") / max(1, int(bin_size))).astype("Int64")
    cdf["__bin_z"] = np.floor(pd.to_numeric(cdf[zcol], errors="coerce") / max(1, int(bin_size))).astype("Int64")

    col_set = set(data.columns)
    cdf = cdf[cdf[name_col].astype(str).isin(col_set)]
    grouped = cdf.groupby(["__bin_x", "__bin_y", "__bin_z"], dropna=True)

    reduced_cols = {}
    mapping_rows = []
    for i, (_bin, g) in enumerate(grouped):
        src = [str(v) for v in g[name_col].tolist() if str(v) in col_set]
        if not src:
            continue
        tname = f"bin_{i:04d}"
        reduced_cols[tname] = data[src].mean(axis=1)
        w = 1.0 / max(1, len(src))
        mapping_rows.extend({"source": s, "target": tname, "weight": w} for s in src)

    reduced = pd.DataFrame(reduced_cols, index=data.index)
    mapping = pd.DataFrame(mapping_rows)
    if reduced.shape[1] == 0:
        return _variance_select(data, target_n or n_features)
    if reduced.shape[1] > int(target_n):
        vars_ = reduced.var(axis=0).fillna(0.0).sort_values(ascending=False)
        keep = list(vars_.head(int(target_n)).index)
        reduced = reduced[keep].copy()
        mapping = mapping[mapping["target"].isin(keep)].reset_index(drop=True)
    return DimRedResult(
        reduced=reduced,
        mapping=mapping,
        meta={"method": "spatial", "k": int(reduced.shape[1]), "bin_size": int(bin_size)},
    )


def _pca_reduce(data: pd.DataFrame, target_n: int | None, target_var: float | None, seed: int) -> DimRedResult:
    """Снижает размерность с помощью PCA по оси колонок (рядам)."""
    from sklearn.decomposition import PCA

    x = data.to_numpy(dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = int(x.shape[1])
    if target_var is not None and 0.0 < float(target_var) <= 1.0:
        n_components: int | float = float(target_var)
    else:
        k = int(target_n or 0)
        n_components = int(max(1, min(k, n_features)))

    pca = PCA(n_components=n_components, random_state=int(seed), svd_solver="full")
    scores = pca.fit_transform(x)

    k_out = int(scores.shape[1])
    cols = [f"pc_{i + 1:04d}" for i in range(k_out)]
    reduced = pd.DataFrame(scores, index=data.index, columns=cols)

    mapping_rows = []
    if hasattr(pca, "components_") and pca.components_ is not None:
        sources = [str(c) for c in data.columns]
        for ci, cname in enumerate(cols):
            for source, weight in zip(sources, pca.components_[ci].tolist()):
                mapping_rows.append({"source": source, "target": cname, "weight": float(weight)})

    meta = {
        "method": "pca",
        "k": k_out,
        "seed": int(seed),
        "target_n": (int(target_n) if target_n is not None else None),
        "target_var": (float(target_var) if target_var is not None else None),
        "explained_var": float(pca.explained_variance_ratio_.sum()) if hasattr(pca, "explained_variance_ratio_") else None,
    }
    return DimRedResult(reduced=reduced, mapping=pd.DataFrame(mapping_rows), meta=meta)


def _kmeans_reduce(data: pd.DataFrame, target_n: int, seed: int, batch_size: int) -> DimRedResult:
    from sklearn.cluster import MiniBatchKMeans

    x = data.T.to_numpy(dtype=float)
    n = x.shape[0]
    k = int(max(1, min(int(target_n), n)))
    if k >= n:
        mapping = pd.DataFrame({"source": list(data.columns), "target": list(data.columns), "weight": 1.0})
        return DimRedResult(reduced=data.copy(), mapping=mapping, meta={"method": "kmeans", "k": n, "batch_size": int(batch_size)})

    km = MiniBatchKMeans(n_clusters=k, batch_size=int(max(32, batch_size)), random_state=int(seed), n_init=3)
    labels = km.fit_predict(x)

    reduced_cols = {}
    mapping_rows = []
    for ci in range(k):
        idx = np.where(labels == ci)[0]
        if idx.size == 0:
            continue
        src = [str(data.columns[j]) for j in idx]
        tname = f"cluster_{ci:04d}"
        reduced_cols[tname] = data.iloc[:, idx].mean(axis=1)
        w = 1.0 / float(idx.size)
        mapping_rows.extend({"source": s, "target": tname, "weight": w} for s in src)

    reduced = pd.DataFrame(reduced_cols, index=data.index)
    mapping = pd.DataFrame(mapping_rows)
    return DimRedResult(
        reduced=reduced,
        mapping=mapping,
        meta={"method": "kmeans", "k": int(reduced.shape[1]), "seed": int(seed), "batch_size": int(batch_size)},
    )


def apply_dimred(
    data: pd.DataFrame,
    *,
    method: str = "variance",
    target_n: int | None = 500,
    target_var: float | None = None,
    seed: int = 0,
    coords_df: pd.DataFrame | None = None,
    kmeans_batch: int = 2048,
    spatial_bin: int = 2,
) -> DimRedResult:
    """Применяет выбранный метод уменьшения размерности к колонкам DataFrame."""
    if data is None or data.empty:
        return DimRedResult(reduced=pd.DataFrame(), mapping=pd.DataFrame(), meta={"method": "none", "reason": "empty"})

    method_norm = str(method or "variance").strip().lower()
    n_features = int(data.shape[1])
    tn = int(target_n or 0)
    target_n = int(max(1, min(tn, n_features))) if tn > 0 else None
    try:
        target_var = float(target_var) if target_var is not None and str(target_var).strip() != "" else None
    except Exception:
        target_var = None

    if method_norm == "variance":
        return _variance_select(data, target_n or n_features)
    if method_norm == "random":
        return _random_select(data, target_n or n_features, seed)
    if method_norm == "spatial":
        return _spatial_bin(data, coords_df, target_n or n_features, spatial_bin)
    if method_norm == "kmeans":
        return _kmeans_reduce(data, target_n or n_features, seed, kmeans_batch)
    if method_norm == "pca":
        return _pca_reduce(data, target_n, target_var, seed)

    return _variance_select(data, target_n or n_features)
