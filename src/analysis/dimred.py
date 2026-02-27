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


def _standardize_matrix(data: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Возвращает стандартизованную матрицу X (T x N) и метаданные.

    Принято: строки — наблюдения (время), колонки — признаки (ряды/каналы).
    NaN/inf приводим к 0, чтобы алгоритмы не падали на «дырках».
    """
    x = data.to_numpy(dtype=float, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Центрирование/масштабирование по колонкам (признакам).
    mu = np.mean(x, axis=0, keepdims=True)
    x = x - mu
    sigma = np.std(x, axis=0, ddof=0, keepdims=True)
    sigma_safe = np.where(sigma == 0.0, 1.0, sigma)
    x = x / sigma_safe

    meta = {
        "centered": True,
        "scaled": True,
        "mu_mean": float(np.mean(mu)) if mu.size else 0.0,
        "sigma_mean": float(np.mean(sigma)) if sigma.size else 0.0,
    }
    return x, meta


def _choose_k_from_priority(
    *,
    eigvals_desc: np.ndarray | None,
    n_features: int,
    target_n: int | None,
    target_var: float | None,
    priority: str,
) -> tuple[int, str, float | None, list[float]]:
    """Выбирает k по приоритету.

    priority:
      - "explained_variance": при наличии target_var берём минимальный k, чтобы cum>=target_var
      - "n_components": при наличии target_n берём k=target_n
      - "auto": если задан target_var — explained_variance, иначе n_components

    Возвращает: (k, priority_used, explained_sum, explained_list_topk)
    """
    pr = (priority or "auto").strip().lower()
    if pr in {"var", "variance", "explained", "explained_variance", "evr"}:
        pr = "explained_variance"
    elif pr in {"k", "n", "n_components", "components"}:
        pr = "n_components"
    elif pr not in {"auto", "explained_variance", "n_components"}:
        pr = "auto"

    tv = None
    try:
        tv = float(target_var) if target_var is not None else None
    except Exception:
        tv = None
    if tv is not None and not (0.0 < tv <= 1.0):
        tv = None

    tn = None
    try:
        tn = int(target_n) if target_n is not None and int(target_n) > 0 else None
    except Exception:
        tn = None

    # auto: если есть target_var — приоритет на дисперсию, иначе на k
    pr_used = pr
    if pr == "auto":
        pr_used = "explained_variance" if tv is not None else "n_components"

    k = 1
    ev_sum = None
    ev_top: list[float] = []

    if pr_used == "n_components":
        if tn is not None:
            k = int(max(1, min(int(tn), int(n_features))))
        else:
            k = int(max(1, min(10, int(n_features))))
        return k, pr_used, ev_sum, ev_top

    # explained_variance
    if eigvals_desc is None or eigvals_desc.size == 0:
        # нет спектра — фолбэк
        if tn is not None:
            k = int(max(1, min(int(tn), int(n_features))))
            return k, "n_components", ev_sum, ev_top
        k = int(max(1, min(10, int(n_features))))
        return k, "n_components", ev_sum, ev_top

    total = float(np.sum(eigvals_desc))
    if total <= 0:
        # деградация: всё нулевое
        if tn is not None:
            k = int(max(1, min(int(tn), int(n_features))))
            return k, "n_components", 0.0, [0.0] * k
        return 1, "n_components", 0.0, [0.0]

    ratios = (eigvals_desc / total).astype(float)
    cum = np.cumsum(ratios)

    if tv is None:
        # нет target_var -> фолбэк на k
        if tn is not None:
            k = int(max(1, min(int(tn), int(n_features))))
            ev_top = [float(v) for v in ratios[:k].tolist()]
            return k, "n_components", float(np.sum(ev_top)), ev_top
        k = int(max(1, min(10, int(n_features))))
        ev_top = [float(v) for v in ratios[:k].tolist()]
        return k, "n_components", float(np.sum(ev_top)), ev_top

    # минимальный k, чтобы cum>=tv
    k = int(np.searchsorted(cum, tv, side="left") + 1)
    k = int(max(1, min(k, int(n_features))))
    ev_top = [float(v) for v in ratios[:k].tolist()]
    ev_sum = float(np.sum(ev_top))
    return k, "explained_variance", ev_sum, ev_top


def _pca_reduce(
    data: pd.DataFrame,
    target_n: int | None,
    target_var: float | None,
    seed: int,
    *,
    priority: str = "auto",
    solver: str = "full",
) -> DimRedResult:
    """PCA по оси колонок (рядам): вход T×N -> выход T×K.

    solver:
      - "full": sklearn PCA svd_solver='full'
      - "randomized": sklearn PCA svd_solver='randomized' (быстро при больших N)
      - "gram": PCA через грам-матрицу XX^T (T×T), полезно когда N >> T

    priority:
      - "explained_variance" | "n_components" | "auto"
      Если заданы и target_n, и target_var, выбираем согласно priority.
    """
    solver_norm = (solver or "full").strip().lower()
    if solver_norm in {"gram", "gram_matrix", "gram-matrix", "grammatrix"}:
        solver_norm = "gram"
    elif solver_norm in {"rand", "random", "randomized", "randomised"}:
        solver_norm = "randomized"
    else:
        solver_norm = "full"

    x, x_meta = _standardize_matrix(data)
    n_samples = int(x.shape[0])
    n_features = int(x.shape[1])
    # k не может превышать min(T, N)
    k_max = int(max(1, min(n_samples, n_features)))

    # Для выбора k по explained variance нам нужен спектр (eigvals_desc).
    eigvals_desc: np.ndarray | None = None

    if solver_norm == "gram" or ((priority or "auto").strip().lower() in {"explained_variance", "variance", "evr", "explained"} and target_var is not None):
        # Грам-матрица: K = X X^T (n_samples x n_samples), её собственные значения = S^2
        k_mat = (x @ x.T).astype(float, copy=False)
        k_mat = 0.5 * (k_mat + k_mat.T)
        w, u = np.linalg.eigh(k_mat)
        w = np.clip(w, 0.0, None)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        u = u[:, idx]
        eigvals_desc = w.copy()

        # k по приоритету
        k, pr_used, ev_sum, ev_top = _choose_k_from_priority(
            eigvals_desc=eigvals_desc,
            n_features=k_max,
            target_n=target_n,
            target_var=target_var,
            priority=priority,
        )

        # Scores = U_k * S_k
        s = np.sqrt(np.maximum(w[:k], 0.0))
        scores = u[:, :k] * s.reshape(1, -1)

        # Components V^T = (S^{-1} U^T X)
        denom = np.where(s == 0.0, 1.0, s)
        comps = (u[:, :k].T @ x) / denom.reshape(-1, 1)  # (k x N)

        cols = [f"pc_{i + 1:04d}" for i in range(int(k))]
        reduced = pd.DataFrame(scores, index=data.index, columns=cols)

        mapping_rows = []
        sources = [str(c) for c in data.columns]
        for ci, cname in enumerate(cols):
            for source, weight in zip(sources, comps[ci].tolist()):
                mapping_rows.append({"source": source, "target": cname, "weight": float(weight)})

        meta = {
            "method": "pca",
            "solver": "gram",
            "k": int(k),
            "seed": int(seed),
            "target_n": (int(target_n) if target_n is not None else None),
            "target_var": (float(target_var) if target_var is not None else None),
            "priority": pr_used,
            "explained_var": ev_sum,
            "explained_var_top": ev_top,
            "n_samples": n_samples,
            "n_features": n_features,
            **x_meta,
        }
        return DimRedResult(reduced=reduced, mapping=pd.DataFrame(mapping_rows), meta=meta)

    # Иначе — sklearn PCA (full/randomized)
    from sklearn.decomposition import PCA

    pr_norm = (priority or "auto").strip().lower()
    if pr_norm in {"var", "variance", "explained", "explained_variance", "evr"}:
        pr_norm = "explained_variance"
    elif pr_norm in {"k", "n", "n_components", "components"}:
        pr_norm = "n_components"
    elif pr_norm != "auto":
        pr_norm = "auto"

    if pr_norm == "auto":
        pr_norm = "explained_variance" if target_var is not None else "n_components"

    if pr_norm == "explained_variance" and target_var is not None and solver_norm == "full":
        n_components: int | float = float(target_var)
        pr_used = "explained_variance"
    else:
        k = int(target_n or 0)
        k = int(max(1, min(k if k > 0 else min(10, k_max), k_max)))
        n_components = k
        pr_used = "n_components" if pr_norm == "n_components" or target_var is None else pr_norm

    pca = PCA(n_components=n_components, random_state=int(seed), svd_solver=("randomized" if solver_norm == "randomized" else "full"))
    scores = pca.fit_transform(x)

    k_out = int(scores.shape[1])
    cols = [f"pc_{i + 1:04d}" for i in range(k_out)]
    reduced = pd.DataFrame(scores, index=data.index, columns=cols)

    mapping_rows = []
    if getattr(pca, "components_", None) is not None:
        sources = [str(c) for c in data.columns]
        for ci, cname in enumerate(cols):
            for source, weight in zip(sources, pca.components_[ci].tolist()):
                mapping_rows.append({"source": source, "target": cname, "weight": float(weight)})

    evr_list: list[float] = []
    if getattr(pca, "explained_variance_ratio_", None) is not None:
        try:
            evr_list = [float(v) for v in pca.explained_variance_ratio_.tolist()]
        except Exception:
            evr_list = []

    meta = {
        "method": "pca",
        "solver": solver_norm,
        "k": k_out,
        "seed": int(seed),
        "target_n": (int(target_n) if target_n is not None else None),
        "target_var": (float(target_var) if target_var is not None else None),
        "priority": pr_used,
        "explained_var": float(sum(evr_list)) if evr_list else None,
        "explained_var_top": evr_list,
        "n_samples": n_samples,
        "n_features": n_features,
        **x_meta,
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
    pca_priority: str = "auto",
    pca_solver: str = "full",
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
    if method_norm in {"pca", "pca_full"}:
        return _pca_reduce(data, target_n, target_var, seed, priority=pca_priority, solver=pca_solver)
    if method_norm in {"pca_gram", "pca_gram_matrix"}:
        return _pca_reduce(data, target_n, target_var, seed, priority=pca_priority, solver="gram")
    if method_norm in {"pca_randomized", "pca_rand", "pca_random"}:
        return _pca_reduce(data, target_n, target_var, seed, priority=pca_priority, solver="randomized")

    return _variance_select(data, target_n or n_features)
