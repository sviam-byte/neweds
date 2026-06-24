"""Информационные метрики: взаимная информация (KSG), distance correlation, AH-ratio."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..defaults import DEFAULT_EMBED_DIM, DEFAULT_EMBED_TAU, DEFAULT_K_MI
from ._shared import (
    _enforce_pairwise_guardrail,
    _get_effective_pairs,
    _init_matrix,
    _iter_pairs,
    _prepare_numpy,
    _residualize_1d,
    _residualize_df,
    _try_parallel,
    get_module_seed,
)
from .registry import register_metric

# ---------------------------------------------------------------------------
# Взаимная информация (KSG kNN-оценка)
# ---------------------------------------------------------------------------


def _neighbor_counts_with_fallback(tree, points: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Считает число соседей для каждой точки в Chebyshev-метрике.

    На старых версиях SciPy ``query_ball_point`` не принимает массив радиусов —
    в этом случае откатываемся на поточечный вызов.
    """
    try:
        neighbors = tree.query_ball_point(points, r=eps, p=np.inf)
        return np.array([max(0, len(lst) - 1) for lst in neighbors], dtype=float)
    except (TypeError, ValueError):
        n = int(points.shape[0])
        return np.fromiter(
            (
                max(0, len(tree.query_ball_point(points[i], r=float(eps[i]), p=np.inf)) - 1)
                for i in range(n)
            ),
            dtype=float,
            count=n,
        )


def _knn_mutual_info(x: np.ndarray, y: np.ndarray, k: int = DEFAULT_K_MI) -> float:
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(x.size, y.size))
    if n <= k or n <= 3:
        warnings.warn(
            f"Mutual information requires more than max(k={k}, 3) observations; got {n}.",
            stacklevel=2,
        )
        return float("nan")
    x = x[:n]
    y = y[:n]

    xy = np.c_[x, y]
    tree_xy = cKDTree(xy)
    distances, _ = tree_xy.query(xy, k=int(k) + 1, p=np.inf)
    eps = np.nextafter(distances[:, int(k)], 0.0)

    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    nx = _neighbor_counts_with_fallback(tree_x, x.reshape(-1, 1), eps)
    ny = _neighbor_counts_with_fallback(tree_y, y.reshape(-1, 1), eps)
    mi = digamma(n) + digamma(int(k)) - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0))
    return float(max(0.0, mi)) if np.isfinite(mi) else float("nan")


def _knn_conditional_mutual_info(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = DEFAULT_K_MI
) -> float:
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = int(min(x.size, y.size, z.shape[0]))
    if n <= k or n <= 3:
        warnings.warn(
            f"Conditional mutual information requires more than max(k={k}, 3) observations; got {n}.",
            stacklevel=2,
        )
        return float("nan")
    x = x[:n]
    y = y[:n]
    z = z[:n, :]

    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    tree_xyz = cKDTree(xyz)
    distances, _ = tree_xyz.query(xyz, k=int(k) + 1, p=np.inf)
    eps = np.nextafter(distances[:, int(k)], 0.0)

    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)

    nxz = _neighbor_counts_with_fallback(tree_xz, xz, eps)
    nyz = _neighbor_counts_with_fallback(tree_yz, yz, eps)
    nz = _neighbor_counts_with_fallback(tree_z, z, eps)

    cmi = digamma(int(k)) - np.mean(digamma(nxz + 1.0) + digamma(nyz + 1.0) - digamma(nz + 1.0))
    return float(max(0.0, cmi)) if np.isfinite(cmi) else float("nan")


def mutual_info_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    k: int = DEFAULT_K_MI,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """Взаимная информация (KSG kNN)."""
    n_vars = int(len(data.columns))
    _enforce_pairwise_guardrail(
        n_vars,
        pairs,
        directed=False,
        metric_name="mutinf_full",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    mi_matrix = _init_matrix(n_vars, np.nan, diag=0.0)
    X = _prepare_numpy(data)
    col_finite = np.isfinite(X)

    def _compute_mi_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        mask = col_finite[:, i] & col_finite[:, j]
        n_valid = int(mask.sum())
        if n_valid <= k:
            warnings.warn(
                f"Mutual information skipped for pair ({i}, {j}): "
                f"{n_valid} valid observations <= k={k}.",
                stacklevel=2,
            )
            return i, j, float("nan")
        v = _knn_mutual_info(X[mask, i], X[mask, j], k=k)
        return i, j, float(v) if np.isfinite(v) else float("nan")

    for i, j, value in _try_parallel(_compute_mi_pair, effective, heavy=True):
        mi_matrix[i, j] = mi_matrix[j, i] = value
    return mi_matrix


def mutual_info_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    k: int = DEFAULT_K_MI,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """Частичная взаимная информация: либо резидуализуем по ``control`` и считаем MI,
    либо для каждой пары считаем условную MI при контроле остальных каналов.
    """
    control_matrix = extra.get("control_matrix")
    if control_matrix is not None:
        sub = data.copy()
        sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
        return mutual_info_matrix(sub, lag=lag, control=None, k=k, pairs=pairs, **extra)
    cols = list(data.columns)
    n_cols = len(cols)
    pmi = _init_matrix(n_cols, np.nan, diag=0.0)
    _enforce_pairwise_guardrail(
        n_cols,
        pairs,
        directed=False,
        metric_name="mutinf_partial",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    it = (
        _iter_pairs(n_cols, pairs, directed=False)
        if pairs is not None
        else [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    )
    for i, j in it:
        xi, xj = cols[i], cols[j]
        z_cols = control if control is not None else [c for c in cols if c not in (xi, xj)]
        z_cols = [c for c in z_cols if c in data.columns and c not in (xi, xj)]
        if not z_cols:
            pair = data[[xi, xj]].dropna()
            if pair.shape[0] <= k:
                warnings.warn(
                    f"Partial mutual information skipped for pair ({i}, {j}): "
                    f"{pair.shape[0]} valid observations <= k={k}.",
                    stacklevel=2,
                )
                value = float("nan")
            else:
                value = float(_knn_mutual_info(pair[xi].values, pair[xj].values, k=k))
        else:
            sub = data[[xi, xj] + z_cols].dropna()
            if sub.shape[0] <= k:
                warnings.warn(
                    f"Conditional mutual information skipped for pair ({i}, {j}): "
                    f"{sub.shape[0]} valid observations <= k={k}.",
                    stacklevel=2,
                )
                value = float("nan")
            else:
                value = float(
                    _knn_conditional_mutual_info(
                        sub[xi].values, sub[xj].values, sub[z_cols].values, k=k
                    )
                )
        pmi[i, j] = pmi[j, i] = value if np.isfinite(value) else np.nan
    return pmi


# ---------------------------------------------------------------------------
# Дистанционная корреляция (dcor)
# ---------------------------------------------------------------------------

_dcor_subsampled: dict[str, bool] = {}


def _dcor_pkg_or_none():
    try:
        import dcor as _dcor_pkg

        return _dcor_pkg
    except ImportError:
        return None


def _dcov_sq(x: np.ndarray, y: np.ndarray) -> float:
    """Квадрат дистанционной ковариации (Székely et al. 2007)."""
    n = x.size
    if n < 4:
        return np.nan
    max_n = 5000
    if n > max_n:
        rng = np.random.default_rng(get_module_seed())
        idx = rng.choice(n, size=max_n, replace=False)
        x = x[idx]
        y = y[idx]
        n = max_n
        _dcor_subsampled["last"] = True
    else:
        _dcor_subsampled["last"] = False
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand
    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_grand = b.mean()
    B = b - b_row - b_col + b_grand
    return float(np.einsum("ij,ij->", A, B) / (n * n))


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Дистанционная корреляция. dCor=0 ⟺ независимость (для конечных моментов)."""
    pkg = _dcor_pkg_or_none()
    if pkg is not None:
        try:
            val = float(pkg.distance_correlation(x, y, method="AVL"))
            return val if np.isfinite(val) else np.nan
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            logging.debug("dcor package failed; using local distance correlation: %s", exc)

    dcov2 = _dcov_sq(x, y)
    dvar_x = _dcov_sq(x, x)
    dvar_y = _dcov_sq(y, y)
    if not np.isfinite(dcov2) or dvar_x <= 0 or dvar_y <= 0:
        return np.nan
    return float(np.sqrt(max(0.0, dcov2) / np.sqrt(dvar_x * dvar_y)))


def dcor_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    n_vars = int(data.shape[1])
    _enforce_pairwise_guardrail(
        n_vars,
        pairs,
        directed=False,
        metric_name="dcor_full",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    out = _init_matrix(n_vars, 0.0, diag=1.0)
    X = _prepare_numpy(data)

    def _compute_dcor_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        xi = X[:, i]
        xj = X[:, j]
        mask = np.isfinite(xi) & np.isfinite(xj)
        if int(mask.sum()) < 8:
            return i, j, 0.0
        v = _dcor(xi[mask], xj[mask])
        return i, j, float(v) if np.isfinite(v) else 0.0

    for i, j, value in _try_parallel(_compute_dcor_pair, effective, heavy=True):
        out[i, j] = out[j, i] = value
    dcor_matrix._subsampled = bool(_dcor_subsampled.get("last", False))  # type: ignore[attr-defined]
    return out


def dcor_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    control_matrix = extra.get("control_matrix")
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub, _desc = _residualize_df(data, control=control, control_matrix=control_matrix)
        return dcor_matrix(sub, lag=lag, control=None, pairs=pairs, **extra)
    n_cols = len(data.columns)
    _enforce_pairwise_guardrail(
        n_cols,
        pairs,
        directed=False,
        metric_name="dcor_partial",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_cols, pairs, directed=False)
    out = _init_matrix(n_cols, 0.0, diag=1.0)
    X = _prepare_numpy(data)
    for i, j in effective:
        z_idx = [k for k in range(n_cols) if k not in (i, j)]
        sub_idx = np.array([i, j] + z_idx)
        sub_data = X[:, sub_idx]
        valid = np.isfinite(sub_data).all(axis=1)
        if int(valid.sum()) < 8:
            out[i, j] = out[j, i] = 0.0
            continue
        x_ctrl = sub_data[valid, 2:]
        xr = _residualize_1d(sub_data[valid, 0], x_ctrl)
        yr = _residualize_1d(sub_data[valid, 1], x_ctrl)
        v = _dcor(xr, yr)
        out[i, j] = out[j, i] = float(v) if np.isfinite(v) else 0.0
    return out


def dcor_matrix_directed(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    lag = int(max(1, lag))
    n_cols = len(data.columns)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    _enforce_pairwise_guardrail(
        n_cols,
        pairs,
        directed=True,
        metric_name="dcor_directed",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(data)
    for i, j in effective:
        x = X[:, i]
        y_shifted = np.roll(X[:, j], -lag)
        if lag > 0:
            y_shifted[-lag:] = np.nan
        mask = np.isfinite(x) & np.isfinite(y_shifted)
        if int(mask.sum()) < 8:
            continue
        v = _dcor(x[mask], y_shifted[mask])
        out[i, j] = float(v) if np.isfinite(v) else 0.0
    return out


# ---------------------------------------------------------------------------
# AH-отношение (Arnhold-H)
# ---------------------------------------------------------------------------


def _H_ratio_direction(
    source: np.ndarray,
    target: np.ndarray,
    m: int = DEFAULT_EMBED_DIM,
    tau: int = DEFAULT_EMBED_TAU,
) -> float | None:
    """Возвращает Arnhold-H ratio (ближайших соседей) для одной направленной пары."""
    from scipy.spatial import cKDTree

    source_values = np.asarray(source, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if source_values.size != target_values.size or source_values.size < 2:
        return None

    embed_dim = int(max(1, m))
    embed_tau = int(max(1, tau))
    embedded_len = int(source_values.size - (embed_dim - 1) * embed_tau)
    if embedded_len < 2:
        return None

    source_embedding = np.empty((embedded_len, embed_dim), dtype=np.float64)
    target_embedding = np.empty((embedded_len, embed_dim), dtype=np.float64)
    for offset in range(embed_dim):
        start = offset * embed_tau
        stop = start + embedded_len
        source_embedding[:, offset] = source_values[start:stop]
        target_embedding[:, offset] = target_values[start:stop]

    valid_rows = np.isfinite(source_embedding).all(axis=1) & np.isfinite(target_embedding).all(
        axis=1
    )
    if not np.any(valid_rows):
        return None

    source_valid = source_embedding[valid_rows]
    target_valid = target_embedding[valid_rows]
    if source_valid.shape[0] < 2:
        return None

    source_tree = cKDTree(source_valid)
    _, source_neighbor_idx = source_tree.query(source_valid, k=2)
    if source_neighbor_idx.ndim != 2 or source_neighbor_idx.shape[1] < 2:
        return None

    source_neighbor = source_neighbor_idx[:, 1]
    target_distance_by_source_neighbor = np.linalg.norm(
        target_valid - target_valid[source_neighbor],
        axis=1,
    )

    target_tree = cKDTree(target_valid)
    target_distances, _ = target_tree.query(target_valid, k=2)
    if target_distances.ndim != 2 or target_distances.shape[1] < 2:
        return None

    target_nearest_distance = np.where(target_distances[:, 1] == 0.0, 1e-10, target_distances[:, 1])
    ratios = target_distance_by_source_neighbor / target_nearest_distance
    ratios = ratios[np.isfinite(ratios)]
    return float(np.mean(ratios)) if ratios.size > 0 else None


def AH_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    embed_dim: int = DEFAULT_EMBED_DIM,
    tau: int = DEFAULT_EMBED_TAU,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """Направленная связность через Arnhold-H ratio."""

    df = data.dropna(axis=0, how="any")
    n_cols = int(df.shape[1])
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    if n_cols < 2 or df.empty:
        return out

    _enforce_pairwise_guardrail(
        n_cols,
        pairs,
        directed=True,
        metric_name="ah_full",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    values = df.to_numpy(dtype=np.float64, copy=False)

    def _compute_ah_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        src, tgt = pair
        ratio = _H_ratio_direction(values[:, src], values[:, tgt], m=embed_dim, tau=tau)
        if ratio is None or ratio <= 0.0:
            return src, tgt, 0.0
        return src, tgt, float(min(1.0, 1.0 / ratio))

    for src, tgt, value in _try_parallel(_compute_ah_pair, effective, heavy=True):
        out[src, tgt] = value
    return out


def compute_partial_AH_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    embed_dim: int = DEFAULT_EMBED_DIM,
    tau: int = DEFAULT_EMBED_TAU,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """AH-ratio на остатках: либо после регрессии по контролям, либо после ``VAR(p)``-чистки."""

    df = data.dropna(axis=0, how="any")
    n_cols = int(df.shape[1])
    if n_cols < 2:
        return _init_matrix(n_cols, 0.0, diag=0.0)

    control_matrix = extra.get("control_matrix")
    if control_matrix is not None or (control is not None and len(control) > 0):
        residualized, _desc = _residualize_df(df, control=control, control_matrix=control_matrix)
        return AH_matrix(residualized, embed_dim=embed_dim, tau=tau, pairs=pairs, **extra)

    try:
        from statsmodels.tsa.vector_ar.var_model import VAR

        model = VAR(df.values).fit(int(max(1, lag)), ic=None)
        residualized = pd.DataFrame(model.resid, columns=df.columns)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        logging.warning("[AH] VAR residualization failed; returning NaN matrix: %s", exc)
        return _init_matrix(n_cols, np.nan, diag=0.0)
    return AH_matrix(residualized, embed_dim=embed_dim, tau=tau, pairs=pairs, **extra)


def AH_matrix_directed(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    **kwargs: dict,
) -> np.ndarray:
    """Обёртка совместимости для направленного варианта AH (с ``control`` или без)."""
    if control:
        return compute_partial_AH_matrix(data, lag=lag, control=control, **kwargs)
    return AH_matrix(data, lag=lag, control=control, **kwargs)


def _register() -> None:
    register_metric(
        "mutinf_full",
        category="information",
        description="Взаимная информация между парами каналов.",
        experimental=True,
    )(mutual_info_matrix)
    register_metric(
        "mutinf_partial",
        category="information",
        description="MI при контроле других каналов.",
        supports_control=True,
        experimental=True,
        partial_mode="explicit_controls_residualization",
    )(mutual_info_matrix_partial)
    register_metric(
        "dcor_full",
        category="information",
        description="Дистанционная корреляция dCor, [0, 1]; 0 ⟺ независимость.",
        stable=True,
    )(dcor_matrix)
    register_metric(
        "dcor_partial",
        category="information",
        description="Частичная dCor через резидуализацию по контрольным.",
        supports_control=True,
        experimental=True,
        partial_mode="explicit_controls_residualization",
    )(dcor_matrix_partial)
    register_metric(
        "dcor_directed",
        category="information",
        description="Лаговая dCor (направленная).",
        directed=True,
        experimental=True,
    )(dcor_matrix_directed)
    register_metric(
        "ah_full",
        category="information",
        description="Active information storage (AH).",
        directed=True,
        experimental=True,
    )(AH_matrix)
    register_metric(
        "ah_partial",
        category="information",
        description="Частичная AH с контрольными переменными.",
        directed=True,
        supports_control=True,
        experimental=True,
        partial_mode="explicit_controls_residualization",
    )(compute_partial_AH_matrix)
    register_metric(
        "ah_directed",
        category="information",
        description="Направленная AH.",
        directed=True,
        experimental=True,
    )(AH_matrix_directed)


__all__ = [
    "mutual_info_matrix",
    "mutual_info_matrix_partial",
    "dcor_matrix",
    "dcor_matrix_partial",
    "dcor_matrix_directed",
    "AH_matrix",
    "AH_matrix_directed",
    "compute_partial_AH_matrix",
]
