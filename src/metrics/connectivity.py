"""Метрики связности, выделенные из основного движка анализа."""

from __future__ import annotations

import importlib
import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial import cKDTree
from scipy.special import digamma
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from ..config import DEFAULT_BINS, DEFAULT_K_MI, DEFAULT_MAX_LAG, PYINFORM_AVAILABLE

# Порог пар, с которого пробуем распараллеливание тяжёлых покомпонентных расчётов.
_PARALLEL_PAIR_THRESHOLD = 800
# Верхняя граница автоматически сэмплируемых пар для очень больших матриц.
_MAX_AUTO_RANDOM_PAIRS = 500_000


def _init_matrix(n: int, default: float, *, diag: Optional[float] = None, dtype=np.float64) -> np.ndarray:
    out = np.full((n, n), float(default), dtype=dtype)
    if diag is not None:
        np.fill_diagonal(out, float(diag))
    return out


def _iter_pairs(n: int, pairs: Optional[list[tuple[int, int]]], *, directed: bool) -> list[tuple[int, int]]:
    """Normalize user pairs.

    For undirected metrics we return unique (min,max) pairs.
    """
    if pairs is None:
        return []
    out: list[tuple[int, int]] = []
    seen = set()
    for i, j in pairs:
        try:
            i = int(i)
            j = int(j)
        except Exception:
            continue
        if i == j or i < 0 or j < 0 or i >= n or j >= n:
            continue
        if not directed:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
        else:
            out.append((i, j))
    return out


def _get_effective_pairs(
    n: int,
    pairs: Optional[list[tuple[int, int]]],
    *,
    directed: bool,
) -> list[tuple[int, int]]:
    """Возвращает итоговый список пар; при huge-N и pairs=None делает auto-sampling."""
    if pairs is not None:
        return _iter_pairs(n, pairs, directed=directed)
    n_full = n * (n - 1) if directed else n * (n - 1) // 2
    if n_full <= 10_000_000:
        if directed:
            return [(i, j) for i in range(n) for j in range(n) if i != j]
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    max_pairs = min(_MAX_AUTO_RANDOM_PAIRS, max(1, n * 5))
    logging.warning(
        "[connectivity] N=%d -> %d pairs, auto random sample %d.",
        n,
        n_full,
        max_pairs,
    )
    rng = np.random.default_rng(42)
    result: set[tuple[int, int]] = set()
    bi = rng.integers(0, n, size=max_pairs * 3)
    bj = rng.integers(0, n, size=max_pairs * 3)
    for ii, jj in zip(bi, bj):
        if ii == jj:
            continue
        key = (int(ii), int(jj)) if directed else (int(min(ii, jj)), int(max(ii, jj)))
        result.add(key)
        if len(result) >= max_pairs:
            break
    return list(result)


def _prepare_numpy(data: pd.DataFrame) -> np.ndarray:
    """Преобразует DataFrame -> float64 numpy без лишних копий где возможно."""
    return data.to_numpy(dtype=np.float64, copy=False)


def _try_parallel(func, pairs: list[tuple[int, int]], n_jobs: int = -1):
    """joblib-parallel для списка пар, с безопасным fallback в последовательный режим."""
    if len(pairs) < _PARALLEL_PAIR_THRESHOLD:
        return [func(p) for p in pairs]
    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs, backend="loky")(delayed(func)(p) for p in pairs)
    except ImportError:
        return [func(p) for p in pairs]


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask].astype(np.float64, copy=False)
    yy = y[mask].astype(np.float64, copy=False)
    sx = float(xx.std())
    sy = float(yy.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _as_2d_controls(df: pd.DataFrame, control: Optional[list[str]] = None, control_matrix: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[str]]:
    """Возвращает (X_ctrl, desc).

    - control: список колонок из df
    - control_matrix: внешняя матрица регрессоров (time × k)
    """
    if control_matrix is not None:
        X = np.asarray(control_matrix, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, [f"ctrl[{i}]" for i in range(X.shape[1])]
    if control:
        cols = [c for c in control if c in df.columns]
        if cols:
            X = df[cols].to_numpy(dtype=np.float64)
            return X, [str(c) for c in cols]
    return np.empty((len(df), 0), dtype=np.float64), []


def _residualize_1d(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X is None or X.size == 0:
        return y
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = int(min(y.size, X.shape[0]))
    y = y[:n]
    X = X[:n, :]
    # добавляем константу
    A = np.c_[np.ones(n), X]
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta
    except Exception:
        return y


def _residualize_df(df: pd.DataFrame, control: Optional[list[str]] = None, control_matrix: Optional[np.ndarray] = None) -> tuple[pd.DataFrame, list[str]]:
    X_ctrl, desc = _as_2d_controls(df, control=control, control_matrix=control_matrix)
    if X_ctrl.size == 0:
        return df, []
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        # общий dropna по y и X (иначе рассинхрон времени)
        y = s.to_numpy(dtype=np.float64)
        mask = np.isfinite(y)
        if X_ctrl.size:
            mask = mask & np.isfinite(X_ctrl).all(axis=1)
        if int(mask.sum()) < 8:
            out[c] = np.nan
            continue
        y_res = _residualize_1d(y[mask], X_ctrl[mask])
        # вернем на исходную длину
        tmp = np.full_like(y, np.nan, dtype=np.float64)
        tmp[mask] = y_res
        out[c] = tmp
    return out, desc


def correlation_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    """Корреляция Пирсона.

    Если pairs задан, считаем только эти пары (упрощение для больших N).
    """
    n_cols = int(data.shape[1])
    if pairs is not None:
        X = _prepare_numpy(data)
        out = _init_matrix(n_cols, 0.0, diag=1.0)
        for i, j in _iter_pairs(n_cols, pairs, directed=False):
            out[i, j] = out[j, i] = _corr_1d(X[:, i], X[:, j])
        return out
    if n_cols <= 20_000:
        return data.corr().values

    effective = _get_effective_pairs(n_cols, None, directed=False)
    X = _prepare_numpy(data)
    out = _init_matrix(n_cols, 0.0, diag=1.0)
    for i, j in effective:
        out[i, j] = out[j, i] = _corr_1d(X[:, i], X[:, j])
    return out

def partial_correlation_matrix(
    df: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    control_matrix: Optional[np.ndarray] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    """Частная корреляция."""
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n_cols = len(cols)
    n_rows = len(df)
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub = df[cols].copy()
        sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
        if pairs is None:
            return sub.corr().values
        n = int(len(cols))
        out = _init_matrix(n, 0.0, diag=1.0)
        X = sub.to_numpy(dtype=np.float64, copy=False)
        for i, j in _iter_pairs(n, pairs, directed=False):
            out[i, j] = out[j, i] = _corr_1d(X[:, i], X[:, j])
        return out
    if n_rows <= n_cols + 2:
        logging.warning("Слишком мало данных для Partial Correlation: строк %s <= колонок %s. Возвращаю NaN.", n_rows, n_cols)
        return np.full((n_cols, n_cols), np.nan)
    if n_cols > 200 and control is None and pairs is None:
        try:
            R = df[cols].corr().values
            P = np.linalg.pinv(R)
            d = np.sqrt(np.abs(np.diag(P)))
            d[d < 1e-12] = 1.0
            pcor = -P / np.outer(d, d)
            np.fill_diagonal(pcor, 1.0)
            return pcor
        except Exception:
            pass
    out = _init_matrix(n_cols, 0.0, diag=1.0)
    effective = _get_effective_pairs(n_cols, pairs, directed=False)
    for i, j in effective:
        xi, xj = cols[i], cols[j]
        ctrl_vars = control if control is not None else [c for c in cols if c not in (xi, xj)]
        sub_cols = [xi, xj] + [c for c in ctrl_vars if c in cols and c not in (xi, xj)]
        sub = df[sub_cols].dropna()
        if sub.shape[0] < len(sub_cols) + 1:
            pcor = np.nan
        else:
            try:
                corr_matrix = sub.corr().values
                precision = np.linalg.pinv(corr_matrix)
                pcor = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
            except Exception:
                pcor = np.nan
        out[i, j] = out[j, i] = float(pcor) if np.isfinite(pcor) else 0.0
    return out

def partial_h2_matrix(df: pd.DataFrame, lag: int = 1, control: Optional[list[str]] = None, **kwargs: dict) -> np.ndarray:
    """Вычисляет квадрат частной корреляции (приближение частного H²)."""
    return partial_correlation_matrix(df, lag=lag, control=control, **kwargs) ** 2


def lagged_directed_correlation(
    df: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    """Вычисляет направленную лаговую корреляцию, где M[src, tgt] = corr(src(t), tgt(t+lag))."""
    lag = int(max(1, lag))
    n_cols = len(df.columns)
    X = _prepare_numpy(df)
    T = X.shape[0]
    if T <= lag + 3:
        return _init_matrix(n_cols, 0.0, diag=0.0)

    # Быстрый векторизованный путь для dense-случая без NaN.
    if pairs is None and n_cols <= 10_000:
        x_past = X[: T - lag]
        x_future = X[lag:]
        if not (np.any(~np.isfinite(x_past)) or np.any(~np.isfinite(x_future))):
            t_eff = T - lag
            past_centered = x_past - x_past.mean(axis=0, keepdims=True)
            fut_centered = x_future - x_future.mean(axis=0, keepdims=True)
            sp = np.sqrt((past_centered**2).sum(axis=0, keepdims=True))
            sf = np.sqrt((fut_centered**2).sum(axis=0, keepdims=True))
            sp[sp < 1e-12] = 1.0
            sf[sf < 1e-12] = 1.0
            out = (past_centered / sp).T @ (fut_centered / sf) / t_eff
            np.fill_diagonal(out, 0.0)
            return out

    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    for i, j in effective:
        out[i, j] = _corr_1d(X[: T - lag, i], X[lag:, j])
    return out

def _knn_mutual_info(x: np.ndarray, y: np.ndarray, k: int = DEFAULT_K_MI) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(x.size, y.size))
    if n <= k or n <= 3:
        return 0.0
    x = x[:n]
    y = y[:n]

    xy = np.c_[x, y]
    tree_xy = cKDTree(xy)
    distances, _ = tree_xy.query(xy, k=int(k) + 1, p=np.inf)
    eps = np.nextafter(distances[:, int(k)], 0.0)

    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))

    nx = np.fromiter(
        (max(0, len(tree_x.query_ball_point([x[i]], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)),
        dtype=float,
        count=n,
    )
    ny = np.fromiter(
        (max(0, len(tree_y.query_ball_point([y[i]], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)),
        dtype=float,
        count=n,
    )
    mi = digamma(n) + digamma(int(k)) - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0))
    return float(max(0.0, mi)) if np.isfinite(mi) else float("nan")


def _knn_conditional_mutual_info(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = DEFAULT_K_MI) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = int(min(x.size, y.size, z.shape[0]))
    if n <= k or n <= 3:
        return 0.0
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

    nxz = np.fromiter((max(0, len(tree_xz.query_ball_point(xz[i], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)), dtype=float, count=n)
    nyz = np.fromiter((max(0, len(tree_yz.query_ball_point(yz[i], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)), dtype=float, count=n)
    nz = np.fromiter((max(0, len(tree_z.query_ball_point(z[i], r=float(eps[i]), p=np.inf)) - 1) for i in range(n)), dtype=float, count=n)

    cmi = digamma(int(k)) - np.mean(digamma(nxz + 1.0) + digamma(nyz + 1.0) - digamma(nz + 1.0))
    return float(max(0.0, cmi)) if np.isfinite(cmi) else float("nan")


def mutual_info_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    k: int = DEFAULT_K_MI,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    """Взаимная информация (KSG kNN). Если pairs задан, считаем только эти пары."""
    n_vars = int(len(data.columns))
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    mi_matrix = _init_matrix(n_vars, 0.0, diag=0.0)
    X = _prepare_numpy(data)

    def _compute_mi_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        xi = X[:, i]
        xj = X[:, j]
        mask = np.isfinite(xi) & np.isfinite(xj)
        if int(mask.sum()) <= k:
            return i, j, 0.0
        v = _knn_mutual_info(xi[mask], xj[mask], k=k)
        return i, j, float(v) if np.isfinite(v) else 0.0

    for i, j, value in _try_parallel(_compute_mi_pair, effective):
        mi_matrix[i, j] = mi_matrix[j, i] = value
    return mi_matrix

def mutual_info_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    k: int = DEFAULT_K_MI,
    pairs: Optional[list[tuple[int, int]]] = None,
    **extra: dict,
) -> np.ndarray:
    """Partial MI."""
    control_matrix = extra.get("control_matrix", None)
    if control_matrix is not None:
        sub = data.copy(); sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
        return mutual_info_matrix(sub, lag=lag, control=None, k=k, pairs=pairs)
    cols = list(data.columns); n_cols = len(cols)
    pmi = _init_matrix(n_cols, 0.0, diag=0.0)
    it = _iter_pairs(n_cols, pairs, directed=False) if pairs is not None else [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    for i, j in it:
        xi, xj = cols[i], cols[j]
        z_cols = control if control is not None else [c for c in cols if c not in (xi, xj)]
        z_cols = [c for c in z_cols if c in data.columns and c not in (xi, xj)]
        if not z_cols:
            pair = data[[xi, xj]].dropna()
            value = float(_knn_mutual_info(pair[xi].values, pair[xj].values, k=k)) if pair.shape[0] > k else 0.0
        else:
            sub = data[[xi, xj] + z_cols].dropna()
            value = float(_knn_conditional_mutual_info(sub[xi].values, sub[xj].values, sub[z_cols].values, k=k)) if sub.shape[0] > k else 0.0
        pmi[i, j] = pmi[j, i] = value if np.isfinite(value) else 0.0
    return pmi

def coherence_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    fs: float = 1.0,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    fs = fs if np.isfinite(fs) and fs > 0 else 1.0
    n_vars = int(data.shape[1])
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    coh = _init_matrix(n_vars, 0.0, diag=1.0)
    X = _prepare_numpy(data)
    for i, j in effective:
        xi = X[:, i]
        xj = X[:, j]
        mask = np.isfinite(xi) & np.isfinite(xj)
        n = int(mask.sum())
        if n <= 3:
            continue
        s1 = xi[mask].astype(np.float64)
        s2 = xj[mask].astype(np.float64)
        nperseg = int(max(8, min(64, n // 2)))
        try:
            _, cxy = signal.coherence(s1, s2, fs=fs, nperseg=nperseg, detrend="constant")
            cxy = np.clip(np.asarray(cxy, dtype=np.float64), 0.0, 1.0)
            cxy[~np.isfinite(cxy)] = np.nan
            coh[i, j] = coh[j, i] = float(np.nanmean(cxy)) if np.isfinite(cxy).any() else 0.0
        except Exception:
            coh[i, j] = coh[j, i] = 0.0
    return coh

def granger_matrix(
    df: pd.DataFrame,
    lag: int = DEFAULT_MAX_LAG,
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    n_cols = int(df.shape[1])
    out = _init_matrix(n_cols, 1.0, diag=0.0)
    columns = df.columns.tolist()
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(df)

    def _compute_granger_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        src, tgt = pair
        x_src = X[:, src]
        x_tgt = X[:, tgt]
        mask = np.isfinite(x_src) & np.isfinite(x_tgt)
        if int(mask.sum()) <= lag * 2 + 5:
            return src, tgt, 1.0
        pair_df = pd.DataFrame({columns[tgt]: x_tgt[mask], columns[src]: x_src[mask]})
        try:
            tests = grangercausalitytests(pair_df, maxlag=int(lag), verbose=False)
            return src, tgt, float(tests[int(lag)][0]["ssr_ftest"][1])
        except Exception:
            return src, tgt, 1.0

    for src, tgt, value in _try_parallel(_compute_granger_pair, effective):
        out[src, tgt] = value
    return out

def granger_matrix_partial(
    df: pd.DataFrame,
    lag: int = DEFAULT_MAX_LAG,
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **extra: dict,
) -> np.ndarray:
    control_matrix = extra.get("control_matrix", None)
    if control_matrix is not None:
        sub = df.copy(); sub, _ = _residualize_df(sub, control=control, control_matrix=control_matrix)
        return granger_matrix(sub, lag=lag, control=None, pairs=pairs)
    columns = list(df.columns); n_cols = len(columns); out = _init_matrix(n_cols, 1.0, diag=0.0)
    if len(df) <= n_cols + 2:
        logging.warning("Слишком мало данных для Granger partial: строк %s <= колонок %s. Возвращаю NaN.", len(df), n_cols)
        return out
    if n_cols > 50 and control is None:
        logging.warning("[granger_partial] N=%d > 50, VAR infeasible. Fallback.", n_cols)
        return granger_matrix(df, lag=lag, control=None, pairs=pairs)

    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    for src_i, tgt_j in effective:
        src = columns[src_i]; tgt = columns[tgt_j]
        control_cols = control if control is not None else [c for c in columns if c not in (src, tgt)]
        control_cols = [c for c in control_cols if c in df.columns and c not in (src, tgt)]
        use_cols = [tgt, src] + control_cols; sub = df[use_cols].dropna(); p = int(max(1, lag))
        if sub.shape[0] < max(30, 5 * p * len(use_cols)): continue
        try:
            result = VAR(sub).fit(maxlags=p, ic=None, trend="c")
            causality = result.test_causality(caused=tgt, causing=[src], kind="f")
            out[src_i, tgt_j] = float(causality.pvalue) if np.isfinite(causality.pvalue) else 1.0
        except Exception:
            out[src_i, tgt_j] = 1.0
    return out

def _load_pyinform():
    if not PYINFORM_AVAILABLE:
        return None
    return importlib.import_module("pyinform")


def _transfer_entropy_discrete(source_d: np.ndarray, target_d: np.ndarray, k: int = 1) -> float:
    k = max(1, int(k))
    source_d = np.asarray(source_d, dtype=int).ravel()
    target_d = np.asarray(target_d, dtype=int).ravel()
    n = min(source_d.size, target_d.size)
    if n <= k + 1:
        return float("nan")
    source_d = source_d[:n]
    target_d = target_d[:n]

    c_xyz, c_xx, c_xpast_ypast, c_xpast = Counter(), Counter(), Counter(), Counter()
    for t in range(k, n):
        x_t = int(target_d[t])
        x_past = tuple(int(v) for v in target_d[t - k : t])
        y_past = tuple(int(v) for v in source_d[t - k : t])
        c_xyz[(x_t, x_past, y_past)] += 1
        c_xx[(x_t, x_past)] += 1
        c_xpast_ypast[(x_past, y_past)] += 1
        c_xpast[x_past] += 1

    n_eff = n - k
    te = 0.0
    for (x_t, x_past, y_past), count in c_xyz.items():
        p_xyz = count / n_eff
        p_x_given_x_y = count / c_xpast_ypast[(x_past, y_past)]
        p_x_given_x = c_xx[(x_t, x_past)] / c_xpast[x_past]
        if p_x_given_x_y > 0 and p_x_given_x > 0:
            te += p_xyz * np.log2(p_x_given_x_y / p_x_given_x)
    return float(te)


def compute_te_jitter(source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = DEFAULT_BINS) -> float:
    """Вычисляет Transfer Entropy с использованием z-score, jitter и квантильной дискретизации."""

    def _zscore_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        mean = np.nanmean(x)
        std = np.nanstd(x)
        return x - mean if (not np.isfinite(std) or std <= 0) else (x - mean) / std

    def _add_tiny_jitter(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size <= 3:
            return x
        uniq = np.unique(x[np.isfinite(x)])
        if uniq.size < max(3, int(0.2 * x.size)):
            rng = np.random.default_rng(0)
            scale = (np.nanstd(x) if np.nanstd(x) > 0 else 1.0) * 1e-10
            x = x + rng.normal(0.0, scale, size=x.shape)
        return x

    def discretize_quantile(series: np.ndarray, num_bins: int) -> np.ndarray:
        s = _add_tiny_jitter(_zscore_1d(series))
        s = s[np.isfinite(s)]
        if s.size == 0:
            return np.array([], dtype=int)
        if float(np.nanmin(s)) == float(np.nanmax(s)):
            return np.zeros(int(np.asarray(series).size), dtype=int)
        edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(num_bins) + 1)))
        if edges.size <= 2:
            return np.zeros(int(np.asarray(series).size), dtype=int)
        edges[-1] = np.nextafter(edges[-1], edges[-1] + 1.0)
        disc = np.digitize(_add_tiny_jitter(_zscore_1d(np.asarray(series))), bins=edges[1:-1], right=False)
        return np.clip(disc, 0, int(num_bins) - 1).astype(int)

    try:
        source_discrete = discretize_quantile(source, bins)
        target_discrete = discretize_quantile(target, bins)
        pyinform = _load_pyinform()
        k = int(max(1, lag))
        if pyinform is not None:
            return float(pyinform.transfer_entropy(source_discrete, target_discrete, k=k))
        return _transfer_entropy_discrete(source_discrete, target_discrete, k=k)
    except Exception as exc:
        logging.error("[TE] Ошибка вычисления: %s", exc)
        return float("nan")


def transfer_entropy_matrix(
    df: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    bins: int = DEFAULT_BINS,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    n_cols = int(df.shape[1])
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(df)

    def _compute_te_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        src, tgt = pair
        x_src = X[:, src]
        x_tgt = X[:, tgt]
        mask = np.isfinite(x_src) & np.isfinite(x_tgt)
        if int(mask.sum()) <= lag:
            return src, tgt, 0.0
        v = compute_te_jitter(x_src[mask], x_tgt[mask], lag=lag, bins=bins)
        return src, tgt, float(v) if np.isfinite(v) else 0.0

    for src, tgt, value in _try_parallel(_compute_te_pair, effective):
        out[src, tgt] = value
    return out

def transfer_entropy_matrix_partial(
    df: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    bins: int = DEFAULT_BINS,
    pairs: Optional[list[tuple[int, int]]] = None,
    **extra: dict,
) -> np.ndarray:
    cols = list(df.columns); n_cols = len(cols); out = _init_matrix(n_cols, 0.0, diag=0.0)
    control_matrix = extra.get("control_matrix", None)
    def residualize(y, x_ctrl):
        if x_ctrl is None or x_ctrl.size == 0: return y
        x_aug = np.c_[np.ones(len(y)), x_ctrl]
        beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return y - x_aug @ beta
    it = _iter_pairs(n_cols, pairs, directed=True) if pairs is not None else [(i, j) for i in range(n_cols) for j in range(n_cols) if i != j]
    for i, j in it:
        src = cols[i]; tgt = cols[j]
        if control_matrix is not None:
            pair_vals = X[:, [i, j]]
            X_ctrl = np.asarray(control_matrix, dtype=np.float64)
            valid = np.isfinite(pair_vals).all(axis=1)
            if X_ctrl.size and X_ctrl.ndim == 1: X_ctrl = X_ctrl.reshape(-1, 1)
            if X_ctrl.shape[1] > 0: valid &= np.isfinite(X_ctrl).all(axis=1)
            if int(valid.sum()) <= lag + 1: out[i, j] = 0.0; continue
            src_vals = pair_vals[valid, 0]; tgt_vals = pair_vals[valid, 1]; x_ctrl = X_ctrl[valid, :] if X_ctrl.size else np.empty((int(valid.sum()), 0))
            src_res = residualize(src_vals, x_ctrl); tgt_res = residualize(tgt_vals, x_ctrl)
            v = compute_te_jitter(src_res, tgt_res, lag=lag, bins=bins); out[i, j] = float(v) if np.isfinite(v) else 0.0; continue
        control_cols = control if control is not None else [c for c in cols if c not in (src, tgt)]
        control_cols = [c for c in control_cols if c in df.columns and c not in (src, tgt)]
        sub = df[[src, tgt] + control_cols].dropna()
        if sub.shape[0] <= lag + 1: out[i, j] = 0.0; continue
        x_ctrl = sub[control_cols].values if control_cols else np.empty((len(sub), 0))
        try:
            src_res = residualize(sub[src].values, x_ctrl); tgt_res = residualize(sub[tgt].values, x_ctrl)
            v = compute_te_jitter(src_res, tgt_res, lag=lag, bins=bins)
            out[i, j] = float(v) if np.isfinite(v) else 0.0
        except Exception:
            out[i, j] = 0.0
    return out

def coherence_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    fs: float = 1.0,
    pairs: Optional[list[tuple[int, int]]] = None,
    **extra: dict,
) -> np.ndarray:
    """Partial coherence: когерентность на резидуалах после регрессии на контроли."""
    control_matrix = extra.get("control_matrix", None)
    sub = data.copy()
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
    return coherence_matrix(sub, lag=lag, control=None, fs=fs, pairs=pairs)

def _dcov_sq(x: np.ndarray, y: np.ndarray) -> float:
    """Квадрат дистанционной ковариации (Székely et al. 2007)."""
    n = x.size
    if n < 4:
        return np.nan
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    return float(np.mean(A * B))


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Дистанционная корреляция.  dCor=0 ⟺ независимость (для конечных моментов)."""
    dcov2 = _dcov_sq(x, y)
    dvar_x = _dcov_sq(x, x)
    dvar_y = _dcov_sq(y, y)
    if not np.isfinite(dcov2) or dvar_x <= 0 or dvar_y <= 0:
        return np.nan
    return float(np.sqrt(max(0.0, dcov2) / np.sqrt(dvar_x * dvar_y)))


def dcor_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    n_vars = int(data.shape[1])
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

    for i, j, value in _try_parallel(_compute_dcor_pair, effective):
        out[i, j] = out[j, i] = value
    return out

def dcor_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **extra: dict,
) -> np.ndarray:
    control_matrix = extra.get("control_matrix", None)
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub, _desc = _residualize_df(data, control=control, control_matrix=control_matrix)
        return dcor_matrix(sub, lag=lag, control=None, pairs=pairs)
    n_cols = len(data.columns)
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
    control: Optional[list[str]] = None,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    lag = int(max(1, lag))
    n_cols = len(data.columns)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
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

def _ordinal_pattern(x: np.ndarray, order: int = 3, delay: int = 1) -> np.ndarray:
    """Возвращает массив порядковых паттернов (Bandt-Pompe) как целых чисел."""
    n = x.size
    idx = np.arange(order) * delay
    if n <= idx[-1]:
        return np.array([], dtype=int)
    patterns = []
    for t in range(n - idx[-1]):
        w = x[t + idx]
        if not np.all(np.isfinite(w)):
            patterns.append(-1)
            continue
        # Кодируем перестановку как единственное целое
        rank = np.argsort(np.argsort(w, kind="mergesort"), kind="mergesort")
        code = 0
        for r in rank:
            code = code * order + int(r)
        patterns.append(code)
    return np.array(patterns, dtype=int)


def _ordinal_mi(x: np.ndarray, y: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Permutation mutual information через частоты совместных паттернов."""
    px = _ordinal_pattern(x, order=order, delay=delay)
    py = _ordinal_pattern(y, order=order, delay=delay)
    n = min(px.size, py.size)
    if n < 20:
        return np.nan
    px, py = px[:n], py[:n]
    valid = (px >= 0) & (py >= 0)
    px, py = px[valid], py[valid]
    n = px.size
    if n < 20:
        return np.nan
    from collections import Counter
    cxy = Counter(zip(px.tolist(), py.tolist()))
    cx = Counter(px.tolist())
    cy = Counter(py.tolist())
    mi = 0.0
    for (a, b), nab in cxy.items():
        p_ab = nab / n
        p_a = cx[a] / n
        p_b = cy[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * np.log2(p_ab / (p_a * p_b))
    return float(max(0.0, mi))


def ordinal_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    order: int = 3,
    delay: int = 1,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    n_vars = int(data.shape[1])
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    out = _init_matrix(n_vars, 0.0, diag=0.0)
    X = _prepare_numpy(data)
    for i, j in effective:
        xi = X[:, i]
        xj = X[:, j]
        mask = np.isfinite(xi) & np.isfinite(xj)
        if int(mask.sum()) < 20:
            out[i, j] = out[j, i] = 0.0
            continue
        v = _ordinal_mi(xi[mask], xj[mask], order=order, delay=delay)
        out[i, j] = out[j, i] = float(v) if np.isfinite(v) else 0.0
    return out

def ordinal_matrix_directed(
    data: pd.DataFrame,
    lag: int = 1,
    control: Optional[list[str]] = None,
    order: int = 3,
    delay: int = 1,
    pairs: Optional[list[tuple[int, int]]] = None,
    **_: dict,
) -> np.ndarray:
    lag = int(max(1, lag))
    n_cols = len(data.columns)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(data)
    for i, j in effective:
        x = X[:, i]
        y_shifted = np.roll(X[:, j], -lag)
        if lag > 0:
            y_shifted[-lag:] = np.nan
        mask = np.isfinite(x) & np.isfinite(y_shifted)
        if int(mask.sum()) < 20:
            continue
        v = _ordinal_mi(x[mask], y_shifted[mask], order=order, delay=delay)
        out[i, j] = float(v) if np.isfinite(v) else 0.0
    return out
