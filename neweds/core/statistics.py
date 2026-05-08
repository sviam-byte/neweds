"""Статистические утилиты общего назначения."""

from __future__ import annotations

import numpy as np


def as_float64_1d(x) -> np.ndarray:
    """1D float64 без NaN/inf."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """FDR-коррекция Бенджамини-Хохберга. Возвращает q-значения той же формы."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q
    pv = p[mask].ravel()
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_raw = ranked * m / (np.arange(1, m + 1))
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.clip(q_mono, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = q_mono
    q[mask] = out
    return q


def apply_pvalue_correction_matrix(mat: np.ndarray, directed: bool) -> np.ndarray:
    """FDR-коррекция матрицы p-значений (внедиагональные элементы)."""
    M = np.array(mat, dtype=float, copy=True)
    n = M.shape[0]
    if n == 0:
        return M
    fmask = np.isfinite(M)
    np.fill_diagonal(fmask, False)
    if not directed:
        tri = np.triu(fmask, 1)
        q = fdr_bh(M[tri])
        M[tri] = q
        M = M + M.T
        np.fill_diagonal(M, 0.0)
        return M
    else:
        q = fdr_bh(M[fmask])
        M[fmask] = q
        np.fill_diagonal(M, 0.0)
        return M


def residualize_series(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Остатки регрессии y ~ X (со сдвигом)."""
    y = np.asarray(y, dtype=float)
    if X is None or np.size(X) == 0:
        return y - np.nanmean(y)
    X = np.asarray(X, dtype=float)

    mask = np.isfinite(y)
    if X.ndim == 1:
        mask &= np.isfinite(X)
    else:
        mask &= np.all(np.isfinite(X), axis=1)

    y2, X2 = y[mask], X[mask]
    if y2.size < 5:
        return y - np.nanmean(y)

    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    X2 = np.column_stack([np.ones(len(X2)), X2])

    try:
        beta, *_ = np.linalg.lstsq(X2, y2, rcond=None)
        resid = np.full_like(y, np.nan, dtype=float)
        resid[mask] = y2 - (X2 @ beta)
        m = np.nanmean(resid)
        resid = np.where(np.isfinite(resid), resid, m)
        return resid
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return y - np.nanmean(y)


def lag_quality(variant: str, mat: np.ndarray, is_pvalue: bool) -> float:
    """Скалярная метрика качества лага: больше — лучше."""
    if mat is None or not isinstance(mat, np.ndarray) or mat.size == 0:
        return np.nan
    n = mat.shape[0]
    if n < 2:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    vals = mat[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if is_pvalue:
        vals = np.clip(vals, 1e-12, 1.0)
        return float(np.mean(-np.log10(vals)))
    return float(np.mean(np.abs(vals)))


def pair_score(
    variant: str, mat: np.ndarray, i: int, j: int, is_directed: bool, is_pvalue: bool
) -> float:
    """Скалярная оценка пары (i,j) из матрицы связности."""
    if mat is None or not isinstance(mat, np.ndarray) or mat.size == 0:
        return float("nan")
    n = int(mat.shape[0])
    if i < 0 or j < 0 or i >= n or j >= n or i == j:
        return float("nan")
    try:
        if is_directed:
            v = float(mat[i, j])
        else:
            v = float(max(abs(float(mat[i, j])), abs(float(mat[j, i]))))
        if is_pvalue:
            if not np.isfinite(v):
                return float("nan")
            return float(1.0 - np.clip(v, 0.0, 1.0))
        return float(abs(v)) if np.isfinite(v) else float("nan")
    except (TypeError, ValueError, IndexError):
        return float("nan")


def select_best_median_worst(items: list[dict], *, key: str = "metric") -> dict:
    """Индексы лучшего, медианного и худшего элемента по ключу key."""
    if not items:
        return {"best": None, "median": None, "worst": None}

    vals: list[tuple[int, float]] = []
    for i, it in enumerate(items):
        try:
            v = float(it.get(key, float("nan")))
        except (TypeError, ValueError):
            v = float("nan")
        if np.isfinite(v):
            vals.append((i, v))

    if not vals:
        return {"best": None, "median": None, "worst": None}

    vals_sorted = sorted(vals, key=lambda t: t[1])
    worst_i = int(vals_sorted[0][0])
    best_i = int(vals_sorted[-1][0])
    med_val = float(np.median([v for _, v in vals_sorted]))
    median_i = int(min(vals_sorted, key=lambda t: abs(t[1] - med_val))[0])
    return {"best": best_i, "median": median_i, "worst": worst_i}
