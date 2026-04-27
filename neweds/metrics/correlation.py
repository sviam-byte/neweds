"""Метрики семейства корреляций: Пирсон, Спирмен, Кендалл, частичная, лаговая, H².

Всё в этом модуле — линейная или ранговая корреляция. Сюда сознательно не тащатся
statsmodels и scipy.signal: импорт модуля остаётся лёгким.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._shared import (
    _corr_1d,
    _fast_corr_matrix,
    _get_effective_pairs,
    _init_matrix,
    _iter_pairs,
    _kendall_1d,
    _prepare_numpy,
    _residualize_df,
    _spearman_1d,
)
from .registry import register_metric


def correlation_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
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
    X = _prepare_numpy(data)
    if not np.isfinite(X).all():
        return data.corr().values
    return _fast_corr_matrix(X)


def spearman_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    """Матрица корреляции Спирмена (ранговая монотонная связь)."""
    n_cols = int(data.shape[1])
    if pairs is not None:
        X = _prepare_numpy(data)
        out = _init_matrix(n_cols, 0.0, diag=1.0)
        for i, j in _iter_pairs(n_cols, pairs, directed=False):
            val = _spearman_1d(X[:, i], X[:, j])
            out[i, j] = out[j, i] = float(val) if np.isfinite(val) else 0.0
        return out
    try:
        return data.corr(method="spearman").values
    except Exception:
        X = _prepare_numpy(data)
        out = _init_matrix(n_cols, 0.0, diag=1.0)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                val = _spearman_1d(X[:, i], X[:, j])
                out[i, j] = out[j, i] = float(val) if np.isfinite(val) else 0.0
        return out


def kendall_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    """Матрица Kendall tau-b (ранговая согласованность пар)."""
    n_cols = int(data.shape[1])
    if pairs is not None:
        X = _prepare_numpy(data)
        out = _init_matrix(n_cols, 0.0, diag=1.0)
        for i, j in _iter_pairs(n_cols, pairs, directed=False):
            val = _kendall_1d(X[:, i], X[:, j])
            out[i, j] = out[j, i] = float(val) if np.isfinite(val) else 0.0
        return out
    try:
        return data.corr(method="kendall").values
    except Exception:
        X = _prepare_numpy(data)
        out = _init_matrix(n_cols, 0.0, diag=1.0)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                val = _kendall_1d(X[:, i], X[:, j])
                out[i, j] = out[j, i] = float(val) if np.isfinite(val) else 0.0
        return out


def spearman_correlation_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **kwargs: dict,
) -> np.ndarray:
    """Совместимый алиас: ранговая корреляция Спирмена."""
    return spearman_matrix(data, lag=lag, control=control, pairs=pairs, **kwargs)


def kendall_correlation_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **kwargs: dict,
) -> np.ndarray:
    """Совместимый алиас: ранговая корреляция Кендалла tau-b."""
    return kendall_matrix(data, lag=lag, control=control, pairs=pairs, **kwargs)


def partial_correlation_matrix(
    df: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    control_matrix: np.ndarray | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    """Частная корреляция.

    Когда явные controls не заданы — через precision matrix всех каналов.
    Когда заданы — через резидуализацию по controls + Pearson на остатках.
    """
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n_cols = len(cols)
    n_rows = len(df)
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub = df[cols].copy()
        sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
        if pairs is None:
            X_sub = sub.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(X_sub).all():
                return sub.corr().values
            return _fast_corr_matrix(X_sub)
        n = int(len(cols))
        out = _init_matrix(n, 0.0, diag=1.0)
        X = sub.to_numpy(dtype=np.float64, copy=False)
        for i, j in _iter_pairs(n, pairs, directed=False):
            out[i, j] = out[j, i] = _corr_1d(X[:, i], X[:, j])
        return out
    if n_rows <= n_cols + 2:
        logging.warning(
            "Слишком мало данных для Partial Correlation: строк %s <= колонок %s. Возвращаю NaN.",
            n_rows,
            n_cols,
        )
        return np.full((n_cols, n_cols), np.nan)
    if pairs is None or n_cols <= 500:
        try:
            R = _fast_corr_matrix(_prepare_numpy(df[cols]))
            P = np.linalg.pinv(R)
            d = np.sqrt(np.abs(np.diag(P)))
            d[d < 1e-12] = 1.0
            pcor = -P / np.outer(d, d)
            np.fill_diagonal(pcor, 1.0)
            if pairs is not None:
                out = _init_matrix(n_cols, 0.0, diag=1.0)
                for i, j in _iter_pairs(n_cols, pairs, directed=False):
                    out[i, j] = out[j, i] = pcor[i, j]
                return out
            return pcor
        except Exception:
            pass

    out = _init_matrix(n_cols, 0.0, diag=1.0)
    effective = _get_effective_pairs(n_cols, pairs, directed=False)
    X = _prepare_numpy(df[cols])
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


def partial_h2_matrix(
    df: pd.DataFrame, lag: int = 1, control: list[str] | None = None, **kwargs: dict
) -> np.ndarray:
    """Вычисляет квадрат частной корреляции (приближение частного H²)."""
    return partial_correlation_matrix(df, lag=lag, control=control, **kwargs) ** 2


def lagged_directed_correlation(
    df: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    """Направленная лаговая корреляция: M[src, tgt] = corr(src(t), tgt(t+lag))."""
    lag = int(max(1, lag))
    n_cols = len(df.columns)
    X = _prepare_numpy(df)
    T = X.shape[0]
    if lag + 3 >= T:
        return _init_matrix(n_cols, 0.0, diag=0.0)

    x_past = X[: T - lag]
    x_future = X[lag:]

    if pairs is None and n_cols <= 10_000:
        past = x_past.copy()
        future = x_future.copy()
        for arr in (past, future):
            nan_mask = ~np.isfinite(arr)
            if nan_mask.any():
                col_means = np.nanmean(arr, axis=0)
                for j in range(arr.shape[1]):
                    arr[nan_mask[:, j], j] = col_means[j]

        past_centered = past - past.mean(axis=0, keepdims=True)
        fut_centered = future - future.mean(axis=0, keepdims=True)
        sp = np.sqrt((past_centered**2).sum(axis=0, keepdims=True))
        sf = np.sqrt((fut_centered**2).sum(axis=0, keepdims=True))
        sp[sp < 1e-12] = 1.0
        sf[sf < 1e-12] = 1.0
        out = (past_centered / sp).T @ (fut_centered / sf)
        np.fill_diagonal(out, 0.0)
        return out

    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    col_finite_past = np.isfinite(x_past)
    col_finite_future = np.isfinite(x_future)
    for i, j in effective:
        mask = col_finite_past[:, i] & col_finite_future[:, j]
        if int(mask.sum()) < 4:
            continue
        out[i, j] = _corr_1d(x_past[mask, i], x_future[mask, j])
    return out


def _h2_full(
    df: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    **kw,
) -> np.ndarray:
    return correlation_matrix(df, lag=lag, control=control, **kw) ** 2


def _h2_directed(
    df: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    **kw,
) -> np.ndarray:
    return lagged_directed_correlation(df, lag=lag, control=control, **kw) ** 2


def _register() -> None:
    register_metric(
        "correlation_full",
        category="correlation",
        description="Корреляция Пирсона, [-1, 1].",
        stable=True,
    )(correlation_matrix)
    register_metric(
        "correlation_spearman",
        category="correlation",
        description="Ранговая корреляция Спирмена, [-1, 1].",
        stable=True,
    )(spearman_correlation_matrix)
    register_metric(
        "correlation_kendall",
        category="correlation",
        description="Кендалл tau-b — ранговая согласованность, [-1, 1].",
        stable=True,
    )(kendall_correlation_matrix)
    register_metric(
        "correlation_partial",
        category="correlation",
        description="Корреляция Пирсона при контроле остальных переменных.",
        supports_control=True,
        partial_mode="precision_matrix",
        stable=True,
    )(partial_correlation_matrix)
    register_metric(
        "correlation_directed",
        category="correlation",
        description="Лаговая направленная корреляция между каналами.",
        directed=True,
    )(lagged_directed_correlation)
    register_metric(
        "h2_full",
        category="correlation",
        description="Квадрат корреляции Пирсона (улавливает нелинейность).",
    )(_h2_full)
    register_metric(
        "h2_partial",
        category="correlation",
        description="H² при контроле других каналов.",
        supports_control=True,
        partial_mode="precision_matrix",
    )(partial_h2_matrix)
    register_metric(
        "h2_directed",
        category="correlation",
        description="Направленная H².",
        directed=True,
    )(_h2_directed)


__all__ = [
    "correlation_matrix",
    "spearman_matrix",
    "kendall_matrix",
    "spearman_correlation_matrix",
    "kendall_correlation_matrix",
    "partial_correlation_matrix",
    "partial_h2_matrix",
    "lagged_directed_correlation",
]
