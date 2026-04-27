"""Порядковые метрики: перестановочная взаимная информация (Bandt-Pompe)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ._shared import (
    _get_effective_pairs,
    _init_matrix,
    _prepare_numpy,
)
from .registry import register_metric


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
    control: list[str] | None = None,
    order: int = 3,
    delay: int = 1,
    pairs: list[tuple[int, int]] | None = None,
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
    control: list[str] | None = None,
    order: int = 3,
    delay: int = 1,
    pairs: list[tuple[int, int]] | None = None,
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


def _register() -> None:
    register_metric(
        "ordinal_full",
        category="ordinal",
        description="Ordinal MI по Bandt–Pompe (порядковые паттерны).",
        stable=True,
    )(ordinal_matrix)
    register_metric(
        "ordinal_directed",
        category="ordinal",
        description="Направленная ordinal MI.",
        directed=True,
        experimental=True,
    )(ordinal_matrix_directed)


__all__ = ["ordinal_matrix", "ordinal_matrix_directed"]
