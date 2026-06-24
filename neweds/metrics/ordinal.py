"""Порядковые метрики: перестановочная взаимная информация (Bandt-Pompe)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ._shared import (
    _enforce_pairwise_guardrail,
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


def _ordinal_mi_from_patterns(px: np.ndarray, py: np.ndarray) -> float:
    """Permutation mutual information from precomputed Bandt-Pompe pattern codes."""
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
    **extra: dict,
) -> np.ndarray:
    n_vars = int(data.shape[1])
    _enforce_pairwise_guardrail(
        n_vars,
        pairs,
        directed=False,
        metric_name="ordinal_full",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    out = _init_matrix(n_vars, 0.0, diag=0.0)
    X = _prepare_numpy(data)
    cached_patterns = None
    if np.isfinite(X).all():
        cached_patterns = [_ordinal_pattern(X[:, i], order=order, delay=delay) for i in range(n_vars)]
    for i, j in effective:
        if cached_patterns is not None:
            v = _ordinal_mi_from_patterns(cached_patterns[i], cached_patterns[j])
        else:
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
    **extra: dict,
) -> np.ndarray:
    lag = int(max(1, lag))
    n_cols = len(data.columns)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    _enforce_pairwise_guardrail(
        n_cols,
        pairs,
        directed=True,
        metric_name="ordinal_directed",
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(data)
    source_patterns = None
    target_patterns = None
    if lag < X.shape[0] and np.isfinite(X).all():
        source_patterns = [
            _ordinal_pattern(X[:-lag, i], order=order, delay=delay) for i in range(n_cols)
        ]
        target_patterns = [
            _ordinal_pattern(X[lag:, j], order=order, delay=delay) for j in range(n_cols)
        ]
    for i, j in effective:
        if source_patterns is not None and target_patterns is not None:
            v = _ordinal_mi_from_patterns(source_patterns[i], target_patterns[j])
        else:
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
