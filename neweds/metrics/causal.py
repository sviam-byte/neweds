"""Causal metrics: Granger F-test, Transfer Entropy.

statsmodels (VAR / grangercausalitytests) грузится лениво — только когда метрика реально
вычисляется.
"""

from __future__ import annotations

import importlib
import logging
from collections import Counter

import numpy as np
import pandas as pd

from ..defaults import DEFAULT_BINS, DEFAULT_MAX_LAG, PYINFORM_AVAILABLE
from ._shared import (
    _get_effective_pairs,
    _init_matrix,
    _iter_pairs,
    _prepare_numpy,
    _residualize_df,
    _try_parallel,
    get_module_seed,
)
from .registry import register_metric


def granger_matrix(
    df: pd.DataFrame,
    lag: int = DEFAULT_MAX_LAG,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    from statsmodels.tsa.stattools import grangercausalitytests

    n_cols = int(df.shape[1])
    out = _init_matrix(n_cols, 1.0, diag=0.0)
    columns = df.columns.tolist()
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(df)
    col_finite = np.isfinite(X)
    min_obs = lag * 2 + 5

    def _compute_granger_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        src, tgt = pair
        mask = col_finite[:, src] & col_finite[:, tgt]
        n_valid = int(mask.sum())
        if n_valid <= min_obs:
            return src, tgt, 1.0
        pair_df = pd.DataFrame(
            {columns[tgt]: X[mask, tgt], columns[src]: X[mask, src]}, copy=False
        )
        try:
            tests = grangercausalitytests(pair_df, maxlag=int(lag), verbose=False)
            p_values = [float(tests[l][0]["ssr_ftest"][1]) for l in range(1, int(lag) + 1)]
            p_min = min(p_values) if p_values else 1.0
            p_corr = min(1.0, p_min * max(1, len(p_values)))
            return src, tgt, p_corr
        except Exception:
            return src, tgt, 1.0

    for src, tgt, value in _try_parallel(_compute_granger_pair, effective):
        out[src, tgt] = value
    return out


def granger_matrix_partial(
    df: pd.DataFrame,
    lag: int = DEFAULT_MAX_LAG,
    control: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    from statsmodels.tsa.vector_ar.var_model import VAR

    control_matrix = extra.get("control_matrix")
    if control_matrix is not None:
        sub = df.copy()
        sub, _ = _residualize_df(sub, control=control, control_matrix=control_matrix)
        return granger_matrix(sub, lag=lag, control=None, pairs=pairs)
    columns = list(df.columns)
    n_cols = len(columns)
    out = _init_matrix(n_cols, 1.0, diag=0.0)
    if len(df) <= n_cols + 2:
        logging.warning(
            "Слишком мало данных для Granger partial: строк %s <= колонок %s. Возвращаю NaN.",
            len(df),
            n_cols,
        )
        return out
    if n_cols > 50 and control is None:
        logging.warning("[granger_partial] N=%d > 50, VAR infeasible. Fallback.", n_cols)
        return granger_matrix(df, lag=lag, control=None, pairs=pairs)

    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    p = int(max(1, lag))

    if control is None and n_cols <= 40:
        sub_all = df.dropna()
        if sub_all.shape[0] >= max(30, 5 * p * n_cols):
            try:
                result = VAR(sub_all).fit(maxlags=p, ic=None, trend="c")
                for src_i, tgt_j in effective:
                    try:
                        causality = result.test_causality(
                            caused=columns[tgt_j], causing=[columns[src_i]], kind="f"
                        )
                        out[src_i, tgt_j] = (
                            float(causality.pvalue) if np.isfinite(causality.pvalue) else 1.0
                        )
                    except Exception:
                        out[src_i, tgt_j] = 1.0
                return out
            except Exception:
                pass

    for src_i, tgt_j in effective:
        src = columns[src_i]
        tgt = columns[tgt_j]
        control_cols = (
            control if control is not None else [c for c in columns if c not in (src, tgt)]
        )
        control_cols = [c for c in control_cols if c in df.columns and c not in (src, tgt)]
        use_cols = [tgt, src] + control_cols
        sub = df[use_cols].dropna()
        if sub.shape[0] < max(30, 5 * p * len(use_cols)):
            continue
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


def compute_te_jitter(
    source: np.ndarray, target: np.ndarray, lag: int = 1, bins: int = DEFAULT_BINS
) -> float:
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
            rng = np.random.default_rng(get_module_seed())
            scale = (np.nanstd(x) if np.nanstd(x) > 0 else 1.0) * 1e-10
            x = x + rng.normal(0.0, scale, size=x.shape)
        return x

    def discretize_quantile(series: np.ndarray, num_bins: int) -> np.ndarray:
        s_full = _add_tiny_jitter(_zscore_1d(np.asarray(series, dtype=np.float64).ravel()))
        s = s_full[np.isfinite(s_full)]
        if s.size == 0:
            return np.array([], dtype=int)
        if float(np.nanmin(s)) == float(np.nanmax(s)):
            return np.zeros(s_full.size, dtype=int)
        edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(num_bins) + 1)))
        if edges.size <= 2:
            return np.zeros(s_full.size, dtype=int)
        edges[-1] = np.nextafter(edges[-1], edges[-1] + 1.0)
        disc = np.digitize(s_full, bins=edges[1:-1], right=False)
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
    control: list[str] | None = None,
    bins: int = DEFAULT_BINS,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    n_cols = int(df.shape[1])
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    effective = _get_effective_pairs(n_cols, pairs, directed=True)
    X = _prepare_numpy(df)
    col_finite = np.isfinite(X)

    def _compute_te_pair(pair: tuple[int, int]) -> tuple[int, int, float]:
        src, tgt = pair
        mask = col_finite[:, src] & col_finite[:, tgt]
        n_valid = int(mask.sum())
        if n_valid <= lag:
            return src, tgt, 0.0
        v = compute_te_jitter(X[mask, src], X[mask, tgt], lag=lag, bins=bins)
        return src, tgt, float(v) if np.isfinite(v) else 0.0

    for src, tgt, value in _try_parallel(_compute_te_pair, effective, heavy=True):
        out[src, tgt] = value
    return out


def transfer_entropy_matrix_partial(
    df: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    bins: int = DEFAULT_BINS,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    cols = list(df.columns)
    n_cols = len(cols)
    out = _init_matrix(n_cols, 0.0, diag=0.0)
    X = _prepare_numpy(df)
    control_matrix = extra.get("control_matrix")

    def residualize(y, x_ctrl):
        if x_ctrl is None or x_ctrl.size == 0:
            return y
        x_aug = np.c_[np.ones(len(y)), x_ctrl]
        beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        return y - x_aug @ beta

    it = (
        _iter_pairs(n_cols, pairs, directed=True)
        if pairs is not None
        else [(i, j) for i in range(n_cols) for j in range(n_cols) if i != j]
    )
    for i, j in it:
        src = cols[i]
        tgt = cols[j]
        if control_matrix is not None:
            pair_vals = X[:, [i, j]]
            X_ctrl = np.asarray(control_matrix, dtype=np.float64)
            valid = np.isfinite(pair_vals).all(axis=1)
            if X_ctrl.size and X_ctrl.ndim == 1:
                X_ctrl = X_ctrl.reshape(-1, 1)
            if X_ctrl.shape[1] > 0:
                valid &= np.isfinite(X_ctrl).all(axis=1)
            if int(valid.sum()) <= lag + 1:
                out[i, j] = 0.0
                continue
            src_vals = pair_vals[valid, 0]
            tgt_vals = pair_vals[valid, 1]
            x_ctrl = X_ctrl[valid, :] if X_ctrl.size else np.empty((int(valid.sum()), 0))
            src_res = residualize(src_vals, x_ctrl)
            tgt_res = residualize(tgt_vals, x_ctrl)
            v = compute_te_jitter(src_res, tgt_res, lag=lag, bins=bins)
            out[i, j] = float(v) if np.isfinite(v) else 0.0
            continue
        control_cols = control if control is not None else [c for c in cols if c not in (src, tgt)]
        control_cols = [c for c in control_cols if c in df.columns and c not in (src, tgt)]
        sub = df[[src, tgt] + control_cols].dropna()
        if sub.shape[0] <= lag + 1:
            out[i, j] = 0.0
            continue
        x_ctrl = sub[control_cols].values if control_cols else np.empty((len(sub), 0))
        try:
            src_res = residualize(sub[src].values, x_ctrl)
            tgt_res = residualize(sub[tgt].values, x_ctrl)
            v = compute_te_jitter(src_res, tgt_res, lag=lag, bins=bins)
            out[i, j] = float(v) if np.isfinite(v) else 0.0
        except Exception:
            out[i, j] = 0.0
    return out


def _register() -> None:
    register_metric(
        "granger_full",
        category="causal",
        description="p-values F-теста причинности по Грейнджеру.",
        directed=True,
        pvalue_based=True,
        stable=True,
    )(granger_matrix)
    register_metric(
        "granger_partial",
        category="causal",
        description="Грейнджер после линейной регрессии по контрольным переменным.",
        directed=True,
        pvalue_based=True,
        supports_control=True,
        partial_mode="explicit_controls_residualization",
    )(granger_matrix_partial)
    register_metric(
        "te_full",
        category="information",
        description="Transfer entropy между каналами.",
        directed=True,
        experimental=True,
    )(transfer_entropy_matrix)
    register_metric(
        "te_partial",
        category="information",
        description="Transfer entropy при контроле других каналов.",
        directed=True,
        supports_control=True,
        experimental=True,
        partial_mode="explicit_controls_residualization",
    )(transfer_entropy_matrix_partial)


__all__ = [
    "compute_te_jitter",
    "granger_matrix",
    "granger_matrix_partial",
    "transfer_entropy_matrix",
    "transfer_entropy_matrix_partial",
]
