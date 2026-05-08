"""Спектральные метрики: квадратичная когерентность и её частичная версия."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._shared import (
    _get_effective_pairs,
    _init_matrix,
    _prepare_numpy,
    _residualize_df,
)
from .registry import register_metric


def coherence_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    fs: float = 1.0,
    pairs: list[tuple[int, int]] | None = None,
    **_: dict,
) -> np.ndarray:
    """Средняя magnitude-squared coherence по парам каналов через ``scipy.signal.coherence``."""
    import scipy.signal as signal

    fs = fs if np.isfinite(fs) and fs > 0 else 1.0
    n_vars = int(data.shape[1])
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    coh = _init_matrix(n_vars, 0.0, diag=1.0)
    X = _prepare_numpy(data)
    col_finite = np.isfinite(X)
    for i, j in effective:
        mask = col_finite[:, i] & col_finite[:, j]
        n = int(mask.sum())
        if n <= 3:
            continue
        s1 = X[mask, i]
        s2 = X[mask, j]
        nperseg = int(max(8, min(64, n // 2)))
        try:
            _, cxy = signal.coherence(s1, s2, fs=fs, nperseg=nperseg, detrend="constant")
            cxy = np.clip(np.asarray(cxy, dtype=np.float64), 0.0, 1.0)
            cxy[~np.isfinite(cxy)] = np.nan
            coh[i, j] = coh[j, i] = float(np.nanmean(cxy)) if np.isfinite(cxy).any() else 0.0
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"Coherence failed for pair ({i}, {j}): {exc}", stacklevel=2)
            coh[i, j] = coh[j, i] = np.nan
    return coh


def coherence_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    fs: float = 1.0,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """Partial coherence: когерентность на резидуалах после регрессии на контроли."""
    control_matrix = extra.get("control_matrix")
    sub = data.copy()
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
    return coherence_matrix(sub, lag=lag, control=None, fs=fs, pairs=pairs)


def _register() -> None:
    register_metric(
        "coherence_full",
        category="spectral",
        description="Magnitude-squared coherence, [0, 1].",
        stable=True,
    )(coherence_matrix)
    register_metric(
        "coherence_partial",
        category="spectral",
        description="Частичная когерентность с контрольными переменными.",
        supports_control=True,
        partial_mode="explicit_controls_residualization",
    )(coherence_matrix_partial)


__all__ = ["coherence_matrix", "coherence_matrix_partial"]
