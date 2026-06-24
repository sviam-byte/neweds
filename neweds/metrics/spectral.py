"""Спектральные и многомасштабные метрики связности."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._shared import (
    _enforce_pairwise_guardrail,
    _fast_corr_matrix,
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
    return coherence_matrix(sub, lag=lag, control=None, fs=fs, pairs=pairs, **extra)


def _prepare_wavelet_series(values: np.ndarray) -> np.ndarray | None:
    """Интерполирует пропуски и стандартизует один ряд перед Haar-разложением."""
    x = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(x)
    if int(finite.sum()) < 16:
        return None
    if not finite.all():
        positions = np.arange(x.size)
        x = np.interp(positions, positions[finite], x[finite])
    x = x - float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    return x / scale


def _haar_detail_coefficients(
    values: np.ndarray,
    *,
    levels: int | None = None,
    min_coefficients: int = 8,
) -> list[np.ndarray]:
    """Возвращает detail-коэффициенты ортонормированного Haar DWT по масштабам."""
    approx = np.asarray(values, dtype=np.float64).ravel()
    max_levels = (
        int(max(1, levels))
        if levels is not None
        else int(max(1, np.floor(np.log2(max(2, approx.size))) - 3))
    )
    details: list[np.ndarray] = []
    norm = np.sqrt(2.0)
    for _ in range(max_levels):
        even_size = int(approx.size // 2 * 2)
        if even_size < 2:
            break
        pairs = approx[:even_size].reshape(-1, 2)
        detail = (pairs[:, 0] - pairs[:, 1]) / norm
        approx = (pairs[:, 0] + pairs[:, 1]) / norm
        if detail.size < int(max(4, min_coefficients)):
            break
        details.append(detail)
    return details


def _wavelet_pair_score(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> float:
    """Coefficient-count weighted mean r² across matching Haar detail scales."""
    scores: list[float] = []
    weights: list[float] = []
    for x_detail, y_detail in zip(left, right):
        n = int(min(x_detail.size, y_detail.size))
        if n < 4:
            continue
        x = x_detail[:n]
        y = y_detail[:n]
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std <= 1e-12 or y_std <= 1e-12:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr):
            scores.append(float(np.clip(corr * corr, 0.0, 1.0)))
            weights.append(float(n))
    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def wavelet_matrix(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    levels: int | None = None,
    min_coefficients: int = 8,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """Многомасштабная Haar-связность как среднее r² detail-коэффициентов.

    Результат лежит в ``[0, 1]`` и симметричен. Это компактная DWT-метрика
    сходства по масштабам, а не continuous-wavelet coherence.
    """
    del lag, control
    n_vars = int(data.shape[1])
    _enforce_pairwise_guardrail(
        n_vars,
        pairs,
        directed=False,
        metric_name=str(extra.get("_metric_name", "wavelet_full")),
        max_pairwise_pairs=extra.get("max_pairwise_pairs"),
        performance_guardrails=extra.get("performance_guardrails", True),
    )
    effective = _get_effective_pairs(n_vars, pairs, directed=False)
    out = _init_matrix(n_vars, np.nan, diag=1.0)
    values = _prepare_numpy(data)
    coefficients: list[list[np.ndarray]] = []
    for idx in range(n_vars):
        series = _prepare_wavelet_series(values[:, idx])
        coefficients.append(
            []
            if series is None
            else _haar_detail_coefficients(
                series,
                levels=levels,
                min_coefficients=min_coefficients,
            )
        )
    if pairs is None:
        score_sum = np.zeros((n_vars, n_vars), dtype=np.float64)
        weight_sum = np.zeros((n_vars, n_vars), dtype=np.float64)
        max_scales = max((len(item) for item in coefficients), default=0)
        for level in range(max_scales):
            valid_indices = [idx for idx, item in enumerate(coefficients) if len(item) > level]
            if len(valid_indices) < 2:
                continue
            detail_matrix = np.column_stack(
                [coefficients[idx][level] for idx in valid_indices]
            )
            corr = _fast_corr_matrix(detail_matrix)
            squared = np.clip(corr * corr, 0.0, 1.0)
            weight = float(detail_matrix.shape[0])
            index = np.ix_(valid_indices, valid_indices)
            score_sum[index] += squared * weight
            weight_sum[index] += weight
        for i, j in effective:
            if weight_sum[i, j] > 0:
                out[i, j] = out[j, i] = score_sum[i, j] / weight_sum[i, j]
    else:
        for i, j in effective:
            score = _wavelet_pair_score(coefficients[i], coefficients[j])
            out[i, j] = out[j, i] = score
    return out


def wavelet_matrix_partial(
    data: pd.DataFrame,
    lag: int = 1,
    control: list[str] | None = None,
    levels: int | None = None,
    min_coefficients: int = 8,
    pairs: list[tuple[int, int]] | None = None,
    **extra: dict,
) -> np.ndarray:
    """Haar multiscale coupling на резидуалах после регрессии на контроли."""
    control_matrix = extra.get("control_matrix")
    sub = data.copy()
    if control_matrix is not None or (control is not None and len(control) > 0):
        sub, _desc = _residualize_df(sub, control=control, control_matrix=control_matrix)
    forwarded = dict(extra)
    forwarded["_metric_name"] = "wavelet_partial"
    return wavelet_matrix(
        sub,
        lag=lag,
        control=None,
        levels=levels,
        min_coefficients=min_coefficients,
        pairs=pairs,
        **forwarded,
    )


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
    register_metric(
        "wavelet_full",
        category="spectral",
        description=(
            "Haar multiscale coupling: coefficient-count weighted mean r² "
            "between detail coefficients, [0, 1]."
        ),
        experimental=True,
    )(wavelet_matrix)
    register_metric(
        "wavelet_partial",
        category="spectral",
        description="Частичная Haar multiscale coupling после residualization на контролях.",
        supports_control=True,
        experimental=True,
        partial_mode="explicit_controls_residualization",
    )(wavelet_matrix_partial)


__all__ = [
    "coherence_matrix",
    "coherence_matrix_partial",
    "wavelet_matrix",
    "wavelet_matrix_partial",
]
