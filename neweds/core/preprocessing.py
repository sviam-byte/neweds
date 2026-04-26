#!/usr/bin/env python3

"""
Модуль предобработки временных рядов.

Каждая функция:
  - принимает DataFrame (time × channels)
  - возвращает (DataFrame, описание шага)
  - не мутирует входные данные
"""

import numpy as np
import pandas as pd


def spatial_bin_channels(
    df: pd.DataFrame,
    bin_size: int = 5,
    method: str = "mean",
) -> tuple[pd.DataFrame, str]:
    """Пространственная бинизация каналов (ROI/voxel → агрегированные каналы).

    Параметры:
        df: Матрица ``time × channels``.
        bin_size: Сколько соседних каналов объединять в один бин.
        method: Агрегация внутри бина: ``mean`` | ``median`` | ``sum``.

    Возвращает:
        ``(out_df, desc)``, где ``out_df`` имеет колонки ``bin_0..bin_N``.
    """
    # Для bin_size<=1 сохраняем обратную совместимость: исходные ряды без изменений.
    if int(bin_size) <= 1:
        return df.copy(), "spatial binning skipped"

    cols = list(df.columns)
    groups = [cols[i : i + int(bin_size)] for i in range(0, len(cols), int(bin_size))]
    agg_method = str(method or "mean").strip().lower()

    out: dict[str, pd.Series] = {}
    for i, group_cols in enumerate(groups):
        block = df[group_cols]

        if agg_method == "median":
            out[f"bin_{i}"] = block.median(axis=1)
        elif agg_method == "sum":
            out[f"bin_{i}"] = block.sum(axis=1)
        else:
            # Любое неизвестное значение трактуем как mean, чтобы не падать в UI.
            out[f"bin_{i}"] = block.mean(axis=1)

    out_df = pd.DataFrame(out, index=df.index)
    desc = f"spatial binning: size={int(bin_size)}, method={agg_method}, bins={len(out_df.columns)}"
    return out_df, desc


# 6.1 Нормализация масштаба


def normalize_zscore(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Классический z-score по каждому каналу (mean=0, std=1)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            m, sd = s.mean(), s.std()
            if np.isfinite(sd) and sd > 1e-12:
                df[col] = (s - m) / sd
    return df, "z-score (mean/std)"


def normalize_robust_zscore(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Robust z-score через медиану и MAD (устойчив к выбросам)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            med = s.median()
            mad = (s - med).abs().median() * 1.4826 + 1e-12
            df[col] = (s - med) / mad
    return df, "robust z-score (median/MAD)"


# 6.2 Детрендинг / удаление дрейфа


def detrend_linear(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Линейный детрендинг каждого канала (OLS x ~ t)."""
    df = df.copy()
    n = len(df)
    if n < 3:
        return df, "detrend_linear (skipped: T<3)"
    t = np.arange(n, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)
    A = np.c_[np.ones(n), t]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            y = df[col].to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            if mask.sum() < 3:
                continue
            try:
                beta, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
                y[mask] -= A[mask] @ beta
                df[col] = y
            except Exception:
                pass
    return df, "detrend_linear"


def detrend_highpass(df: pd.DataFrame, window: int = 50) -> tuple[pd.DataFrame, str]:
    """High-pass через вычитание скользящего среднего."""
    df = df.copy()
    window = max(3, int(window))
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            trend = s.rolling(window=window, center=True, min_periods=1).mean()
            df[col] = s - trend
    return df, f"highpass (window={window})"


# 6.3 Обработка выбросов


def clip_outliers_robust(df: pd.DataFrame, z_threshold: float = 5.0) -> tuple[pd.DataFrame, str]:
    """Клиппинг по robust z-score: значения с |robust_z| > threshold → порог."""
    df = df.copy()
    total_clipped = 0
    total_points = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            med = s.median()
            mad = (s - med).abs().median() * 1.4826 + 1e-12
            rz = (s - med) / mad
            mask = rz.abs() > z_threshold
            n_clip = int(mask.sum())
            total_clipped += n_clip
            total_points += int(s.notna().sum())
            if n_clip > 0:
                upper = med + z_threshold * mad
                lower = med - z_threshold * mad
                df[col] = s.clip(lower=lower, upper=upper)
    pct = 100.0 * total_clipped / max(1, total_points)
    desc = f"clip_outliers (robust_z>{z_threshold}): {total_clipped} pts ({pct:.2f}%)"
    return df, desc


def replace_spikes_median(
    df: pd.DataFrame, z_threshold: float = 5.0, window: int = 5
) -> tuple[pd.DataFrame, str]:
    """Замена спайков (|Δx| > threshold*MAD) на локальную медиану."""
    df = df.copy()
    total_replaced = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            dx = s.diff().abs()
            med_dx = dx.median()
            mad_dx = (dx - med_dx).abs().median() * 1.4826 + 1e-12
            spike_mask = (dx / mad_dx) > z_threshold
            n_spikes = int(spike_mask.sum())
            total_replaced += n_spikes
            if n_spikes > 0:
                local_med = s.rolling(window=window, center=True, min_periods=1).median()
                df.loc[spike_mask, col] = local_med[spike_mask]
    return df, f"spike_replace (Δx>{z_threshold}*MAD): {total_replaced} pts"


# 6.4 Работа с автокорреляцией


def remove_ar1(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Prewhitening: удаление AR(1) компоненты для каждого канала.

    x_new[t] = x[t] - φ·x[t-1], где φ = AR(1) коэффициент.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            vals = s.values
            if vals.size < 10:
                continue
            x0 = vals[:-1]
            x1 = vals[1:]
            mask = np.isfinite(x0) & np.isfinite(x1)
            if mask.sum() < 5:
                continue
            phi = float(np.corrcoef(x0[mask], x1[mask])[0, 1])
            if np.isfinite(phi) and abs(phi) > 0.01:
                residual = vals.copy()
                residual[1:] = vals[1:] - phi * vals[:-1]
                residual[0] = vals[0]
                df[col] = residual
    return df, "remove_ar1 (prewhitening)"


# Существующие функции (сохраняем обратную совместимость)


def additional_preprocessing(
    df: pd.DataFrame,
    unique_thresh: float = 0.05,
    *,
    low_variance_eps: float = 1e-12,
    skip_unique_filter: bool = False,
    drop_all_nan: bool = True,
) -> pd.DataFrame:
    """
    Дополнительная предобработка данных:
    - Удаление вырожденных колонок (all-NaN, нулевая дисперсия/диапазон)
    - Опционально: legacy-фильтр по числу уникальных значений

    Args:
        df: Исходный DataFrame
        unique_thresh: Порог уникальности для legacy-фильтра
        low_variance_eps: Порог дисперсии/диапазона для удаления вырожденных колонок
        skip_unique_filter: Если True, не применять фильтр по unique ratio
        drop_all_nan: Если True, удалять колонки, где после coercion нет finite-значений

    Returns:
        pd.DataFrame: Предобработанный DataFrame
    """
    df = df.copy()

    cols_to_drop: list[str] = []
    for col in list(df.columns):
        s = pd.to_numeric(df[col], errors="coerce")
        vals = s.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(vals)
        n_finite = int(finite.sum())

        if drop_all_nan and n_finite == 0:
            cols_to_drop.append(col)
            continue

        if n_finite == 0:
            continue

        finite_vals = vals[finite]
        var = float(np.nanvar(finite_vals))
        vrange = float(np.nanmax(finite_vals) - np.nanmin(finite_vals))

        # Для voxel-like рядов используем численно устойчивый критерий вырожденности.
        is_degenerate = (
            (not np.isfinite(var))
            or (var <= float(low_variance_eps))
            or (vrange <= float(low_variance_eps))
        )
        if is_degenerate:
            cols_to_drop.append(col)
            continue

        # Legacy-логика: для классических табличных данных можно отсекать
        # почти-константные признаки по доле уникальных значений.
        if not skip_unique_filter:
            uniq_ratio = float(pd.Series(finite_vals).nunique(dropna=True)) / max(1, n_finite)
            if uniq_ratio < float(unique_thresh):
                cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=list(dict.fromkeys(cols_to_drop)), errors="ignore")

    return df


def configure_warnings(quiet: bool = False) -> None:
    """
    Настраивает предупреждения без глобального подавления.

    Args:
        quiet: Если True, подавляет все предупреждения
    """
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="statsmodels.tsa.stattools",
    )
    warnings.filterwarnings(
        "ignore",
        message="nperseg = 256 is greater than input length",
    )
    if quiet:
        warnings.filterwarnings("ignore")


def spatial_grid_bin_fmri(
    volume4d: np.ndarray,
    grid_size: int = 5,
    method: str = "mean",
) -> pd.DataFrame:
    """Пространственная бинизация fMRI 4D-объёма в матрицу ``time × bins``.

    Args:
        volume4d: Массив формы ``(X, Y, Z, T)``.
        grid_size: Размер пространственного куба агрегации.
        method: Агрегатор внутри куба: ``mean`` | ``median`` | ``sum``.

    Returns:
        DataFrame формы ``(T, K)``, где ``K`` — число spatial-бинов.
    """
    arr = np.asarray(volume4d)
    if arr.ndim != 4:
        raise ValueError(f"spatial_grid_bin_fmri expects 4D array, got shape={arr.shape}")

    X, Y, Z, T = arr.shape
    g = max(1, int(grid_size))
    method_eff = str(method or "mean").strip().lower()

    bins: list[np.ndarray] = []
    for x in range(0, X, g):
        for y in range(0, Y, g):
            for z in range(0, Z, g):
                block = arr[x : x + g, y : y + g, z : z + g, :]
                flat = block.reshape(-1, T)
                if flat.size == 0:
                    continue

                if method_eff == "median":
                    ts = np.nanmedian(flat, axis=0)
                elif method_eff == "sum":
                    ts = np.nansum(flat, axis=0)
                else:
                    ts = np.nanmean(flat, axis=0)
                bins.append(np.asarray(ts, dtype=np.float64))

    if not bins:
        return pd.DataFrame(index=np.arange(T))

    mat = np.asarray(bins, dtype=np.float64)
    return pd.DataFrame(mat.T, columns=[f"bin_{i}" for i in range(mat.shape[0])])
