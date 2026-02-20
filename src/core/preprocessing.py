#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль предобработки временных рядов.

Каждая функция:
  - принимает DataFrame (time × channels)
  - возвращает (DataFrame, описание шага)
  - не мутирует входные данные
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# 6.1 Нормализация масштаба

def normalize_zscore(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Классический z-score по каждому каналу (mean=0, std=1)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(np.float64)
            m, sd = s.mean(), s.std()
            if np.isfinite(sd) and sd > 1e-12:
                df[col] = (s - m) / sd
    return df, "z-score (mean/std)"


def normalize_robust_zscore(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
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

def detrend_linear(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
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
                y[mask] -= (A[mask] @ beta)
                df[col] = y
            except Exception:
                pass
    return df, "detrend_linear"


def detrend_highpass(df: pd.DataFrame, window: int = 50) -> Tuple[pd.DataFrame, str]:
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

def clip_outliers_robust(df: pd.DataFrame, z_threshold: float = 5.0) -> Tuple[pd.DataFrame, str]:
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


def replace_spikes_median(df: pd.DataFrame, z_threshold: float = 5.0, window: int = 5) -> Tuple[pd.DataFrame, str]:
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

def remove_ar1(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
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

def additional_preprocessing(df: pd.DataFrame, unique_thresh: float = 0.05) -> pd.DataFrame:
    """
    Дополнительная предобработка данных:
    - Удаление почти константных колонок
    - Лог-преобразование для снижения асимметрии

    Args:
        df: Исходный DataFrame
        unique_thresh: Порог уникальности для удаления константных колонок

    Returns:
        pd.DataFrame: Предобработанный DataFrame
    """
    df = df.copy()

    # Удаляем почти константные колонки
    for col in df.columns:
        if len(df[col]) > 0 and pd.api.types.is_numeric_dtype(df[col]):
            uniq_ratio = df[col].nunique() / len(df[col])
            if uniq_ratio < unique_thresh:
                logging.info(
                    f"[Preproc] Столбец {col} почти константный (uniq_ratio={uniq_ratio:.3f}), удаляем."
                )
                df.drop(columns=[col], inplace=True)

    # Лог-преобразование для снижения асимметрии
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and (df[col] > 0).all():
            skew_before = stats.skew(df[col].dropna())
            if not np.isnan(skew_before):
                transformed = np.log(df[col])
                skew_after = stats.skew(transformed.dropna())
                if not np.isnan(skew_after) and abs(skew_after) < abs(skew_before):
                    logging.info(
                        f"[Preproc] Лог-преобразование для {col}: skew {skew_before:.3f} -> {skew_after:.3f}."
                    )
                    df[col] = transformed

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
