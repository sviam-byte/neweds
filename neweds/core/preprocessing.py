#!/usr/bin/env python3

"""
Модуль предобработки временных рядов.

Каждая функция:
  - принимает DataFrame (time × channels)
  - возвращает (DataFrame, описание шага)
  - не мутирует входные данные

Также содержит:
- ``PreprocessReport`` — структурированный отчёт о применённых шагах.
- ``preprocess_timeseries`` — оркестратор всех шагов препроцессинга,
  используется ``data_loader.load_or_generate``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


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
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                logging.warning("Linear detrend skipped for column %s: %s", col, exc)
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


# Дальше — исторические функции; имена сохраняем ради обратной совместимости.


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
    - Опционально: дополнительный фильтр по числу уникальных значений

    Args:
        df: Исходный DataFrame
        unique_thresh: Порог уникальности для дополнительный фильтра
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

        # Историческая логика: для классических табличных данных можно отсекать
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
        raise ValueError(f"spatial_grid_bin_fmri ждёт 4D массив, получил shape={arr.shape}")

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


# ---------------------------------------------------------------------------
# Уровень оркестрации пайплайна: PreprocessReport + preprocess_timeseries.
# Раньше всё это лежало в ``data_loader.py``, теперь по смыслу здесь.
# ---------------------------------------------------------------------------


@dataclass
class PreprocessReport:
    """Структурированный отчёт о предобработке временных рядов.

    Используется в UI/HTML-отчёте, чтобы явно показать применённые шаги.
    """

    enabled: bool = True
    steps_global: list[str] = field(default_factory=list)
    steps_by_column: dict[str, list[str]] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def add(self, msg: str, col: str | None = None) -> None:
        """Добавляет шаг в глобальный список или к конкретной колонке."""
        if col is None:
            self.steps_global.append(msg)
        else:
            self.steps_by_column.setdefault(col, []).append(msg)


def _rank_normalize_1d(x: np.ndarray, *, mode: str = "dense", ties: str = "average") -> np.ndarray:
    """Ранговая нормализация (структурная): значения -> ранги.

    Пример (dense): ``[100, 33, 98, 2] -> [4, 2, 3, 1]``.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out

    ties = str(ties or "average").strip().lower()
    if ties not in {"average", "min", "max", "dense", "ordinal", "first"}:
        ties = "average"

    if ties == "first":
        idx = np.where(m)[0]
        vals = x[idx]
        order = np.lexsort((idx, vals))
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, order.size + 1, dtype=float)
        out[idx] = ranks
    else:
        method = "ordinal" if ties == "ordinal" else ties
        out[m] = stats.rankdata(x[m], method=method)

    mode = str(mode or "dense").strip().lower()
    if mode in {"pct", "percent", "percentile"}:
        denom = max(1.0, float(np.nanmax(out) - 1.0))
        out[m] = (out[m] - 1.0) / denom
    return out


def _apply_outliers_1d(
    x: np.ndarray,
    *,
    rule: str = "robust_z",
    action: str = "mask",
    z: float = 5.0,
    k: float = 1.5,
    abs_thr: float | None = None,
    p_low: float = 0.5,
    p_high: float = 99.5,
    hampel_window: int = 7,
    jump_thr: float | None = None,
    local_median_window: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Находит/обрабатывает выбросы в 1D и возвращает ``(new_x, mask)``."""
    x = np.asarray(x, dtype=float)
    y = x.copy()
    m = np.isfinite(x)
    mask = np.zeros_like(m, dtype=bool)
    rule = str(rule or "robust_z").strip().lower()
    action = str(action or "mask").strip().lower()

    if m.sum() == 0:
        return y, mask

    def _apply_action(msk: np.ndarray) -> None:
        nonlocal y
        if not msk.any():
            return
        if action in {"mask", "nan"}:
            y[msk] = np.nan
            return
        if action in {"median"}:
            med = float(np.nanmedian(y[m]))
            y[msk] = med
            return
        if action in {"local_median"}:
            s = pd.Series(y)
            local = (
                s.rolling(window=int(max(3, local_median_window)), center=True, min_periods=1)
                .median()
                .to_numpy()
            )
            y[msk] = local[msk]
            return
        if action in {"clip", "winsorize"}:
            vals = y[m]
            lo = float(np.nanpercentile(vals, float(p_low)))
            hi = float(np.nanpercentile(vals, float(p_high)))
            y[msk] = np.clip(y[msk], lo, hi)
            return
        y[msk] = np.nan

    vals = y[m]

    if rule in {"z", "zscore"}:
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals)) + 1e-12
        mask[m] = np.abs((vals - mu) / sd) > float(z)
        _apply_action(mask)
        return y, mask

    if rule in {"robust", "robust_z", "mad"}:
        med = float(np.nanmedian(vals))
        mad = float(np.nanmedian(np.abs(vals - med))) * 1.4826 + 1e-12
        mask[m] = np.abs((vals - med) / mad) > float(z)
        _apply_action(mask)
        return y, mask

    if rule in {"iqr"}:
        q1 = float(np.nanpercentile(vals, 25))
        q3 = float(np.nanpercentile(vals, 75))
        iqr = (q3 - q1) + 1e-12
        lo, hi = q1 - float(k) * iqr, q3 + float(k) * iqr
        mask[m] = (vals < lo) | (vals > hi)
        _apply_action(mask)
        return y, mask

    if rule in {"abs", "absolute"}:
        if abs_thr is None or not np.isfinite(abs_thr):
            return y, mask
        mask[m] = np.abs(vals) > float(abs_thr)
        _apply_action(mask)
        return y, mask

    if rule in {"percentile", "pct"}:
        lo = float(np.nanpercentile(vals, float(p_low)))
        hi = float(np.nanpercentile(vals, float(p_high)))
        mask[m] = (vals < lo) | (vals > hi)
        _apply_action(mask)
        return y, mask

    if rule in {"hampel"}:
        s = pd.Series(y)
        w = int(max(3, hampel_window))
        med = s.rolling(window=w, center=True, min_periods=1).median()
        abs_dev = (s - med).abs()
        mad = abs_dev.rolling(window=w, center=True, min_periods=1).median() * 1.4826 + 1e-12
        rz = (s - med) / mad
        mask = np.asarray(np.isfinite(rz) & (rz.abs() > float(z)))
        _apply_action(mask)
        return y, mask

    if rule in {"jump", "diff"}:
        d = np.abs(np.diff(y, prepend=np.nan))
        if jump_thr is None or not np.isfinite(jump_thr):
            dv = d[np.isfinite(d)]
            if dv.size == 0:
                return y, mask
            med = float(np.nanmedian(dv))
            mad = float(np.nanmedian(np.abs(dv - med))) * 1.4826 + 1e-12
            thr = med + float(z) * mad
        else:
            thr = float(jump_thr)
        mask = np.isfinite(d) & (d > thr)
        _apply_action(mask)
        return y, mask

    return y, mask


def preprocess_timeseries(
    df: pd.DataFrame,
    *,
    enabled: bool = True,
    log_transform: bool = False,
    # выбросы
    remove_outliers: bool = True,
    outlier_rule: str = "robust_z",
    outlier_action: str = "mask",
    outlier_z: float = 5.0,
    outlier_k: float = 1.5,
    outlier_abs: float | None = None,
    outlier_p_low: float = 0.5,
    outlier_p_high: float = 99.5,
    outlier_hampel_window: int = 7,
    outlier_jump_thr: float | None = None,
    outlier_local_median_window: int = 7,
    # нормализация
    normalize: bool = True,
    normalize_mode: str = "zscore",
    rank_mode: str = "dense",
    rank_ties: str = "average",
    # пропуски/структурные шаги
    fill_missing: bool = True,
    remove_ar1: bool = False,
    remove_ar_order: int = 1,
    ar_diagnostics: bool = True,
    remove_seasonality: bool = False,
    season_period: int | None = None,
    check_stationarity: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, PreprocessReport]:
    """Предобработка матрицы (можно полностью отключить ``enabled=False``)."""
    from neweds.analysis import stats as analysis_stats

    # ВАЖНО: df может содержать тяжёлые метаданные в attrs (например coords для вокселей).
    # Pandas при доступе к df[col] может делать deepcopy attrs, что взрывает память.
    # Поэтому на время предобработки очищаем attrs и восстанавливаем в конце.
    _saved_attrs = dict(getattr(df, "attrs", {}) or {})
    out = df.copy()
    try:
        out.attrs = {}
    except (AttributeError, TypeError, ValueError) as exc:
        logging.debug("attrs reset via assignment failed; trying clear(): %s", exc)
        try:
            out.attrs.clear()
        except (AttributeError, TypeError, ValueError) as clear_exc:
            logging.debug("attrs clear skipped: %s", clear_exc)
    report = PreprocessReport(enabled=bool(enabled))
    if not enabled:
        report.add("[Preprocess] disabled: using raw numeric matrix as-is.")
        try:
            if _saved_attrs:
                out.attrs.update(_saved_attrs)
        except (AttributeError, TypeError, ValueError) as exc:
            logging.debug("attrs restore skipped for disabled preprocessing: %s", exc)
        return (out, report) if return_report else out

    report.add("[Preprocess] enabled")

    input_format = str(_saved_attrs.get("format") or "").strip().lower()
    source_kind = str(_saved_attrs.get("source_kind") or "").strip().lower()
    has_voxel_coords = _saved_attrs.get("coords") is not None

    is_voxel_like = input_format == "voxel_wide" or source_kind == "h5_4d" or bool(has_voxel_coords)

    if is_voxel_like:
        report.add("[Preprocess] voxel-like input detected: disabling unique-ratio filter")

    before_cols = list(out.columns)
    out = additional_preprocessing(
        out,
        skip_unique_filter=is_voxel_like,
        low_variance_eps=1e-12,
    )
    after_cols = list(out.columns)

    dropped = [c for c in before_cols if c not in after_cols]
    if dropped:
        report.dropped_columns.extend(dropped)
        report.add(f"[Preprocess] dropped degenerate columns: {dropped[:100]}")
        if len(dropped) > 100:
            report.add(f"[Preprocess] ... and {len(dropped) - 100} more dropped columns")

    if out.shape[1] == 0:
        raise ValueError(
            "All features were removed during preprocessing. "
            "Likely cause: too aggressive filtering on voxel/H5 data."
        )

    out = out.fillna(out.mean(numeric_only=True))
    report.add("[Preprocess] fillna: column means")

    if log_transform:
        report.add("[Preprocess] log-transform: applied to positive values")

        def fn(x):
            return np.log(x) if x is not None and not np.isnan(x) and x > 0 else x

        try:
            out = out.map(fn)  # type: ignore[attr-defined]
        except AttributeError:
            out = out.applymap(fn)

    if remove_outliers:
        rule = str(outlier_rule or "robust_z")
        action = str(outlier_action or "mask")
        report.add(f"[Preprocess] outliers: rule={rule}, action={action}")
        total = 0
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if rule in {"robust", "robust_z", "mad"} and num_cols:
            arr = out[num_cols].to_numpy(dtype=np.float64)
            medians = np.nanmedian(arr, axis=0)
            mad = np.nanmedian(np.abs(arr - medians[np.newaxis, :]), axis=0) * 1.4826 + 1e-12
            z_scores = np.abs((arr - medians[np.newaxis, :]) / mad[np.newaxis, :])
            outlier_mask = z_scores > float(outlier_z)
            outlier_mask[~np.isfinite(arr)] = False
            total = int(outlier_mask.sum())
            if total > 0:
                if action in {"mask", "nan"}:
                    arr[outlier_mask] = np.nan
                elif action == "median":
                    for j in range(arr.shape[1]):
                        arr[outlier_mask[:, j], j] = medians[j]
                elif action in {"clip", "winsorize"}:
                    lo = np.nanpercentile(arr, float(outlier_p_low), axis=0)
                    hi = np.nanpercentile(arr, float(outlier_p_high), axis=0)
                    for j in range(arr.shape[1]):
                        col_mask = outlier_mask[:, j]
                        arr[col_mask, j] = np.clip(arr[col_mask, j], lo[j], hi[j])
                else:
                    arr[outlier_mask] = np.nan
                out[num_cols] = arr

            for j, c in enumerate(num_cols):
                n_col = int(outlier_mask[:, j].sum())
                if n_col:
                    report.add(f"[Preprocess] outliers: n={n_col}", col=c)
        else:
            for col in out.columns:
                if not pd.api.types.is_numeric_dtype(out[col]):
                    continue
                x = out[col].astype(float).to_numpy()
                y, msk = _apply_outliers_1d(
                    x,
                    rule=rule,
                    action=action,
                    z=float(outlier_z),
                    k=float(outlier_k),
                    abs_thr=(None if outlier_abs is None else float(outlier_abs)),
                    p_low=float(outlier_p_low),
                    p_high=float(outlier_p_high),
                    hampel_window=int(outlier_hampel_window),
                    jump_thr=(None if outlier_jump_thr is None else float(outlier_jump_thr)),
                    local_median_window=int(outlier_local_median_window),
                )
                n = int(np.sum(msk))
                if n:
                    total += n
                    out[col] = y
                    report.add(f"[Preprocess] outliers: n={n}", col=col)
        report.add(f"[Preprocess] outliers total: {total}")

    if fill_missing:
        report.add("[Preprocess] fill_missing: linear interpolate + bfill/ffill")
        out = (
            out.interpolate(method="linear", limit_direction="both", axis=0)
            .bfill()
            .ffill()
            .fillna(0)
        )

    if remove_ar1:
        p_order = int(max(1, int(remove_ar_order or 1)))
        ac_note: dict = {
            "enabled": True,
            "order": int(p_order),
            "sampled": False,
            "n_series": int(out.shape[1]),
        }

        def _safe_corr_at_lag(x: np.ndarray, lag: int) -> float:
            lag = int(max(1, lag))
            if x.size <= lag + 2:
                return float("nan")
            a = x[:-lag]
            b = x[lag:]
            m = np.isfinite(a) & np.isfinite(b)
            if int(m.sum()) < 5:
                return float("nan")
            aa = a[m]
            bb = b[m]
            sa = float(np.std(aa))
            sb = float(np.std(bb))
            if not np.isfinite(sa) or not np.isfinite(sb) or sa * sb < 1e-12:
                return float("nan")
            return float(np.corrcoef(aa, bb)[0, 1])

        def _phi1_ols(x: np.ndarray) -> float:
            if x.size < 6:
                return float("nan")
            x0 = x[:-1]
            x1 = x[1:]
            m = np.isfinite(x0) & np.isfinite(x1)
            if int(m.sum()) < 5:
                return float("nan")
            a = x0[m]
            b = x1[m]
            den = float(np.dot(a, a))
            if not np.isfinite(den) or den < 1e-12:
                return float("nan")
            return float(np.dot(a, b) / den)

        def _ljung_box_p(x: np.ndarray, lag: int) -> float:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
            except ImportError:
                return float("nan")
            lag = int(max(1, lag))
            xx = np.asarray(x, dtype=float)
            xx = xx[np.isfinite(xx)]
            if xx.size < max(12, 3 * lag):
                return float("nan")
            try:
                res = acorr_ljungbox(xx, lags=[lag], return_df=True)
                pv = float(res["lb_pvalue"].iloc[0])
                return pv if np.isfinite(pv) else float("nan")
            except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
                logging.debug("Ljung-Box diagnostic skipped: %s", exc)
                return float("nan")

        cols_all = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        max_diag_cols = 2000
        cols_diag = cols_all
        if len(cols_all) > max_diag_cols:
            cols_diag = list(cols_all[:max_diag_cols])
            ac_note["sampled"] = True
            ac_note["n_series_sampled"] = int(len(cols_diag))
        else:
            ac_note["n_series_sampled"] = int(len(cols_diag))

        if bool(ar_diagnostics):
            phi1_list = []
            corr_lists_by_lag = {int(k): [] for k in range(1, p_order + 1)}
            lb_lists_by_lag = {int(k): [] for k in range(1, p_order + 1)}

            top = []
            for col in cols_diag:
                x = out[col].astype(float).to_numpy(copy=False)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                phi1 = _phi1_ols(x)
                corr_by_lag_cur = {}
                lb_by_lag_cur = {}
                for lag_k in range(1, p_order + 1):
                    rk = _safe_corr_at_lag(x, lag_k)
                    lbk = _ljung_box_p(x, lag_k)
                    corr_by_lag_cur[int(lag_k)] = rk
                    lb_by_lag_cur[int(lag_k)] = lbk

                if np.isfinite(phi1):
                    phi1_list.append(float(phi1))
                    top.append((abs(float(phi1)), col, float(phi1)))
                for lag_k in range(1, p_order + 1):
                    rk = corr_by_lag_cur.get(int(lag_k))
                    lbk = lb_by_lag_cur.get(int(lag_k))
                    if np.isfinite(rk):
                        corr_lists_by_lag[int(lag_k)].append(float(rk))
                    if np.isfinite(lbk):
                        lb_lists_by_lag[int(lag_k)].append(float(lbk))

            top.sort(reverse=True, key=lambda t: t[0])
            top = top[:3]
            ac_note["examples_cols"] = [t[1] for t in top]

            def _summary(vals: list[float]) -> dict:
                if not vals:
                    return {"n": 0}
                a = np.asarray(vals, dtype=float)
                return {
                    "n": int(a.size),
                    "median": float(np.nanmedian(a)),
                    "mean": float(np.nanmean(a)),
                    "p25": float(np.nanpercentile(a, 25)),
                    "p75": float(np.nanpercentile(a, 75)),
                }

            corr_summary_by_lag = {
                f"lag{int(k)}": _summary(vals) for k, vals in corr_lists_by_lag.items()
            }
            lb_summary_by_lag = {
                f"lag{int(k)}": _summary(vals) for k, vals in lb_lists_by_lag.items()
            }
            frac_lb_bad_by_lag = {
                f"lag{int(k)}": (
                    float(np.mean(np.asarray(vals, dtype=float) < 0.05)) if vals else float("nan")
                )
                for k, vals in lb_lists_by_lag.items()
            }

            ac_note["before"] = {
                "phi1": _summary(phi1_list),
                "corr_lag1": corr_summary_by_lag.get("lag1", {"n": 0}),
                "corr_lagp": corr_summary_by_lag.get(f"lag{int(p_order)}", {"n": 0}),
                "ljungbox_p_lag1": lb_summary_by_lag.get("lag1", {"n": 0}),
                "ljungbox_p_lagp": lb_summary_by_lag.get(f"lag{int(p_order)}", {"n": 0}),
                "frac_lb_p_lt_0_05_lag1": frac_lb_bad_by_lag.get("lag1", float("nan")),
                "frac_lb_p_lt_0_05_lagp": frac_lb_bad_by_lag.get(
                    f"lag{int(p_order)}", float("nan")
                ),
                "corr_by_lag": corr_summary_by_lag,
                "ljungbox_p_by_lag": lb_summary_by_lag,
                "frac_lb_p_lt_0_05_by_lag": frac_lb_bad_by_lag,
            }

            ex = {}
            for _, col, phi1 in top:
                try:
                    xb = out[col].astype(float).to_numpy(copy=False)
                    xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                    ex[str(col)] = {
                        "phi1": float(phi1),
                        "before": out[col].astype(float).to_numpy(copy=True).tolist(),
                        "before_corr_lag1": _safe_corr_at_lag(xb, 1),
                        "before_corr_lagp": _safe_corr_at_lag(xb, p_order),
                        "before_corr_by_lag": {
                            f"lag{int(k)}": _safe_corr_at_lag(xb, int(k))
                            for k in range(1, p_order + 1)
                        },
                    }
                except (ValueError, TypeError, FloatingPointError) as exc:
                    report.add(f"[Preprocess] AR diagnostics example skipped for {col}: {exc}")
            ac_note["examples"] = ex
        report.add(f"[Preprocess] remove AR(p): p={p_order} (OLS on lags)")
        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):
                continue
            x = out[col].astype(float).to_numpy(copy=True)
            if x.size < (p_order + 4):
                continue
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            if p_order == 1:
                x0, x1 = x[:-1], x[1:]
                den = float(np.dot(x0, x0))
                phi = float(np.dot(x0, x1) / den) if np.isfinite(den) and den > 1e-12 else 0.0
                if not np.isfinite(phi):
                    phi = 0.0
                y = np.empty_like(x)
                y[0] = 0.0
                y[1:] = x1 - phi * x0
                out[col] = y
                report.add(f"[Preprocess] AR(1) phi≈{phi:.3f}", col=col)
                continue

            t_size = int(x.size)
            y_target = x[p_order:]
            x_lags = np.column_stack([x[p_order - i : t_size - i] for i in range(1, p_order + 1)])
            try:
                phi, *_ = np.linalg.lstsq(x_lags, y_target, rcond=None)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                report.add(f"[Preprocess] AR(p) skipped for {col}: {exc}")
                continue
            phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

            y = np.zeros_like(x)
            y[p_order:] = y_target - (x_lags @ phi)
            out[col] = y
            coeffs = [float(v) for v in phi[: min(5, p_order)]]
            suffix = "..." if p_order > 5 else ""
            report.add(f"[Preprocess] AR(p) coeffs≈{coeffs}{suffix}", col=col)

        if bool(ar_diagnostics):
            cols_diag2 = cols_diag
            phi1_list = []
            corr_lists_by_lag = {int(k): [] for k in range(1, p_order + 1)}
            lb_lists_by_lag = {int(k): [] for k in range(1, p_order + 1)}

            for col in cols_diag2:
                x = out[col].astype(float).to_numpy(copy=False)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                phi1 = _phi1_ols(x)
                corr_by_lag_cur = {}
                lb_by_lag_cur = {}
                for lag_k in range(1, p_order + 1):
                    rk = _safe_corr_at_lag(x, lag_k)
                    lbk = _ljung_box_p(x, lag_k)
                    corr_by_lag_cur[int(lag_k)] = rk
                    lb_by_lag_cur[int(lag_k)] = lbk
                if np.isfinite(phi1):
                    phi1_list.append(float(phi1))
                for lag_k in range(1, p_order + 1):
                    rk = corr_by_lag_cur.get(int(lag_k))
                    lbk = lb_by_lag_cur.get(int(lag_k))
                    if np.isfinite(rk):
                        corr_lists_by_lag[int(lag_k)].append(float(rk))
                    if np.isfinite(lbk):
                        lb_lists_by_lag[int(lag_k)].append(float(lbk))

            def _summary2(vals: list[float]) -> dict:
                if not vals:
                    return {"n": 0}
                a = np.asarray(vals, dtype=float)
                return {
                    "n": int(a.size),
                    "median": float(np.nanmedian(a)),
                    "mean": float(np.nanmean(a)),
                    "p25": float(np.nanpercentile(a, 25)),
                    "p75": float(np.nanpercentile(a, 75)),
                }

            corr_summary_by_lag = {
                f"lag{int(k)}": _summary2(vals) for k, vals in corr_lists_by_lag.items()
            }
            lb_summary_by_lag = {
                f"lag{int(k)}": _summary2(vals) for k, vals in lb_lists_by_lag.items()
            }
            frac_lb_bad_by_lag = {
                f"lag{int(k)}": (
                    float(np.mean(np.asarray(vals, dtype=float) < 0.05)) if vals else float("nan")
                )
                for k, vals in lb_lists_by_lag.items()
            }

            ac_note["after"] = {
                "phi1": _summary2(phi1_list),
                "corr_lag1": corr_summary_by_lag.get("lag1", {"n": 0}),
                "corr_lagp": corr_summary_by_lag.get(f"lag{int(p_order)}", {"n": 0}),
                "ljungbox_p_lag1": lb_summary_by_lag.get("lag1", {"n": 0}),
                "ljungbox_p_lagp": lb_summary_by_lag.get(f"lag{int(p_order)}", {"n": 0}),
                "frac_lb_p_lt_0_05_lag1": frac_lb_bad_by_lag.get("lag1", float("nan")),
                "frac_lb_p_lt_0_05_lagp": frac_lb_bad_by_lag.get(
                    f"lag{int(p_order)}", float("nan")
                ),
                "corr_by_lag": corr_summary_by_lag,
                "ljungbox_p_by_lag": lb_summary_by_lag,
                "frac_lb_p_lt_0_05_by_lag": frac_lb_bad_by_lag,
            }

            ex = ac_note.get("examples") or {}
            if isinstance(ex, dict):
                for col, item in list(ex.items()):
                    try:
                        if isinstance(item, dict):
                            item["after"] = out[col].astype(float).to_numpy(copy=True).tolist()
                            xa = out[col].astype(float).to_numpy(copy=False)
                            xa = np.nan_to_num(xa, nan=0.0, posinf=0.0, neginf=0.0)
                            item["after_corr_lag1"] = _safe_corr_at_lag(xa, 1)
                            item["after_corr_lagp"] = _safe_corr_at_lag(xa, p_order)
                            item["after_corr_by_lag"] = {
                                f"lag{int(k)}": _safe_corr_at_lag(xa, int(k))
                                for k in range(1, p_order + 1)
                            }
                    except (KeyError, ValueError, TypeError, FloatingPointError) as exc:
                        report.add(
                            f"[Preprocess] AR diagnostics after-example skipped for {col}: {exc}"
                        )
            ac_note["examples"] = ex

            try:
                b = (ac_note.get("before") or {}).get("corr_lag1") or {}
                a = (ac_note.get("after") or {}).get("corr_lag1") or {}
                bmed = float(b.get("median")) if b.get("median") is not None else float("nan")
                amed = float(a.get("median")) if a.get("median") is not None else float("nan")
                if np.isfinite(bmed) and np.isfinite(amed):
                    ac_note["lag1_reduction_median"] = float(
                        1.0 - (abs(amed) / max(1e-12, abs(bmed)))
                    )
            except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as exc:
                report.add(f"[Preprocess] AR lag1 reduction summary skipped: {exc}")

            try:
                before_corr = (ac_note.get("before") or {}).get("corr_by_lag") or {}
                after_corr = (ac_note.get("after") or {}).get("corr_by_lag") or {}
                lag_reduction = {}
                for lag_k in range(1, p_order + 1):
                    kb = f"lag{int(lag_k)}"
                    b = before_corr.get(kb) or {}
                    a = after_corr.get(kb) or {}
                    bmed = float(b.get("median")) if b.get("median") is not None else float("nan")
                    amed = float(a.get("median")) if a.get("median") is not None else float("nan")
                    if np.isfinite(bmed) and np.isfinite(amed):
                        lag_reduction[kb] = float(1.0 - (abs(amed) / max(1e-12, abs(bmed))))
                    else:
                        lag_reduction[kb] = float("nan")
                ac_note["lag_reduction_median_by_lag"] = lag_reduction
            except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as exc:
                report.add(f"[Preprocess] AR lag reduction summary skipped: {exc}")

            report.notes["autocorr"] = ac_note

    if remove_seasonality:
        report.add("[Preprocess] remove seasonality: STL (if period detected)")
        try:
            from statsmodels.tsa.seasonal import STL

            from neweds.analysis import stats as seasonality_stats
        except ImportError:
            STL = None
            seasonality_stats = None

        if STL is not None and seasonality_stats is not None:
            for col in out.columns:
                if not pd.api.types.is_numeric_dtype(out[col]):
                    continue
                x = out[col].astype(float)
                if x.size < 30:
                    continue
                per = (
                    int(season_period)
                    if season_period is not None and int(season_period) >= 2
                    else None
                )
                if per is None:
                    try:
                        ss = seasonality_stats.detect_seasonality(x)
                        cand = ss.get("acf_period")
                        strength = ss.get("acf_strength")
                        if cand is not None and strength is not None and float(strength) >= 0.2:
                            per = int(cand)
                    except (ValueError, TypeError, FloatingPointError, KeyError) as exc:
                        report.add(f"[Preprocess] seasonality detection skipped for {col}: {exc}")
                        per = None
                if per is None or per < 2:
                    continue
                try:
                    stl = STL(x, period=int(per), robust=True).fit()
                    out[col] = (x - stl.seasonal).to_numpy()
                    report.add(f"[Preprocess] STL period={int(per)}", col=col)
                except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    report.add(f"[Preprocess] STL skipped for {col}: {exc}")
                    continue

    if normalize:
        mode = str(normalize_mode or "zscore").strip().lower()
        cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if not cols:
            mode = "none"

        if mode in {"none", "off", "false"}:
            report.add("[Preprocess] normalize: off")
        elif mode in {"z", "zscore", "standard"}:
            report.add("[Preprocess] normalize: z-score (mean/std) per series")
            arr = out[cols].to_numpy(dtype=np.float64)
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            stds[stds < 1e-12] = 1.0
            out[cols] = (arr - means[np.newaxis, :]) / stds[np.newaxis, :]
            report.notes["normalization"] = {
                "mode": "zscore",
                "n_series": len(cols),
                "col_names": [str(c) for c in cols],
                "means": means.tolist(),
                "stds": stds.tolist(),
            }
        elif mode in {"robust", "robust_z", "mad"}:
            report.add("[Preprocess] normalize: robust z-score (median/MAD) per series")
            medians_list: list[float] = []
            mads_list: list[float] = []
            for col in cols:
                s = out[col].astype(float)
                med = float(s.median())
                mad = float((s - med).abs().median()) * 1.4826 + 1e-12
                out[col] = (s - med) / mad
                medians_list.append(med)
                mads_list.append(mad)
            report.notes["normalization"] = {
                "mode": "robust_zscore",
                "n_series": len(cols),
                "col_names": [str(c) for c in cols],
                "medians": medians_list,
                "mads": mads_list,
            }
        elif mode in {"rank", "rank_dense", "rank_pct", "rank_percentile"}:
            rmode = str(rank_mode or "dense").strip().lower()
            if mode in {"rank_pct", "rank_percentile"}:
                rmode = "pct"
            report.add(f"[Preprocess] normalize: rank ({rmode}, ties={rank_ties})")
            for col in cols:
                x = out[col].astype(float).to_numpy()
                out[col] = _rank_normalize_1d(x, mode=rmode, ties=str(rank_ties))
        else:
            report.add(f"[Preprocess] normalize: unknown mode '{mode}', fallback to z-score")
            arr = out[cols].to_numpy(dtype=np.float64)
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            stds[stds < 1e-12] = 1.0
            out[cols] = (arr - means[np.newaxis, :]) / stds[np.newaxis, :]
            report.notes["normalization"] = {
                "mode": "zscore",
                "n_series": len(cols),
                "col_names": [str(c) for c in cols],
                "means": means.tolist(),
                "stds": stds.tolist(),
            }

    if check_stationarity:
        report.add("[Preprocess] stationarity check: ADF")
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                series = out[col].dropna()
                if len(series) > 10:
                    _, pvalue = analysis_stats.test_stationarity(series)
                    if pvalue is not None:
                        logging.info(
                            "Ряд '%s' %s (p-value ADF=%.3f).",
                            col,
                            "стационарен" if pvalue <= 0.05 else "вероятно нестационарен",
                            pvalue,
                        )
                    else:
                        logging.debug(
                            "ADF skipped for column '%s' (constant/short/degenerate series).",
                            col,
                        )
    try:
        if _saved_attrs:
            out.attrs.update(_saved_attrs)
    except (AttributeError, TypeError, ValueError) as exc:
        logging.debug("attrs restore skipped after preprocessing: %s", exc)
    return (out, report) if return_report else out
