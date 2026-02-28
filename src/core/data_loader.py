#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль загрузки и парсинга данных из файлов.
"""

import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .preprocessing import additional_preprocessing
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy import stats


@dataclass
class PreprocessReport:
    """Структурированный отчёт о предобработке временных рядов.

    Используется в UI/HTML-отчёте, чтобы явно показать применённые шаги.
    """

    enabled: bool = True
    steps_global: List[str] = field(default_factory=list)
    steps_by_column: Dict[str, List[str]] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def add(self, msg: str, col: Optional[str] = None) -> None:
        """Добавляет шаг в глобальный список или к конкретной колонке."""
        if col is None:
            self.steps_global.append(msg)
        else:
            self.steps_by_column.setdefault(col, []).append(msg)


def _is_mostly_numeric_row(row) -> bool:
    """Проверяет, что в строке >=80% непустых значений приводятся к float."""
    vals = []
    for v in row:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        vals.append(v)
    if not vals:
        return False
    numeric = 0
    for v in vals:
        try:
            float(v)
            numeric += 1
        except Exception:
            pass
    return numeric / max(1, len(vals)) >= 0.8


def _detect_header(df_raw: pd.DataFrame) -> bool:
    """Если 1-я строка нечисловая, а 2-я числовая — считаем 1-ю заголовком."""
    if df_raw.shape[0] < 2:
        return False
    r0 = df_raw.iloc[0].tolist()
    r1 = df_raw.iloc[1].tolist()
    return (not _is_mostly_numeric_row(r0)) and _is_mostly_numeric_row(r1)


def _maybe_split_single_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Поддержка формата: «CSV в ячейке».

    Встречается в XLSX, когда каждая строка лежит в одной ячейке и содержит
    `x,y,z,t0,t1,...` (или `;`/`	` разделители). Также бывает вариант, когда
    Excel разнес 1–2 первых колонок, а остальное пустое.
    """
    try:
        # 1) строго одна колонка строк
        if df_raw.shape[1] == 1 and isinstance(df_raw.iloc[0, 0], str):
            return df_raw[0].astype(str).str.split(r"[,;\t]", expand=True)

        # 2) «почти одна колонка»: >80% значений в первой колонке непустые, остальные почти пустые
        if df_raw.shape[1] > 1:
            nonnull = df_raw.notna().mean(axis=0)
            if float(nonnull.iloc[0]) >= 0.8 and bool((nonnull.iloc[1:] <= 0.05).all()):
                if isinstance(df_raw.iloc[0, 0], str):
                    return df_raw.iloc[:, [0]].copy().iloc[:, 0].astype(str).str.split(r"[,;\t]", expand=True)

        # 3) строка целиком в одной ячейке, но не в первой колонке (редко)
        if df_raw.shape[1] > 1:
            best_j = None
            best_score = 0.0
            for j in range(df_raw.shape[1]):
                col = df_raw.iloc[:, j]
                is_str = col.apply(lambda v: isinstance(v, str) and ("," in v or ";" in v or "\t" in v))
                score = float(is_str.mean())
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is not None and best_score >= 0.8:
                return df_raw.iloc[:, [best_j]].copy().iloc[:, 0].astype(str).str.split(r"[,;\t]", expand=True)
    except Exception:
        pass
    return df_raw



def _detect_voxel_wide(df: pd.DataFrame) -> tuple[bool, dict[str, str]]:
    """Проверяет формат вида x,y,z,t0..tN."""
    cols = list(df.columns)
    lower = {str(c).strip().lower(): str(c) for c in cols}
    if not {"x", "y", "z"}.issubset(set(lower.keys())):
        return False, lower
    other = [c for c in cols if str(c).strip().lower() not in {"x", "y", "z"}]
    if len(other) < 2:
        return False, lower
    return True, lower


def voxel_wide_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Конвертирует таблицу x,y,z,t0..tN в матрицу time × voxel.

    Метаданные координат сохраняются в out.attrs['coords'] как DataFrame.
    """
    is_vox, lower = _detect_voxel_wide(df)
    if not is_vox:
        return df

    xcol, ycol, zcol = lower["x"], lower["y"], lower["z"]
    time_cols = [c for c in df.columns if str(c).strip().lower() not in {"x", "y", "z"}]

    coords = df[[xcol, ycol, zcol]].copy()
    coords.columns = ["x", "y", "z"]
    for c in ["x", "y", "z"]:
        coords[c] = pd.to_numeric(coords[c], errors="coerce")

    ts = df[time_cols].copy().apply(pd.to_numeric, errors="coerce")

    def _t_index(name: str) -> int | None:
        s = str(name).strip().lower()
        if s.startswith("t") and s[1:].isdigit():
            return int(s[1:])
        if s.isdigit():
            return int(s)
        return None

    t_ids = [_t_index(c) for c in time_cols]
    if all(v is not None for v in t_ids):
        order = np.argsort(np.asarray(t_ids, dtype=int))
        ts = ts.iloc[:, order]
        time_cols_sorted = [time_cols[i] for i in order]
    else:
        time_cols_sorted = list(time_cols)

    voxel_ids: list[str] = []
    for i in range(len(coords)):
        x, y, z = coords.iloc[i].tolist()
        voxel_ids.append(
            f"v{i:04d}_x{int(x) if np.isfinite(x) else 'nan'}_y{int(y) if np.isfinite(y) else 'nan'}_z{int(z) if np.isfinite(z) else 'nan'}"
        )
    coords.insert(0, "voxel_id", voxel_ids)
    try:
        dup = coords.duplicated(subset=["x", "y", "z"], keep=False)
        coords["coord_duplicate"] = dup.astype(int)
    except Exception:
        coords["coord_duplicate"] = 0

    ts.index = voxel_ids
    out = ts.T
    out.columns = voxel_ids
    out.attrs["coords"] = coords
    out.attrs["voxel_time_cols"] = [str(c) for c in time_cols_sorted]
    return out


def _detect_time_like_col(col: pd.Series) -> bool:
    """Эвристика для авто-обнаружения временной/индексной колонки."""
    try:
        dt = pd.to_datetime(col, errors="coerce", utc=False)
        if dt.notna().mean() >= 0.9:
            return dt.is_monotonic_increasing or dt.is_monotonic_decreasing
    except Exception:
        pass

    c = pd.to_numeric(col, errors="coerce")
    if c.notna().mean() >= 0.95:
        dif = c.dropna().diff().dropna()
        if len(dif) >= 3 and (dif.abs() > 0).mean() >= 0.9:
            return True
    return False


def read_input_table(
    filepath: str,
    header: str = "auto",
    *,
    usecols: Any = "auto",
    csv_engine: str = "auto",
) -> pd.DataFrame:
    """Чтение CSV/XLSX/PARQUET с поддержкой автодетекта заголовка и «CSV в ячейке».

    Для больших файлов:
    - XLSX + usecols="auto": сначала читаем только 1-ю колонку (частый кейс, когда
      каждая строка лежит в одной ячейке и содержит `x,y,z,t0,...`). Это в разы
      снижает память.
    - CSV + csv_engine="pyarrow": ускоряет чтение больших CSV (если установлен pyarrow).
    """
    fp = str(filepath)
    low = fp.lower()

    if low.endswith(".parquet"):
        df0 = pd.read_parquet(fp)
        if header not in {"auto", "yes", "no"}:
            raise ValueError("header must be one of: auto|yes|no")
        return df0

    if low.endswith(".csv"):
        # Важно: low_memory=False выключает покусковую догадку типов в pandas
        # и снижает риск нестабильной типизации на mixed-type CSV.
        kw: Dict[str, Any] = {"header": None, "low_memory": False}
        if csv_engine in {"pyarrow", "c", "python"}:
            kw["engine"] = csv_engine
        elif csv_engine == "auto":
            # Автовыбор: pyarrow обычно существенно быстрее на больших CSV.
            # Если пакет недоступен, мягко падаем на pandas default-engine.
            try:
                import pyarrow  # noqa: F401

                kw["engine"] = "pyarrow"
            except ImportError:
                pass
        # pandas+pyarrow не поддерживает low_memory.
        if kw.get("engine") == "pyarrow":
            kw.pop("low_memory", None)
        df0 = pd.read_csv(fp, **kw)
    else:
        xl_usecols = usecols
        if usecols == "auto":
            xl_usecols = [0]
        try:
            df0 = pd.read_excel(fp, header=None, usecols=xl_usecols)
        except Exception:
            # Запасной вариант: читаем целиком
            df0 = pd.read_excel(fp, header=None)
    df0 = _maybe_split_single_column(df0)

    if header not in {"auto", "yes", "no"}:
        raise ValueError("header must be one of: auto|yes|no")
    has_header = _detect_header(df0) if header == "auto" else (header == "yes")
    if has_header:
        hdr = df0.iloc[0].astype(str).tolist()
        df = df0.iloc[1:].copy()
        df.columns = [h if h.strip() else f"c{i+1}" for i, h in enumerate(hdr)]
    else:
        df = df0.copy()
        df.columns = [f"c{i+1}" for i in range(df.shape[1])]
    return df


def tidy_timeseries_table(
    df: pd.DataFrame,
    time_col: str = "auto",
    transpose: str = "auto",
    *,
    dtype: str | None = None,
    time_start: int | None = None,
    time_end: int | None = None,
    time_stride: int | None = None,
    feature_limit: int | None = None,
    feature_sampling: str = "first",
    feature_seed: int = 13,
) -> pd.DataFrame:
    """Превращает сырую таблицу в numeric матрицу вида time × features."""
    out = df.copy()
    out = out.dropna(axis=1, how="all")

    # Спец-кейс: x,y,z,t0..tN (воксельный wide-формат)
    try:
        out = voxel_wide_to_timeseries(out)
    except Exception:
        pass

    # Важно: voxel_wide_to_timeseries кладёт в out.attrs['coords'] DataFrame координат.
    # Pandas при многих операциях (например, Series.notna()) делает deepcopy attrs,
    # что на больших данных взрывает память (deepcopy координат ДЛЯ КАЖДОЙ колонки).
    # Поэтому: временно выносим attrs наружу и очищаем их на время чистки таблицы.
    _saved_attrs: Dict[str, Any] = dict(getattr(out, "attrs", {}) or {})
    try:
        out.attrs = {}
    except Exception:
        try:
            out.attrs.clear()
        except Exception:
            pass

    # Если есть coords — это уже time×voxel. Авто-транспонирование запрещаем,
    # иначе можно случайно перевернуть огромные данные и убить память.
    has_coords = bool(_saved_attrs.get("coords") is not None)

    if time_col not in {"auto", "none"} and time_col not in out.columns:
        raise ValueError(f"time_col '{time_col}' not found in columns")
    if time_col == "auto":
        if out.shape[1] >= 2 and _detect_time_like_col(out.iloc[:, 0]):
            out = out.iloc[:, 1:].copy()
    elif time_col != "none":
        out = out.drop(columns=[time_col])

    out = out.apply(pd.to_numeric, errors="coerce")
    # Векторная фильтрация колонок: одна операция по всему numpy-массиву.
    # Это быстрее и меньше нагружает attrs/deepcopy-путь pandas.
    try:
        arr = out.to_numpy()
        if arr.dtype.kind == "f":
            col_frac = np.isfinite(arr).mean(axis=0)
        else:
            col_frac = pd.notna(arr).mean(axis=0)
        out = out.loc[:, col_frac >= 0.2]
    except Exception:
        col_frac = out.notna().mean(axis=0)
        out = out.loc[:, col_frac >= 0.2]

    if transpose not in {"auto", "yes", "no"}:
        raise ValueError("transpose must be one of: auto|yes|no")
    if has_coords and transpose == "auto":
        do_t = False
    else:
        do_t = (out.shape[0] < out.shape[1]) if transpose == "auto" else (transpose == "yes")
    if do_t:
        out = out.T
        out.columns = [f"c{i+1}" for i in range(out.shape[1])]

    # Обрезка больших данных (до предобработки)
    # Обрезка по времени
    try:
        t0 = int(time_start) if time_start is not None else None
        t1 = int(time_end) if time_end is not None else None
        ts = int(time_stride) if time_stride is not None else None
        if ts is not None and ts <= 0:
            ts = None
        if t0 is not None or t1 is not None or ts is not None:
            out = out.iloc[slice(t0, t1, ts), :]
    except Exception:
        pass

    # Ограничение числа признаков
    try:
        if feature_limit is not None and int(feature_limit) > 0 and out.shape[1] > int(feature_limit):
            mode = str(feature_sampling or "first").strip().lower()
            k = int(feature_limit)
            if mode in {"random", "rand"}:
                rng = np.random.default_rng(int(feature_seed))
                cols = list(out.columns)
                pick = rng.choice(len(cols), size=k, replace=False)
                out = out.loc[:, [cols[i] for i in sorted(pick)]]
            elif mode in {"variance", "var", "topvar"}:
                sub = out
                if sub.shape[0] > 2000:
                    sub = sub.iloc[:: max(1, sub.shape[0] // 2000), :]
                v = sub.var(axis=0, skipna=True).to_numpy(dtype=float)
                order = np.argsort(-np.nan_to_num(v, nan=-np.inf))
                keep = [out.columns[i] for i in order[:k]]
                out = out.loc[:, keep]
            else:
                out = out.iloc[:, :k]
    except Exception:
        pass

    # Приведение типа (последним, чтобы не плодить копии)
    if dtype:
        dt = str(dtype).strip().lower()
        if dt in {"float32", "f4"}:
            out = out.astype(np.float32)
        elif dt in {"float64", "f8"}:
            out = out.astype(np.float64)

    out = out.dropna(axis=0, how="all")

    # Восстановление attrs (coords и прочее) ПОСЛЕ всех операций,
    # чтобы избежать гигантских deepcopy во время обработки.
    try:
        if _saved_attrs:
            coords = _saved_attrs.get("coords")
            if isinstance(coords, pd.DataFrame) and "voxel_id" in coords.columns:
                # синхронизируем coords с текущими колонками (после фильтрации/feature_limit)
                try:
                    idx = coords.set_index("voxel_id", drop=False)
                    idx = idx.loc[[c for c in out.columns if c in idx.index]]
                    _saved_attrs["coords"] = idx.reset_index(drop=True)
                except Exception:
                    pass
            out.attrs.update(_saved_attrs)
    except Exception:
        pass
    return out


# Утилиты предобработки: выбросы и ранговая нормализация

def _rank_normalize_1d(x: np.ndarray, *, mode: str = "dense", ties: str = "average") -> np.ndarray:
    """Ранговая нормализация (структурная): значения -> ранги.

    Пример (dense): [100, 33, 98, 2] -> [4, 2, 3, 1]
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
    """Находит/обрабатывает выбросы в 1D и возвращает (new_x, mask)."""
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
            local = s.rolling(window=int(max(3, local_median_window)), center=True, min_periods=1).median().to_numpy()
            y[msk] = local[msk]
            return
        if action in {"clip", "winsorize"}:
            # Клиппинг к заданным перцентильным границам (задаётся p_low/p_high)
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
    """Предобработка матрицы (можно полностью отключить enabled=False)."""
    # ВАЖНО: df может содержать тяжёлые метаданные в attrs (например coords для вокселей).
    # Pandas при доступе к df[col] может делать deepcopy attrs, что взрывает память.
    # Поэтому на время предобработки очищаем attrs и восстанавливаем в конце.
    _saved_attrs = dict(getattr(df, "attrs", {}) or {})
    out = df.copy()
    try:
        out.attrs = {}
    except Exception:
        try:
            out.attrs.clear()
        except Exception:
            pass
    report = PreprocessReport(enabled=bool(enabled))
    if not enabled:
        report.add("[Preprocess] disabled: using raw numeric matrix as-is.")
        try:
            if _saved_attrs:
                out.attrs.update(_saved_attrs)
        except Exception:
            pass
        return (out, report) if return_report else out

    report.add("[Preprocess] enabled")

    before_cols = list(out.columns)
    out = additional_preprocessing(out)
    after_cols = list(out.columns)
    dropped = [c for c in before_cols if c not in after_cols]
    if dropped:
        report.dropped_columns.extend(dropped)
        report.add(f"[Preprocess] dropped near-constant columns: {dropped}")

    out = out.fillna(out.mean(numeric_only=True))
    report.add("[Preprocess] fillna: column means")

    if log_transform:
        report.add("[Preprocess] log-transform: applied to positive values")
        fn = lambda x: np.log(x) if x is not None and not np.isnan(x) and x > 0 else x
        # pandas 2.2+ постепенно уводит applymap, предпочтительнее DataFrame.map.
        try:
            out = out.map(fn)  # type: ignore[attr-defined]
        except Exception:
            out = out.applymap(fn)

    if remove_outliers:
        rule = str(outlier_rule or "robust_z")
        action = str(outlier_action or "mask")
        report.add(f"[Preprocess] outliers: rule={rule}, action={action}")
        total = 0
        # Быстрый векторный путь для robust_z/mad на всех numeric-колонках.
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
            # Fallback: построчная обработка для остальных правил.
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
        out = out.interpolate(method="linear", limit_direction="both", axis=0).bfill().ffill().fillna(0)

    if remove_ar1:
        p_order = int(max(1, int(remove_ar_order or 1)))

        # --- Диагностика автокорреляции (до очистки) ---
        # Явный акцент на лаг 1 + (при необходимости) лаг p.
        # Важно: это нужно не для красоты, а чтобы понимать, не убиваем ли мы сигнал.
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
            # OLS оценка AR(1): phi = <x[t-1], x[t]> / <x[t-1], x[t-1]>
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
            # Ljung–Box p-value на заданном числе лагов.
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
            except Exception:
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
            except Exception:
                return float("nan")

        # Для очень больших матриц диагностику ограничиваем сэмплом колонок.
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
            r1_list = []
            rp_list = []
            lb1_list = []
            lbp_list = []

            # Сохраним 3 примера: максимальная |phi1| (до очистки).
            top = []  # list[tuple[abs_phi, col, phi1]]
            for col in cols_diag:
                x = out[col].astype(float).to_numpy(copy=False)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                phi1 = _phi1_ols(x)
                r1 = _safe_corr_at_lag(x, 1)
                rp = _safe_corr_at_lag(x, p_order)
                lb1 = _ljung_box_p(x, 1)
                lbp = _ljung_box_p(x, p_order)

                if np.isfinite(phi1):
                    phi1_list.append(float(phi1))
                    top.append((abs(float(phi1)), col, float(phi1)))
                if np.isfinite(r1):
                    r1_list.append(float(r1))
                if np.isfinite(rp):
                    rp_list.append(float(rp))
                if np.isfinite(lb1):
                    lb1_list.append(float(lb1))
                if np.isfinite(lbp):
                    lbp_list.append(float(lbp))

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

            ac_note["before"] = {
                "phi1": _summary(phi1_list),
                "corr_lag1": _summary(r1_list),
                "corr_lagp": _summary(rp_list),
                "ljungbox_p_lag1": _summary(lb1_list),
                "ljungbox_p_lagp": _summary(lbp_list),
                "frac_lb_p_lt_0_05_lag1": float(np.mean(np.asarray(lb1_list) < 0.05)) if lb1_list else float("nan"),
                "frac_lb_p_lt_0_05_lagp": float(np.mean(np.asarray(lbp_list) < 0.05)) if lbp_list else float("nan"),
            }

            # Сохраним сами ряды для примеров (до очистки).
            ex = {}
            for _, col, phi1 in top:
                try:
                    ex[str(col)] = {
                        "phi1": float(phi1),
                        "before": out[col].astype(float).to_numpy(copy=True).tolist(),
                    }
                except Exception:
                    pass
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
                # OLS-оценка AR(1) коэффициента (не corr!).
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

            # OLS: x[t] ~ sum_{i=1..p} phi_i * x[t-i]
            t_size = int(x.size)
            y_target = x[p_order:]
            x_lags = np.column_stack([x[p_order - i : t_size - i] for i in range(1, p_order + 1)])
            try:
                phi, *_ = np.linalg.lstsq(x_lags, y_target, rcond=None)
            except Exception:
                phi = np.zeros((p_order,), dtype=float)
            phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

            y = np.zeros_like(x)
            y[p_order:] = y_target - (x_lags @ phi)
            out[col] = y
            coeffs = [float(v) for v in phi[: min(5, p_order)]]
            suffix = "..." if p_order > 5 else ""
            report.add(f"[Preprocess] AR(p) coeffs≈{coeffs}{suffix}", col=col)

        # --- Диагностика автокорреляции (после очистки) + примеры рядов ---
        if bool(ar_diagnostics):
            cols_diag2 = cols_diag
            phi1_list = []
            r1_list = []
            rp_list = []
            lb1_list = []
            lbp_list = []

            for col in cols_diag2:
                x = out[col].astype(float).to_numpy(copy=False)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                phi1 = _phi1_ols(x)
                r1 = _safe_corr_at_lag(x, 1)
                rp = _safe_corr_at_lag(x, p_order)
                lb1 = _ljung_box_p(x, 1)
                lbp = _ljung_box_p(x, p_order)
                if np.isfinite(phi1):
                    phi1_list.append(float(phi1))
                if np.isfinite(r1):
                    r1_list.append(float(r1))
                if np.isfinite(rp):
                    rp_list.append(float(rp))
                if np.isfinite(lb1):
                    lb1_list.append(float(lb1))
                if np.isfinite(lbp):
                    lbp_list.append(float(lbp))

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

            ac_note["after"] = {
                "phi1": _summary(phi1_list),
                "corr_lag1": _summary(r1_list),
                "corr_lagp": _summary(rp_list),
                "ljungbox_p_lag1": _summary(lb1_list),
                "ljungbox_p_lagp": _summary(lbp_list),
                "frac_lb_p_lt_0_05_lag1": float(np.mean(np.asarray(lb1_list) < 0.05)) if lb1_list else float("nan"),
                "frac_lb_p_lt_0_05_lagp": float(np.mean(np.asarray(lbp_list) < 0.05)) if lbp_list else float("nan"),
            }

            # допишем примеры "после"
            ex = ac_note.get("examples") or {}
            if isinstance(ex, dict):
                for col, item in list(ex.items()):
                    try:
                        if isinstance(item, dict):
                            item["after"] = out[col].astype(float).to_numpy(copy=True).tolist()
                            # остаточная автокорреляция на лаг 1 / lag p
                            xa = out[col].astype(float).to_numpy(copy=False)
                            xa = np.nan_to_num(xa, nan=0.0, posinf=0.0, neginf=0.0)
                            item["after_corr_lag1"] = _safe_corr_at_lag(xa, 1)
                            item["after_corr_lagp"] = _safe_corr_at_lag(xa, p_order)
                    except Exception:
                        pass
            ac_note["examples"] = ex

            # краткая метрика «насколько упала автокор на лаг1» по медиане
            try:
                b = (ac_note.get("before") or {}).get("corr_lag1") or {}
                a = (ac_note.get("after") or {}).get("corr_lag1") or {}
                bmed = float(b.get("median")) if b.get("median") is not None else float("nan")
                amed = float(a.get("median")) if a.get("median") is not None else float("nan")
                if np.isfinite(bmed) and np.isfinite(amed):
                    ac_note["lag1_reduction_median"] = float(1.0 - (abs(amed) / max(1e-12, abs(bmed))))
            except Exception:
                pass

            report.notes["autocorr"] = ac_note

    if remove_seasonality:
        # STL сезонность: либо заданный период, либо пробуем оценить.
        report.add("[Preprocess] remove seasonality: STL (if period detected)")
        try:
            from statsmodels.tsa.seasonal import STL
            from ..analysis import stats as analysis_stats
        except Exception:
            STL = None
            analysis_stats = None

        if STL is not None and analysis_stats is not None:
            for col in out.columns:
                if not pd.api.types.is_numeric_dtype(out[col]):
                    continue
                x = out[col].astype(float)
                if x.size < 30:
                    continue
                per = int(season_period) if season_period is not None and int(season_period) >= 2 else None
                if per is None:
                    try:
                        ss = analysis_stats.detect_seasonality(x)
                        cand = ss.get("acf_period")
                        strength = ss.get("acf_strength")
                        if cand is not None and strength is not None and float(strength) >= 0.2:
                            per = int(cand)
                    except Exception:
                        per = None
                if per is None or per < 2:
                    continue
                try:
                    stl = STL(x, period=int(per), robust=True).fit()
                    out[col] = (x - stl.seasonal).to_numpy()
                    report.add(f"[Preprocess] STL period={int(per)}", col=col)
                except Exception:
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
        elif mode in {"robust", "robust_z", "mad"}:
            report.add("[Preprocess] normalize: robust z-score (median/MAD) per series")
            for col in cols:
                s = out[col].astype(float)
                med = float(s.median())
                mad = float((s - med).abs().median()) * 1.4826 + 1e-12
                out[col] = (s - med) / mad
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

    if check_stationarity:
        report.add("[Preprocess] stationarity check: ADF")
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                series = out[col].dropna()
                if len(series) > 10:
                    pvalue = adfuller(series, autolag="AIC")[1]
                    logging.info(
                        f"Ряд '{col}' {'стационарен' if pvalue <= 0.05 else 'вероятно нестационарен'} (p-value ADF={pvalue:.3f})."
                    )
    try:
        if _saved_attrs:
            out.attrs.update(_saved_attrs)
    except Exception:
        pass
    return (out, report) if return_report else out


def load_or_generate(
    filepath: str,
    *,
    header: str = "auto",
    time_col: str = "auto",
    transpose: str = "auto",
    # Параметры для больших данных и производительности
    dtype: str | None = None,
    # Если dtype=None, то для очень широких/больших таблиц автоматически
    # понижаем тип до float32, чтобы снизить пик памяти при загрузке/clean-up.
    auto_float32: bool = True,
    time_start: int | None = None,
    time_end: int | None = None,
    time_stride: int | None = None,
    feature_limit: int | None = None,
    feature_sampling: str = "first",
    feature_seed: int = 13,
    usecols: Any = "auto",
    csv_engine: str = "auto",
    preprocess: bool = True,
    log_transform: bool = False,
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
    normalize: bool = True,
    normalize_mode: str = "zscore",
    rank_mode: str = "dense",
    rank_ties: str = "average",
    fill_missing: bool = True,
    remove_ar1: bool = False,
    remove_ar_order: int = 1,
    ar_diagnostics: bool = True,
    remove_seasonality: bool = False,
    season_period: int | None = None,
    check_stationarity: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, PreprocessReport]:
    """
    Главная функция загрузки и предобработки данных из файла.
    
    Args:
        filepath: Путь к CSV или Excel файлу
        header: Режим заголовка ('auto', 'yes', 'no')
        time_col: Колонка времени ('auto', 'none', или название)
        transpose: Транспонирование ('auto', 'yes', 'no')
        auto_float32: Автопонижение к float32 для очень больших таблиц при dtype=None
        preprocess: Включить предобработку
        log_transform: Применить логарифм
        remove_outliers: Удалить выбросы
        normalize: Нормализовать данные
        fill_missing: Заполнить пропуски
        check_stationarity: Проверить стационарность
        
    Returns:
        Если ``return_report=False``: ``pd.DataFrame`` с матрицей временных рядов.
        Если ``return_report=True``: кортеж ``(pd.DataFrame, PreprocessReport)``
        для последующей визуализации шагов предобработки в UI/HTML-отчёте.
    """
    try:
        raw = read_input_table(filepath, header=header, usecols=usecols, csv_engine=csv_engine)

        # Автопонижение типа: лучше осознанно перейти в float32,
        # чем получить OOM на неявных копиях в pandas/numpy при больших матрицах.
        dtype_eff = dtype
        if dtype_eff is None and auto_float32:
            try:
                n_rows, n_cols = int(raw.shape[0]), int(raw.shape[1])
                n_cells = n_rows * n_cols
                if n_cols >= 128 or n_cells >= 2_000_000:
                    dtype_eff = "float32"
            except Exception:
                pass

        df = tidy_timeseries_table(
            raw,
            time_col=time_col,
            transpose=transpose,
            dtype=dtype_eff,
            time_start=time_start,
            time_end=time_end,
            time_stride=time_stride,
            feature_limit=feature_limit,
            feature_sampling=feature_sampling,
            feature_seed=feature_seed,
        )
        coords_df = None
        try:
            coords_df = df.attrs.get("coords")
        except Exception:
            coords_df = None

        df_out = preprocess_timeseries(
            df,
            enabled=preprocess,
            log_transform=log_transform,
            remove_outliers=remove_outliers,
            outlier_rule=outlier_rule,
            outlier_action=outlier_action,
            outlier_z=outlier_z,
            outlier_k=outlier_k,
            outlier_abs=outlier_abs,
            outlier_p_low=outlier_p_low,
            outlier_p_high=outlier_p_high,
            outlier_hampel_window=outlier_hampel_window,
            outlier_jump_thr=outlier_jump_thr,
            outlier_local_median_window=outlier_local_median_window,
            normalize=normalize,
            normalize_mode=normalize_mode,
            rank_mode=rank_mode,
            rank_ties=rank_ties,
            fill_missing=fill_missing,
            remove_ar1=remove_ar1,
            remove_ar_order=remove_ar_order,
            ar_diagnostics=ar_diagnostics,
            remove_seasonality=remove_seasonality,
            season_period=season_period,
            check_stationarity=check_stationarity,
            return_report=bool(return_report),
        )
        if return_report:
            df, report = df_out  # type: ignore[misc]
        else:
            df, report = df_out, None

        # Прокидываем метаданные (например, координаты вокселей) в report.notes.
        # Далее очищаем attrs у итогового DataFrame: тяжелые объекты уже сериализованы в report,
        # и их повторное хранение повышает риск deepcopy(attrs) при df[col].
        if report is not None and coords_df is not None:
            try:
                report.notes["format"] = "voxel_wide"
                report.notes["n_voxels"] = int(getattr(coords_df, "shape", [0])[0])
                report.notes["coords"] = coords_df.to_dict(orient="records")
            except Exception:
                pass

        try:
            if hasattr(df, "attrs"):
                df.attrs.pop("coords", None)
                df.attrs.pop("voxel_time_cols", None)
                if len(df.attrs) == 0:
                    df.attrs = {}
        except Exception:
            pass
        logging.info(
            f"[Load] OK shape={df.shape} header={header} time_col={time_col} transpose={transpose} preprocess={preprocess}"
        )
        return (df, report) if return_report else df
    except Exception as e:
        logging.error(f"[Load] Ошибка загрузки: {e}")
        raise
