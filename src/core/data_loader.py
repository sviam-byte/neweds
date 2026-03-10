#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль загрузки и парсинга данных из файлов.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .preprocessing import additional_preprocessing, spatial_grid_bin_fmri
from ..analysis import stats as analysis_stats
import numpy as np
from scipy import stats
from scipy.io import loadmat

from src.io.loaders import load_h5_spatial_binned_lazy

try:
    import h5py as _h5py
except ImportError:
    _h5py = None  # type: ignore[assignment]


MAX_RAW_VOXELS_FOR_GUI = 5000
"""Безопасный верхний предел числа сырых voxel-рядов для GUI-пайплайна.

Ограничение применяется на этапе извлечения из 4D H5 до построения полного
DataFrame time×voxel, чтобы избежать взрывного роста памяти.
"""


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


def _select_voxels_wide(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    zcol: str,
    time_cols: list,
    feature_limit: int | None,
    feature_sampling: str,
    feature_seed: int,
) -> pd.DataFrame:
    """Сэмплинг/сокращение вокселей ДО транспонирования.

    Для формата x,y,z,t0..tN это критично: транспонирование превращает N_voxels в число колонок.
    При N~250k это резко увеличивает память и время даже до dimred.

    Режимы:
      - first: первые K строк;
      - random: случайные K строк;
      - variance: топ-K по дисперсии по time_cols.
    """
    _ = (xcol, ycol, zcol)  # Явно фиксируем сигнатуру для читаемости вызова.
    if feature_limit is None:
        return df
    try:
        k = int(feature_limit)
    except Exception:
        return df
    if k <= 0 or int(df.shape[0]) <= k:
        return df

    mode = str(feature_sampling or "first").strip().lower()
    if mode in {"first", "head"}:
        return df.iloc[:k, :].copy()

    if mode in {"random", "rand"}:
        rng = np.random.default_rng(int(feature_seed))
        idx = rng.choice(int(df.shape[0]), size=k, replace=False)
        idx = np.sort(idx)
        return df.iloc[idx, :].copy()

    if mode in {"variance", "var", "topvar"}:
        arr = df[time_cols].to_numpy(dtype=np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.var(arr, axis=1, ddof=0)
        idx = np.argpartition(v, -k)[-k:]
        idx = idx[np.argsort(-v[idx])]
        return df.iloc[idx, :].copy()

    return df.iloc[:k, :].copy()


def voxel_wide_to_timeseries(
    df: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    feature_sampling: str = "first",
    feature_seed: int = 13,
    spatial_bin_size: int = 0,
    spatial_bin_method: str = "mean",
    spatial_bin_range: tuple | None = None,
) -> pd.DataFrame:
    """Конвертирует таблицу x,y,z,t0..tN в матрицу time × voxel (или time × bin).

    Метаданные координат сохраняются в out.attrs['coords'] как DataFrame.

    Если ``spatial_bin_size > 1``: пространственный биннинг по координатам.
    Абсолютно детерминирован — зависит только от (x,y,z) и bin_size.
    Контроль и эксперимент с одной геометрией → одни и те же бины.

    Иначе: обрезка feature_limit ДО транспонирования (legacy-поведение).
    """
    is_vox, lower = _detect_voxel_wide(df)
    if not is_vox:
        return df

    xcol, ycol, zcol = lower["x"], lower["y"], lower["z"]
    time_cols = [c for c in df.columns if str(c).strip().lower() not in {"x", "y", "z"}]

    _bs = int(spatial_bin_size or 0)
    if _bs > 1:
        return _voxel_wide_spatial_bin(
            df,
            xcol=xcol,
            ycol=ycol,
            zcol=zcol,
            time_cols=time_cols,
            bin_size=_bs,
            method=spatial_bin_method,
            bin_range=spatial_bin_range,
        )

    work = df[[xcol, ycol, zcol] + time_cols].copy()
    for c in [xcol, ycol, zcol]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[time_cols] = work[time_cols].apply(pd.to_numeric, errors="coerce")

    work = _select_voxels_wide(
        work,
        xcol=xcol,
        ycol=ycol,
        zcol=zcol,
        time_cols=time_cols,
        feature_limit=feature_limit,
        feature_sampling=feature_sampling,
        feature_seed=feature_seed,
    )

    coords = work[[xcol, ycol, zcol]].copy()
    coords.columns = ["x", "y", "z"]

    ts = work[time_cols].copy()

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

    n = int(coords.shape[0])
    x = coords["x"].to_numpy(dtype=float, copy=False)
    y = coords["y"].to_numpy(dtype=float, copy=False)
    z = coords["z"].to_numpy(dtype=float, copy=False)

    def _int_or_nan(a: np.ndarray) -> np.ndarray:
        m = np.isfinite(a)
        out = np.empty(a.shape[0], dtype=object)
        out[m] = a[m].astype(np.int64).astype(str)
        out[~m] = "nan"
        return out

    xs = _int_or_nan(x)
    ys = _int_or_nan(y)
    zs = _int_or_nan(z)

    i_str = np.char.zfill(np.arange(n, dtype=np.int64).astype(str), 4)
    voxel_ids = np.char.add(
        np.char.add(
            np.char.add(
                np.char.add(
                    np.char.add(np.char.add(np.char.add("v", i_str), "_x"), xs),
                    "_y",
                ),
                ys,
            ),
            "_z",
        ),
        zs,
    ).astype(object)

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
    out.attrs["format"] = "voxel_wide"
    return out


def _voxel_wide_spatial_bin(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    time_cols: list,
    *,
    bin_size: int = 5,
    method: str = "mean",
    eps: float = 1e-12,
    min_voxels_per_bin: int = 1,
    bin_range: tuple | None = None,
) -> pd.DataFrame:
    """Spatial binning для CSV voxel-wide формата (строки=воксели).

    Bin key: ``floor(coord / bin_size)`` — целочисленная решётка.

    Абсолютно детерминирован: результат зависит **только** от координат (x,y,z),
    ``bin_size`` и ``bin_range``, а не от значений временных рядов.

    Args:
        bin_range: Фиксированная сетка ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``
            в координатах вокселей (не бинов). Если задан — **все** бины в этом
            диапазоне будут в выходе (пустые = NaN). Это гарантирует идентичный
            набор колонок между файлами даже при разном покрытии вокселей.
            Если ``None`` — диапазон берётся из данных текущего файла.

    Возвращает DataFrame time × bins с ``out.attrs['coords']`` и
    ``out.attrs['spatial_bin_report']``.
    """
    b = max(1, int(bin_size))
    method_eff = str(method or "mean").strip().lower()

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[zcol], errors="coerce").to_numpy(dtype=float)

    bx = np.floor(x / b).astype(np.int32)
    by = np.floor(y / b).astype(np.int32)
    bz = np.floor(z / b).astype(np.int32)

    if bin_range is not None:
        (rx0, rx1), (ry0, ry1), (rz0, rz1) = bin_range
        grid_bx_min = int(np.floor(rx0 / b))
        grid_bx_max = int(np.floor(rx1 / b))
        grid_by_min = int(np.floor(ry0 / b))
        grid_by_max = int(np.floor(ry1 / b))
        grid_bz_min = int(np.floor(rz0 / b))
        grid_bz_max = int(np.floor(rz1 / b))
    else:
        grid_bx_min, grid_bx_max = int(bx.min()), int(bx.max())
        grid_by_min, grid_by_max = int(by.min()), int(by.max())
        grid_bz_min, grid_bz_max = int(bz.min()), int(bz.max())

    grid_rx = grid_bx_max - grid_bx_min + 1
    grid_ry = grid_by_max - grid_by_min + 1
    grid_rz = grid_bz_max - grid_bz_min + 1
    n_grid = grid_rx * grid_ry * grid_rz

    bx_off = (bx - grid_bx_min).astype(np.int64)
    by_off = (by - grid_by_min).astype(np.int64)
    bz_off = (bz - grid_bz_min).astype(np.int64)

    in_grid = (
        (bx_off >= 0)
        & (bx_off < grid_rx)
        & (by_off >= 0)
        & (by_off < grid_ry)
        & (bz_off >= 0)
        & (bz_off < grid_rz)
    )

    packed = bx_off * (grid_ry * grid_rz) + by_off * grid_rz + bz_off
    packed[~in_grid] = -1

    n_voxels = len(bx)
    logging.info(
        "[CSV spatial bin] %d вокселей → сетка %d×%d×%d = %d бинов (bin_size=%d, method=%s, range=%s)",
        n_voxels,
        grid_rx,
        grid_ry,
        grid_rz,
        n_grid,
        b,
        method_eff,
        "fixed" if bin_range is not None else "auto",
    )

    ts_arr = df[time_cols].to_numpy(dtype=np.float32)
    n_time = ts_arr.shape[1]
    voxel_var = np.nanvar(ts_arr, axis=1)
    alive = np.isfinite(voxel_var) & (voxel_var > eps) & in_grid

    sums = np.zeros((n_grid, n_time), dtype=np.float64)
    counts = np.zeros(n_grid, dtype=np.int64)
    inv_alive = packed[alive].astype(np.int64)

    if method_eff in {"mean", "sum"}:
        np.add.at(sums, inv_alive, ts_arr[alive].astype(np.float64))
        np.add.at(counts, inv_alive, 1)
        bin_active = counts >= max(1, min_voxels_per_bin)
        if method_eff == "mean":
            result = np.where(
                bin_active[:, None],
                sums / np.maximum(counts[:, None], 1),
                np.nan,
            ).astype(np.float32)
        else:
            result = np.where(bin_active[:, None], sums, np.nan).astype(np.float32)
    else:
        result = np.full((n_grid, n_time), np.nan, dtype=np.float32)
        counts = np.zeros(n_grid, dtype=np.int64)
        for bi in range(n_grid):
            mask = (packed == bi) & alive
            cnt = int(np.sum(mask))
            if cnt < max(1, min_voxels_per_bin):
                continue
            counts[bi] = cnt
            result[bi, :] = np.nanmedian(ts_arr[mask], axis=0)
        bin_active = counts >= max(1, min_voxels_per_bin)

    x_sums = np.zeros(n_grid, dtype=np.float64)
    y_sums = np.zeros(n_grid, dtype=np.float64)
    z_sums = np.zeros(n_grid, dtype=np.float64)
    coord_counts = np.zeros(n_grid, dtype=np.int64)
    inv_all = packed[in_grid].astype(np.int64)
    np.add.at(x_sums, inv_all, x[in_grid])
    np.add.at(y_sums, inv_all, y[in_grid])
    np.add.at(z_sums, inv_all, z[in_grid])
    np.add.at(coord_counts, inv_all, 1)

    grid_indices = np.arange(n_grid)
    all_bx = grid_indices // (grid_ry * grid_rz) + grid_bx_min
    all_by = (grid_indices % (grid_ry * grid_rz)) // grid_rz + grid_by_min
    all_bz = grid_indices % grid_rz + grid_bz_min

    if bin_range is not None:
        keep_idx = np.arange(n_grid)
        n_active = int(np.sum(bin_active))
    else:
        keep_idx = np.where(bin_active)[0]
        n_active = len(keep_idx)

    if n_active == 0:
        raise ValueError(
            f"Spatial binning: все {n_grid} бинов пусты "
            f"(bin_size={b}, вокселей={n_voxels}). Попробуй увеличить bin_size."
        )

    result = result[keep_idx, :]
    bin_names = [f"bin_{all_bx[i]}_{all_by[i]}_{all_bz[i]}" for i in keep_idx]

    coords_rows = []
    for j, i in enumerate(keep_idx):
        cc = max(1, int(coord_counts[i]))
        coords_rows.append(
            {
                "voxel_id": bin_names[j],
                "x": float(x_sums[i] / cc) if cc > 0 else float(all_bx[i] * b + b / 2),
                "y": float(y_sums[i] / cc) if cc > 0 else float(all_by[i] * b + b / 2),
                "z": float(z_sums[i] / cc) if cc > 0 else float(all_bz[i] * b + b / 2),
                "bin_key": f"{all_bx[i]}_{all_by[i]}_{all_bz[i]}",
                "n_voxels": int(coord_counts[i]),
                "n_active": int(counts[i]),
            }
        )

    out = pd.DataFrame(result.T, columns=bin_names)
    coords_df = pd.DataFrame(coords_rows)
    out.attrs["coords"] = coords_df
    out.attrs["format"] = "spatial_bins"
    out.attrs["source_kind"] = "csv_voxel_spatial"
    out.attrs["feature_axis"] = "spatial_bin"
    out.attrs["bin_size"] = b
    out.attrs["spatial_bin_report"] = {
        "original_voxels": int(n_voxels),
        "alive_voxels": int(np.sum(alive)),
        "total_grid_bins": int(n_grid),
        "output_bins": len(keep_idx),
        "active_bins": int(n_active),
        "bin_size": b,
        "method": method_eff,
        "bin_range": bin_range,
        "grid_range_used": (
            (int(grid_bx_min * b), int((grid_bx_max + 1) * b)),
            (int(grid_by_min * b), int((grid_by_max + 1) * b)),
            (int(grid_bz_min * b), int((grid_bz_max + 1) * b)),
        ),
        "bin_key_formula": "floor(coord / bin_size)",
        "deterministic": True,
        "fixed_range": bin_range is not None,
    }

    logging.info(
        "[CSV spatial bin] Результат: %d×%d (из %d вокселей, %d живых → %d/%d бинов, range=%s)",
        out.shape[0],
        out.shape[1],
        n_voxels,
        int(np.sum(alive)),
        n_active,
        len(keep_idx),
        "fixed" if bin_range is not None else "auto",
    )
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




def _mat_value_candidates(obj: Any) -> list[tuple[str, np.ndarray]]:
    """Извлекает числовые 2D/1D массивы из MAT-словаря/объекта."""
    out: list[tuple[str, np.ndarray]] = []

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                try:
                    if value.size == 1:
                        _walk(prefix, value.reshape(-1)[0])
                    return
                except Exception:
                    return
            if value.ndim in (1, 2) and np.issubdtype(value.dtype, np.number):
                out.append((prefix, value))
            return

        if isinstance(value, dict):
            for k, v in value.items():
                if str(k).startswith("__"):
                    continue
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
            return

    _walk("", obj)
    return out


def _mat_to_dataframe(filepath: str) -> pd.DataFrame:
    """Загружает .mat и выбирает наиболее подходящую числовую матрицу."""
    blob = loadmat(filepath, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    cand = _mat_value_candidates(blob)
    if not cand:
        raise ValueError("No numeric arrays found in MAT file")

    def _score(item: tuple[str, np.ndarray]) -> tuple[int, int, int]:
        _name, arr = item
        a = np.asarray(arr)
        return (1 if a.ndim == 2 else 0, int(a.size), int(min(a.shape)) if a.ndim >= 1 else 0)

    name, arr = max(cand, key=_score)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    df = pd.DataFrame(arr)
    try:
        df.attrs["mat_source"] = str(name)
    except Exception:
        pass
    return df




def h5_4d_to_voxel_wide(
    arr4d: np.ndarray,
    *,
    nonzero_mode: str = "any",
    eps: float = 0.0,
    max_voxels: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Преобразует 4D массив (X,Y,Z,T) в DataFrame формата (T, N_voxels).

    Колонки получают имена с координатами вокселей, а в attrs добавляются
    метаданные для downstream-пайплайна (`coords`, `format`, `source_kind`).

    Args:
        arr4d: Входной 4D массив формы (X, Y, Z, T).
        nonzero_mode: Критерий отбора валидных вокселей:
            - "any": хотя бы одна точка по времени имеет |value| > eps;
            - "var": дисперсия ряда > eps.
        eps: Числовой порог для `nonzero_mode`.
        max_voxels: Если задан, ограничивает число вокселей случайной подвыборкой.
        seed: Seed для воспроизводимой подвыборки.

    Returns:
        DataFrame формы (T, N_voxels).

    Raises:
        ValueError: если вход не 4D, временная ось слишком короткая
            или после фильтрации не осталось валидных вокселей.
    """
    if not isinstance(arr4d, np.ndarray):
        arr4d = np.asarray(arr4d)

    if arr4d.ndim != 4:
        raise ValueError(f"h5_4d_to_voxel_wide expects 4D array, got shape={arr4d.shape}")

    x, y, z, t = arr4d.shape
    if t < 2:
        raise ValueError(f"Invalid time axis length: {t}")

    # View без копии: (voxels, time).
    flat = arr4d.reshape(x * y * z, t)

    mode = str(nonzero_mode or "any").strip().lower()
    if mode == "any":
        keep_mask = np.any(np.abs(flat) > float(eps), axis=1)
    elif mode == "var":
        keep_mask = np.nanvar(flat, axis=1) > float(eps)
    else:
        raise ValueError(f"Unknown nonzero_mode: {nonzero_mode}")

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        raise ValueError("No nonzero/valid voxels found in 4D H5 dataset")

    if max_voxels is not None and keep_idx.size > int(max_voxels):
        rng = np.random.default_rng(int(seed))
        keep_idx = np.sort(rng.choice(keep_idx, size=int(max_voxels), replace=False))

    kept = flat[keep_idx, :]
    xs, ys, zs = np.unravel_index(keep_idx, (x, y, z))
    colnames = [f"v{i:06d}_x{xi}_y{yi}_z{zi}" for i, (xi, yi, zi) in enumerate(zip(xs, ys, zs))]

    df = pd.DataFrame(kept.T, columns=colnames)

    coords = [(int(xi), int(yi), int(zi)) for xi, yi, zi in zip(xs, ys, zs)]
    df.attrs["coords"] = coords
    df.attrs["format"] = "voxel_wide"
    df.attrs["source_kind"] = "h5_4d"
    df.attrs["source_shape"] = tuple(arr4d.shape)
    df.attrs["time_axis"] = 3
    df.attrs["feature_axis"] = "voxel"
    return df




def h5_4d_to_spatial_bins(
    arr4d: np.ndarray,
    *,
    bin_size: int = 5,
    eps: float = 1e-12,
    min_voxels_per_bin: int = 1,
) -> pd.DataFrame:
    """Агрегирует 4D массив (X,Y,Z,T) в фиксированные spatial bins и возвращает DataFrame T×K.

    Схема одинакова для всех субъектов при одинаковой геометрии и одинаковом bin_size.
    Это предпочтительный режим для межсубъектного сравнения, когда важна
    вычислимость и детерминированность без атласа.
    """
    arr4d = np.asarray(arr4d)
    if arr4d.ndim != 4:
        raise ValueError(f"h5_4d_to_spatial_bins expects 4D array, got shape={arr4d.shape}")

    b = max(1, int(bin_size))
    x_dim, y_dim, z_dim, t_dim = [int(v) for v in arr4d.shape]

    reduced_cols: dict[str, np.ndarray] = {}
    coords_rows: list[dict[str, Any]] = []

    for x0 in range(0, x_dim, b):
        x1 = min(x0 + b, x_dim)
        for y0 in range(0, y_dim, b):
            y1 = min(y0 + b, y_dim)
            for z0 in range(0, z_dim, b):
                z1 = min(z0 + b, z_dim)

                block = arr4d[x0:x1, y0:y1, z0:z1, :]
                flat = block.reshape(-1, t_dim)
                if flat.size == 0:
                    continue

                var = np.nanvar(flat, axis=1)
                keep = np.isfinite(var) & (var > float(eps))
                if int(np.sum(keep)) < int(max(1, min_voxels_per_bin)):
                    continue

                ts = np.nanmean(flat[keep], axis=0).astype(np.float32, copy=False)
                name = f"bin_x{x0}_{x1}_y{y0}_{y1}_z{z0}_{z1}"
                reduced_cols[name] = ts

                coords_rows.append({
                    "voxel_id": name,
                    "x": float((x0 + x1 - 1) / 2.0),
                    "y": float((y0 + y1 - 1) / 2.0),
                    "z": float((z0 + z1 - 1) / 2.0),
                    "bin_x0": int(x0),
                    "bin_x1": int(x1),
                    "bin_y0": int(y0),
                    "bin_y1": int(y1),
                    "bin_z0": int(z0),
                    "bin_z1": int(z1),
                    "n_voxels": int(np.sum(keep)),
                })

    if not reduced_cols:
        raise ValueError("No spatial bins survived variance filtering in HDF5 volume")

    df = pd.DataFrame(reduced_cols)
    df.attrs["coords"] = pd.DataFrame(coords_rows)
    df.attrs["format"] = "spatial_bins"
    df.attrs["source_kind"] = "h5_4d_spatial"
    df.attrs["time_axis"] = 3
    df.attrs["feature_axis"] = "spatial_bin"
    df.attrs["bin_size"] = int(bin_size)
    return df

def _build_aggregated_h5_path(
    source_path: str,
    *,
    output_dir: str | None = None,
    bin_size: int = 5,
    aggregation: str = "mean",
) -> str:
    """Строит путь для сохранения агрегированного H5."""
    src = Path(source_path)
    root = Path(output_dir) if output_dir else (src.parent / "results" / "aggregated_h5")
    folder = root / f"spatialbin_b{int(bin_size)}_{aggregation}"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / f"{src.stem}.h5")


def save_aggregated_h5(
    out_path: str,
    df: pd.DataFrame,
    *,
    source_path: str,
    bin_size: int,
    original_shape: tuple[int, ...] | list[int] | None = None,
    aggregation: str = "mean",
) -> str:
    """Сохраняет агрегированные spatial-bins в отдельный H5."""
    import h5py

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    coords = df.attrs.get("coords")
    if isinstance(coords, pd.DataFrame):
        coords_df = coords.copy()
    elif isinstance(coords, list):
        coords_df = pd.DataFrame(coords)
    else:
        coords_df = pd.DataFrame()

    with h5py.File(out_path, "w") as f:
        g_agg = f.create_group("agg")
        g_meta = f.create_group("meta")

        arr = df.to_numpy(dtype=np.float32, copy=False)
        g_agg.create_dataset("timeseries", data=arr, compression="gzip")

        if not coords_df.empty:
            xyz_cols = [c for c in ["x", "y", "z"] if c in coords_df.columns]
            if len(xyz_cols) == 3:
                g_agg.create_dataset(
                    "bin_xyz",
                    data=coords_df[xyz_cols].to_numpy(dtype=np.float32, copy=False),
                    compression="gzip",
                )

            count_col = "n_voxels"
            if count_col in coords_df.columns:
                g_agg.create_dataset(
                    "bin_counts",
                    data=coords_df[count_col].to_numpy(dtype=np.int32, copy=False),
                    compression="gzip",
                )

            bound_cols = [c for c in ["bin_x0", "bin_x1", "bin_y0", "bin_y1", "bin_z0", "bin_z1"] if c in coords_df.columns]
            if len(bound_cols) == 6:
                g_agg.create_dataset(
                    "bin_bounds",
                    data=coords_df[bound_cols].to_numpy(dtype=np.int32, copy=False),
                    compression="gzip",
                )

        g_meta.attrs["representation"] = "spatial_bins"
        g_meta.attrs["aggregation"] = str(aggregation)
        g_meta.attrs["bin_size"] = int(bin_size)
        g_meta.attrs["source_file"] = str(source_path)
        g_meta.attrs["orientation"] = "time_by_bins"
        if original_shape is not None:
            g_meta.attrs["original_shape"] = list(original_shape)

    return out_path


def load_aggregated_h5(filepath: str) -> pd.DataFrame:
    """Читает ранее сохранённый aggregated H5 и возвращает DataFrame T×K."""
    import h5py

    with h5py.File(filepath, "r") as f:
        arr = np.asarray(f["agg/timeseries"], dtype=np.float32)

        cols = [f"bin_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)

        coords_df = pd.DataFrame()
        if "agg/bin_xyz" in f:
            xyz = np.asarray(f["agg/bin_xyz"])
            coords_df["x"] = xyz[:, 0]
            coords_df["y"] = xyz[:, 1]
            coords_df["z"] = xyz[:, 2]
        if "agg/bin_counts" in f:
            coords_df["n_voxels"] = np.asarray(f["agg/bin_counts"])
        if "agg/bin_bounds" in f:
            bb = np.asarray(f["agg/bin_bounds"])
            coords_df["bin_x0"] = bb[:, 0]
            coords_df["bin_x1"] = bb[:, 1]
            coords_df["bin_y0"] = bb[:, 2]
            coords_df["bin_y1"] = bb[:, 3]
            coords_df["bin_z0"] = bb[:, 4]
            coords_df["bin_z1"] = bb[:, 5]

        if not coords_df.empty:
            coords_df.insert(0, "voxel_id", cols)
            df.attrs["coords"] = coords_df

        df.attrs["format"] = "spatial_bins"
        df.attrs["source_kind"] = "aggregated_h5"
        return df


def _load_h5_neuroimaging(
    filepath: str,
    *,
    feature_limit: int | None = None,
    feature_sampling: str = "spatial",
    h5_spatial_bin: int | None = None,
    spatial_grid_size: int | None = None,
    spatial_grid_method: str = "mean",
    lazy_spatial_bin: bool = False,
    time_chunk: int = 50,
    feature_seed: int = 13,
    time_start: int | None = None,
    time_end: int | None = None,
    time_stride: int | None = None,
    nonzero_threshold: float = 1e-6,
    default_feature_limit: int = 0,
) -> pd.DataFrame:
    """Загрузка 4D нейровизуализационного HDF5 (X,Y,Z,T) в DataFrame time×voxel.

    Поддерживает формат с dataset shape=(X,Y,Z,T) dtype=float32.
    Возвращает DataFrame shape=(T_eff, N_voxels) с attrs["coords"].

    При feature_sampling="spatial" (или при заданном h5_spatial_bin>1) применяется
    прямая детерминированная 3D-агрегация без случайной подвыборки вокселей.
    Если feature_limit не задан/<=0, дополнительный post-cap не применяется.
    """
    if _h5py is None:
        raise ImportError(
            "h5py не установлен. Для загрузки HDF5 файлов: pip install h5py"
        )

    with _h5py.File(filepath, "r") as f:
        # Ищем наибольший 4D dataset
        ds = None
        ds_name = None

        def _scan_datasets(group, prefix=""):
            nonlocal ds, ds_name
            for name in group:
                path = f"{prefix}/{name}" if prefix else name
                obj = group[name]
                if isinstance(obj, _h5py.Dataset):
                    if obj.ndim == 4:
                        if ds is None or obj.size > ds.size:
                            ds = obj
                            ds_name = path
                elif isinstance(obj, _h5py.Group):
                    _scan_datasets(obj, path)

        _scan_datasets(f)

        # Fallback: 2D dataset (уже time×features или features×time)
        if ds is None:
            ds2d = None
            ds2d_name = None
            for name in f:
                obj = f[name]
                if isinstance(obj, _h5py.Dataset) and obj.ndim == 2:
                    if ds2d is None or obj.size > ds2d.size:
                        ds2d = obj
                        ds2d_name = name
            if ds2d is not None:
                logging.info("[HDF5] Fallback to 2D dataset '%s' shape=%s", ds2d_name, ds2d.shape)
                arr = np.asarray(ds2d[()], dtype=np.float32)
                if arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                df = pd.DataFrame(arr, columns=[f"c{i+1}" for i in range(arr.shape[1])])
                return df
            raise ValueError(
                f"Нет подходящего 4D/2D dataset в HDF5: {filepath}. "
                f"Доступные ключи: {list(f.keys())}"
            )

        shape = ds.shape
        logging.info("[HDF5] Dataset '%s' shape=%s dtype=%s", ds_name, shape, ds.dtype)

        # Определяем ось времени. Конвенция нейровизуализации: (X,Y,Z,T) → последняя ось.
        # Проверяем: если последняя размерность ≥ каждой из остальных, это T.
        # Иначе берём ось с наибольшим значением.
        spatial_dims = list(shape[:3])
        t_dim = shape[3] if len(shape) == 4 else shape[-1]
        T_axis = 3
        if t_dim < max(spatial_dims):
            # Нестандартный порядок осей: T может быть первой (T,X,Y,Z)
            T_axis = int(np.argmax(shape))
            logging.warning(
                "[HDF5] Нестандартный порядок осей: T_axis=%d (shape=%s)", T_axis, shape
            )

        T = int(shape[T_axis])

        # Ленивый путь: spatial-агрегация читается чанками по времени,
        # чтобы не загружать весь 4D объём в память сразу.
        feature_mode = str(feature_sampling or "spatial").strip().lower()
        grid_size_eff = int(spatial_grid_size) if spatial_grid_size is not None else 0
        if grid_size_eff <= 0:
            grid_size_eff = int(h5_spatial_bin) if h5_spatial_bin is not None and int(h5_spatial_bin) > 1 else 0
        spatial_mode = feature_mode in {"spatial", "spatial_bin", "bins", "auto", "deterministic"} or (grid_size_eff > 1)
        if bool(lazy_spatial_bin) and spatial_mode and T_axis == 3:
            bin_size = grid_size_eff if grid_size_eff > 1 else 5
            df_lazy = load_h5_spatial_binned_lazy(
                filepath,
                dataset=str(ds_name),
                grid_size=bin_size,
                method=str(spatial_grid_method or "mean"),
                time_chunk=int(time_chunk or 50),
            )
            # Локальный временной срез применяем уже к агрегированным рядам.
            t0 = int(time_start) if time_start is not None else 0
            t1 = int(time_end) if time_end is not None else int(df_lazy.shape[0])
            ts = int(time_stride) if time_stride is not None and int(time_stride) > 0 else 1
            t0 = max(0, min(t0, int(df_lazy.shape[0])))
            t1 = max(t0, min(t1, int(df_lazy.shape[0])))
            df_lazy = df_lazy.iloc[t0:t1:ts, :].copy()
            var_eps = max(1e-12, float(nonzero_threshold))
            if not df_lazy.empty:
                v = np.nanvar(df_lazy.to_numpy(dtype=np.float64, copy=False), axis=0)
                keep = np.isfinite(v) & (v > float(var_eps))
                if keep.any():
                    df_lazy = df_lazy.loc[:, keep].copy()
            df_lazy.attrs["original_shape"] = list(shape)
            df_lazy.attrs["format"] = "spatial_bins"
            df_lazy.attrs["source_kind"] = "h5_4d_spatial"
            df_lazy.attrs["spatial_bin_size"] = int(bin_size)
            df_lazy.attrs["feature_axis"] = "spatial_bin"
            return df_lazy

        # Slicing по времени (до загрузки всего массива в память)
        t0 = int(time_start) if time_start is not None else 0
        t1 = int(time_end) if time_end is not None else T
        ts = int(time_stride) if time_stride is not None and int(time_stride) > 0 else 1
        t0 = max(0, min(t0, T))
        t1 = max(t0, min(t1, T))

        idx = [slice(None)] * len(shape)
        idx[T_axis] = slice(t0, t1, ts)

        logging.info("[HDF5] Loading slice %s ...", idx)
        arr4d = np.asarray(ds[tuple(idx)], dtype=np.float32)

    # Перемещаем ось T в конец если она не там
    if T_axis != len(arr4d.shape) - 1:
        arr4d = np.moveaxis(arr4d, T_axis, -1)

    *spatial, T_actual = arr4d.shape
    n_total = int(np.prod(spatial))

    logging.info(
        "[HDF5] Spatial=%s T=%d, total voxels=%d, array=%.1f MB",
        spatial, T_actual, n_total,
        arr4d.nbytes / (1024**2),
    )

    var_eps = max(1e-12, float(nonzero_threshold))
    feature_mode = str(feature_sampling or "spatial").strip().lower()
    # Совместимость: новый алиас spatial_grid_size/method для 4D fMRI биннинга.
    grid_size_eff = int(spatial_grid_size) if spatial_grid_size is not None else 0
    if grid_size_eff <= 0:
        grid_size_eff = int(h5_spatial_bin) if h5_spatial_bin is not None and int(h5_spatial_bin) > 1 else 0

    spatial_mode = feature_mode in {"spatial", "spatial_bin", "bins", "auto", "deterministic"} or (grid_size_eff > 1)

    if spatial_mode:
        # Используем общий helper из preprocessing, чтобы логика spatial-grid была
        # одинаковой для H5 и потенциальных прямых вызовов из других модулей.
        bin_size = grid_size_eff if grid_size_eff > 1 else 5
        df_h5 = spatial_grid_bin_fmri(
            arr4d,
            grid_size=bin_size,
            method=str(spatial_grid_method or "mean"),
        )
        # Отбрасываем почти-константные бины, чтобы защититься от пустых областей.
        if not df_h5.empty:
            v = np.nanvar(df_h5.to_numpy(dtype=np.float64, copy=False), axis=0)
            keep = np.isfinite(v) & (v > float(var_eps))
            if keep.any():
                df_h5 = df_h5.loc[:, keep].copy()
        df_h5.attrs["original_shape"] = list(shape)
        df_h5.attrs["format"] = "spatial_bins"
        df_h5.attrs["source_kind"] = "h5_4d_spatial"
        df_h5.attrs["spatial_bin_size"] = int(bin_size)
        df_h5.attrs["feature_axis"] = "spatial_bin"
        del arr4d
        logging.info(
            "[HDF5] Spatial binning applied directly on 4D volume: %s -> %s (bin=%d)",
            shape,
            df_h5.shape,
            bin_size,
        )
        return df_h5

    # Безопасный pre-cap для GUI: не раздуваем DataFrame до сотен тысяч колонок.
    df_h5 = h5_4d_to_voxel_wide(
        arr4d,
        nonzero_mode="var",
        eps=var_eps,
        max_voxels=MAX_RAW_VOXELS_FOR_GUI,
        seed=int(feature_seed),
    )
    del arr4d

    n_selected = int(df_h5.shape[1])
    logging.info("[HDF5] Pre-capped voxel rows for GUI: %d (limit=%d)", n_selected, MAX_RAW_VOXELS_FOR_GUI)

    # Дополнительное ограничение с учётом пользовательского feature_limit.
    k = feature_limit
    if k is None or int(k) <= 0:
        k = n_selected if int(default_feature_limit or 0) <= 0 else default_feature_limit
    k = int(min(int(k), n_selected))

    if n_selected > k:
        mode = str(feature_sampling or "variance").strip().lower()
        if mode in {"random", "rand"}:
            rng = np.random.default_rng(int(feature_seed))
            idx = np.sort(rng.choice(n_selected, size=k, replace=False))
        elif mode in {"variance", "var", "topvar"}:
            vals = df_h5.to_numpy(dtype=np.float64, copy=False)
            v = np.nanvar(vals, axis=0)
            idx = np.argpartition(v, -k)[-k:]
            idx = idx[np.argsort(-v[idx])]
        elif mode in {"activity", "act"}:
            vals = df_h5.to_numpy(dtype=np.float64, copy=False)
            a = np.nansum(np.abs(vals), axis=0)
            idx = np.argpartition(a, -k)[-k:]
            idx = idx[np.argsort(-a[idx])]
        else:
            idx = np.arange(k, dtype=int)

        df_h5 = df_h5.iloc[:, idx].copy()
        coords_attr = df_h5.attrs.get("coords")
        if isinstance(coords_attr, list):
            df_h5.attrs["coords"] = [coords_attr[i] for i in idx]
        logging.info("[HDF5] Subsampled after pre-cap: %d -> %d voxels (mode=%s)", n_selected, k, mode)

    # Нормализуем attrs['coords'] к DataFrame для совместимости с остальным кодом.
    coords_attr = df_h5.attrs.get("coords")
    if isinstance(coords_attr, list):
        coords_df = pd.DataFrame(coords_attr, columns=["x", "y", "z"])
        coords_df.insert(0, "voxel_id", [str(c) for c in df_h5.columns])
        df_h5.attrs["coords"] = coords_df
    elif isinstance(coords_attr, pd.DataFrame):
        # Гарантируем наличие voxel_id и согласованность порядка с колонками.
        coords_df = coords_attr.copy()
        if "voxel_id" not in coords_df.columns or len(coords_df) != len(df_h5.columns):
            coords_df = pd.DataFrame(
                {
                    "voxel_id": [str(c) for c in df_h5.columns],
                    "x": [np.nan] * len(df_h5.columns),
                    "y": [np.nan] * len(df_h5.columns),
                    "z": [np.nan] * len(df_h5.columns),
                }
            )
        df_h5.attrs["coords"] = coords_df

    logging.info("[HDF5] Output DataFrame: %s (%.1f MB)", df_h5.shape, df_h5.memory_usage(deep=True).sum() / (1024**2))
    return df_h5


def _csv_probe_ncols(filepath: str, *, nrows: int = 2) -> int:
    """Быстрый probe числа колонок CSV без полной загрузки.

    Читает только ``nrows`` строк, чтобы определить ширину таблицы.
    Возвращает 0 при ошибке.
    """
    try:
        probe = pd.read_csv(filepath, header=None, nrows=nrows)
        return int(probe.shape[1])
    except Exception:
        return 0

def read_input_table(
    filepath: str,
    header: str = "auto",
    *,
    usecols: Any = "auto",
    csv_engine: str = "auto",
) -> pd.DataFrame:
    """Чтение CSV/XLSX/PARQUET/MAT с поддержкой автодетекта заголовка и «CSV в ячейке».

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

    if low.endswith(".mat"):
        if header not in {"auto", "yes", "no"}:
            raise ValueError("header must be one of: auto|yes|no")
        return _mat_to_dataframe(fp)

    if low.endswith((".h5", ".hdf5", ".hdf")):
        # HDF5 обрабатывается целиком в load_or_generate → _load_h5_neuroimaging.
        # Если вызвали read_input_table напрямую, делаем базовую загрузку.
        return _load_h5_neuroimaging(fp)

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
        # Поддержка usecols для CSV: ограничение колонок на этапе чтения,
        # чтобы избежать OOM на очень широких файлах (сотни тысяч вокселей).
        if usecols != "auto" and usecols is not None:
            kw["usecols"] = usecols
        df0 = pd.read_csv(fp, **kw)
    else:
        xl_usecols = usecols
        excel_probe_single_col = False
        excel_engine = None

        # Явно выбираем движок по расширению, чтобы чтение Excel было стабильнее.
        # openpyxl покрывает .xlsx/.xlsm, а xlrd нужен для legacy .xls.
        if low.endswith(".xls"):
            excel_engine = "xlrd"
        elif low.endswith((".xlsx", ".xlsm")):
            excel_engine = "openpyxl"

        if usecols == "auto":
            # Лёгкий probe первой колонки нужен для кейса «CSV в одной ячейке».
            # Если это обычный XLS/XLSX с несколькими колонками, ниже перечитаем лист целиком.
            xl_usecols = [0]
            excel_probe_single_col = True

        try:
            df0 = pd.read_excel(fp, header=None, usecols=xl_usecols, engine=excel_engine)
        except ImportError as exc:
            if low.endswith(".xls"):
                raise ImportError(
                    "Для чтения .xls нужен пакет xlrd. Установи: pip install xlrd>=2.0.1"
                ) from exc
            raise
        except Exception:
            # Если чтение с выбором колонок не удалось, мягко падаем на полное чтение.
            df0 = pd.read_excel(fp, header=None, engine=excel_engine)
            excel_probe_single_col = False

        if excel_probe_single_col:
            probe_df = df0
            split_probe = _maybe_split_single_column(probe_df)

            # Если одиночная колонка распалась на несколько полей,
            # это действительно «CSV в ячейке», полный reread не нужен.
            # Иначе считаем, что это обычный Excel, и читаем весь лист.
            probe_is_embedded_csv = (
                split_probe.shape[1] > probe_df.shape[1]
                or (probe_df.shape[1] == 1 and split_probe.shape[1] > 1)
            )

            if not probe_is_embedded_csv:
                try:
                    df0 = pd.read_excel(fp, header=None, engine=excel_engine)
                except Exception:
                    df0 = probe_df
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
    spatial_bin_size: int = 0,
    spatial_bin_method: str = "mean",
    spatial_bin_range: tuple | None = None,
) -> pd.DataFrame:
    """Превращает сырую таблицу в numeric матрицу вида time × features."""
    out = df.copy()
    out = out.dropna(axis=1, how="all")

    # Спец-кейс: x,y,z,t0..tN (воксельный wide-формат)
    try:
        out = voxel_wide_to_timeseries(
            out,
            feature_limit=feature_limit,
            feature_sampling=feature_sampling,
            feature_seed=feature_seed,
            spatial_bin_size=spatial_bin_size,
            spatial_bin_method=spatial_bin_method,
            spatial_bin_range=spatial_bin_range,
        )
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

    input_format = str(_saved_attrs.get("format") or "").strip().lower()
    source_kind = str(_saved_attrs.get("source_kind") or "").strip().lower()
    has_voxel_coords = _saved_attrs.get("coords") is not None

    is_voxel_like = (
        input_format == "voxel_wide"
        or source_kind == "h5_4d"
        or bool(has_voxel_coords)
    )

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
            corr_lists_by_lag = {int(k): [] for k in range(1, p_order + 1)}
            lb_lists_by_lag = {int(k): [] for k in range(1, p_order + 1)}

            # Сохраним 3 примера: максимальная |phi1| (до очистки).
            top = []  # list[tuple[abs_phi, col, phi1]]
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
                    float(np.mean(np.asarray(vals, dtype=float) < 0.05))
                    if vals else float("nan")
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
                "frac_lb_p_lt_0_05_lagp": frac_lb_bad_by_lag.get(f"lag{int(p_order)}", float("nan")),
                "corr_by_lag": corr_summary_by_lag,
                "ljungbox_p_by_lag": lb_summary_by_lag,
                "frac_lb_p_lt_0_05_by_lag": frac_lb_bad_by_lag,
            }

            # Сохраним сами ряды для примеров (до очистки).
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
                    float(np.mean(np.asarray(vals, dtype=float) < 0.05))
                    if vals else float("nan")
                )
                for k, vals in lb_lists_by_lag.items()
            }

            ac_note["after"] = {
                "phi1": _summary(phi1_list),
                "corr_lag1": corr_summary_by_lag.get("lag1", {"n": 0}),
                "corr_lagp": corr_summary_by_lag.get(f"lag{int(p_order)}", {"n": 0}),
                "ljungbox_p_lag1": lb_summary_by_lag.get("lag1", {"n": 0}),
                "ljungbox_p_lagp": lb_summary_by_lag.get(f"lag{int(p_order)}", {"n": 0}),
                "frac_lb_p_lt_0_05_lag1": frac_lb_bad_by_lag.get("lag1", float("nan")),
                "frac_lb_p_lt_0_05_lagp": frac_lb_bad_by_lag.get(f"lag{int(p_order)}", float("nan")),
                "corr_by_lag": corr_summary_by_lag,
                "ljungbox_p_by_lag": lb_summary_by_lag,
                "frac_lb_p_lt_0_05_by_lag": frac_lb_bad_by_lag,
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
                            item["after_corr_by_lag"] = {
                                f"lag{int(k)}": _safe_corr_at_lag(xa, int(k))
                                for k in range(1, p_order + 1)
                            }
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

            # И то же самое по всем лагам 1..p.
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
            except Exception:
                pass

            report.notes["autocorr"] = ac_note

    if remove_seasonality:
        # STL сезонность: либо заданный период, либо пробуем оценить.
        report.add("[Preprocess] remove seasonality: STL (if period detected)")
        try:
            from statsmodels.tsa.seasonal import STL
            from ..analysis import stats as seasonality_stats
        except Exception:
            STL = None
            seasonality_stats = None

        if STL is not None and seasonality_stats is not None:
            for col in out.columns:
                if not pd.api.types.is_numeric_dtype(out[col]):
                    continue
                x = out[col].astype(float)
                if x.size < 30:
                    continue
                per = int(season_period) if season_period is not None and int(season_period) >= 2 else None
                if per is None:
                    try:
                        ss = seasonality_stats.detect_seasonality(x)
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
                    _, pvalue = analysis_stats.test_stationarity(series)
                    if pvalue is not None:
                        logging.info(
                            f"Ряд '{col}' {'стационарен' if pvalue <= 0.05 else 'вероятно нестационарен'} (p-value ADF={pvalue:.3f})."
                        )
                    else:
                        logging.debug(
                            "ADF skipped for column '%s' (constant/short/degenerate series).",
                            col,
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
    h5_spatial_bin: int | None = None,
    spatial_grid_size: int | None = None,
    spatial_grid_method: str = "mean",
    spatial_bin_range: tuple | None = None,
    lazy_spatial_bin: bool = False,
    time_chunk: int = 50,
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
    save_aggregated_h5: bool = False,
    aggregated_h5_dir: str | None = None,
    reuse_existing_aggregated_h5: bool = True,
    usecols: Any = "auto",
    csv_engine: str = "auto",
    csv_stream_spatial_bin: bool = True,
    csv_chunk_rows: int = 4096,
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
        _fp_low = str(filepath).lower()

        # HDF5 neuroimaging: отдельный путь, т.к. 4D→2D конверсия
        # принципиально отличается от табличного CSV/Excel пайплайна.
        if _fp_low.endswith((".h5", ".hdf5", ".hdf")):
            h5_feature_sampling = str(feature_sampling or "").strip().lower()
            if h5_feature_sampling in {"", "first", "auto"}:
                h5_feature_sampling = "spatial"

            aggregated_path = None
            if h5_feature_sampling in {"spatial", "spatial_bin", "bins", "deterministic", "auto"}:
                _bin_size = int(spatial_grid_size) if spatial_grid_size is not None and int(spatial_grid_size) > 1 else (int(h5_spatial_bin) if h5_spatial_bin is not None and int(h5_spatial_bin) > 1 else 5)
                aggregated_path = _build_aggregated_h5_path(
                    filepath,
                    output_dir=aggregated_h5_dir,
                    bin_size=_bin_size,
                    aggregation="mean",
                )
                if bool(reuse_existing_aggregated_h5) and os.path.exists(aggregated_path):
                    logging.info("[HDF5] Reusing aggregated H5: %s", aggregated_path)
                    df = load_aggregated_h5(aggregated_path)
                    df.attrs["aggregated_h5_path"] = aggregated_path
                else:
                    df = _load_h5_neuroimaging(
                        filepath,
                        feature_limit=feature_limit,
                        feature_sampling=h5_feature_sampling,
                        feature_seed=feature_seed,
                        h5_spatial_bin=h5_spatial_bin,
                        spatial_grid_size=spatial_grid_size,
                        spatial_grid_method=spatial_grid_method,
                        lazy_spatial_bin=lazy_spatial_bin,
                        time_chunk=time_chunk,
                        time_start=time_start,
                        time_end=time_end,
                        time_stride=time_stride,
                    )
                    if bool(save_aggregated_h5) and str(getattr(df, "attrs", {}).get("format", "")).lower() == "spatial_bins":
                        try:
                            globals()["save_aggregated_h5"](
                                aggregated_path,
                                df,
                                source_path=filepath,
                                bin_size=_bin_size,
                                original_shape=getattr(df, "attrs", {}).get("original_shape"),
                                aggregation="mean",
                            )
                            df.attrs["aggregated_h5_path"] = aggregated_path
                        except Exception as exc:
                            logging.warning("Failed to save aggregated H5: %s", exc)
            else:
                df = _load_h5_neuroimaging(
                    filepath,
                    feature_limit=feature_limit,
                    feature_sampling=h5_feature_sampling,
                    feature_seed=feature_seed,
                    h5_spatial_bin=h5_spatial_bin,
                    spatial_grid_size=spatial_grid_size,
                    spatial_grid_method=spatial_grid_method,
                    lazy_spatial_bin=lazy_spatial_bin,
                    time_chunk=time_chunk,
                    time_start=time_start,
                    time_end=time_end,
                    time_stride=time_stride,
                )
        else:
            _csv_spatial_bin_size = int(spatial_grid_size or h5_spatial_bin or 0)
            _fp_low_csv = str(filepath).lower()
            df = None

            _try_stream_csv_spatial = (
                _fp_low_csv.endswith(".csv")
                and bool(csv_stream_spatial_bin)
                and _csv_spatial_bin_size > 1
                and spatial_bin_range is None
                and str(spatial_grid_method or "mean").strip().lower() in {"mean", "sum"}
                and usecols in {"auto", None}
            )
            if _try_stream_csv_spatial:
                try:
                    _layout = _probe_csv_voxel_wide_layout(filepath, header=header, csv_engine=csv_engine)
                    if bool(_layout.get("is_voxel_wide")):
                        df = stream_csv_voxel_wide_to_timeseries(
                            filepath,
                            header=header,
                            csv_engine=csv_engine,
                            chunksize=int(csv_chunk_rows),
                            spatial_bin_size=_csv_spatial_bin_size,
                            spatial_bin_method=str(spatial_grid_method or "mean"),
                            spatial_bin_range=None,
                        )
                except Exception as exc:
                    logging.warning("[CSV spatial bin stream] fallback to regular loader: %s", exc)
                    df = None

            if df is None:
                # Защита от OOM на очень широких CSV (сотни тысяч колонок-вокселей):
                # пробуем определить число колонок быстрым probe и, если их слишком
                # много, ограничиваем usecols ДО полной загрузки — аналогично тому,
                # как H5-путь применяет spatial binning / MAX_RAW_VOXELS_FOR_GUI.
                _usecols_eff = usecols
                if _fp_low_csv.endswith(".csv") and _usecols_eff in {"auto", None}:
                    _csv_ncols = _csv_probe_ncols(str(filepath))
                    # Максимум колонок для безопасной загрузки CSV.
                    # feature_limit, если задан, имеет приоритет.
                    _csv_col_cap = MAX_RAW_VOXELS_FOR_GUI
                    if feature_limit is not None and int(feature_limit) > 0:
                        _csv_col_cap = int(feature_limit)
                    if _csv_ncols > _csv_col_cap > 0:
                        logging.warning(
                            "[CSV] Файл содержит %d колонок (лимит=%d). "
                            "Ограничиваем usecols при чтении для предотвращения OOM.",
                            _csv_ncols,
                            _csv_col_cap,
                        )
                        _mode = str(feature_sampling or "first").strip().lower()
                        if _mode in {"random", "rand"}:
                            _rng = np.random.default_rng(int(feature_seed))
                            _pick = sorted(_rng.choice(_csv_ncols, size=_csv_col_cap, replace=False).tolist())
                            _usecols_eff = _pick
                        else:
                            # "first" / default — первые N колонок
                            _usecols_eff = list(range(_csv_col_cap))

                raw = read_input_table(filepath, header=header, usecols=_usecols_eff, csv_engine=csv_engine)

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
                    spatial_bin_size=_csv_spatial_bin_size,
                    spatial_bin_method=str(spatial_grid_method or "mean"),
                    spatial_bin_range=spatial_bin_range,
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
                report.notes["format"] = str(df.attrs.get("format", "voxel_wide"))
                report.notes["n_voxels"] = int(getattr(coords_df, "shape", [0])[0])
                report.notes["coords"] = coords_df.to_dict(orient="records")
            except Exception:
                pass

        if report is not None:
            try:
                sb_report = df.attrs.get("spatial_bin_report")
                if isinstance(sb_report, dict):
                    report.notes["spatial_bin_report"] = sb_report
                    report.add(
                        f"Spatial binning: {sb_report.get('original_voxels', '?')} вокселей → "
                        f"{sb_report.get('active_bins', '?')} бинов "
                        f"(bin_size={sb_report.get('bin_size', '?')}, "
                        f"метод={sb_report.get('method', '?')}, детерминирован)"
                    )
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


def _csv_choose_stream_engine(csv_engine: str = "auto") -> str:
    """Выбирает движок CSV, совместимый с chunksize/итерацией."""
    eng = str(csv_engine or "auto").strip().lower()
    if eng in {"c", "python"}:
        return eng
    return "c"


def _read_csv_probe_raw(
    filepath: str,
    *,
    nrows: int = 8,
    csv_engine: str = "auto",
) -> pd.DataFrame:
    """Читает небольшой сырой probe CSV без полной загрузки файла."""
    kw: Dict[str, Any] = {"header": None, "nrows": int(nrows), "low_memory": False}
    kw["engine"] = _csv_choose_stream_engine(csv_engine)
    return pd.read_csv(filepath, **kw)


def _probe_csv_voxel_wide_layout(
    filepath: str,
    *,
    header: str = "auto",
    csv_engine: str = "auto",
) -> dict[str, Any]:
    """Пробует дешёво определить, что CSV имеет формат x,y,z,t0..tN."""
    if header not in {"auto", "yes", "no"}:
        raise ValueError("header must be one of: auto|yes|no")

    raw = _read_csv_probe_raw(filepath, nrows=8, csv_engine=csv_engine)
    raw = _maybe_split_single_column(raw)
    has_header = _detect_header(raw) if header == "auto" else (header == "yes")
    if has_header:
        hdr = raw.iloc[0].astype(str).tolist()
        df = raw.iloc[1:].copy()
        df.columns = [h if h.strip() else f"c{i+1}" for i, h in enumerate(hdr)]
    else:
        df = raw.copy()
        df.columns = [f"c{i+1}" for i in range(df.shape[1])]

    is_vox, lower = _detect_voxel_wide(df)
    layout: dict[str, Any] = {
        "has_header": bool(has_header),
        "is_voxel_wide": bool(is_vox),
        "columns": list(df.columns),
        "lower": lower,
    }
    if is_vox:
        layout["xcol"] = lower["x"]
        layout["ycol"] = lower["y"]
        layout["zcol"] = lower["z"]
        layout["time_cols"] = [c for c in df.columns if str(c).strip().lower() not in {"x", "y", "z"}]
    return layout


def _iter_csv_voxel_wide_chunks(
    filepath: str,
    *,
    header: str = "auto",
    csv_engine: str = "auto",
    chunksize: int = 4096,
):
    """Итератор по чанкам CSV для формата x,y,z,t0..tN."""
    layout = _probe_csv_voxel_wide_layout(filepath, header=header, csv_engine=csv_engine)
    if not bool(layout.get("is_voxel_wide")):
        raise ValueError("CSV is not in voxel-wide format x,y,z,t0..tN")

    kw: Dict[str, Any] = {
        "chunksize": max(1, int(chunksize)),
        "low_memory": False,
        "engine": _csv_choose_stream_engine(csv_engine),
    }
    if bool(layout.get("has_header")):
        kw["header"] = 0
    else:
        kw["header"] = None
        kw["names"] = list(layout["columns"])

    for chunk in pd.read_csv(filepath, **kw):
        yield chunk


def stream_csv_voxel_wide_to_timeseries(
    filepath: str,
    *,
    header: str = "auto",
    csv_engine: str = "auto",
    chunksize: int = 4096,
    spatial_bin_size: int = 5,
    spatial_bin_method: str = "mean",
    spatial_bin_range: tuple | None = None,
    eps: float = 1e-12,
    min_voxels_per_bin: int = 1,
) -> pd.DataFrame:
    """Потоковая загрузка giant CSV x,y,z,t0..tN → time×bins.

    Режим ориентирован на локально-детерминированную биннизацию (вариант A):
    бин каждого вокселя зависит только от его координат и ``bin_size``.
    Пустые бины не форсятся в выход — число бинов может отличаться между файлами,
    но одинаковые воксели всегда получают одинаковый ``bin_key``.

    Для скорости потоково поддерживаются ``mean`` и ``sum``. ``median`` требует
    хранения всех рядов внутри бина и здесь сознательно не поддерживается.
    """
    layout = _probe_csv_voxel_wide_layout(filepath, header=header, csv_engine=csv_engine)
    if not bool(layout.get("is_voxel_wide")):
        raise ValueError("CSV is not in voxel-wide format x,y,z,t0..tN")

    method_eff = str(spatial_bin_method or "mean").strip().lower()
    if method_eff not in {"mean", "sum"}:
        raise ValueError(
            "Streaming CSV spatial binning currently supports only mean/sum; "
            f"got method={spatial_bin_method!r}"
        )
    if spatial_bin_range is not None:
        raise ValueError(
            "Streaming CSV spatial binning currently implements only local deterministic mode "
            "(spatial_bin_range=None)."
        )

    b = max(1, int(spatial_bin_size or 1))
    xcol = str(layout["xcol"])
    ycol = str(layout["ycol"])
    zcol = str(layout["zcol"])
    time_cols = list(layout["time_cols"])

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
        time_cols = [time_cols[i] for i in order]
    else:
        order = np.arange(len(time_cols), dtype=int)

    bin_to_idx: Dict[tuple[int, int, int], int] = {}
    bin_keys: list[tuple[int, int, int]] = []
    sums_rows: list[np.ndarray] = []
    counts_rows: list[int] = []
    xsum_rows: list[float] = []
    ysum_rows: list[float] = []
    zsum_rows: list[float] = []
    coord_count_rows: list[int] = []

    n_chunks = 0
    n_voxels_total = 0
    n_alive_total = 0
    n_time: int | None = None

    for chunk in _iter_csv_voxel_wide_chunks(
        filepath,
        header=header,
        csv_engine=csv_engine,
        chunksize=chunksize,
    ):
        n_chunks += 1
        work = chunk[[xcol, ycol, zcol] + time_cols].copy()
        x = pd.to_numeric(work[xcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        y = pd.to_numeric(work[ycol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        z = pd.to_numeric(work[zcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        ts_arr = work[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if order.size:
            ts_arr = ts_arr[:, order]
        if n_time is None:
            n_time = int(ts_arr.shape[1])
        n_voxels_total += int(ts_arr.shape[0])
        if ts_arr.size == 0:
            continue

        voxel_var = np.nanvar(ts_arr, axis=1)
        alive = np.isfinite(voxel_var) & (voxel_var > eps) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(alive):
            continue
        n_alive_total += int(np.sum(alive))

        xa = x[alive]
        ya = y[alive]
        za = z[alive]
        ta = ts_arr[alive].astype(np.float64, copy=False)

        bx = np.floor(xa / b).astype(np.int32)
        by = np.floor(ya / b).astype(np.int32)
        bz = np.floor(za / b).astype(np.int32)
        coords = np.stack([bx, by, bz], axis=1)
        uniq, inv = np.unique(coords, axis=0, return_inverse=True)
        n_uniq = int(uniq.shape[0])
        if n_uniq == 0:
            continue

        chunk_sums = np.zeros((n_uniq, ta.shape[1]), dtype=np.float64)
        np.add.at(chunk_sums, inv, ta)
        chunk_counts = np.bincount(inv, minlength=n_uniq).astype(np.int64)
        chunk_xsum = np.bincount(inv, weights=xa, minlength=n_uniq).astype(np.float64)
        chunk_ysum = np.bincount(inv, weights=ya, minlength=n_uniq).astype(np.float64)
        chunk_zsum = np.bincount(inv, weights=za, minlength=n_uniq).astype(np.float64)

        for j in range(n_uniq):
            key = (int(uniq[j, 0]), int(uniq[j, 1]), int(uniq[j, 2]))
            idx = bin_to_idx.get(key)
            if idx is None:
                idx = len(bin_keys)
                bin_to_idx[key] = idx
                bin_keys.append(key)
                sums_rows.append(chunk_sums[j].copy())
                counts_rows.append(int(chunk_counts[j]))
                xsum_rows.append(float(chunk_xsum[j]))
                ysum_rows.append(float(chunk_ysum[j]))
                zsum_rows.append(float(chunk_zsum[j]))
                coord_count_rows.append(int(chunk_counts[j]))
            else:
                sums_rows[idx] += chunk_sums[j]
                counts_rows[idx] += int(chunk_counts[j])
                xsum_rows[idx] += float(chunk_xsum[j])
                ysum_rows[idx] += float(chunk_ysum[j])
                zsum_rows[idx] += float(chunk_zsum[j])
                coord_count_rows[idx] += int(chunk_counts[j])

    if not bin_keys or n_time is None:
        raise ValueError(
            f"Streaming CSV spatial binning produced no active bins (bin_size={b}, file={filepath})."
        )

    keep = [i for i, c in enumerate(counts_rows) if int(c) >= max(1, int(min_voxels_per_bin))]
    if not keep:
        raise ValueError(
            f"Streaming CSV spatial binning: all bins are below min_voxels_per_bin={min_voxels_per_bin}."
        )

    keep_sorted = sorted(keep, key=lambda i: bin_keys[i])
    result = np.vstack([sums_rows[i] for i in keep_sorted]).astype(np.float64, copy=False)
    counts_arr = np.asarray([counts_rows[i] for i in keep_sorted], dtype=np.int64)
    if method_eff == "mean":
        result = (result / np.maximum(counts_arr[:, None], 1)).astype(np.float32)
    else:
        result = result.astype(np.float32)

    bin_names = [f"bin_{bin_keys[i][0]}_{bin_keys[i][1]}_{bin_keys[i][2]}" for i in keep_sorted]
    coords_rows = []
    for pos, i in enumerate(keep_sorted):
        cc = max(1, int(coord_count_rows[i]))
        bx_i, by_i, bz_i = bin_keys[i]
        coords_rows.append(
            {
                "voxel_id": bin_names[pos],
                "x": float(xsum_rows[i] / cc),
                "y": float(ysum_rows[i] / cc),
                "z": float(zsum_rows[i] / cc),
                "bin_key": f"{bx_i}_{by_i}_{bz_i}",
                "n_voxels": int(coord_count_rows[i]),
                "n_active": int(counts_rows[i]),
            }
        )

    out = pd.DataFrame(result.T, columns=bin_names)
    out.attrs["coords"] = pd.DataFrame(coords_rows)
    out.attrs["format"] = "spatial_bins"
    out.attrs["source_kind"] = "csv_voxel_spatial_stream"
    out.attrs["feature_axis"] = "spatial_bin"
    out.attrs["bin_size"] = b
    out.attrs["voxel_time_cols"] = [str(c) for c in time_cols]
    out.attrs["spatial_bin_report"] = {
        "original_voxels": int(n_voxels_total),
        "alive_voxels": int(n_alive_total),
        "output_bins": len(keep_sorted),
        "bin_size": b,
        "method": method_eff,
        "bin_range": None,
        "bin_key_formula": "floor(coord / bin_size)",
        "deterministic": True,
        "fixed_range": False,
        "streaming": True,
        "chunksize": int(chunksize),
        "n_chunks": int(n_chunks),
    }
    logging.info(
        "[CSV spatial bin stream] Result: %d×%d from %d voxels (%d alive), %d bins, chunks=%d, chunk_rows=%d",
        out.shape[0],
        out.shape[1],
        int(n_voxels_total),
        int(n_alive_total),
        len(keep_sorted),
        int(n_chunks),
        int(chunksize),
    )
    return out
