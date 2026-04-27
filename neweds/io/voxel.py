"""Хелперы для voxel-wide CSV-формата.

Voxel-wide раскладка: ``x, y, z, t0, t1, ..., tN`` — одна строка на воксель,
каждый столбец после ``z`` это значение сигнала в соответствующий момент времени.

Здесь живут:

- автодетект voxel-wide формата (``detect_voxel_wide``);
- транспонирование к стандартной матрице ``time × voxel`` с детерминированным
  ``voxel_id`` (``voxel_wide_to_timeseries``);
- пространственный биннинг по целочисленной решётке ``floor(coord / bin_size)``
  (``voxel_wide_spatial_bin``);
- эвристика для time-like колонки (``detect_time_like_col``).

Никаких тяжёлых импортов — модуль работает только на numpy и pandas.
"""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd


def detect_voxel_wide(df: pd.DataFrame) -> tuple[bool, dict[str, str]]:
    """Проверяет, что таблица в формате ``x, y, z, t0..tN``.

    Возвращает пару ``(is_voxel_wide, lower_to_original)``: первый флаг —
    подходит ли формат, второй — словарь «нижний регистр имени → оригинал»,
    чтобы дальше можно было обращаться к колонкам без оглядки на регистр.
    """
    cols = list(df.columns)
    lower = {str(c).strip().lower(): str(c) for c in cols}
    if not {"x", "y", "z"}.issubset(set(lower.keys())):
        return False, lower
    other = [c for c in cols if str(c).strip().lower() not in {"x", "y", "z"}]
    if len(other) < 2:
        return False, lower
    return True, lower


def detect_time_like_col(col: pd.Series) -> bool:
    """Эвристика для авто-обнаружения временной/индексной колонки.

    Сначала проверяем простой числовой индекс (это дёшево), и только потом
    парсим datetime на коротком префиксе. Так мы не тратим dateutil впустую
    на сотни тысяч строк voxel-wide матрицы.
    """
    sample = col.dropna()
    if sample.empty:
        return False

    c = pd.to_numeric(sample, errors="coerce")
    if c.notna().mean() >= 0.95:
        dif = c.dropna().diff().dropna()
        return bool(len(dif) >= 3 and (dif.abs() > 0).mean() >= 0.9)

    probe = sample.iloc[: min(len(sample), 2048)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt = pd.to_datetime(probe, errors="coerce", utc=False)
        if dt.notna().mean() >= 0.9:
            return dt.is_monotonic_increasing or dt.is_monotonic_decreasing
    except Exception:
        pass
    return False


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
    """Сэмплирует/сокращает воксели ДО транспонирования.

    Для формата ``x, y, z, t0..tN`` это принципиально: после транспонирования
    число вокселей становится числом колонок. При N≈250k это очень больно
    бьёт по памяти, поэтому режем сразу.

    Режимы:

    - ``first`` — первые K строк;
    - ``random`` — случайные K строк (детерминировано через ``feature_seed``);
    - ``variance`` — топ-K по дисперсии вдоль ``time_cols``.
    """
    _ = (xcol, ycol, zcol)
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


def _validate_voxel_uniqueness(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    *,
    on_duplicate: str = "error",
) -> np.ndarray:
    """Проверяет уникальность ``(x, y, z)`` в отсортированном наборе координат.

    Возвращает булеву маску ``keep`` той же длины, что и входные массивы.
    """
    dup_mask = (xi[1:] == xi[:-1]) & (yi[1:] == yi[:-1]) & (zi[1:] == zi[:-1])
    n_dupes = int(np.sum(dup_mask))
    if n_dupes == 0:
        return np.ones(len(xi), dtype=bool)

    dup_pos = np.where(dup_mask)[0]
    examples = [(int(xi[i + 1]), int(yi[i + 1]), int(zi[i + 1])) for i in dup_pos[:3]]
    msg = (
        f"[voxel_wide] Обнаружено {n_dupes} повторяющихся координат (x,y,z). "
        f"Примеры: {examples}. Это делает voxel_id неоднозначным."
    )

    mode = str(on_duplicate or "error").strip().lower()
    if mode == "error":
        raise ValueError(msg)
    if mode == "warn":
        logging.warning(msg)
        return np.ones(len(xi), dtype=bool)
    if mode == "drop_first":
        logging.warning(msg + " Применяется drop_first: оставляем последнее вхождение.")
        keep = np.ones(len(xi), dtype=bool)
        keep[np.concatenate((dup_mask, [False]))] = False
        return keep
    raise ValueError(f"Неизвестный режим on_duplicate={on_duplicate!r}")


def voxel_wide_to_timeseries(
    df: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    feature_sampling: str = "first",
    feature_seed: int = 13,
    spatial_bin_size: int = 0,
    spatial_bin_method: str = "mean",
    spatial_bin_range: tuple | None = None,
    on_duplicate_voxels: str = "error",
) -> pd.DataFrame:
    """Конвертирует таблицу ``x, y, z, t0..tN`` в матрицу ``time × voxel``.

    Координаты вокселей складываются в ``out.attrs['coords']`` как отдельный
    DataFrame — чтобы downstream-код мог восстановить пространственный смысл.

    Если ``spatial_bin_size > 1`` — пространственный биннинг по координатам.
    Биннинг абсолютно детерминирован: результат зависит только от ``(x, y, z)``
    и ``bin_size``, а не от значений временных рядов.
    """
    is_vox, lower = detect_voxel_wide(df)
    if not is_vox:
        return df

    xcol, ycol, zcol = lower["x"], lower["y"], lower["z"]
    _t_re = re.compile(r"^t\d+$")
    time_cols = [
        c for c in df.columns if _t_re.match(str(c).strip().lower()) or str(c).strip().isdigit()
    ]

    _bs = int(spatial_bin_size or 0)
    if _bs > 1:
        return voxel_wide_spatial_bin(
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

    x = coords["x"].to_numpy(dtype=float, copy=False)
    y = coords["y"].to_numpy(dtype=float, copy=False)
    z = coords["z"].to_numpy(dtype=float, copy=False)

    def _to_int_safe(a: np.ndarray) -> np.ndarray:
        """Аккуратно приводит float-координаты к int64: NaN/inf → 0, дробные усекаются."""
        m = np.isfinite(a)
        out = np.full(a.shape[0], 0, dtype=np.int64)
        if np.any(~m):
            logging.warning(
                "[voxel_wide] %d нефинитных координат заменены на 0.",
                int(np.sum(~m)),
            )
        frac_mask = m & (a != np.floor(a))
        if np.any(frac_mask):
            logging.warning(
                "[voxel_wide] Координаты содержат дробные значения и будут усечены до int64."
            )
        out[m] = a[m].astype(np.int64)
        return out

    xi = _to_int_safe(x)
    yi = _to_int_safe(y)
    zi = _to_int_safe(z)

    # Детерминированный порядок строк: сортировка по (x, y, z).
    row_order = np.lexsort((zi, yi, xi))
    xi = xi[row_order]
    yi = yi[row_order]
    zi = zi[row_order]
    coords = coords.iloc[row_order].copy().reset_index(drop=True)
    ts = ts.iloc[row_order].copy()

    keep = _validate_voxel_uniqueness(
        xi,
        yi,
        zi,
        on_duplicate=on_duplicate_voxels,
    )
    if not keep.all():
        xi = xi[keep]
        yi = yi[keep]
        zi = zi[keep]
        coords = coords.iloc[np.where(keep)[0]].copy().reset_index(drop=True)
        ts = ts.iloc[np.where(keep)[0]].copy()

    dup_mask = (
        (xi[1:] == xi[:-1]) & (yi[1:] == yi[:-1]) & (zi[1:] == zi[:-1])
        if len(xi) > 1
        else np.array([], dtype=bool)
    )

    # voxel_id строим только из координат — одинаковая (x,y,z) у разных
    # субъектов даёт одинаковый voxel_id. Это нужно для канонического
    # выравнивания в group pipeline.
    voxel_ids = np.array(
        [f"x{xi[i]}_y{yi[i]}_z{zi[i]}" for i in range(len(xi))],
        dtype=object,
    )

    coords.insert(0, "voxel_id", voxel_ids)
    if len(xi) > 1:
        left = np.concatenate(([False], dup_mask))
        right = np.concatenate((dup_mask, [False]))
        coords["coord_duplicate"] = (left | right).astype(int)
    else:
        coords["coord_duplicate"] = 0

    ts.index = voxel_ids
    out = ts.T
    out.columns = voxel_ids
    out.attrs["coords"] = coords
    out.attrs["voxel_time_cols"] = [str(c) for c in time_cols_sorted]
    out.attrs["format"] = "voxel_wide"
    return out


def voxel_wide_spatial_bin(
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
    """Пространственный биннинг для voxel-wide CSV (строки = воксели).

    Ключ бина: ``floor(coord / bin_size)`` — целочисленная решётка по каждой оси.
    Биннинг абсолютно детерминирован: результат зависит **только** от координат
    ``(x, y, z)``, параметра ``bin_size`` и явного ``bin_range``.

    Если ``bin_range`` задан — в выходе будут **все** бины из этого диапазона,
    включая пустые (заполненные NaN). Это гарантирует одинаковый набор колонок
    между файлами даже при разном покрытии вокселей — необходимо для
    канонического выравнивания между субъектами.
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
        "[CSV spatial bin] %d вокселей → сетка %d×%d×%d = %d бинов "
        "(bin_size=%d, method=%s, range=%s)",
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


__all__ = [
    "detect_time_like_col",
    "detect_voxel_wide",
    "voxel_wide_spatial_bin",
    "voxel_wide_to_timeseries",
]
