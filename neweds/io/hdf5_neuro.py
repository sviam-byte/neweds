"""Загрузчики нейровизуализационных HDF5: 4D ``(X, Y, Z, T)`` → DataFrame ``T × voxels``.

Здесь живут:

- ``h5_4d_to_voxel_wide`` — «плоский» voxel-wide формат, координаты
  сохраняются в ``attrs``;
- ``h5_4d_to_spatial_bins`` — детерминированная агрегация по spatial bins;
- ``save_aggregated_h5`` / ``load_aggregated_h5`` — кеш агрегированных
  субъектов в отдельный H5 (чтобы не пересчитывать биннинг на каждом прогоне
  group pipeline);
- ``load_h5_neuroimaging`` — полный путь загрузки 4D HDF5 с обрезкой по
  времени, spatial-grid агрегацией и pre-cap'ом числа сырых вокселей.

На уровне модуля импортируется только ``h5py`` (в try/except). Всё остальное —
``scipy``, ленивая spatial-агрегация — подгружается уже внутри функций.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import h5py as _h5py
except ImportError:
    _h5py = None  # type: ignore[assignment]


MAX_RAW_VOXELS_FOR_GUI = 5000
"""Безопасный верхний предел числа сырых voxel-рядов для GUI-пайплайна.

Ограничение применяется на этапе извлечения из 4D H5 до построения полного
DataFrame time×voxel, чтобы избежать взрывного роста памяти.
"""


def h5_4d_to_voxel_wide(
    arr4d: np.ndarray,
    *,
    nonzero_mode: str = "any",
    eps: float = 0.0,
    max_voxels: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Превращает 4D-массив ``(X, Y, Z, T)`` в DataFrame ``(T, N_voxels)``.

    В именах колонок зашиты координаты вокселей, а в ``attrs`` кладутся
    метаданные для downstream-пайплайна (``coords``, ``format``, ``source_kind``)
    — чтобы потом можно было восстановить пространственное расположение.
    """
    if not isinstance(arr4d, np.ndarray):
        arr4d = np.asarray(arr4d)

    if arr4d.ndim != 4:
        raise ValueError(f"h5_4d_to_voxel_wide expects 4D array, got shape={arr4d.shape}")

    x, y, z, t = arr4d.shape
    if t < 2:
        raise ValueError(f"Слишком короткая временная ось: {t}")

    # View без копии: (voxels, time).
    flat = arr4d.reshape(x * y * z, t)

    mode = str(nonzero_mode or "any").strip().lower()
    if mode == "any":
        keep_mask = np.any(np.abs(flat) > float(eps), axis=1)
    elif mode == "var":
        keep_mask = np.nanvar(flat, axis=1) > float(eps)
    else:
        raise ValueError(f"Неизвестный nonzero_mode: {nonzero_mode}")

    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        raise ValueError("В 4D HDF5 не найдено ни одного валидного вокселя")

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
    """Агрегирует 4D массив ``(X, Y, Z, T)`` в фиксированные spatial bins → DataFrame ``T × K``.

    При одинаковой геометрии и ``bin_size`` схема будет одинакова у всех
    субъектов — поэтому этот режим предпочтителен для межсубъектного сравнения.
    """
    arr4d = np.asarray(arr4d)
    if arr4d.ndim != 4:
        raise ValueError(f"h5_4d_to_spatial_bins ждёт 4D массив, получил shape={arr4d.shape}")

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

                coords_rows.append(
                    {
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
                    }
                )

    if not reduced_cols:
        raise ValueError("Ни один spatial bin не прошёл фильтр по дисперсии — пустой объём?")

    df = pd.DataFrame(reduced_cols)
    df.attrs["coords"] = pd.DataFrame(coords_rows)
    df.attrs["format"] = "spatial_bins"
    df.attrs["source_kind"] = "h5_4d_spatial"
    df.attrs["time_axis"] = 3
    df.attrs["feature_axis"] = "spatial_bin"
    df.attrs["bin_size"] = int(bin_size)
    return df


def build_aggregated_h5_path(
    source_path: str,
    *,
    output_dir: str | None = None,
    bin_size: int = 5,
    aggregation: str = "mean",
) -> str:
    """Строит канонический путь для сохранения агрегированного H5-кэша."""
    src_path = Path(source_path)
    root = Path(output_dir) if output_dir else (src_path.parent / "results" / "aggregated_h5")
    folder = root / f"spatialbin_b{int(bin_size)}_{aggregation}"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / f"{src_path.stem}.h5")


def save_aggregated_h5(
    out_path: str,
    df: pd.DataFrame,
    *,
    source_path: str,
    bin_size: int,
    original_shape: tuple[int, ...] | list[int] | None = None,
    aggregation: str = "mean",
) -> str:
    """Сохраняет агрегированные spatial-bins в отдельный HDF5 (для повторных прогонов)."""
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

            bound_cols = [
                c
                for c in ["bin_x0", "bin_x1", "bin_y0", "bin_y1", "bin_z0", "bin_z1"]
                if c in coords_df.columns
            ]
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
    """Читает ранее сохранённый агрегированный HDF5-кэш и возвращает DataFrame ``T × K``."""
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


def load_h5_neuroimaging(
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
    Если ``feature_limit`` не задан/<=0, дополнительный post-cap не применяется.
    """
    if _h5py is None:
        raise ImportError("h5py не установлен. Для загрузки HDF5 файлов: pip install h5py")

    # Импортируем lazy: эти зависимости относятся к нейровизуализации, не нужны другим путям.
    from neweds.core.preprocessing import spatial_grid_bin_fmri
    from neweds.io.h5 import load_h5_spatial_binned_lazy

    with _h5py.File(filepath, "r") as f:
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
                df = pd.DataFrame(arr, columns=[f"c{i + 1}" for i in range(arr.shape[1])])
                return df
            raise ValueError(
                f"Нет подходящего 4D/2D dataset в HDF5: {filepath}. "
                f"Доступные ключи: {list(f.keys())}"
            )

        shape = ds.shape
        logging.info("[HDF5] Dataset '%s' shape=%s dtype=%s", ds_name, shape, ds.dtype)

        # Конвенция нейровизуализации: (X,Y,Z,T) → последняя ось.
        spatial_dims = list(shape[:3])
        t_dim = shape[3] if len(shape) == 4 else shape[-1]
        T_axis = 3
        if t_dim < max(spatial_dims):
            T_axis = int(np.argmax(shape))
            logging.warning(
                "[HDF5] Нестандартный порядок осей: T_axis=%d (shape=%s)", T_axis, shape
            )

        T = int(shape[T_axis])

        feature_mode = str(feature_sampling or "spatial").strip().lower()
        grid_size_eff = int(spatial_grid_size) if spatial_grid_size is not None else 0
        if grid_size_eff <= 0:
            grid_size_eff = (
                int(h5_spatial_bin) if h5_spatial_bin is not None and int(h5_spatial_bin) > 1 else 0
            )
        spatial_mode = feature_mode in {
            "spatial",
            "spatial_bin",
            "bins",
            "auto",
            "deterministic",
        } or (grid_size_eff > 1)
        if bool(lazy_spatial_bin) and spatial_mode and T_axis == 3:
            bin_size = grid_size_eff if grid_size_eff > 1 else 5
            df_lazy = load_h5_spatial_binned_lazy(
                filepath,
                dataset=str(ds_name),
                grid_size=bin_size,
                method=str(spatial_grid_method or "mean"),
                time_chunk=int(time_chunk or 50),
            )
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

        t0 = int(time_start) if time_start is not None else 0
        t1 = int(time_end) if time_end is not None else T
        ts = int(time_stride) if time_stride is not None and int(time_stride) > 0 else 1
        t0 = max(0, min(t0, T))
        t1 = max(t0, min(t1, T))

        idx = [slice(None)] * len(shape)
        idx[T_axis] = slice(t0, t1, ts)

        logging.info("[HDF5] Loading slice %s ...", idx)
        arr4d = np.asarray(ds[tuple(idx)], dtype=np.float32)

    if T_axis != len(arr4d.shape) - 1:
        arr4d = np.moveaxis(arr4d, T_axis, -1)

    *spatial, T_actual = arr4d.shape
    n_total = int(np.prod(spatial))

    logging.info(
        "[HDF5] Spatial=%s T=%d, total voxels=%d, array=%.1f MB",
        spatial,
        T_actual,
        n_total,
        arr4d.nbytes / (1024**2),
    )

    var_eps = max(1e-12, float(nonzero_threshold))
    feature_mode = str(feature_sampling or "spatial").strip().lower()
    grid_size_eff = int(spatial_grid_size) if spatial_grid_size is not None else 0
    if grid_size_eff <= 0:
        grid_size_eff = (
            int(h5_spatial_bin) if h5_spatial_bin is not None and int(h5_spatial_bin) > 1 else 0
        )

    spatial_mode = feature_mode in {"spatial", "spatial_bin", "bins", "auto", "deterministic"} or (
        grid_size_eff > 1
    )

    if spatial_mode:
        bin_size = grid_size_eff if grid_size_eff > 1 else 5
        df_h5 = spatial_grid_bin_fmri(
            arr4d,
            grid_size=bin_size,
            method=str(spatial_grid_method or "mean"),
        )
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

    df_h5 = h5_4d_to_voxel_wide(
        arr4d,
        nonzero_mode="var",
        eps=var_eps,
        max_voxels=MAX_RAW_VOXELS_FOR_GUI,
        seed=int(feature_seed),
    )
    del arr4d

    n_selected = int(df_h5.shape[1])
    logging.info(
        "[HDF5] Pre-capped voxel rows for GUI: %d (limit=%d)", n_selected, MAX_RAW_VOXELS_FOR_GUI
    )

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
        logging.info(
            "[HDF5] Subsampled after pre-cap: %d -> %d voxels (mode=%s)", n_selected, k, mode
        )

    coords_attr = df_h5.attrs.get("coords")
    if isinstance(coords_attr, list):
        coords_df = pd.DataFrame(coords_attr, columns=["x", "y", "z"])
        coords_df.insert(0, "voxel_id", [str(c) for c in df_h5.columns])
        df_h5.attrs["coords"] = coords_df
    elif isinstance(coords_attr, pd.DataFrame):
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

    logging.info(
        "[HDF5] Output DataFrame: %s (%.1f MB)",
        df_h5.shape,
        df_h5.memory_usage(deep=True).sum() / (1024**2),
    )
    return df_h5


__all__ = [
    "MAX_RAW_VOXELS_FOR_GUI",
    "build_aggregated_h5_path",
    "h5_4d_to_spatial_bins",
    "h5_4d_to_voxel_wide",
    "load_aggregated_h5",
    "load_h5_neuroimaging",
    "save_aggregated_h5",
]
