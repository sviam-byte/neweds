#!/usr/bin/env python3

"""Загрузка и парсинг данных: CSV, Excel, Parquet, HDF5, voxel-wide-форматы."""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

# Реализации I/O разъехались по io/* и core/preprocessing.py.
# Этот модуль — диспетчер (read_input_table, tidy_timeseries_table, load_or_generate)
# плюс backward-compat re-exports для исторических импортов из core.data_loader.
from neweds.io.hdf5_neuro import (  # noqa: F401
    MAX_RAW_VOXELS_FOR_GUI,
    build_aggregated_h5_path as _build_aggregated_h5_path,
    h5_4d_to_spatial_bins,
    h5_4d_to_voxel_wide,
    load_aggregated_h5,
    load_h5_neuroimaging as _load_h5_neuroimaging,
    save_aggregated_h5 as _save_aggregated_h5,
)
from neweds.io.tabular import (  # noqa: F401
    CSV_ENCODING_CANDIDATES,
    csv_probe_ncols as _csv_probe_ncols,
    detect_header as _detect_header,
    maybe_split_single_column as _maybe_split_single_column,
    probe_csv_voxel_wide_layout as _probe_csv_voxel_wide_layout,
    read_csv_with_encoding_fallback as _read_csv_with_encoding_fallback,
    stream_csv_voxel_wide_to_timeseries,
)
from neweds.io.voxel import (  # noqa: F401
    detect_time_like_col as _detect_time_like_col,
    voxel_wide_to_timeseries,
)

from .preprocessing import (  # noqa: F401
    PreprocessReport,
    additional_preprocessing,
    spatial_grid_bin_fmri,
)


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
        from neweds.io.mat import mat_to_dataframe

        return mat_to_dataframe(fp)

    if low.endswith((".h5", ".hdf5", ".hdf")):
        # HDF5 обрабатывается целиком в load_or_generate → _load_h5_neuroimaging.
        # Если вызвали read_input_table напрямую, делаем базовую загрузку.
        return _load_h5_neuroimaging(fp)

    if low.endswith(".csv"):
        # Важно: low_memory=False выключает покусковую догадку типов в pandas
        # и снижает риск нестабильной типизации на mixed-type CSV.
        kw: dict[str, Any] = {"header": None, "low_memory": False}
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
        df0 = _read_csv_with_encoding_fallback(fp, **kw)
    else:
        xl_usecols = usecols
        excel_probe_single_col = False
        excel_engine = None

        # Явно выбираем движок по расширению, чтобы чтение Excel было стабильнее.
        # openpyxl покрывает .xlsx/.xlsm, а xlrd нужен для старых .xls.
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
        except (ValueError, TypeError, OSError) as exc:
            # Если чтение с выбором колонок не удалось, мягко падаем на полное чтение.
            logging.debug("Excel usecols read failed; reading full sheet: %s", exc)
            df0 = pd.read_excel(fp, header=None, engine=excel_engine)
            excel_probe_single_col = False

        if excel_probe_single_col:
            probe_df = df0
            split_probe = _maybe_split_single_column(probe_df)

            # Если одиночная колонка распалась на несколько полей,
            # это действительно «CSV в ячейке», полный reread не нужен.
            # Иначе считаем, что это обычный Excel, и читаем весь лист.
            probe_is_embedded_csv = split_probe.shape[1] > probe_df.shape[1] or (
                probe_df.shape[1] == 1 and split_probe.shape[1] > 1
            )

            if not probe_is_embedded_csv:
                try:
                    df0 = pd.read_excel(fp, header=None, engine=excel_engine)
                except (ValueError, TypeError, OSError) as exc:
                    logging.debug("Excel full reread failed; using probe data: %s", exc)
                    df0 = probe_df
    df0 = _maybe_split_single_column(df0)

    if header not in {"auto", "yes", "no"}:
        raise ValueError("header must be one of: auto|yes|no")
    has_header = _detect_header(df0) if header == "auto" else (header == "yes")
    if has_header:
        hdr = df0.iloc[0].astype(str).tolist()
        df = df0.iloc[1:].copy()
        df.columns = [h if h.strip() else f"c{i + 1}" for i, h in enumerate(hdr)]
    else:
        df = df0.copy()
        df.columns = [f"c{i + 1}" for i in range(df.shape[1])]
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
    on_duplicate_voxels: str = "error",
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
            on_duplicate_voxels=on_duplicate_voxels,
        )
    except (ValueError, TypeError, KeyError) as exc:
        logging.debug("voxel-wide autodetection skipped: %s", exc)

    # Важно: voxel_wide_to_timeseries кладёт в out.attrs['coords'] DataFrame координат.
    # Pandas при многих операциях (например, Series.notna()) делает deepcopy attrs,
    # что на больших данных взрывает память (deepcopy координат ДЛЯ КАЖДОЙ колонки).
    # Поэтому: временно выносим attrs наружу и очищаем их на время чистки таблицы.
    _saved_attrs: dict[str, Any] = dict(getattr(out, "attrs", {}) or {})
    try:
        out.attrs = {}
    except (AttributeError, TypeError, ValueError) as exc:
        logging.debug("attrs reset via assignment failed; trying clear(): %s", exc)
        try:
            out.attrs.clear()
        except (AttributeError, TypeError, ValueError) as clear_exc:
            logging.debug("attrs clear skipped: %s", clear_exc)

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
    except (ValueError, TypeError) as exc:
        logging.debug("fast finite-column filter skipped; using pandas fallback: %s", exc)
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
        out.columns = [f"c{i + 1}" for i in range(out.shape[1])]

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
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid time slicing options: {exc}") from exc

    # Ограничение числа признаков
    try:
        if (
            feature_limit is not None
            and int(feature_limit) > 0
            and out.shape[1] > int(feature_limit)
        ):
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
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid feature_limit/feature_sampling options: {exc}") from exc

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
                except (KeyError, ValueError, TypeError) as exc:
                    logging.debug("coords metadata sync skipped: %s", exc)
            out.attrs.update(_saved_attrs)
    except (AttributeError, TypeError, ValueError) as exc:
        logging.debug("attrs restore skipped: %s", exc)
    return out


# Утилиты предобработки: выбросы и ранговая нормализация


# preprocess_timeseries / _rank_normalize_1d / _apply_outliers_1d moved to
# neweds.core.preprocessing.
from .preprocessing import (  # noqa: E402, F401
    _apply_outliers_1d,
    _rank_normalize_1d,
    preprocess_timeseries,
)


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
    on_duplicate_voxels: str = "error",
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
    csv_chunk_rows: int = 32768,
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
                _bin_size = (
                    int(spatial_grid_size)
                    if spatial_grid_size is not None and int(spatial_grid_size) > 1
                    else (
                        int(h5_spatial_bin)
                        if h5_spatial_bin is not None and int(h5_spatial_bin) > 1
                        else 5
                    )
                )
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
                    if (
                        bool(save_aggregated_h5)
                        and str(getattr(df, "attrs", {}).get("format", "")).lower()
                        == "spatial_bins"
                    ):
                        try:
                            _save_aggregated_h5(
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
                and str(spatial_grid_method or "mean").strip().lower() in {"mean", "sum"}
                and usecols in {"auto", None}
            )
            if _try_stream_csv_spatial:
                try:
                    _layout = _probe_csv_voxel_wide_layout(
                        filepath, header=header, csv_engine=csv_engine
                    )
                    if bool(_layout.get("is_voxel_wide")):
                        df = stream_csv_voxel_wide_to_timeseries(
                            filepath,
                            header=header,
                            csv_engine=csv_engine,
                            chunksize=int(csv_chunk_rows),
                            spatial_bin_size=_csv_spatial_bin_size,
                            spatial_bin_method=str(spatial_grid_method or "mean"),
                            spatial_bin_range=spatial_bin_range,
                        )
                except Exception as exc:
                    _csv_ncols_probe = _csv_probe_ncols(str(filepath))
                    if _csv_ncols_probe >= 50000:
                        raise ValueError(
                            "Streaming CSV spatial binning failed on a very wide CSV "
                            f"(ncols≈{_csv_ncols_probe}). Regular fallback is blocked to avoid OOM. "
                            f"Original error: {exc}"
                        ) from exc
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
                            _pick = sorted(
                                _rng.choice(_csv_ncols, size=_csv_col_cap, replace=False).tolist()
                            )
                            _usecols_eff = _pick
                        else:
                            # "first" / default — первые N колонок
                            _usecols_eff = list(range(_csv_col_cap))

                raw = read_input_table(
                    filepath, header=header, usecols=_usecols_eff, csv_engine=csv_engine
                )

                # Автопонижение типа: лучше осознанно перейти в float32,
                # чем получить OOM на неявных копиях в pandas/numpy при больших матрицах.
                dtype_eff = dtype
                if dtype_eff is None and auto_float32:
                    try:
                        n_rows, n_cols = int(raw.shape[0]), int(raw.shape[1])
                        n_cells = n_rows * n_cols
                        if n_cols >= 128 or n_cells >= 2_000_000:
                            dtype_eff = "float32"
                    except (ValueError, TypeError, OverflowError) as exc:
                        logging.debug("auto_float32 size probe skipped: %s", exc)

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
                    on_duplicate_voxels=on_duplicate_voxels,
                )
        coords_df = None
        try:
            coords_df = df.attrs.get("coords")
        except (AttributeError, TypeError):
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
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logging.debug("voxel metadata report notes skipped: %s", exc)

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
            except (AttributeError, TypeError, ValueError) as exc:
                logging.debug("spatial bin report notes skipped: %s", exc)

        try:
            if hasattr(df, "attrs"):
                df.attrs.pop("coords", None)
                df.attrs.pop("voxel_time_cols", None)
                if len(df.attrs) == 0:
                    df.attrs = {}
        except (AttributeError, TypeError, ValueError) as exc:
            logging.debug("final attrs cleanup skipped: %s", exc)
        logging.info(
            f"[Load] OK shape={df.shape} header={header} time_col={time_col} transpose={transpose} preprocess={preprocess}"
        )
        return (df, report) if return_report else df
    except Exception as e:
        logging.error(f"[Load] Ошибка загрузки: {e}")
        raise
