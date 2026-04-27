"""I/O табличных форматов: CSV/Excel, автодетект кодировки, потоковая voxel-wide загрузка.

Здесь живут:

- ``CSV_ENCODING_CANDIDATES`` — список типовых кодировок Windows/UTF, по которым
  делается перебор;
- ``probe_csv_encoding`` / ``read_csv_with_encoding_fallback`` — устойчивое
  чтение CSV даже когда файл не в UTF-8;
- ``detect_header`` / ``maybe_split_single_column`` — эвристики для CSV без
  явной шапки и для XLSX, в котором вся строка лежит в одной ячейке;
- ``probe_csv_voxel_wide_layout``, ``iter_csv_voxel_wide_chunks``,
  ``stream_csv_voxel_wide_to_timeseries`` — потоковая загрузка очень больших
  CSV формата ``x, y, z, t0..tN`` с детерминированным пространственным биннингом.

Ничего тяжёлого из ML тут нет: только pandas, numpy и scipy.sparse.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix as _csr_matrix

from neweds.io.voxel import detect_voxel_wide

CSV_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin1")


def looks_like_encoding_error(exc: Exception) -> bool:
    """Проверяет, что исключение похоже на проблему с кодировкой CSV."""
    msg = str(exc).lower()
    return (
        isinstance(exc, UnicodeDecodeError)
        or "codec can't decode" in msg
        or "unicode" in msg
        or "utf-8" in msg
        and "decode" in msg
    )


def probe_csv_encoding(filepath: str, *, engine: str = "") -> str:
    """Быстро подбирает рабочую кодировку CSV по первым 32 строкам файла.

    Перебираем кандидатов из ``CSV_ENCODING_CANDIDATES`` и возвращаем первую,
    с которой pandas смог прочитать префикс. Так в потоковом режиме мы не
    повторяем тяжёлый fallback-перебор на каждом куске.
    """
    base_engine = str(engine or "").strip().lower()
    for enc in CSV_ENCODING_CANDIDATES:
        kw: dict[str, Any] = {"header": None, "nrows": 32, "encoding": enc}
        if base_engine == "pyarrow" and enc not in {"utf-8", "utf8", "utf-8-sig"}:
            kw["engine"] = "c"
        else:
            kw["engine"] = base_engine or None
        if kw.get("engine") is None:
            kw.pop("engine", None)
        try:
            pd.read_csv(filepath, **kw)
            if enc != "utf-8":
                logging.info("[CSV] encoding probe: %s -> %s", filepath, enc)
            return enc
        except Exception as exc:
            if looks_like_encoding_error(exc):
                continue
            raise
    return "utf-8"


def read_csv_with_encoding_fallback(filepath: str, **kw):
    """Читает CSV (или итератор по чанкам) с перебором типовых кодировок.

    Нужно для гигантских voxel-wide CSV, где один битый не-UTF-8 байт ломал
    потоковую загрузку и заставлял всё проваливаться в обычный pandas-путь
    с реальным риском OOM.

    Если ``encoding`` уже передан явно — fallback-перебор не выполняется.
    """
    if "encoding" in kw:
        return pd.read_csv(filepath, **kw)

    last_exc: Exception | None = None
    base_engine = str(kw.get("engine", "") or "").strip().lower()
    tried: list[str] = []

    for enc in CSV_ENCODING_CANDIDATES:
        trial = dict(kw)
        trial["encoding"] = enc
        # pyarrow стабильно работает только с UTF-8; для legacy-кодировок
        # принудительно уходим на pandas C-engine.
        if base_engine == "pyarrow" and enc not in {"utf-8", "utf8", "utf-8-sig"}:
            trial["engine"] = "c"
        try:
            obj = pd.read_csv(filepath, **trial)
            if enc != "utf-8":
                logging.info("[CSV] encoding fallback: %s -> %s", filepath, enc)
            return obj
        except Exception as exc:
            if looks_like_encoding_error(exc):
                last_exc = exc
                tried.append(enc)
                continue
            raise

    if last_exc is not None:
        raise UnicodeDecodeError(
            "csv",
            b"",
            0,
            1,
            f"Не удалось распознать кодировку CSV из {tried}: {last_exc}",
        )
    return pd.read_csv(filepath, **kw)


def csv_probe_ncols(filepath: str, *, nrows: int = 2) -> int:
    """Быстро узнаёт число колонок CSV без полной загрузки файла; 0 при ошибке."""
    try:
        probe = read_csv_with_encoding_fallback(filepath, header=None, nrows=nrows, low_memory=False)
        return int(probe.shape[1])
    except Exception:
        return 0


def is_mostly_numeric_row(row) -> bool:
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


def detect_header(df_raw: pd.DataFrame) -> bool:
    """Если 1-я строка нечисловая, а 2-я числовая — считаем 1-ю заголовком."""
    if df_raw.shape[0] < 2:
        return False
    r0 = df_raw.iloc[0].tolist()
    r1 = df_raw.iloc[1].tolist()
    return (not is_mostly_numeric_row(r0)) and is_mostly_numeric_row(r1)


def maybe_split_single_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Разбирает «CSV внутри одной ячейки».

    Часто встречается в XLSX: вся строка лежит в одной ячейке как текст
    ``x,y,z,t0,t1,...`` (с разделителями ``,``, ``;`` или табом). Бывает
    и так, что Excel разнёс 1-2 первых поля по колонкам, а остальное в первом
    столбце пустое — тогда тоже надо распарсить вручную.
    """
    try:
        # Случай 1: ровно одна колонка, в ней строка с разделителями.
        if df_raw.shape[1] == 1 and isinstance(df_raw.iloc[0, 0], str):
            return df_raw[0].astype(str).str.split(r"[,;\t]", expand=True)

        # Случай 2: первая колонка плотно заполнена, остальные почти пустые.
        if df_raw.shape[1] > 1:
            nonnull = df_raw.notna().mean(axis=0)
            if float(nonnull.iloc[0]) >= 0.8 and bool((nonnull.iloc[1:] <= 0.05).all()):
                if isinstance(df_raw.iloc[0, 0], str):
                    return (
                        df_raw.iloc[:, [0]]
                        .copy()
                        .iloc[:, 0]
                        .astype(str)
                        .str.split(r"[,;\t]", expand=True)
                    )

        # Случай 3: «настоящая» колонка где-то правее. Ищем колонку, где
        # большинство значений выглядят как строка с разделителями.
        if df_raw.shape[1] > 1:
            best_j = None
            best_score = 0.0
            for j in range(df_raw.shape[1]):
                col = df_raw.iloc[:, j]
                is_str = col.apply(
                    lambda v: isinstance(v, str) and ("," in v or ";" in v or "\t" in v)
                )
                score = float(is_str.mean())
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is not None and best_score >= 0.8:
                return (
                    df_raw.iloc[:, [best_j]]
                    .copy()
                    .iloc[:, 0]
                    .astype(str)
                    .str.split(r"[,;\t]", expand=True)
                )
    except Exception:
        pass
    return df_raw


def csv_choose_stream_engine(csv_engine: str = "auto") -> str:
    """Выбирает движок pandas для CSV, который совместим с ``chunksize``-итерацией.

    pyarrow-движок не умеет в потоковый режим, поэтому для стриминга всегда
    откатываемся на ``c`` (или ``python``, если явно попросили).
    """
    eng = str(csv_engine or "auto").strip().lower()
    if eng in {"c", "python"}:
        return eng
    return "c"


def read_csv_probe_raw(
    filepath: str,
    *,
    nrows: int = 8,
    csv_engine: str = "auto",
) -> pd.DataFrame:
    """Читает короткий «сырой» префикс CSV (без шапки) для дальнейшей детекции формата."""
    kw: dict[str, Any] = {"header": None, "nrows": int(nrows), "low_memory": False}
    kw["engine"] = csv_choose_stream_engine(csv_engine)
    return read_csv_with_encoding_fallback(filepath, **kw)


def probe_csv_voxel_wide_layout(
    filepath: str,
    *,
    header: str = "auto",
    csv_engine: str = "auto",
) -> dict[str, Any]:
    """По первым строкам CSV пытается понять, что формат — ``x, y, z, t0..tN``."""
    if header not in {"auto", "yes", "no"}:
        raise ValueError("header должен быть одним из: auto|yes|no")

    raw = read_csv_probe_raw(filepath, nrows=8, csv_engine=csv_engine)
    raw = maybe_split_single_column(raw)
    has_header = detect_header(raw) if header == "auto" else (header == "yes")
    if has_header:
        hdr = raw.iloc[0].astype(str).tolist()
        df = raw.iloc[1:].copy()
        df.columns = [h if h.strip() else f"c{i + 1}" for i, h in enumerate(hdr)]
    else:
        df = raw.copy()
        df.columns = [f"c{i + 1}" for i in range(df.shape[1])]

    is_vox, lower = detect_voxel_wide(df)
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
        _t_re = re.compile(r"^t\d+$")
        layout["time_cols"] = [
            c for c in df.columns if _t_re.match(str(c).strip().lower()) or str(c).strip().isdigit()
        ]
    return layout


def iter_csv_voxel_wide_chunks(
    filepath: str,
    *,
    header: str = "auto",
    csv_engine: str = "auto",
    chunksize: int = 4096,
    _layout: dict[str, Any] | None = None,
    _encoding: str | None = None,
):
    """Возвращает итератор по чанкам CSV формата ``x, y, z, t0..tN``."""
    layout = _layout or probe_csv_voxel_wide_layout(filepath, header=header, csv_engine=csv_engine)
    if not bool(layout.get("is_voxel_wide")):
        raise ValueError("CSV не в voxel-wide формате (ожидался x,y,z,t0..tN)")

    stream_engine = csv_choose_stream_engine(csv_engine)
    encoding = _encoding or probe_csv_encoding(filepath, engine=stream_engine)

    kw: dict[str, Any] = {
        "chunksize": max(1, int(chunksize)),
        "low_memory": False,
        "engine": stream_engine,
        "encoding": encoding,
    }
    if bool(layout.get("has_header")):
        kw["header"] = 0
    else:
        kw["header"] = None
        kw["names"] = list(layout["columns"])

    yield from read_csv_with_encoding_fallback(filepath, **kw)


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
    """Потоково загружает огромный voxel-wide CSV (``x, y, z, t0..tN``) → ``time × bins``.

    Биннинг локально-детерминирован: бин для каждого вокселя зависит только
    от его координат и ``bin_size``. В стриминговом режиме сознательно
    поддерживаем только ``mean`` и ``sum`` — для медианы пришлось бы хранить
    все значения внутри бина в памяти.
    """
    layout = probe_csv_voxel_wide_layout(filepath, header=header, csv_engine=csv_engine)
    if not bool(layout.get("is_voxel_wide")):
        raise ValueError("CSV не в voxel-wide формате (ожидался x,y,z,t0..tN)")

    # Кодировку определяем один раз на файл и переиспользуем во всех чанках.
    _stream_engine = csv_choose_stream_engine(csv_engine)
    _encoding = probe_csv_encoding(filepath, engine=_stream_engine)

    method_eff = str(spatial_bin_method or "mean").strip().lower()
    if method_eff not in {"mean", "sum"}:
        raise ValueError(
            "Потоковый spatial binning поддерживает только mean/sum; "
            f"получено method={spatial_bin_method!r}"
        )
    b = max(1, int(spatial_bin_size or 1))
    fixed_range = spatial_bin_range is not None
    if fixed_range:
        (rx0, rx1), (ry0, ry1), (rz0, rz1) = spatial_bin_range
        grid_bx_min = int(np.floor(rx0 / b))
        grid_bx_max = int(np.floor(rx1 / b))
        grid_by_min = int(np.floor(ry0 / b))
        grid_by_max = int(np.floor(ry1 / b))
        grid_bz_min = int(np.floor(rz0 / b))
        grid_bz_max = int(np.floor(rz1 / b))
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

    bin_to_idx: dict[tuple[int, int, int], int] = {}
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

    for chunk in iter_csv_voxel_wide_chunks(
        filepath,
        header=header,
        csv_engine=csv_engine,
        chunksize=chunksize,
        _layout=layout,
        _encoding=_encoding,
    ):
        n_chunks += 1
        x = pd.to_numeric(chunk[xcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        y = pd.to_numeric(chunk[ycol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        z = pd.to_numeric(chunk[zcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        _tc_block = chunk[time_cols]
        if _tc_block.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all():
            ts_arr = _tc_block.to_numpy(dtype=np.float32, copy=False)
        else:
            ts_arr = _tc_block.apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=np.float32, copy=False
            )
        if order.size:
            ts_arr = ts_arr[:, order]
        if n_time is None:
            n_time = int(ts_arr.shape[1])
        n_voxels_total += int(ts_arr.shape[0])
        if ts_arr.size == 0:
            continue

        voxel_var = np.nanvar(ts_arr, axis=1)
        alive = (
            np.isfinite(voxel_var)
            & (voxel_var > eps)
            & np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
        )
        if not np.any(alive):
            continue
        n_alive_total += int(np.sum(alive))

        xa = x[alive]
        ya = y[alive]
        za = z[alive]
        ta = ts_arr[alive].astype(np.float64, copy=False)

        bx = np.floor(xa / b).astype(np.int64)
        by = np.floor(ya / b).astype(np.int64)
        bz = np.floor(za / b).astype(np.int64)
        if fixed_range:
            in_range = (
                (bx >= grid_bx_min)
                & (bx <= grid_bx_max)
                & (by >= grid_by_min)
                & (by <= grid_by_max)
                & (bz >= grid_bz_min)
                & (bz <= grid_bz_max)
            )
            bx = bx[in_range]
            by = by[in_range]
            bz = bz[in_range]
            xa = xa[in_range]
            ya = ya[in_range]
            za = za[in_range]
            ta = ta[in_range]
            if bx.size == 0:
                continue

        by_min = int(by.min())
        bz_min = int(bz.min())
        by_span = int(by.max() - by_min + 1)
        bz_span = int(bz.max() - bz_min + 1)
        by_off = by - by_min
        bz_off = bz - bz_min
        hash_keys = ((bx * by_span) + by_off) * bz_span + bz_off
        uniq_hash, inv = np.unique(hash_keys, return_inverse=True)
        n_uniq = int(uniq_hash.shape[0])
        if n_uniq == 0:
            continue

        u_bx = uniq_hash // (by_span * bz_span)
        rem = uniq_hash % (by_span * bz_span)
        u_by = (rem // bz_span) + by_min
        u_bz = (rem % bz_span) + bz_min

        n_rows = int(ta.shape[0])
        group_mat = _csr_matrix(
            (np.ones(n_rows, dtype=np.float64), (inv, np.arange(n_rows, dtype=np.int32))),
            shape=(n_uniq, n_rows),
        )
        chunk_sums = np.asarray(group_mat @ ta, dtype=np.float64)
        chunk_counts = np.asarray(group_mat.sum(axis=1), dtype=np.int64).ravel()
        chunk_xsum = np.asarray(group_mat @ xa.reshape(-1, 1), dtype=np.float64).ravel()
        chunk_ysum = np.asarray(group_mat @ ya.reshape(-1, 1), dtype=np.float64).ravel()
        chunk_zsum = np.asarray(group_mat @ za.reshape(-1, 1), dtype=np.float64).ravel()

        for j in range(n_uniq):
            key = (int(u_bx[j]), int(u_by[j]), int(u_bz[j]))
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

    if fixed_range:
        for ibx in range(grid_bx_min, grid_bx_max + 1):
            for iby in range(grid_by_min, grid_by_max + 1):
                for ibz in range(grid_bz_min, grid_bz_max + 1):
                    key = (int(ibx), int(iby), int(ibz))
                    if key in bin_to_idx:
                        continue
                    idx = len(bin_keys)
                    bin_to_idx[key] = idx
                    bin_keys.append(key)
                    sums_rows.append(np.zeros((n_time or 0,), dtype=np.float64))
                    counts_rows.append(0)
                    xsum_rows.append(float(ibx * b + b / 2.0))
                    ysum_rows.append(float(iby * b + b / 2.0))
                    zsum_rows.append(float(ibz * b + b / 2.0))
                    coord_count_rows.append(0)

    if not bin_keys or n_time is None:
        raise ValueError(
            f"Streaming CSV spatial binning produced no active bins (bin_size={b}, file={filepath})."
        )

    if fixed_range:
        keep = list(range(len(bin_keys)))
    else:
        keep = [i for i, c in enumerate(counts_rows) if int(c) >= max(1, int(min_voxels_per_bin))]
        if not keep:
            raise ValueError(
                f"Streaming CSV spatial binning: all bins are below "
                f"min_voxels_per_bin={min_voxels_per_bin}."
            )

    keep_sorted = sorted(keep, key=lambda i: bin_keys[i])

    result_rows: list[np.ndarray] = []
    n_t = int(n_time or 0)
    for i in keep_sorted:
        cnt = int(counts_rows[i])
        if cnt == 0:
            result_rows.append(np.full(n_t, np.nan, dtype=np.float32))
        elif method_eff == "mean":
            result_rows.append((sums_rows[i] / cnt).astype(np.float32))
        else:
            result_rows.append(sums_rows[i].astype(np.float32))
    result = np.vstack(result_rows) if result_rows else np.empty((0, n_t), dtype=np.float32)

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
        "bin_range": spatial_bin_range,
        "bin_key_formula": "floor(coord / bin_size)",
        "deterministic": True,
        "fixed_range": bool(fixed_range),
        "streaming": True,
        "chunksize": int(chunksize),
        "n_chunks": int(n_chunks),
    }
    logging.info(
        "[CSV spatial bin stream] Result: %d×%d from %d voxels (%d alive), "
        "%d bins, chunks=%d, chunk_rows=%d",
        out.shape[0],
        out.shape[1],
        int(n_voxels_total),
        int(n_alive_total),
        len(keep_sorted),
        int(n_chunks),
        int(chunksize),
    )
    return out


__all__ = [
    "CSV_ENCODING_CANDIDATES",
    "csv_choose_stream_engine",
    "csv_probe_ncols",
    "detect_header",
    "is_mostly_numeric_row",
    "iter_csv_voxel_wide_chunks",
    "looks_like_encoding_error",
    "maybe_split_single_column",
    "probe_csv_encoding",
    "probe_csv_voxel_wide_layout",
    "read_csv_probe_raw",
    "read_csv_with_encoding_fallback",
    "stream_csv_voxel_wide_to_timeseries",
]
