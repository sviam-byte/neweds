"""Загрузчики больших датасетов, в т.ч. ленивая обработка HDF5 чанками по времени."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd


def load_h5_spatial_binned_lazy(
    path,
    dataset: str = "timeseries",
    grid_size: int = 7,
    method: str = "mean",
    time_chunk: int = 50,
) -> pd.DataFrame:
    """Ленивое чтение HDF5 + пространственная бинизация.

    Обрабатывает fMRI чанками по времени.

    Args:
        path: Путь к HDF5 файлу.
        dataset: Имя/путь dataset внутри HDF5 (ожидается ``(X,Y,Z,T)``).
        grid_size: Размер пространственного блока агрегации.
        method: ``mean`` | ``median`` | ``sum``.
        time_chunk: Размер временного чанка для поэтапной загрузки.

    Returns:
        DataFrame формы ``time × bins``.
    """
    g = max(1, int(grid_size))
    chunk = max(1, int(time_chunk))
    agg = str(method or "mean").strip().lower()

    with h5py.File(path, "r") as f:
        data = f[dataset]
        if data.ndim != 4:
            raise ValueError(
                f"load_h5_spatial_binned_lazy ждёт 4D dataset, получил shape={data.shape}"
            )

        X, Y, Z, T = map(int, data.shape)

        bins_x = range(0, X, g)
        bins_y = range(0, Y, g)
        bins_z = range(0, Z, g)

        n_bins = len(bins_x) * len(bins_y) * len(bins_z)
        out: list[np.ndarray] = []

        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            block = np.asarray(data[:, :, :, t0:t1], dtype=np.float32)
            chunk_len = int(block.shape[3])

            chunk_bins = np.zeros((chunk_len, n_bins), dtype=np.float32)
            b = 0

            for x in bins_x:
                for y in bins_y:
                    for z in bins_z:
                        sub = block[x : x + g, y : y + g, z : z + g, :]
                        sub = sub.reshape(-1, chunk_len)
                        if sub.size == 0:
                            b += 1
                            continue

                        if agg == "median":
                            ts = np.nanmedian(sub, axis=0)
                        elif agg == "sum":
                            ts = np.nansum(sub, axis=0)
                        else:
                            ts = np.nanmean(sub, axis=0)

                        chunk_bins[:, b] = np.asarray(ts, dtype=np.float32)
                        b += 1

            out.append(chunk_bins)

    mat = np.vstack(out) if out else np.zeros((0, n_bins), dtype=np.float32)
    return pd.DataFrame(mat, columns=[f"bin_{i}" for i in range(n_bins)])
