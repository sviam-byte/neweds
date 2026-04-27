"""MATLAB ``.mat`` file loader.

Извлекает наиболее подходящую числовую 2D/1D матрицу из MAT-структуры
и возвращает её как ``pd.DataFrame``. Импорт ``scipy.io.loadmat`` ленив,
чтобы не платить при ``import neweds`` за то, чем пользуются единицы.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def mat_to_dataframe(filepath: str) -> pd.DataFrame:
    """Загружает ``.mat`` и выбирает наиболее подходящую числовую матрицу.

    Скоринг кандидатов: 2D > 1D, по убыванию ``size``, затем по убыванию
    короткой стороны (предпочитаем «менее вытянутые» матрицы — обычно это
    реальные сигналы, а не индексы/метки).
    """
    from scipy.io import loadmat

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


__all__ = ["mat_to_dataframe"]
