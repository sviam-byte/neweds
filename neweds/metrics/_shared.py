"""Общая инфраструктура для модулей метрик: мат-утилиты, parallel, residualization.

Этот модуль НЕ тянет тяжёлые зависимости (statsmodels, dcor, pyinform) на импорте —
они импортируются лениво внутри конкретных категорийных модулей.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

# Порог пар, с которого пробуем параллелизацию.
# Для тяжёлых методов (dcor, MI, TE, AH) порог ниже — overhead joblib окупается быстрее.
_PARALLEL_PAIR_THRESHOLD_DEFAULT = 200
_PARALLEL_PAIR_THRESHOLD_HEAVY = 30
# Верхняя граница автоматически сэмплируемых пар для очень больших матриц.
_MAX_AUTO_RANDOM_PAIRS = 500_000

# Глобальный seed для всех стохастических операций.
# Устанавливается из AnalysisConfig.master_seed через set_module_seed().
_MODULE_SEED: int = 42


def set_module_seed(seed: int) -> None:
    global _MODULE_SEED
    _MODULE_SEED = int(seed)


def get_module_seed() -> int:
    return _MODULE_SEED


def _init_matrix(
    n: int, default: float, *, diag: float | None = None, dtype=np.float64
) -> np.ndarray:
    out = np.full((n, n), float(default), dtype=dtype)
    if diag is not None:
        np.fill_diagonal(out, float(diag))
    return out


def _iter_pairs(
    n: int, pairs: list[tuple[int, int]] | None, *, directed: bool
) -> list[tuple[int, int]]:
    """Нормализует список пар, переданный пользователем.

    Для ненаправленных метрик возвращаем уникальные пары в порядке (min, max),
    чтобы одна и та же связь не считалась дважды.
    """
    if pairs is None:
        return []
    out: list[tuple[int, int]] = []
    seen = set()
    for i, j in pairs:
        try:
            i = int(i)
            j = int(j)
        except Exception:
            continue
        if i == j or i < 0 or j < 0 or i >= n or j >= n:
            continue
        if not directed:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
        else:
            out.append((i, j))
    return out


def _get_effective_pairs(
    n: int,
    pairs: list[tuple[int, int]] | None,
    *,
    directed: bool,
) -> list[tuple[int, int]]:
    """Возвращает итоговый список пар; при huge-N и pairs=None делает auto-sampling."""
    if pairs is not None:
        return _iter_pairs(n, pairs, directed=directed)
    n_full = n * (n - 1) if directed else n * (n - 1) // 2
    if n_full <= 10_000_000:
        if directed:
            return [(i, j) for i in range(n) for j in range(n) if i != j]
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    max_pairs = min(_MAX_AUTO_RANDOM_PAIRS, max(1, n * 5))
    logging.warning(
        "[connectivity] N=%d -> %d pairs, auto random sample %d.",
        n,
        n_full,
        max_pairs,
    )
    rng = np.random.default_rng(_MODULE_SEED)
    result: set[tuple[int, int]] = set()
    bi = rng.integers(0, n, size=max_pairs * 3)
    bj = rng.integers(0, n, size=max_pairs * 3)
    for ii, jj in zip(bi, bj):
        if ii == jj:
            continue
        key = (int(ii), int(jj)) if directed else (int(min(ii, jj)), int(max(ii, jj)))
        result.add(key)
        if len(result) >= max_pairs:
            break
    return list(result)


def _prepare_numpy(data: pd.DataFrame) -> np.ndarray:
    """Преобразует DataFrame -> float64 numpy без лишних копий где возможно."""
    return data.to_numpy(dtype=np.float64, copy=False)


def _safe_parallel_backend(n_jobs: int = -1) -> tuple[int, str]:
    """Подбирает безопасный backend/число воркеров с учётом переменных окружения."""
    env_jobs = str(os.getenv("TS_TOOL_N_JOBS", "")).strip()
    env_backend = str(os.getenv("TS_TOOL_PARALLEL_BACKEND", "")).strip().lower()
    try:
        cpu_n = max(1, int(os.cpu_count() or 1))
    except Exception:
        cpu_n = 1
    try:
        nj = int(env_jobs) if env_jobs else int(n_jobs)
    except Exception:
        nj = 1
    if nj == -1:
        nj = max(1, min(cpu_n - 1 if cpu_n > 1 else 1, 4))
    nj = max(1, min(int(nj), max(1, cpu_n)))
    backend = env_backend or "threading"
    if backend not in {"threading", "loky", "multiprocessing", "sequential"}:
        backend = "threading"
    if backend == "sequential" or nj <= 1:
        return 1, "sequential"
    return nj, backend


def _try_parallel(func, pairs: list[tuple[int, int]], n_jobs: int = -1, *, heavy: bool = False):
    """joblib-parallel для списка пар, с безопасным fallback в последовательный режим."""
    threshold = _PARALLEL_PAIR_THRESHOLD_HEAVY if heavy else _PARALLEL_PAIR_THRESHOLD_DEFAULT
    if len(pairs) < threshold:
        return [func(p) for p in pairs]
    nj, backend = _safe_parallel_backend(n_jobs)
    if nj <= 1 or backend == "sequential":
        return [func(p) for p in pairs]
    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=nj, backend=backend)(delayed(func)(p) for p in pairs)
    except ImportError:
        return [func(p) for p in pairs]


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask].astype(np.float64, copy=False)
    yy = y[mask].astype(np.float64, copy=False)
    sx = float(xx.std())
    sy = float(yy.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _spearman_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Безопасная оценка корреляции Спирмена для двух одномерных рядов."""
    from scipy import stats

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = np.asarray(x[mask], dtype=np.float64)
    yy = np.asarray(y[mask], dtype=np.float64)
    if np.nanstd(xx) <= 1e-12 or np.nanstd(yy) <= 1e-12:
        return float("nan")
    try:
        return float(stats.spearmanr(xx, yy, nan_policy="omit").statistic)
    except Exception:
        return float("nan")


def _kendall_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Безопасная оценка Kendall tau-b для двух одномерных рядов."""
    from scipy import stats

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = np.asarray(x[mask], dtype=np.float64)
    yy = np.asarray(y[mask], dtype=np.float64)
    if np.nanstd(xx) <= 1e-12 or np.nanstd(yy) <= 1e-12:
        return float("nan")
    try:
        return float(stats.kendalltau(xx, yy, nan_policy="omit", variant="b").statistic)
    except Exception:
        return float("nan")


def _as_2d_controls(
    df: pd.DataFrame, control: list[str] | None = None, control_matrix: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Возвращает (X_ctrl, desc).

    - control: список колонок из df
    - control_matrix: внешняя матрица регрессоров (time × k)
    """
    if control_matrix is not None:
        X = np.asarray(control_matrix, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, [f"ctrl[{i}]" for i in range(X.shape[1])]
    if control:
        cols = [c for c in control if c in df.columns]
        if cols:
            X = df[cols].to_numpy(dtype=np.float64)
            return X, [str(c) for c in cols]
    return np.empty((len(df), 0), dtype=np.float64), []


def _residualize_1d(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X is None or X.size == 0:
        return y
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = int(min(y.size, X.shape[0]))
    y = y[:n]
    X = X[:n, :]
    # Добавляем константу — чтобы регрессия включала свободный член.
    A = np.c_[np.ones(n), X]
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta
    except Exception:
        return y


def _residualize_df(
    df: pd.DataFrame, control: list[str] | None = None, control_matrix: np.ndarray | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Резидуализует каждую колонку ``df`` относительно контрольных переменных.

    Возвращает новый DataFrame того же размера, где из каждого ряда вычтена
    линейная проекция на ``control`` / ``control_matrix``.
    """
    X_ctrl, desc = _as_2d_controls(df, control=control, control_matrix=control_matrix)
    if X_ctrl.size == 0:
        return df, []
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        # Считаем регрессию только по строкам, где конечны и y, и все контроли —
        # иначе синхронизация по времени съедет.
        y = s.to_numpy(dtype=np.float64)
        mask = np.isfinite(y)
        if X_ctrl.size:
            mask = mask & np.isfinite(X_ctrl).all(axis=1)
        if int(mask.sum()) < 8:
            out[c] = np.nan
            continue
        y_res = _residualize_1d(y[mask], X_ctrl[mask])
        # Возвращаем остатки на исходную длину, дыры остаются NaN.
        tmp = np.full_like(y, np.nan, dtype=np.float64)
        tmp[mask] = y_res
        out[c] = tmp
    return out, desc


def _fast_corr_matrix(X: np.ndarray) -> np.ndarray:
    """Векторная корреляция Пирсона на numpy — быстрее ``pandas.corr()`` для больших матриц."""
    X = np.asarray(X, dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    norms = np.sqrt((X_centered**2).sum(axis=0, keepdims=True))
    norms[norms < 1e-12] = 1.0
    X_normed = X_centered / norms
    C = X_normed.T @ X_normed
    np.clip(C, -1.0, 1.0, out=C)
    return C


__all__ = [
    "_PARALLEL_PAIR_THRESHOLD_DEFAULT",
    "_PARALLEL_PAIR_THRESHOLD_HEAVY",
    "_MAX_AUTO_RANDOM_PAIRS",
    "set_module_seed",
    "get_module_seed",
    "_init_matrix",
    "_iter_pairs",
    "_get_effective_pairs",
    "_prepare_numpy",
    "_safe_parallel_backend",
    "_try_parallel",
    "_corr_1d",
    "_spearman_1d",
    "_kendall_1d",
    "_as_2d_controls",
    "_residualize_1d",
    "_residualize_df",
    "_fast_corr_matrix",
]
