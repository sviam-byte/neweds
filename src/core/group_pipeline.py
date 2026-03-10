"""Групповой pipeline: загрузка субъектов, canonical alignment, connectivity, сравнение групп."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from .data_loader import load_or_generate
from .voxel_space import CanonicalVoxelSpace

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".parquet"}


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def _iter_subject_files(directory: str | Path) -> list[Path]:
    """Возвращает поддерживаемые файлы субъектов, отсортированные по имени."""
    root = Path(directory)
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
    )
    if not files:
        raise FileNotFoundError(f"Нет поддерживаемых файлов в {root}")
    return files


def load_subject(
    filepath: str | Path,
    *,
    spatial_grid_size: int = 10,
    spatial_grid_method: str = "mean",
    csv_chunk_rows: int = 32768,
) -> pd.DataFrame:
    """Загружает одного субъекта → DataFrame (time × bins).

    Всегда использует потоковую пространственную биннизацию для voxel-wide CSV.
    Формат колонок: bin_{bx}_{by}_{bz}  (детерминирован по координатам).
    """
    df = load_or_generate(
        str(filepath),
        spatial_grid_size=spatial_grid_size,
        spatial_grid_method=spatial_grid_method,
        csv_stream_spatial_bin=True,
        csv_chunk_rows=csv_chunk_rows,
        preprocess=True,
        normalize=False,
        remove_outliers=False,
        return_report=False,
    )
    if df.empty:
        raise ValueError(f"Пустой DataFrame после загрузки: {filepath}")
    return df


def load_group(
    directory: str | Path,
    group_label: str,
    *,
    spatial_grid_size: int = 10,
    spatial_grid_method: str = "mean",
    csv_chunk_rows: int = 32768,
) -> dict[str, pd.DataFrame]:
    """Загружает все субъекты из директории.

    Returns:
        dict: ``group_label::stem`` → DataFrame (time × bins)
    """
    files = _iter_subject_files(directory)
    result: dict[str, pd.DataFrame] = {}
    for path in files:
        sid = f"{group_label}::{path.stem}"
        logger.info("[%s] загрузка %s …", group_label, path.name)
        try:
            df = load_subject(
                path,
                spatial_grid_size=spatial_grid_size,
                spatial_grid_method=spatial_grid_method,
                csv_chunk_rows=csv_chunk_rows,
            )
            result[sid] = df
            logger.info(
                "[%s] %s: %d t × %d bins",
                group_label, path.name, df.shape[0], df.shape[1],
            )
        except Exception as exc:
            logger.error("[%s] ПРОПУСК %s: %s", group_label, path.name, exc)
    if not result:
        raise RuntimeError(f"Ни один субъект из {directory} не загружен.")
    return result


# ---------------------------------------------------------------------------
# Canonical space и выравнивание
# ---------------------------------------------------------------------------

def fit_canonical_space(
    dfs: dict[str, pd.DataFrame],
    strategy: str = "intersection",
    fill_value: float = float("nan"),
) -> CanonicalVoxelSpace:
    """Строит CanonicalVoxelSpace по всем субъектам."""
    space = CanonicalVoxelSpace.from_dataframes(
        list(dfs.values()),
        strategy=strategy,
        fill_value=fill_value,
    )
    logger.info(
        "CanonicalVoxelSpace: стратегия=%s, n_bins=%d",
        strategy, space.n_voxels,
    )
    return space


def align_all(
    dfs: dict[str, pd.DataFrame],
    space: CanonicalVoxelSpace,
) -> dict[str, pd.DataFrame]:
    """Выравнивает все субъекты к canonical space."""
    result: dict[str, pd.DataFrame] = {}
    for sid, df in dfs.items():
        aligned = space.align(df)
        cov = aligned.attrs.get("canonical_coverage", float("nan"))
        logger.info("%s: coverage=%.1f%%", sid, cov * 100)
        result[sid] = aligned
    return result


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def _correlation_matrix_fast(df: pd.DataFrame) -> np.ndarray:
    """Pearson correlation (NaN → column mean перед расчётом)."""
    X = df.to_numpy(dtype=np.float64, copy=True)
    col_means = np.nanmean(X, axis=0)
    nan_mask = ~np.isfinite(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X -= X.mean(axis=0, keepdims=True)
    norms = np.sqrt((X ** 2).sum(axis=0, keepdims=True))
    norms[norms < 1e-12] = 1.0
    X /= norms
    C = X.T @ X
    np.clip(C, -1.0, 1.0, out=C)
    return C


_CONNECTIVITY_METHODS = {
    "correlation": _correlation_matrix_fast,
}


def compute_connectivity(df: pd.DataFrame, method: str = "correlation") -> np.ndarray:
    """Вычисляет матрицу connectivity для одного субъекта."""
    fn = _CONNECTIVITY_METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Метод '{method}' не поддерживается. Доступны: {list(_CONNECTIVITY_METHODS)}"
        )
    return fn(df)


def extract_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Верхний треугольник (без диагонали) → 1D вектор признаков."""
    n = matrix.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    return matrix[idx_i, idx_j]


def build_feature_matrix(
    dfs_aligned: dict[str, pd.DataFrame],
    method: str = "correlation",
) -> tuple[np.ndarray, list[str]]:
    """Строит матрицу признаков (n_subjects × n_features).

    Returns:
        features: ndarray (n_subjects, n_features)
        subject_ids: list[str]
    """
    rows: list[np.ndarray] = []
    subject_ids: list[str] = []
    for sid, df in dfs_aligned.items():
        C = compute_connectivity(df, method=method)
        feat = extract_upper_triangle(C)
        rows.append(feat)
        subject_ids.append(sid)
        logger.info(
            "%s: connectivity %dx%d, признаков=%d",
            sid, C.shape[0], C.shape[1], feat.size,
        )
    return np.vstack(rows), subject_ids


# ---------------------------------------------------------------------------
# Статистика: Mann-Whitney + Benjamini-Hochberg (векторизованный)
# ---------------------------------------------------------------------------

def _mannwhitneyu_vectorized(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U, нормальная аппроксимация, векторизованный.

    Args:
        X_a: (n_a, n_features)
        X_b: (n_b, n_features)

    Returns:
        u_stat: (n_features,) — U-статистика для группы A
        p_values: (n_features,) — двусторонние p-значения
    """
    n_a = X_a.shape[0]
    n_b = X_b.shape[0]
    n = n_a + n_b
    n_features = X_a.shape[1]

    X_all = np.vstack([X_a, X_b])  # (n, n_features)

    # Ранги по каждому признаку (average ties)
    order = np.argsort(X_all, axis=0, kind="stable")
    ranks = np.empty((n, n_features), dtype=np.float64)
    rows_idx = np.arange(n)
    for f in range(n_features):
        r = np.empty(n, dtype=np.float64)
        r[order[:, f]] = rows_idx + 1.0
        # average ties: группируем одинаковые значения
        vals = X_all[order[:, f], f]
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[j] == vals[i]:
                j += 1
            if j > i + 1:
                avg = (i + j + 1) / 2.0  # средний ранг 1-based
                r[order[i:j, f]] = avg
            i = j
        ranks[:, f] = r

    rank_sum_a = ranks[:n_a, :].sum(axis=0)
    U_a = rank_sum_a - n_a * (n_a + 1) / 2.0

    mu = n_a * n_b / 2.0
    sigma = np.sqrt(n_a * n_b * (n + 1) / 12.0)
    Z = (U_a - mu) / sigma
    p = 2.0 * _norm.sf(np.abs(Z))
    p = np.clip(p, 0.0, 1.0)

    return U_a, p


def _fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns:
        p_fdr: скорректированные p-значения
        significant: bool-маска (p_fdr <= alpha)
    """
    n = len(p_values)
    order = np.argsort(p_values)
    rank = np.arange(1, n + 1, dtype=np.float64)

    p_adj = np.minimum(1.0, p_values[order] * n / rank)
    # монотонность: снизу вверх
    for i in range(n - 2, -1, -1):
        if p_adj[i] > p_adj[i + 1]:
            p_adj[i] = p_adj[i + 1]

    p_fdr = np.empty(n, dtype=np.float64)
    p_fdr[order] = p_adj
    return p_fdr, p_fdr <= alpha


def group_comparison(
    features_a: np.ndarray,
    features_b: np.ndarray,
    bin_ids: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Сравнение двух групп по всем парам бинов (upper triangle features).

    Args:
        features_a: (n_a, n_features) — матрица признаков группы A
        features_b: (n_b, n_features) — матрица признаков группы B
        bin_ids: canonical voxel_ids, len = N_bins
        alpha: уровень значимости FDR

    Returns:
        DataFrame: bin_i, bin_j, u_stat, p_raw, p_fdr, significant
        Отсортирован по p_fdr (ascending).
    """
    n_bins = len(bin_ids)
    idx_i, idx_j = np.triu_indices(n_bins, k=1)

    logger.info(
        "Mann-Whitney: %d vs %d субъектов, %d признаков …",
        features_a.shape[0], features_b.shape[0], features_a.shape[1],
    )

    u_stat, p_raw = _mannwhitneyu_vectorized(features_a, features_b)

    logger.info("FDR (Benjamini-Hochberg, alpha=%.3f) …", alpha)
    p_fdr, significant = _fdr_bh(p_raw, alpha=alpha)

    n_sig = int(significant.sum())
    logger.info(
        "Значимых пар после FDR: %d / %d (%.3f%%)",
        n_sig, len(p_raw), 100 * n_sig / max(1, len(p_raw)),
    )

    return pd.DataFrame({
        "bin_i":       [bin_ids[k] for k in idx_i],
        "bin_j":       [bin_ids[k] for k in idx_j],
        "u_stat":      u_stat,
        "p_raw":       p_raw,
        "p_fdr":       p_fdr,
        "significant": significant,
    }).sort_values("p_fdr").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Главная точка входа
# ---------------------------------------------------------------------------

def run_group_pipeline(
    schiz_dir: str | Path,
    healthy_dir: str | Path,
    output_dir: str | Path,
    *,
    method: str = "correlation",
    spatial_grid_size: int = 10,
    spatial_grid_method: str = "mean",
    csv_chunk_rows: int = 32768,
    strategy: str = "intersection",
    alpha: float = 0.05,
    save_canonical_space: bool = True,
    save_feature_matrix: bool = True,
) -> dict:
    """Полный pipeline группового сравнения.

    Шаги:
      1. Загрузка субъектов (потоковая биннизация)
      2. Построение canonical voxel space по всем субъектам
      3. Выравнивание субъектов к canonical space
      4. Вычисление connectivity матриц (Pearson correlation)
      5. Извлечение признаков (upper triangle пар бинов)
      6. Mann-Whitney U + Benjamini-Hochberg FDR
      7. Экспорт: group_comparison.csv, canonical_space.json, features_*.npy

    Returns:
        summary: dict с ключевыми метриками
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка
    logger.info("=== Загрузка: шизофрения (%s) ===", schiz_dir)
    dfs_schiz = load_group(
        schiz_dir, "schiz",
        spatial_grid_size=spatial_grid_size,
        spatial_grid_method=spatial_grid_method,
        csv_chunk_rows=csv_chunk_rows,
    )
    logger.info("=== Загрузка: здоровые (%s) ===", healthy_dir)
    dfs_healthy = load_group(
        healthy_dir, "healthy",
        spatial_grid_size=spatial_grid_size,
        spatial_grid_method=spatial_grid_method,
        csv_chunk_rows=csv_chunk_rows,
    )

    all_dfs = {**dfs_schiz, **dfs_healthy}
    logger.info(
        "Загружено: %d шизофрения + %d здоровые = %d субъектов",
        len(dfs_schiz), len(dfs_healthy), len(all_dfs),
    )

    # 2. Canonical space
    logger.info("=== Canonical voxel space (strategy=%s) ===", strategy)
    space = fit_canonical_space(all_dfs, strategy=strategy)

    if save_canonical_space:
        space_path = out / "canonical_space.json"
        space.save(space_path)
        logger.info("Canonical space: %s", space_path)

    # 3. Выравнивание
    logger.info("=== Выравнивание ===")
    dfs_schiz_al = align_all(dfs_schiz, space)
    dfs_healthy_al = align_all(dfs_healthy, space)

    # 4–5. Connectivity + признаки
    logger.info("=== Connectivity (%s) + признаки ===", method)
    feat_schiz, ids_schiz = build_feature_matrix(dfs_schiz_al, method=method)
    feat_healthy, ids_healthy = build_feature_matrix(dfs_healthy_al, method=method)

    if save_feature_matrix:
        np.save(out / "features_schiz.npy", feat_schiz)
        np.save(out / "features_healthy.npy", feat_healthy)
        pd.Series(ids_schiz).to_csv(
            out / "subject_ids_schiz.csv", index=False, header=["subject_id"]
        )
        pd.Series(ids_healthy).to_csv(
            out / "subject_ids_healthy.csv", index=False, header=["subject_id"]
        )
        logger.info("Матрицы признаков → %s", out)

    # 6. Статистика
    logger.info("=== Mann-Whitney + FDR ===")
    results_df = group_comparison(
        feat_schiz, feat_healthy,
        bin_ids=space.voxel_ids,
        alpha=alpha,
    )

    # 7. Экспорт
    results_df.to_csv(out / "group_comparison.csv", index=False)
    logger.info("Результаты → %s/group_comparison.csv", out)

    sig = results_df[results_df["significant"]]
    if not sig.empty:
        sig.head(20).to_csv(out / "top_significant_pairs.csv", index=False)
        logger.info(
            "Топ-%d значимых пар → %s/top_significant_pairs.csv",
            min(20, len(sig)), out,
        )

    n_sig = int(results_df["significant"].sum())
    n_total = len(results_df)
    return {
        "n_schiz":          len(dfs_schiz),
        "n_healthy":        len(dfs_healthy),
        "n_canonical_bins": space.n_voxels,
        "n_features":       n_total,
        "n_significant":    n_sig,
        "pct_significant":  round(100 * n_sig / max(1, n_total), 4),
        "alpha":            alpha,
        "method":           method,
        "strategy":         strategy,
        "spatial_grid_size": spatial_grid_size,
        "output_dir":       str(out.resolve()),
    }
