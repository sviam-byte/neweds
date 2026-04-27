"""Групповой pipeline: загрузка субъектов, canonical alignment, connectivity, сравнение групп.

⚠️ EXPERIMENTAL: это baseline edge-wise сравнение (Mann-Whitney U + Benjamini-Hochberg FDR).
covariate-aware GLM, permutation tests, site-aware design — на дорожной карте, но пока нет.
Не используйте результаты как финальную статистику для публикации.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from .data_loader import load_or_generate
from .voxel_space import CanonicalVoxelSpace

GROUP_PIPELINE_EXPERIMENTAL_NOTICE = (
    "[experimental] neweds-group is a baseline edge-wise group-comparison pipeline. "
    "Covariate-aware GLM, permutation tests, and site-aware design are NOT yet implemented. "
    "Treat outputs as exploratory."
)


@dataclass(frozen=True, slots=True)
class GroupComparisonResult:
    """Структурированный результат группового сравнения.

    Совместим с историческим dict-выводом ``run_group_pipeline`` через ``from_summary``
    и ``as_dict``. ``run_group_pipeline`` пока возвращает dict ради обратной совместимости.
    """

    method: str
    strategy: str
    canonical_reference: str
    n_case: int
    n_control: int
    n_canonical_bins: int
    n_features: int
    n_significant: int
    pct_significant: float
    alpha: float
    output_dir: str
    design_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped_subjects: list[dict[str, str]] = field(default_factory=list)
    missing_bins_diag_corr: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> GroupComparisonResult:
        known = {
            "method",
            "strategy",
            "canonical_reference",
            "n_case",
            "n_control",
            "n_canonical_bins",
            "n_features",
            "n_significant",
            "pct_significant",
            "alpha",
            "output_dir",
            "design_metadata",
            "warnings",
            "skipped_subjects",
            "missing_bins_diag_corr",
        }
        extra = {k: v for k, v in summary.items() if k not in known}
        return cls(
            method=str(summary.get("method", "")),
            strategy=str(summary.get("strategy", "")),
            canonical_reference=str(summary.get("canonical_reference", "")),
            n_case=int(summary.get("n_case", summary.get("n_schiz", 0))),
            n_control=int(summary.get("n_control", summary.get("n_healthy", 0))),
            n_canonical_bins=int(summary.get("n_canonical_bins", 0)),
            n_features=int(summary.get("n_features", 0)),
            n_significant=int(summary.get("n_significant", 0)),
            pct_significant=float(summary.get("pct_significant", 0.0)),
            alpha=float(summary.get("alpha", 0.0)),
            output_dir=str(summary.get("output_dir", "")),
            design_metadata=dict(summary.get("design_metadata", {})),
            warnings=list(summary.get("warnings", [])),
            skipped_subjects=list(summary.get("skipped_subjects", [])),
            missing_bins_diag_corr=summary.get("missing_bins_diag_corr"),
            extra=extra,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "strategy": self.strategy,
            "canonical_reference": self.canonical_reference,
            "n_case": self.n_case,
            "n_control": self.n_control,
            "n_canonical_bins": self.n_canonical_bins,
            "n_features": self.n_features,
            "n_significant": self.n_significant,
            "pct_significant": self.pct_significant,
            "alpha": self.alpha,
            "output_dir": self.output_dir,
            "design_metadata": dict(self.design_metadata),
            "warnings": list(self.warnings),
            "skipped_subjects": list(self.skipped_subjects),
            "missing_bins_diag_corr": self.missing_bins_diag_corr,
            **self.extra,
        }

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".parquet"}
_T_COL_RE = re.compile(r"^t\d+$")
_META_COLS = frozenset({"subject", "group", "iq", "sex"})

# Порог point-biserial корреляции missingness×group для критического предупреждения.
COVERAGE_CONFOUND_THRESHOLD = 0.4


class GroupLoadResult(dict[str, pd.DataFrame]):
    """Mapping of subject id to data plus explicit skipped-subject metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.skipped_subjects: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------


def _iter_subject_files(directory: str | Path) -> list[Path]:
    """Возвращает поддерживаемые файлы субъектов, отсортированные по имени."""
    root = Path(directory)
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS)
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
        # Не применяем subject-wise preprocessing до align(), чтобы не вносить
        # асимметричный dropout бинов между группами.
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        # Не маскируем NaN нулями: отсутствующие бины должны быть явными.
        fill_missing=False,
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
    allow_skip: bool = False,
) -> dict[str, pd.DataFrame]:
    """Загружает все субъекты из директории.

    Returns:
        dict: ``group_label::stem`` → DataFrame (time × bins)
    """
    files = _iter_subject_files(directory)
    result = GroupLoadResult()
    schemas: list[dict] = []
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
            schemas.append(_validate_subject_schema(path, _peek_columns(path)))
            logger.info(
                "[%s] %s: %d t × %d bins",
                group_label,
                path.name,
                df.shape[0],
                df.shape[1],
            )
        except Exception as exc:
            if not allow_skip:
                raise RuntimeError(f"[{group_label}] failed to load {path.name}: {exc}") from exc
            result.skipped_subjects.append({"file": path.name, "error": str(exc)})
            logger.error("[%s] ПРОПУСК %s: %s", group_label, path.name, exc)
    if not result:
        raise RuntimeError(f"Ни один субъект из {directory} не загружен.")

    # Защита от несопоставимых матриц connectivity: у всех субъектов должен быть
    # одинаковый размер временной оси (число TR/таймпоинтов).
    n_times = {sid: int(df.shape[0]) for sid, df in result.items()}
    unique_times = set(n_times.values())
    if len(unique_times) > 1:
        counts = Counter(n_times.values())
        logger.error(
            "[%s] КРИТИЧНО: обнаружено разное число timepoints: %s",
            group_label,
            dict(counts),
        )
        raise ValueError(
            f"[{group_label}] Разное число timepoints: {dict(counts)}. "
            "Перед анализом приведите субъекты к единой временной длине."
        )

    if schemas:
        _cross_validate_group_schemas(schemas, group_label)
    return result


def _peek_columns(filepath: Path) -> list[str]:
    """Читает только имена колонок файла субъекта (дёшево, без полной загрузки)."""
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        probe = pd.read_csv(filepath, nrows=0)
        return [str(c) for c in probe.columns]
    if suffix in {".xlsx", ".xls"}:
        probe = pd.read_excel(filepath, nrows=0)
        return [str(c) for c in probe.columns]
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            return [str(c) for c in pq.ParquetFile(filepath).schema.names]
        except Exception:
            probe = pd.read_parquet(filepath)
            return [str(c) for c in probe.columns]
    return []


def _validate_subject_schema(filepath: Path, columns: list[str]) -> dict:
    """Извлекает ключевые метрики схемы входного файла субъекта."""
    cols = [str(c).strip() for c in columns]
    lower = [c.lower() for c in cols]

    time_cols = [c for c in lower if _T_COL_RE.match(c)]
    meta_cols = [c for c in lower if c in _META_COLS]
    coord_cols = [c for c in lower if c in {"x", "y", "z"}]

    t_ids = sorted(int(c[1:]) for c in time_cols)
    if t_ids:
        t_min, t_max = t_ids[0], t_ids[-1]
        missing = sorted(set(range(t_min, t_max + 1)) - set(t_ids))
    else:
        t_min, t_max, missing = None, None, []

    if missing:
        logger.warning(
            "[schema] %s: пропущены временные точки t* (%d, первые=%s)",
            filepath.name,
            len(missing),
            missing[:10],
        )

    return {
        "file": filepath.name,
        "n_time_cols": len(time_cols),
        "n_meta_cols": len(meta_cols),
        "n_coord_cols": len(coord_cols),
        "t_range": (t_min, t_max),
        "has_meta": bool(meta_cols),
        "missing_t_count": len(missing),
    }


def _cross_validate_group_schemas(schemas: list[dict], group_label: str) -> None:
    """Сверяет схемы субъектов внутри одной группы и пишет диагностические предупреждения."""
    n_times = sorted({int(s["n_time_cols"]) for s in schemas})
    t_ranges = sorted({tuple(s["t_range"]) for s in schemas})
    if len(n_times) > 1:
        logger.warning("[%s] Разное число временных колонок t*: %s", group_label, n_times)
    if len(t_ranges) > 1:
        logger.warning("[%s] Разные диапазоны t*: %s", group_label, t_ranges)

    no_meta = [s["file"] for s in schemas if not bool(s["has_meta"])]
    if no_meta:
        logger.warning(
            "[%s] Субъекты без метаданных subject/group/iq/sex: %s",
            group_label,
            no_meta,
        )

    with_gaps = [s["file"] for s in schemas if int(s["missing_t_count"]) > 0]
    if with_gaps:
        logger.warning("[%s] Обнаружены пропуски t* у файлов: %s", group_label, with_gaps)


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
        strategy,
        space.n_voxels,
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


def _count_fully_missing_bins(df: pd.DataFrame) -> int:
    """Считает число бинов, полностью отсутствующих у субъекта после align()."""
    arr = df.to_numpy(dtype=np.float64, copy=False)
    return int((~np.isfinite(arr)).all(axis=0).sum())


def build_missing_bin_qc_table(dfs_aligned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Строит QC-таблицу покрытия canonical space для набора субъектов."""
    rows: list[dict[str, float | str | int]] = []
    for sid, df in dfs_aligned.items():
        n_bins = int(df.shape[1])
        n_missing = _count_fully_missing_bins(df)
        rows.append(
            {
                "subject_id": sid,
                "n_bins": n_bins,
                "n_missing_bins": n_missing,
                "missing_bin_fraction": float(n_missing / max(1, n_bins)),
            }
        )
    return pd.DataFrame(rows)


def _point_biserial_binary(labels: np.ndarray, values: np.ndarray) -> float:
    """Корреляция бинарной метки и непрерывного признака (через Pearson)."""
    x = np.asarray(labels, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def _correlation_matrix_fast(df: pd.DataFrame) -> np.ndarray:
    """Pearson correlation с корректной обработкой отсутствующих бинов.

    После ``align()`` часть бинов может отсутствовать у субъекта полностью.
    Такие колонки приходят как all-NaN и должны давать нулевые корреляции,
    а не заражать матрицу NaN-ами.
    """
    X = df.to_numpy(dtype=np.float64, copy=True)

    # Колонки, где есть хотя бы одно конечное наблюдение.
    finite_col = np.isfinite(X).any(axis=0)

    # Для обычных колонок заполняем NaN средним; для all-NaN колонок берём 0.
    finite_mask = np.isfinite(X)
    finite_counts = finite_mask.sum(axis=0)
    finite_sums = np.where(finite_mask, X, 0.0).sum(axis=0)
    col_means = np.divide(
        finite_sums,
        np.maximum(finite_counts, 1),
        dtype=np.float64,
    )
    col_means = np.where(finite_col, col_means, 0.0)
    nan_mask = ~np.isfinite(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    X -= X.mean(axis=0, keepdims=True)
    norms = np.sqrt((X**2).sum(axis=0, keepdims=True))
    # Нулевые колонки (в т.ч. бывшие all-NaN) оставляем нулевыми после деления.
    norms[norms < 1e-12] = 1.0
    X /= norms

    C = X.T @ X
    np.clip(C, -1.0, 1.0, out=C)

    # Явно обнуляем строки/столбцы отсутствующих бинов для прозрачности.
    C[~finite_col, :] = 0.0
    C[:, ~finite_col] = 0.0
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
            sid,
            C.shape[0],
            C.shape[1],
            feat.size,
        )
    return np.vstack(rows), subject_ids


def filter_features_by_bin_coverage(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    bin_ids: list[str],
    dfs_a: dict[str, pd.DataFrame],
    dfs_b: dict[str, pd.DataFrame],
    min_coverage: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Фильтрует признаки-пары по покрытию бинов среди всех субъектов.

    Сохраняются только пары (i, j), где оба бина присутствуют как минимум у
    ``min_coverage`` доли субъектов (проверка по наличию хотя бы одного finite
    значения в колонке после align).
    """
    if not (0.0 < min_coverage <= 1.0):
        raise ValueError("min_coverage должен быть в диапазоне (0, 1].")

    n_bins = len(bin_ids)
    all_dfs = list(dfs_a.values()) + list(dfs_b.values())
    n_total = len(all_dfs)

    bin_present_count = np.zeros(n_bins, dtype=np.int64)
    for df in all_dfs:
        arr = df.to_numpy(dtype=np.float64, copy=False)
        bin_present_count += np.isfinite(arr).any(axis=0).astype(np.int64)

    bin_ok = bin_present_count >= (min_coverage * n_total)
    idx_i, idx_j = np.triu_indices(n_bins, k=1)
    feat_mask = bin_ok[idx_i] & bin_ok[idx_j]

    logger.info(
        "Coverage filter: %d/%d бинов >= %.0f%% покрытия; признаков сохранено %d/%d",
        int(bin_ok.sum()),
        n_bins,
        min_coverage * 100,
        int(feat_mask.sum()),
        len(feat_mask),
    )
    if not np.any(feat_mask):
        raise RuntimeError(
            "Coverage filter удалил все признаки. "
            "Снизьте min_bin_coverage или проверьте покрытие данных."
        )

    return feat_a[:, feat_mask], feat_b[:, feat_mask], feat_mask


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

    ranks = np.empty((n, n_features), dtype=np.float64)
    tie_corrections = np.zeros(n_features, dtype=np.float64)
    for f in range(n_features):
        col = X_all[:, f]
        ranks[:, f] = pd.Series(col).rank(method="average").to_numpy(dtype=np.float64)
        _, counts = np.unique(col, return_counts=True)
        tie_corrections[f] = np.sum(counts**3 - counts)

    rank_sum_a = ranks[:n_a, :].sum(axis=0)
    U_a = rank_sum_a - n_a * (n_a + 1) / 2.0

    mu = n_a * n_b / 2.0
    sigma = np.sqrt((n_a * n_b / 12.0) * ((n + 1) - tie_corrections / (n * (n - 1))))
    sigma = np.maximum(sigma, 1e-12)
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
    pair_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Сравнение двух групп по всем парам бинов (upper triangle features).

    Args:
        features_a: (n_a, n_features) — матрица признаков группы A
        features_b: (n_b, n_features) — матрица признаков группы B
        bin_ids: canonical voxel_ids, len = N_bins
        alpha: уровень значимости FDR

    Returns:
        DataFrame: bin_i, bin_j, u_stat, p_raw, p_fdr, effect_size_r, significant
        Отсортирован по p_fdr (ascending).
    """
    n_bins = len(bin_ids)
    idx_i, idx_j = np.triu_indices(n_bins, k=1)
    if pair_mask is not None:
        idx_i = idx_i[pair_mask]
        idx_j = idx_j[pair_mask]

    n_a, n_b = features_a.shape[0], features_b.shape[0]

    logger.info(
        "Mann-Whitney: %d vs %d субъектов, %d признаков …",
        features_a.shape[0],
        features_b.shape[0],
        features_a.shape[1],
    )

    u_stat, p_raw = _mannwhitneyu_vectorized(features_a, features_b)

    logger.info("FDR (Benjamini-Hochberg, alpha=%.3f) …", alpha)
    p_fdr, significant = _fdr_bh(p_raw, alpha=alpha)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n_a * n_b)

    n_sig = int(significant.sum())
    logger.info(
        "Значимых пар после FDR: %d / %d (%.3f%%)",
        n_sig,
        len(p_raw),
        100 * n_sig / max(1, len(p_raw)),
    )

    return (
        pd.DataFrame(
            {
                "bin_i": [bin_ids[k] for k in idx_i],
                "bin_j": [bin_ids[k] for k in idx_j],
                "u_stat": u_stat,
                "p_raw": p_raw,
                "p_fdr": p_fdr,
                "effect_size_r": rank_biserial,
                "significant": significant,
            }
        )
        .sort_values("p_fdr")
        .reset_index(drop=True)
    )


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
    canonical_reference: str = "all",
    min_bin_coverage: float = 0.8,
    allow_skip: bool = False,
) -> dict:
    """Полный pipeline группового сравнения.

    Шаги:
      1. Загрузка субъектов (потоковая биннизация)
      2. Построение canonical voxel space по reference-группе
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
        schiz_dir,
        "schiz",
        spatial_grid_size=spatial_grid_size,
        spatial_grid_method=spatial_grid_method,
        csv_chunk_rows=csv_chunk_rows,
        allow_skip=allow_skip,
    )
    logger.info("=== Загрузка: здоровые (%s) ===", healthy_dir)
    dfs_healthy = load_group(
        healthy_dir,
        "healthy",
        spatial_grid_size=spatial_grid_size,
        spatial_grid_method=spatial_grid_method,
        csv_chunk_rows=csv_chunk_rows,
        allow_skip=allow_skip,
    )

    all_dfs = {**dfs_schiz, **dfs_healthy}
    logger.info(
        "Загружено: %d шизофрения + %d здоровые = %d субъектов",
        len(dfs_schiz),
        len(dfs_healthy),
        len(all_dfs),
    )

    # 2. Canonical space
    logger.info("=== Canonical voxel space (strategy=%s) ===", strategy)
    raw_ref = str(canonical_reference or "all").strip().lower()
    ref = {"case": "schiz", "control": "healthy"}.get(raw_ref, raw_ref)
    if ref == "healthy":
        ref_dfs = dfs_healthy
    elif ref == "schiz":
        ref_dfs = dfs_schiz
    elif ref == "all":
        ref_dfs = all_dfs
        logger.warning("canonical_reference='all' может давать leakage при train/test сценариях.")
    else:
        raise ValueError(
            "canonical_reference must be one of: 'case', 'control', 'healthy', 'schiz', 'all'."
        )

    space = fit_canonical_space(ref_dfs, strategy=strategy)
    space.source_info["reference_group"] = ref
    space.source_info["n_reference_subjects"] = len(ref_dfs)

    if save_canonical_space:
        space_path = out / "canonical_space.json"
        space.save(space_path)
        logger.info("Canonical space: %s", space_path)

    # 3. Выравнивание
    logger.info("=== Выравнивание ===")
    dfs_schiz_al = align_all(dfs_schiz, space)
    dfs_healthy_al = align_all(dfs_healthy, space)

    # 4. QC по отсутствующим бинам (потенциальный конфаундер покрытия)
    qc_schiz = build_missing_bin_qc_table(dfs_schiz_al)
    qc_schiz["group"] = "schiz"
    qc_healthy = build_missing_bin_qc_table(dfs_healthy_al)
    qc_healthy["group"] = "healthy"
    qc_missing = pd.concat([qc_schiz, qc_healthy], axis=0, ignore_index=True)

    labels = np.concatenate(
        [
            np.zeros(len(qc_schiz), dtype=np.float64),
            np.ones(len(qc_healthy), dtype=np.float64),
        ]
    )
    missing_corr = _point_biserial_binary(labels, qc_missing["n_missing_bins"].to_numpy())
    if np.isfinite(missing_corr) and abs(missing_corr) >= 0.3:
        logger.warning(
            "n_missing_bins заметно коррелирует с диагнозом (r=%.4f). "
            "Результаты группового сравнения могут быть конфаундированы покрытием.",
            missing_corr,
        )
    if np.isfinite(missing_corr) and abs(missing_corr) >= COVERAGE_CONFOUND_THRESHOLD:
        logger.error(
            "КРИТИЧНО: n_missing_bins сильно коррелирует с диагнозом (r=%.4f >= %.2f). "
            "Рекомендуется фиксированный spatial_bin_range и/или более строгий coverage-filter.",
            missing_corr,
            COVERAGE_CONFOUND_THRESHOLD,
        )

    qc_missing.to_csv(out / "missing_bin_qc.csv", index=False)

    # 5. Connectivity + признаки
    logger.info("=== Connectivity (%s) + признаки ===", method)
    feat_schiz, ids_schiz = build_feature_matrix(dfs_schiz_al, method=method)
    feat_healthy, ids_healthy = build_feature_matrix(dfs_healthy_al, method=method)

    pair_mask: np.ndarray | None = None
    if min_bin_coverage > 0.0:
        logger.info(
            "=== Фильтрация признаков по покрытию бинов (min=%.0f%%) ===", min_bin_coverage * 100
        )
        feat_schiz, feat_healthy, pair_mask = filter_features_by_bin_coverage(
            feat_schiz,
            feat_healthy,
            bin_ids=space.voxel_ids,
            dfs_a=dfs_schiz_al,
            dfs_b=dfs_healthy_al,
            min_coverage=min_bin_coverage,
        )

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
        feat_schiz,
        feat_healthy,
        bin_ids=space.voxel_ids,
        alpha=alpha,
        pair_mask=pair_mask,
    )

    # 7. Экспорт
    results_df.to_csv(out / "group_comparison.csv", index=False)
    logger.info("Результаты → %s/group_comparison.csv", out)

    sig = results_df[results_df["significant"]]
    if not sig.empty:
        sig.head(20).to_csv(out / "top_significant_pairs.csv", index=False)
        logger.info(
            "Топ-%d значимых пар → %s/top_significant_pairs.csv",
            min(20, len(sig)),
            out,
        )

    n_sig = int(results_df["significant"].sum())
    n_total = len(results_df)

    warnings: list[str] = [GROUP_PIPELINE_EXPERIMENTAL_NOTICE]
    if np.isfinite(missing_corr) and abs(missing_corr) >= COVERAGE_CONFOUND_THRESHOLD:
        warnings.append(
            f"coverage confound: |r(missing_bins, group)|={abs(missing_corr):.3f} "
            f">= {COVERAGE_CONFOUND_THRESHOLD}"
        )

    design_metadata = {
        "method": method,
        "strategy": strategy,
        "canonical_reference": ref,
        "spatial_grid_size": int(spatial_grid_size),
        "spatial_grid_method": str(spatial_grid_method),
        "min_bin_coverage": float(min_bin_coverage),
        "alpha": float(alpha),
        "covariates": [],
        "covariate_model": "none",
        "stat_test": "mann_whitney_u",
        "multiple_comparison": "fdr_bh",
    }

    return {
        "n_case": len(dfs_schiz),
        "n_control": len(dfs_healthy),
        # Backward-compat keys (старые пользователи).
        "n_schiz": len(dfs_schiz),
        "n_healthy": len(dfs_healthy),
        "n_canonical_bins": space.n_voxels,
        "n_features": n_total,
        "n_significant": n_sig,
        "pct_significant": round(100 * n_sig / max(1, n_total), 4),
        "alpha": alpha,
        "method": method,
        "strategy": strategy,
        "canonical_reference": ref,
        "allow_skip": bool(allow_skip),
        "skipped_subjects": list(getattr(dfs_schiz, "skipped_subjects", []))
        + list(getattr(dfs_healthy, "skipped_subjects", [])),
        "min_bin_coverage": min_bin_coverage,
        "spatial_grid_size": spatial_grid_size,
        "missing_bins_diag_corr": (
            None if not np.isfinite(missing_corr) else round(float(missing_corr), 6)
        ),
        "missing_bins_qc_path": str((out / "missing_bin_qc.csv").resolve()),
        "output_dir": str(out.resolve()),
        "design_metadata": design_metadata,
        "warnings": warnings,
        "experimental": True,
    }
