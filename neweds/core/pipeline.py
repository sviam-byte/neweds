"""Публичный пайплайн анализа.

Главный путь данных:

    neweds.cli  →  run_analysis  →  реестр метрик  →  AnalysisResult  →  отчёты
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import pandas as pd

from neweds.config import AnalysisConfig, ComputationContract
from neweds.core.data_loader import load_or_generate
from neweds.core.metric_runner import compute_metric
from neweds.core.results import AnalysisResult, MetricResult
from neweds.metrics.registry import get_metric

DEFAULT_PUBLIC_VARIANTS = [
    "correlation_full",
    "dcor_full",
    "ordinal_full",
]


def _config_hash(config: AnalysisConfig) -> str:
    """Короткий стабильный хэш конфига — попадает в метаданные для воспроизводимости."""

    try:
        payload = asdict(config)
    except TypeError:
        payload = dict(vars(config))
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _missing_fraction(data: pd.DataFrame) -> float:
    arr = data.to_numpy(dtype=float, copy=False)
    return float((~np.isfinite(arr)).mean())


def _matrix_strength(matrix: np.ndarray, *, pvalue_based: bool) -> float:
    """Скорим матрицу одним числом «больше = сильнее».

    Для p-value-метрик знак инвертируем (меньшее p = более сильное свидетельство).
    Диагональ маскируем, чтобы метрики, которые жёстко ставят 1.0 на диагонали,
    не задирали средний модуль.
    """

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return float("-inf")
    mask = ~np.eye(arr.shape[0], dtype=bool)
    finite = arr[mask]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("-inf")
    score = float(np.mean(np.abs(finite)))
    return -score if pvalue_based else score


def _select_lag(
    data: pd.DataFrame,
    variant: str,
    *,
    max_lag: int,
    controls: list[str] | None,
) -> tuple[np.ndarray, int]:
    """Перебираем лаги ``1..max_lag`` и выбираем тот, где матрица сильнее."""

    metric = get_metric(variant)
    pvalue_based = metric.pvalue_based

    best_matrix: np.ndarray | None = None
    best_lag = 1
    best_score = float("-inf")
    for lag in range(1, max(1, max_lag) + 1):
        matrix = np.asarray(compute_metric(data, variant, lag=lag, control=controls))
        score = _matrix_strength(matrix, pvalue_based=pvalue_based)
        if score > best_score:
            best_score = score
            best_matrix = matrix
            best_lag = lag
    assert best_matrix is not None  # max_lag >= 1 — хотя бы одна итерация гарантирована
    return best_matrix, best_lag


def run_analysis(
    input_path: str,
    config: AnalysisConfig,
    *,
    controls: list[str] | None = None,
) -> AnalysisResult:
    """Запускает публичный пайплайн анализа.

    Args:
        input_path: путь к поддерживаемому файлу (CSV / Excel / Parquet / HDF5).
        config: конфиг анализа. ``config.variants`` задаёт метрики,
            ``config.lag_selection`` переключает между фиксированным и подбираемым лагом.
        controls: контрольные переменные для ``*_partial`` метрик.
            Если не передать, берётся ``config.controls``.

    Returns:
        :class:`AnalysisResult` с одним :class:`MetricResult` на каждую метрику.
    """

    data, preprocess_report = load_or_generate(
        str(input_path),
        time_col="none",
        transpose="no",
        preprocess=True,
        normalize=True,
        fill_missing=True,
        check_stationarity=False,
        return_report=True,
    )
    variants: Iterable[str] = config.variants or DEFAULT_PUBLIC_VARIANTS
    if controls is None:
        controls = list(config.controls) if config.controls else None

    logs: list[str] = []
    metrics: dict[str, MetricResult] = {}

    max_lag = int(max(1, config.max_lag))
    optimize_lag = config.lag_selection == "optimize"

    cfg_hash = _config_hash(config)
    missing_frac = _missing_fraction(data)
    preprocess_steps = list(getattr(preprocess_report, "steps_global", []) or [])

    for variant in variants:
        metric = get_metric(variant)
        if optimize_lag and metric.directed:
            matrix_np, used_lag = _select_lag(data, variant, max_lag=max_lag, controls=controls)
        else:
            matrix_np = np.asarray(compute_metric(data, variant, lag=max_lag, control=controls))
            used_lag = max_lag

        metadata = {
            "pipeline": "modern",
            "config_hash": cfg_hash,
            "input_shape": list(data.shape),
            "input_missing_fraction": missing_frac,
            "preprocess_report_type": type(preprocess_report).__name__,
            "preprocess_steps": preprocess_steps,
            "category": metric.category,
            "experimental": metric.experimental,
        }

        contract = ComputationContract(
            variant=variant,
            input_channels=int(data.shape[1]),
            input_T=int(data.shape[0]),
            input_missing_frac=missing_frac,
            preprocess_steps=preprocess_steps,
            controls=list(controls) if controls else [],
            control_strategy="provided" if controls else "none",
            directed=metric.directed,
            directed_lag=used_lag,
            lag_selection=config.lag_selection,
            output_shape=tuple(matrix_np.shape),
            seed=config.master_seed,
            config_hash=cfg_hash,
        )

        metrics[variant] = MetricResult(
            name=variant,
            matrix=matrix_np,
            directed=metric.directed,
            lag=used_lag if metric.directed else None,
            pvalue_based=metric.pvalue_based,
            metadata=metadata,
            contract=contract,
        )
        logs.append(
            f"Посчитан {variant}: shape={matrix_np.shape} "
            f"lag={used_lag if metric.directed else 'нет'}"
        )

    return AnalysisResult(
        input_info={
            "path": str(input_path),
            "shape": list(data.shape),
            "columns": [str(c) for c in data.columns],
            "missing_fraction": missing_frac,
        },
        config=config,
        metrics=metrics,
        logs=logs,
        artifacts={
            "data": data,
            "preprocess_report": preprocess_report,
        },
    )
