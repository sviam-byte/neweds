"""Публичный pipeline анализа связности."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neweds.config import AnalysisConfig, ComputationContract
from neweds.core.data_loader import load_or_generate
from neweds.core.metric_runner import compute_metric
from neweds.core.results import AnalysisResult, MetricResult
from neweds.core.statistics import fdr_bh
from neweds.core.window_scanner import analyze_sliding_windows
from neweds.metrics.registry import get_metric

DEFAULT_PUBLIC_VARIANTS = [
    "correlation_full",
    "dcor_full",
    "ordinal_full",
]

_HDF5_EXTS = {".h5", ".hdf5", ".hdf"}
_HEAVY_WINDOW_PREFIXES = (
    "mutinf",
    "dcor",
    "te",
    "granger",
    "ah",
    "ordinal",
    "wavelet",
)


def _is_heavy_window_metric(variant: str) -> bool:
    return str(variant).startswith(_HEAVY_WINDOW_PREFIXES)


def _config_hash(config: AnalysisConfig) -> str:
    try:
        payload = asdict(config)
    except TypeError:
        payload = dict(vars(config))
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _missing_fraction(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    arr = data.to_numpy(dtype=float, copy=False)
    return float((~np.isfinite(arr)).mean())


def _matrix_strength(matrix: np.ndarray, *, pvalue_based: bool) -> float:
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


def _pvalue_mask(matrix: np.ndarray, *, directed: bool) -> np.ndarray:
    arr = np.asarray(matrix)
    mask = np.isfinite(arr)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return mask
    np.fill_diagonal(mask, False)
    if not directed:
        mask &= np.triu(np.ones(arr.shape, dtype=bool), 1)
    return mask


def _apply_pvalue_correction(matrix: np.ndarray, *, method: str, directed: bool) -> np.ndarray:
    correction = str(method or "none").strip().lower()
    if correction in {"", "none"}:
        return np.asarray(matrix, dtype=float)
    out = np.asarray(matrix, dtype=float).copy()
    mask = _pvalue_mask(out, directed=directed)
    if not np.any(mask):
        return out
    pvals = out[mask]
    if correction == "bonferroni":
        corrected = np.clip(pvals * int(pvals.size), 0.0, 1.0)
    elif correction in {"fdr", "fdr_bh", "bh"}:
        corrected = fdr_bh(pvals)
    else:
        raise ValueError("pvalue_correction must be one of: 'none', 'bonferroni', 'fdr_bh'.")
    out[mask] = corrected
    if not directed and out.ndim == 2 and out.shape[0] == out.shape[1]:
        tri_i, tri_j = np.where(mask)
        out[tri_j, tri_i] = corrected
    return out


def _resolve_controls(
    data: pd.DataFrame,
    controls: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray | None]:
    numeric_columns = [str(c) for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    requested = [str(c) for c in (controls or [])]
    missing = [c for c in requested if c not in data.columns]
    if missing:
        raise ValueError(f"Unknown control columns: {missing}")
    nonnumeric = [
        c for c in requested if c in data.columns and not pd.api.types.is_numeric_dtype(data[c])
    ]
    if nonnumeric:
        raise ValueError(f"Control columns must be numeric: {nonnumeric}")

    control_columns = [c for c in requested if c in numeric_columns]
    signal_columns = [c for c in numeric_columns if c not in set(control_columns)]
    if not signal_columns:
        raise ValueError("No numeric signal columns remain after excluding controls.")

    signal_data = data[signal_columns].copy()
    control_matrix = (
        data[control_columns].to_numpy(dtype=np.float64, copy=False) if control_columns else None
    )
    return signal_data, signal_columns, control_columns, control_matrix


def _compute_variant(
    signal_data: pd.DataFrame,
    variant: str,
    *,
    lag: int,
    controls: list[str],
    control_matrix: np.ndarray | None,
    max_pairwise_pairs: int | None = None,
    performance_guardrails: bool = True,
) -> np.ndarray:
    metric = get_metric(variant)
    params: dict[str, Any] = {}
    control_arg: list[str] | None = None
    if controls and metric.supports_control:
        params["control_matrix"] = control_matrix
        control_arg = controls
    params["max_pairwise_pairs"] = max_pairwise_pairs
    params["performance_guardrails"] = performance_guardrails
    return np.asarray(compute_metric(signal_data, variant, lag=lag, control=control_arg, **params))


def _select_lag(
    signal_data: pd.DataFrame,
    variant: str,
    *,
    max_lag: int,
    controls: list[str],
    control_matrix: np.ndarray | None,
    max_pairwise_pairs: int | None = None,
    performance_guardrails: bool = True,
) -> tuple[np.ndarray, int]:
    metric = get_metric(variant)
    best_matrix: np.ndarray | None = None
    best_lag = 1
    best_score = float("-inf")
    for lag in range(1, max(1, max_lag) + 1):
        matrix = _compute_variant(
            signal_data,
            variant,
            lag=lag,
            controls=controls,
            control_matrix=control_matrix,
            max_pairwise_pairs=max_pairwise_pairs,
            performance_guardrails=performance_guardrails,
        )
        score = _matrix_strength(matrix, pvalue_based=metric.pvalue_based)
        if score > best_score:
            best_score = score
            best_matrix = matrix
            best_lag = lag
    if best_matrix is None:
        raise RuntimeError(f"No connectivity matrix was produced for variant={variant!r}")
    return best_matrix, best_lag


def _load_data(input_path: str, config: AnalysisConfig):
    ext = Path(str(input_path)).suffix.lower()
    h5_grid_size = int(config.spatial_grid_size) if ext in _HDF5_EXTS else None
    ar_order = int(getattr(config, "ar_order", 0))
    return load_or_generate(
        str(input_path),
        time_col="none",
        transpose="no",
        h5_spatial_bin=int(config.spatial_bin_size) if int(config.spatial_bin_size) > 1 else None,
        spatial_grid_size=h5_grid_size,
        spatial_grid_method=str(config.spatial_grid_method),
        lazy_spatial_bin=bool(config.lazy_spatial_bin),
        time_chunk=int(config.time_chunk),
        preprocess=bool(getattr(config, "preprocess", True)),
        normalize=bool(getattr(config, "normalize", True)),
        fill_missing=bool(getattr(config, "fill_missing", True)),
        remove_outliers=bool(getattr(config, "remove_outliers", True)),
        remove_ar1=ar_order > 0,
        remove_ar_order=max(1, ar_order),
        check_stationarity=False,
        return_report=True,
    )


def _run_windows(
    signal_data: pd.DataFrame,
    variants: Iterable[str],
    config: AnalysisConfig,
    *,
    controls: list[str],
    control_matrix: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    if not config.window_sizes:
        return {}
    windows: dict[str, dict[str, Any]] = {}
    stride_default = config.window_stride or max(
        1, min(int(config.window_sizes[0]), len(signal_data)) // 2
    )
    for variant in variants:
        metric = get_metric(variant)
        per_size: dict[int, dict[str, Any]] = {}

        def _compute(chunk: pd.DataFrame, name: str, *, lag: int = 1, **params):
            cm = None
            if control_matrix is not None:
                positions = signal_data.index.get_indexer(chunk.index)
                cm = control_matrix[positions]
            return _compute_variant(
                chunk,
                name,
                lag=lag,
                controls=controls,
                control_matrix=cm,
                max_pairwise_pairs=int(config.max_pairwise_pairs),
                performance_guardrails=bool(config.performance_guardrails),
            )

        for window_size in config.window_sizes:
            max_windows = (
                int(config.heavy_window_max_windows)
                if _is_heavy_window_metric(variant)
                else 400
            )
            per_size[int(window_size)] = analyze_sliding_windows(
                signal_data,
                variant,
                int(window_size),
                int(stride_default),
                compute_variant_func=_compute,
                is_pvalue=metric.pvalue_based,
                lag=int(max(1, config.max_lag)),
                max_windows=max_windows,
                return_matrices=False,
            )
        windows[variant] = {
            "policy": str(config.window_policy),
            "stride": int(stride_default),
            "sizes": per_size,
        }
    return windows


def run_analysis(
    input_path: str,
    config: AnalysisConfig,
    *,
    controls: list[str] | None = None,
) -> AnalysisResult:
    """Запускает публичный pipeline анализа."""

    data, preprocess_report = _load_data(str(input_path), config)
    if config.auto_difference:
        data = data.diff().dropna().reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    variants: list[str] = list(config.variants or DEFAULT_PUBLIC_VARIANTS)
    if controls is None:
        controls = list(config.controls) if config.controls else None

    signal_data, signal_columns, control_columns, control_matrix = _resolve_controls(data, controls)

    logs: list[str] = []
    metrics: dict[str, MetricResult] = {}

    max_lag = int(max(1, config.max_lag))
    optimize_lag = config.lag_selection == "optimize"

    cfg_hash = _config_hash(config)
    missing_frac = _missing_fraction(signal_data)
    preprocess_steps = list(getattr(preprocess_report, "steps_global", []) or [])
    if config.auto_difference:
        preprocess_steps.append("auto_difference")

    for variant in variants:
        metric = get_metric(variant)
        if optimize_lag and metric.directed:
            matrix_np, used_lag = _select_lag(
                signal_data,
                variant,
                max_lag=max_lag,
                controls=control_columns,
                control_matrix=control_matrix,
                max_pairwise_pairs=int(config.max_pairwise_pairs),
                performance_guardrails=bool(config.performance_guardrails),
            )
        else:
            matrix_np = _compute_variant(
                signal_data,
                variant,
                lag=max_lag,
                controls=control_columns,
                control_matrix=control_matrix,
                max_pairwise_pairs=int(config.max_pairwise_pairs),
                performance_guardrails=bool(config.performance_guardrails),
            )
            used_lag = max_lag

        if metric.pvalue_based:
            matrix_np = _apply_pvalue_correction(
                matrix_np,
                method=config.pvalue_correction,
                directed=metric.directed,
            )

        metadata = {
            "pipeline": "modern",
            "config_hash": cfg_hash,
            "input_shape": list(signal_data.shape),
            "input_missing_fraction": missing_frac,
            "preprocess_report_type": type(preprocess_report).__name__,
            "preprocess_steps": preprocess_steps,
            "category": metric.category,
            "experimental": metric.experimental,
            "partial_mode": str(getattr(metric, "partial_mode", "none")),
            "signal_columns": list(signal_columns),
            "control_columns": list(control_columns),
            "matrix_columns": list(signal_columns),
            "pvalue_correction": str(config.pvalue_correction),
        }

        contract = ComputationContract(
            variant=variant,
            input_channels=int(signal_data.shape[1]),
            input_T=int(signal_data.shape[0]),
            input_missing_frac=missing_frac,
            preprocess_steps=preprocess_steps,
            controls=list(control_columns),
            control_strategy="provided" if control_columns else "none",
            partial_mode=str(getattr(metric, "partial_mode", "none")),
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
            f"Computed {variant}: shape={matrix_np.shape} lag={used_lag if metric.directed else 'none'}"
        )

    windows = _run_windows(
        signal_data,
        variants,
        config,
        controls=control_columns,
        control_matrix=control_matrix,
    )

    return AnalysisResult(
        input_info={
            "path": str(input_path),
            "shape": list(signal_data.shape),
            "columns": list(signal_columns),
            "control_columns": list(control_columns),
            "raw_shape": list(data.shape),
            "raw_columns": [str(c) for c in data.columns],
            "missing_fraction": missing_frac,
        },
        config=config,
        metrics=metrics,
        logs=logs,
        windows=windows,
        artifacts={
            "analysis_data": signal_data,
            "data": signal_data,
            "preprocess_report": preprocess_report,
        },
    )
