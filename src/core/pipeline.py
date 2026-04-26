"""Public analysis pipeline.

This is the portfolio-facing path:

interfaces/cli.py -> src.core.pipeline.run_analysis -> src.metrics.registry
-> AnalysisResult -> reporting

The legacy engine is intentionally not used here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Iterable

import numpy as np

from src.config import AnalysisConfig, ComputationContract, is_directed_method, is_pvalue_method
from src.core.data_loader import load_or_generate
from src.core.metric_runner import compute_metric
from src.core.results import AnalysisResult, MetricResult

DEFAULT_PUBLIC_VARIANTS = [
    "correlation_full",
    "dcor_full",
    "ordinal_full",
]


def _config_hash(config: AnalysisConfig) -> str:
    """Stable short hash for reproducibility metadata."""

    try:
        payload = asdict(config)
    except TypeError:
        payload = dict(vars(config))
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _missing_fraction(data) -> float:
    arr = data.to_numpy(dtype=float, copy=False)
    return float((~np.isfinite(arr)).mean())


def run_analysis(input_path: str, config: AnalysisConfig) -> AnalysisResult:
    """Run the modern layered analysis pipeline."""

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

    logs: list[str] = []
    metrics: dict[str, MetricResult] = {}

    lag = int(max(1, config.max_lag))
    if config.lag_selection != "fixed":
        logs.append(
            "Modern public pipeline supports fixed-lag metric execution. "
            f"Using lag={lag}; advanced lag optimization remains legacy-compatible."
        )
    cfg_hash = _config_hash(config)
    missing_frac = _missing_fraction(data)
    preprocess_steps = list(getattr(preprocess_report, "steps_global", []) or [])

    for variant in variants:
        matrix = compute_metric(data, variant, lag=lag)
        matrix_np = np.asarray(matrix)
        directed = is_directed_method(variant)

        metadata = {
            "pipeline": "modern",
            "config_hash": cfg_hash,
            "input_shape": list(data.shape),
            "input_missing_fraction": missing_frac,
            "preprocess_report_type": type(preprocess_report).__name__,
            "preprocess_steps": preprocess_steps,
        }

        contract = ComputationContract(
            variant=variant,
            input_channels=int(data.shape[1]),
            input_T=int(data.shape[0]),
            input_missing_frac=missing_frac,
            preprocess_steps=preprocess_steps,
            directed=directed,
            directed_lag=lag,
            lag_selection=config.lag_selection,
            output_shape=tuple(matrix_np.shape),
            seed=config.master_seed,
            config_hash=cfg_hash,
        )

        metrics[variant] = MetricResult(
            name=variant,
            matrix=matrix_np,
            directed=directed,
            lag=lag if directed else None,
            pvalue_based=is_pvalue_method(variant),
            metadata=metadata,
            contract=contract,
        )
        logs.append(f"Computed {variant}: shape={matrix_np.shape}")

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
