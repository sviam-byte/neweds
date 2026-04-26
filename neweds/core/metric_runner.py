"""Граница вычислений: тонкая прослойка между пайплайнами и реестром метрик.

Стабильный вычислительный слой публичных пайплайнов.
Зависит только от реестра метрик — не тащит за собой ничего лишнего.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neweds.metrics.registry import METRICS_REGISTRY, get_metric_func


def compute_metric(
    data: pd.DataFrame,
    variant: str,
    *,
    lag: int = 1,
    control: list[str] | None = None,
    **params,
) -> np.ndarray:
    """Считает одну метрику через центральный реестр."""

    metric_func = get_metric_func(variant)
    return metric_func(data, lag=int(max(1, lag)), control=control, **params)


def compute_all_metrics(
    data: pd.DataFrame,
    variants: list[str] | None = None,
    *,
    lag: int = 1,
    control: list[str] | None = None,
    **params,
) -> dict[str, np.ndarray]:
    """Считает сразу несколько метрик через центральный реестр."""

    selected = list(variants) if variants is not None else list(METRICS_REGISTRY.keys())
    return {
        variant: compute_metric(data, variant, lag=lag, control=control, **params)
        for variant in selected
    }
