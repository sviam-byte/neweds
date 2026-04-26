"""Metric execution boundary.

This module is the stable computation layer used by public pipelines.
It depends on the metrics registry, not on the legacy engine.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.metrics.registry import METRICS_REGISTRY, get_metric_func


def compute_metric(
    data: pd.DataFrame,
    variant: str,
    *,
    lag: int = 1,
    control: Optional[list[str]] = None,
    **params,
) -> np.ndarray:
    """Compute one metric through the central registry."""

    metric_func = get_metric_func(variant)
    return metric_func(data, lag=int(max(1, lag)), control=control, **params)


def compute_all_metrics(
    data: pd.DataFrame,
    variants: Optional[list[str]] = None,
    *,
    lag: int = 1,
    control: Optional[list[str]] = None,
    **params,
) -> dict[str, np.ndarray]:
    """Compute multiple metrics through the central registry."""

    selected = list(variants) if variants is not None else list(METRICS_REGISTRY.keys())
    return {
        variant: compute_metric(data, variant, lag=lag, control=control, **params)
        for variant in selected
    }
