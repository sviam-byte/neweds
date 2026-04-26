"""Скользящее окно и 3D-сканирование (window × lag × position) для анализа во времени."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .statistics import lag_quality, select_best_median_worst


def analyze_sliding_windows(
    data: pd.DataFrame,
    variant: str,
    window_size: int,
    stride: int,
    *,
    compute_variant_func,
    is_pvalue: bool,
    lag: int = 1,
    pairs: list[tuple[int, int]] | None = None,
    start_min: int | None = None,
    start_max: int | None = None,
    max_windows: int = 400,
    return_matrices: bool = False,
    n_jobs: int | None = None,
    parallel_backend: str | None = None,
) -> dict:
    """Sliding window analysis for a given window_size.

    Returns structure for HTML report:
      {
        "best_window": {"start": int, "end": int, "metric": float, "matrix": ndarray},
        "curve": {"x": [start_idx...], "y": [metric...]},
        "ticks": [{"start":.., "end":.., "metric":.., "matrix":..}, ...],
        "extremes": {"best": idx, "median": idx, "worst": idx}
      }
    """
    if data is None or data.empty:
        return {}

    n = len(data)
    w = int(max(2, min(window_size, n)))
    s = int(max(1, stride))

    best = {"start": 0, "end": w, "metric": float("-inf"), "matrix": None}

    st0 = int(max(0, start_min)) if start_min is not None else 0
    st1 = int(min(n - w, start_max)) if start_max is not None else (n - w)
    st1 = int(max(st0, st1))

    max_windows = int(max(1, max_windows))
    starts = list(range(st0, st1 + 1, s))
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]

    try:
        nj = int(n_jobs) if n_jobs is not None else 1
    except Exception:
        nj = 1
    nj = int(max(1, nj))

    def _compute_one(start: int) -> tuple[int, int, float, np.ndarray | None]:
        end = start + w
        if end > n:
            return int(start), int(end), float("nan"), None
        chunk = data.iloc[start:end]
        try:
            mat = compute_variant_func(chunk, variant, lag=int(max(1, lag)), pairs=pairs)
            score = lag_quality(variant, mat, is_pvalue)
            return (
                int(start),
                int(end),
                float(score) if np.isfinite(score) else float("nan"),
                (mat if return_matrices else None),
            )
        except Exception as ex:
            logging.error("[SlidingWindow] %s win=%d start=%d: %s", variant, w, start, ex)
            return int(start), int(end), float("nan"), None

    if nj == 1 or len(starts) <= 1:
        computed = [_compute_one(st) for st in starts]
    else:
        try:
            from joblib import Parallel, delayed

            backend = str(parallel_backend or "loky")
            computed = Parallel(n_jobs=nj, backend=backend)(
                delayed(_compute_one)(st) for st in starts
            )
        except ImportError:
            computed = [_compute_one(st) for st in starts]

    xs: list[int] = []
    ys: list[float] = []
    ticks: list[dict] = []
    for start, end, score_f, mat in computed:
        xs.append(int(start))
        ys.append(float(score_f) if np.isfinite(score_f) else float("nan"))
        ticks.append(
            {
                "start": int(start),
                "end": int(end),
                "metric": float(score_f) if np.isfinite(score_f) else float("nan"),
                "matrix": mat if return_matrices else None,
            }
        )
        if np.isfinite(score_f) and float(score_f) > float(best["metric"]):
            best = {"start": int(start), "end": int(end), "metric": float(score_f), "matrix": mat}

    return {
        "best_window": best,
        "curve": {"x": xs, "y": ys},
        "ticks": ticks,
        "extremes": select_best_median_worst(ticks, key="metric"),
    }
