"""ROI/bin signal-quality diagnostics for time-by-series matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: float | np.floating | None) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(mask)) < 3:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if float(np.nanstd(aa)) <= eps or float(np.nanstd(bb)) <= eps:
        return float("nan")
    return _safe_float(np.corrcoef(aa, bb)[0, 1])


def _risk_label(
    *,
    n_active: int,
    n_pairs: int,
    median_pairwise_corr: float,
    median_abs_pairwise_corr: float,
    frac_negative_corr: float,
    frac_abs_corr_lt_0_2: float,
    mean_signal_validity: float,
) -> str:
    if n_active < 2 or n_pairs < 1:
        return "bad"

    med = _safe_float(median_pairwise_corr)
    med_abs = _safe_float(median_abs_pairwise_corr)
    neg = _safe_float(frac_negative_corr)
    weak = _safe_float(frac_abs_corr_lt_0_2)
    mean_valid = _safe_float(mean_signal_validity)

    bad_checks = (
        np.isfinite(neg) and neg >= 0.30,
        np.isfinite(med_abs) and med_abs < 0.20,
        np.isfinite(mean_valid) and mean_valid < 0.20,
        np.isfinite(med) and med < 0.0,
    )
    if any(bad_checks):
        return "bad"

    ok_checks = (
        np.isfinite(med) and med >= 0.50,
        np.isfinite(neg) and neg <= 0.05,
        np.isfinite(weak) and weak <= 0.20,
        np.isfinite(mean_valid) and mean_valid >= 0.50,
    )
    if all(ok_checks):
        return "ok"
    return "weak"


def within_region_homogeneity(
    block: pd.DataFrame,
    *,
    weak_abs_corr_threshold: float = 0.2,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Summarize whether a region/bin can be safely aggregated.

    Parameters
    ----------
    block:
        DataFrame with rows as time points and columns as voxels/channels inside
        one ROI or spatial bin.
    weak_abs_corr_threshold:
        Threshold used for ``frac_abs_corr_lt_0_2``. The output key keeps the
        explicit 0.2 name for a stable first QC contract.
    eps:
        Variance floor below which a series is treated as constant.
    """
    if block is None:
        block = pd.DataFrame()
    data = block.apply(pd.to_numeric, errors="coerce") if not block.empty else pd.DataFrame()
    n_series = int(data.shape[1])

    if n_series == 0:
        return {
            "n_series": 0,
            "n_active": 0,
            "n_pairs": 0,
            "median_pairwise_corr": float("nan"),
            "median_abs_pairwise_corr": float("nan"),
            "frac_negative_corr": float("nan"),
            "frac_abs_corr_lt_0_2": float("nan"),
            "mean_signal_validity": float("nan"),
            "median_mean_signal_corr": float("nan"),
            "frac_negative_mean_signal_corr": float("nan"),
            "aggregation_risk": "bad",
        }

    finite_counts = data.notna().sum(axis=0)
    variances = data.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
    active_cols = [
        c for c in data.columns if int(finite_counts[c]) >= 3 and _safe_float(variances[c]) > eps
    ]
    active = data[active_cols].copy()
    n_active = int(active.shape[1])

    pairwise_vals: list[float] = []
    if n_active >= 2:
        corr = active.corr(method="pearson", min_periods=3).to_numpy(dtype=float)
        tri = np.triu_indices(n_active, k=1)
        pairwise_vals = [float(v) for v in corr[tri] if np.isfinite(v)]

    n_pairs = int(len(pairwise_vals))
    if n_pairs:
        pairwise = np.asarray(pairwise_vals, dtype=float)
        median_pairwise_corr = _safe_float(np.nanmedian(pairwise))
        median_abs_pairwise_corr = _safe_float(np.nanmedian(np.abs(pairwise)))
        frac_negative_corr = _safe_float(np.mean(pairwise < 0.0))
        frac_abs_corr_lt_0_2 = _safe_float(np.mean(np.abs(pairwise) < float(weak_abs_corr_threshold)))
    else:
        median_pairwise_corr = float("nan")
        median_abs_pairwise_corr = float("nan")
        frac_negative_corr = float("nan")
        frac_abs_corr_lt_0_2 = float("nan")

    mean_corrs: list[float] = []
    if n_active >= 1:
        mean_signal = active.mean(axis=1, skipna=True).to_numpy(dtype=float)
        for col in active.columns:
            r = _safe_corr(mean_signal, active[col].to_numpy(dtype=float), eps=eps)
            if np.isfinite(r):
                mean_corrs.append(float(r))

    if mean_corrs:
        mean_arr = np.asarray(mean_corrs, dtype=float)
        mean_signal_validity = _safe_float(np.nanmedian(mean_arr))
        median_mean_signal_corr = mean_signal_validity
        frac_negative_mean_signal_corr = _safe_float(np.mean(mean_arr < 0.0))
    else:
        mean_signal_validity = float("nan")
        median_mean_signal_corr = float("nan")
        frac_negative_mean_signal_corr = float("nan")

    risk = _risk_label(
        n_active=n_active,
        n_pairs=n_pairs,
        median_pairwise_corr=median_pairwise_corr,
        median_abs_pairwise_corr=median_abs_pairwise_corr,
        frac_negative_corr=frac_negative_corr,
        frac_abs_corr_lt_0_2=frac_abs_corr_lt_0_2,
        mean_signal_validity=mean_signal_validity,
    )
    return {
        "n_series": n_series,
        "n_active": n_active,
        "n_pairs": n_pairs,
        "median_pairwise_corr": median_pairwise_corr,
        "median_abs_pairwise_corr": median_abs_pairwise_corr,
        "frac_negative_corr": frac_negative_corr,
        "frac_abs_corr_lt_0_2": frac_abs_corr_lt_0_2,
        "mean_signal_validity": mean_signal_validity,
        "median_mean_signal_corr": median_mean_signal_corr,
        "frac_negative_mean_signal_corr": frac_negative_mean_signal_corr,
        "aggregation_risk": risk,
    }


def summarize_region_qc(
    data: pd.DataFrame,
    regions: Mapping[str, Sequence[str]] | pd.Series | None = None,
    *,
    region_name: str = "all",
    **qc_kwargs: Any,
) -> pd.DataFrame:
    """Run homogeneity QC for one matrix or a set of named regions.

    ``regions`` can be either ``{"region": ["col_a", "col_b"]}`` or a Series
    mapping source column names to region labels. If omitted, the whole matrix is
    summarized as one region.
    """
    if data is None:
        data = pd.DataFrame()

    groups: dict[str, list[str]] = {}
    if regions is None:
        groups[str(region_name)] = [str(c) for c in data.columns]
    elif isinstance(regions, pd.Series):
        for source, region in regions.items():
            groups.setdefault(str(region), []).append(str(source))
    elif isinstance(regions, Mapping):
        groups = {str(k): [str(c) for c in v] for k, v in regions.items()}
    else:
        raise TypeError("regions must be None, a mapping, or a pandas Series")

    rows: list[dict[str, Any]] = []
    col_lookup = {str(c): c for c in data.columns}
    for name, cols in groups.items():
        existing = [col_lookup[c] for c in cols if c in col_lookup]
        qc = within_region_homogeneity(data.loc[:, existing], **qc_kwargs)
        row = {"region": name, **qc}
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["summarize_region_qc", "within_region_homogeneity"]
