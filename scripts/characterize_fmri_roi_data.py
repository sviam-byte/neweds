"""Characterize already-extracted fMRI ROI time-series data.

This script is intentionally focused on Stage 1 signal/data characterization,
before broad connectivity metric scans. It reads HC/SZ ROI CSV files, normalizes
valid matrices to time x ROI, computes value QC, distribution, temporal, and
spectral summaries, then writes CSV/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neweds.core.fmri_roi_audit import (
    build_common_bad_rois,
    build_roi_qc,
    build_temporal_qc,
    load_valid_subjects,
    scan_inventory,
)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _quantile(values: np.ndarray, q: float) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    return _safe(np.nanquantile(finite, q))


def _mad(values: np.ndarray) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    med = np.nanmedian(finite)
    return _safe(np.nanmedian(np.abs(finite - med)))


def _distribution_rows(subjects) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for idx, col in enumerate(subject.data_time_roi.columns):
            values = subject.data_time_roi[col].to_numpy(dtype=float)
            finite = _finite(values)
            if finite.size:
                mean = _safe(np.nanmean(finite))
                std = _safe(np.nanstd(finite))
                skew = _safe(stats.skew(finite, bias=False)) if finite.size >= 3 else float("nan")
                kurt = (
                    _safe(stats.kurtosis(finite, fisher=True, bias=False))
                    if finite.size >= 4
                    else float("nan")
                )
                if np.isfinite(std) and std > 1e-12:
                    z = np.abs((finite - mean) / std)
                    outlier_fraction_z3 = _safe(np.mean(z > 3.0))
                else:
                    outlier_fraction_z3 = float("nan")
            else:
                mean = std = skew = kurt = outlier_fraction_z3 = float("nan")
            rows.append(
                {
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "atlas": subject.atlas,
                    "roi_index_0based": idx,
                    "roi_index_1based": idx + 1,
                    "roi_id": str(col),
                    "n_timepoints": int(values.size),
                    "n_finite": int(finite.size),
                    "mean": mean,
                    "std": std,
                    "variance": _safe(np.nanvar(finite)) if finite.size else float("nan"),
                    "skewness": skew,
                    "kurtosis_excess": kurt,
                    "min": _safe(np.nanmin(finite)) if finite.size else float("nan"),
                    "q01": _quantile(values, 0.01),
                    "q05": _quantile(values, 0.05),
                    "q25": _quantile(values, 0.25),
                    "q50": _quantile(values, 0.50),
                    "q75": _quantile(values, 0.75),
                    "q95": _quantile(values, 0.95),
                    "q99": _quantile(values, 0.99),
                    "max": _safe(np.nanmax(finite)) if finite.size else float("nan"),
                    "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
                    "mad": _mad(values),
                    "outlier_fraction_z3": outlier_fraction_z3,
                    "missing_fraction": _safe(np.mean(~np.isfinite(values))) if values.size else float("nan"),
                    "zero_fraction": _safe(np.mean(finite == 0.0)) if finite.size else float("nan"),
                }
            )
    return rows


def build_distribution_tables(subjects) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long = pd.DataFrame(_distribution_rows(subjects))
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame()
    by_roi = (
        long.groupby(["atlas", "group", "roi_index_0based", "roi_index_1based"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_mean=("mean", "median"),
            median_std=("std", "median"),
            median_variance=("variance", "median"),
            median_skewness=("skewness", "median"),
            median_kurtosis_excess=("kurtosis_excess", "median"),
            median_outlier_fraction_z3=("outlier_fraction_z3", "median"),
            median_missing_fraction=("missing_fraction", "median"),
            median_zero_fraction=("zero_fraction", "median"),
        )
        .reset_index(drop=True)
    )
    by_subject = (
        long.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            n_roi_distribution=("roi_index_0based", "nunique"),
            median_mean=("mean", "median"),
            median_std=("std", "median"),
            median_variance=("variance", "median"),
            median_abs_skewness=("skewness", lambda s: _safe(np.nanmedian(np.abs(s)))),
            median_outlier_fraction_z3=("outlier_fraction_z3", "median"),
            median_missing_fraction=("missing_fraction", "median"),
            median_zero_fraction=("zero_fraction", "median"),
        )
        .reset_index(drop=True)
    )
    return long, by_roi, by_subject


def _bandpower(freq: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq < hi) & np.isfinite(power)
    if not np.any(mask):
        return float("nan")
    return _safe(np.sum(power[mask]))


def _spectral_features(values: np.ndarray) -> dict[str, float]:
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) < 8:
        return {
            "total_power": float("nan"),
            "very_low_power": float("nan"),
            "low_power": float("nan"),
            "mid_power": float("nan"),
            "high_power": float("nan"),
            "low_high_power_ratio": float("nan"),
            "dominant_frequency": float("nan"),
            "spectral_centroid": float("nan"),
            "spectral_entropy": float("nan"),
            "spectral_slope": float("nan"),
        }
    x = np.asarray(values, dtype=float)
    if not np.all(finite_mask):
        s = pd.Series(x)
        x = s.interpolate(limit_direction="both").fillna(float(np.nanmean(x))).to_numpy(dtype=float)
    x = x - float(np.nanmean(x))
    if float(np.nanstd(x)) <= 1e-12:
        return {
            "total_power": 0.0,
            "very_low_power": 0.0,
            "low_power": 0.0,
            "mid_power": 0.0,
            "high_power": 0.0,
            "low_high_power_ratio": float("nan"),
            "dominant_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "spectral_slope": float("nan"),
        }
    freq = np.fft.rfftfreq(x.size, d=1.0)
    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2) / max(1, x.size)
    if power.size > 0:
        power[0] = 0.0
    total = _safe(np.sum(power))
    very_low = _bandpower(freq, power, 0.0, 0.01)
    low = _bandpower(freq, power, 0.01, 0.05)
    mid = _bandpower(freq, power, 0.05, 0.15)
    high = _bandpower(freq, power, 0.15, 0.50 + 1e-12)
    nonzero = power > 0
    if np.any(nonzero):
        dom_idx = int(np.nanargmax(power))
        dominant = _safe(freq[dom_idx])
        centroid = _safe(np.sum(freq * power) / (np.sum(power) + 1e-12))
        prob = power[nonzero] / (np.sum(power[nonzero]) + 1e-12)
        entropy = _safe(-np.sum(prob * np.log2(prob + 1e-12)) / max(1.0, math.log2(prob.size)))
    else:
        dominant = centroid = entropy = float("nan")
    slope_mask = (freq > 0.0) & (power > 0.0)
    if int(np.sum(slope_mask)) >= 4:
        slope = _safe(np.polyfit(np.log(freq[slope_mask]), np.log(power[slope_mask]), 1)[0])
    else:
        slope = float("nan")
    return {
        "total_power": total,
        "very_low_power": very_low,
        "low_power": low,
        "mid_power": mid,
        "high_power": high,
        "low_high_power_ratio": _safe((very_low + low) / (high + 1e-12)),
        "dominant_frequency": dominant,
        "spectral_centroid": centroid,
        "spectral_entropy": entropy,
        "spectral_slope": slope,
    }


def build_spectral_tables(subjects) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for idx, col in enumerate(subject.data_time_roi.columns):
            values = subject.data_time_roi[col].to_numpy(dtype=float)
            rows.append(
                {
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "atlas": subject.atlas,
                    "roi_index_0based": idx,
                    "roi_index_1based": idx + 1,
                    "roi_id": str(col),
                    **_spectral_features(values),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame()
    by_roi = (
        long.groupby(["atlas", "group", "roi_index_0based", "roi_index_1based"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_total_power=("total_power", "median"),
            median_very_low_power=("very_low_power", "median"),
            median_low_power=("low_power", "median"),
            median_mid_power=("mid_power", "median"),
            median_high_power=("high_power", "median"),
            median_low_high_power_ratio=("low_high_power_ratio", "median"),
            median_dominant_frequency=("dominant_frequency", "median"),
            median_spectral_centroid=("spectral_centroid", "median"),
            median_spectral_entropy=("spectral_entropy", "median"),
            median_spectral_slope=("spectral_slope", "median"),
        )
        .reset_index(drop=True)
    )
    by_subject = (
        long.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            n_roi_spectral=("roi_index_0based", "nunique"),
            median_total_power=("total_power", "median"),
            median_low_high_power_ratio=("low_high_power_ratio", "median"),
            median_dominant_frequency=("dominant_frequency", "median"),
            median_spectral_centroid=("spectral_centroid", "median"),
            median_spectral_entropy=("spectral_entropy", "median"),
            median_spectral_slope=("spectral_slope", "median"),
        )
        .reset_index(drop=True)
    )
    return long, by_roi, by_subject


def build_subject_qc_summary(
    roi_qc: pd.DataFrame,
    distribution_by_subject: pd.DataFrame,
    spectral_by_subject: pd.DataFrame,
    temporal_long: pd.DataFrame,
) -> pd.DataFrame:
    if roi_qc.empty:
        return pd.DataFrame()
    base = (
        roi_qc.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            n_roi=("roi_index_0based", "nunique"),
            n_zero_roi=("zero_flag", "sum"),
            n_constant_roi=("constant_flag", "sum"),
            n_nan_roi=("nan_flag", "sum"),
            n_short_series_roi=("short_series_flag", "sum"),
            median_roi_std=("std", "median"),
            median_abs_trend_slope=("linear_trend_slope", lambda s: _safe(np.nanmedian(np.abs(s)))),
            median_zero_fraction=("fraction_zero", "median"),
            median_nan_fraction=("fraction_nan", "median"),
            n_extreme_amplitude_roi=("extreme_amplitude_flag", "sum"),
        )
        .reset_index(drop=True)
    )
    temporal_subject = (
        temporal_long.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            median_ar1=("ar1_coeff", "median"),
            median_acf_lag1=("acf_lag_1", "median"),
            median_acf_lag5=("acf_lag_5", "median"),
            median_temporal_trend_r2=("trend_r2", "median"),
        )
        .reset_index(drop=True)
        if not temporal_long.empty
        else pd.DataFrame()
    )
    out = base
    for table in [distribution_by_subject, spectral_by_subject, temporal_subject]:
        if not table.empty:
            out = out.merge(table, on=["atlas", "group", "subject_id"], how="left", suffixes=("", "_extra"))
    warning_cols = [
        "n_zero_roi",
        "n_constant_roi",
        "n_nan_roi",
        "n_short_series_roi",
        "n_extreme_amplitude_roi",
    ]
    out["warning_count"] = out[warning_cols].sum(axis=1)
    return out


def build_feature_table(
    roi_qc: pd.DataFrame,
    distribution_long: pd.DataFrame,
    temporal_long: pd.DataFrame,
    spectral_long: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["atlas", "group", "subject_id", "roi_index_0based", "roi_index_1based"]
    out = roi_qc.copy()
    keep_dist = [
        *keys,
        "skewness",
        "kurtosis_excess",
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
        "iqr",
        "outlier_fraction_z3",
    ]
    if not distribution_long.empty:
        out = out.merge(distribution_long[keep_dist], on=keys, how="left")
    if not temporal_long.empty:
        out = out.merge(temporal_long, on=keys, how="left", suffixes=("", "_temporal"))
    if not spectral_long.empty:
        out = out.merge(spectral_long, on=keys, how="left", suffixes=("", "_spectral"))
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    cols = [str(c) for c in text_df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in text_df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in text_df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_summary_md(
    path: Path,
    *,
    hc_dir: Path,
    sz_dir: Path,
    output_dir: Path,
    inventory: pd.DataFrame,
    subjects,
    roi_qc: pd.DataFrame,
    subject_summary: pd.DataFrame,
    common_bad: dict[str, pd.DataFrame],
) -> None:
    ok = inventory[inventory["status"].eq("ok")]
    lines = [
        "# fMRI ROI Data Characterization Report",
        "",
        "This report summarizes Stage 1 signal/data characterization for already-extracted ROI time series.",
        "",
        "## Inputs",
        "",
        f"- HC directory: `{hc_dir}`",
        f"- SZ directory: `{sz_dir}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Inventory",
        "",
        f"- Files discovered: {len(inventory)}",
        f"- Valid subject files: {len(ok)}",
        f"- Loaded subject matrices: {len(subjects)}",
    ]
    if not inventory.empty:
        lines.extend(["", "### Files By Group/Atlas/Status", ""])
        counts = (
            inventory.groupby(["group", "atlas", "status"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        lines.append(_markdown_table(counts))
    if not roi_qc.empty:
        lines.extend(["", "## ROI QC", ""])
        lines.append(f"- ROI feature rows: {len(roi_qc)}")
        lines.append(f"- Zero ROI observations: {int(roi_qc['zero_flag'].sum())}")
        lines.append(f"- Constant ROI observations: {int(roi_qc['constant_flag'].sum())}")
        lines.append(f"- NaN-containing ROI observations: {int(roi_qc['nan_flag'].sum())}")
    if common_bad:
        lines.extend(["", "## Common Bad ROI", ""])
        for atlas, table in common_bad.items():
            lines.append(f"- {atlas}: {len(table)} conservative common bad ROI")
    if not subject_summary.empty:
        lines.extend(["", "## Subject QC Summary", ""])
        compact = subject_summary.sort_values("warning_count", ascending=False).head(20)
        lines.append(_markdown_table(compact))
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- These are descriptive signal-quality and temporal-structure outputs.",
            "- They do not validate atlas overlay quality or voxel-wise ROI homogeneity.",
            "- They do not establish clinical or diagnostic biomarkers.",
            "- Use these outputs to decide what is safe to pass into Stage 2 connectivity scans.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_characterization(
    hc_dir: Path,
    sz_dir: Path,
    output_dir: Path,
    *,
    atlas: str,
) -> None:
    inv_dir = _mkdir(output_dir / "inventories")
    qc_dir = _mkdir(output_dir / "qc")
    dist_dir = _mkdir(output_dir / "distributions")
    temporal_dir = _mkdir(output_dir / "temporal")
    spectral_dir = _mkdir(output_dir / "spectral")
    reports_dir = _mkdir(output_dir / "reports")

    inventory = scan_inventory(hc_dir, sz_dir, atlas_filter=atlas)
    _write_csv(inventory, inv_dir / "data_inventory.csv")
    subjects = load_valid_subjects(inventory)

    roi_qc = build_roi_qc(subjects)
    _write_csv(roi_qc, qc_dir / "roi_qc_long.csv")
    if not roi_qc.empty:
        by_region = (
            roi_qc.groupby(["atlas", "group", "roi_index_0based", "roi_index_1based"], as_index=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_zero=("zero_flag", "sum"),
                n_constant=("constant_flag", "sum"),
                n_nan=("nan_flag", "sum"),
                median_std=("std", "median"),
                median_variance=("variance", "median"),
                median_abs_trend_slope=("linear_trend_slope", lambda s: _safe(np.nanmedian(np.abs(s)))),
            )
            .reset_index(drop=True)
        )
        _write_csv(by_region, qc_dir / "roi_qc_by_region.csv")

    common_bad: dict[str, pd.DataFrame] = {}
    for atlas_name in sorted(set(roi_qc["atlas"].dropna())) if not roi_qc.empty else []:
        table = build_common_bad_rois(roi_qc, atlas=atlas_name)
        common_bad[atlas_name] = table
        _write_csv(table, qc_dir / f"common_bad_rois_{atlas_name}.csv")

    distribution_long, distribution_by_roi, distribution_by_subject = build_distribution_tables(subjects)
    _write_csv(distribution_long, dist_dir / "signal_distribution_long.csv")
    _write_csv(distribution_by_roi, dist_dir / "signal_distribution_by_roi.csv")
    _write_csv(distribution_by_subject, dist_dir / "signal_distribution_by_subject.csv")

    temporal_long, temporal_group_summary = build_temporal_qc(subjects)
    _write_csv(temporal_long, temporal_dir / "temporal_qc_long.csv")
    _write_csv(temporal_group_summary, temporal_dir / "temporal_qc_group_summary.csv")

    spectral_long, spectral_by_roi, spectral_by_subject = build_spectral_tables(subjects)
    _write_csv(spectral_long, spectral_dir / "spectral_qc_long.csv")
    _write_csv(spectral_by_roi, spectral_dir / "spectral_qc_by_roi.csv")
    _write_csv(spectral_by_subject, spectral_dir / "spectral_qc_by_subject.csv")

    subject_summary = build_subject_qc_summary(
        roi_qc,
        distribution_by_subject,
        spectral_by_subject,
        temporal_long,
    )
    _write_csv(subject_summary, qc_dir / "subject_qc_summary.csv")

    feature_table = build_feature_table(roi_qc, distribution_long, temporal_long, spectral_long)
    _write_csv(feature_table, output_dir / "roi_signal_characterization_all_features.csv")

    _write_summary_md(
        reports_dir / "data_characterization_report.md",
        hc_dir=hc_dir,
        sz_dir=sz_dir,
        output_dir=output_dir,
        inventory=inventory,
        subjects=subjects,
        roi_qc=roi_qc,
        subject_summary=subject_summary,
        common_bad=common_bad,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas", default="all", choices=["all", "AAL3", "HCP"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_characterization(
        Path(args.hc_dir),
        Path(args.sz_dir),
        Path(args.output_dir),
        atlas=str(args.atlas),
    )


if __name__ == "__main__":
    main()
