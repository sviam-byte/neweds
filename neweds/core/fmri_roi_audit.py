"""Experimental audit pipeline for already-extracted fMRI ROI time series."""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neweds.analysis.spatial_adjacency import neighbor_offsets
from neweds.core.group_pipeline import _fdr_bh

logger = logging.getLogger(__name__)

FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE = (
    "[experimental] neweds-fmri-audit analyzes already-extracted ROI time series. "
    "It does not validate voxel-wise ROI homogeneity, atlas overlay quality, or clinical biomarkers."
)

AAL3_EXPECTED_ROI = 167
HCP_EXPECTED_ROI = {360, 361, 379}
SUPPORTED_ATLASES = {"AAL3", "HCP"}
KNOWN_ZERO_AAL3_INDICES_0BASED = {34, 35, 80, 81}
SUBJECT_LEVEL_DENSITY_THRESHOLDS = (0.2, 0.4, 0.6)
HCP_SMALL_REGION_VOXEL_THRESHOLD = 10
DEFAULT_BAD_ROI_THRESHOLDS = (0.05, 0.10, 0.20)


@dataclass(frozen=True, slots=True)
class FmriRoiAuditResult:
    """Structured result for the experimental fMRI ROI audit pipeline."""

    n_hc: int
    n_sz: int
    atlases: list[str]
    branches: list[str]
    n_subjects_by_atlas: dict[str, int]
    n_bad_rois_by_atlas: dict[str, int]
    n_edges_by_atlas_branch: dict[str, int]
    n_significant_by_atlas_branch: dict[str, int]
    output_dir: str
    warnings: list[str] = field(default_factory=list)
    experimental: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_hc": self.n_hc,
            "n_sz": self.n_sz,
            "atlases": list(self.atlases),
            "branches": list(self.branches),
            "n_subjects_by_atlas": dict(self.n_subjects_by_atlas),
            "n_bad_rois_by_atlas": dict(self.n_bad_rois_by_atlas),
            "n_edges_by_atlas_branch": dict(self.n_edges_by_atlas_branch),
            "n_significant_by_atlas_branch": dict(self.n_significant_by_atlas_branch),
            "output_dir": self.output_dir,
            "warnings": list(self.warnings),
            "experimental": self.experimental,
        }


@dataclass(frozen=True, slots=True)
class RoiSubject:
    group: str
    subject_id: str
    atlas: str
    path: Path
    data_time_roi: pd.DataFrame
    orientation: str
    n_rows: int
    n_cols: int


def detect_atlas(filename: str) -> str:
    """Infer atlas family from a subject filename."""
    name = filename.upper()
    if "AAL3" in name or "AAL_3" in name:
        return "AAL3"
    if "HCP" in name or "GLASSER" in name or "MMP" in name:
        return "HCP"
    return "unknown"


def parse_subject_id(filename: str, atlas: str | None = None) -> str:
    """Remove common atlas/time-series suffixes from a subject file stem."""
    stem = Path(filename).stem
    text = stem
    patterns = [
        r"[_-]?AAL3(?:\.2)?[_-]?timeseries$",
        r"[_-]?AAL[_-]?3(?:\.2)?[_-]?timeseries$",
        r"[_-]?HCP[_-]?timeseries$",
        r"[_-]?GLASSER[_-]?timeseries$",
        r"[_-]?MMP1?[_-]?timeseries$",
        r"[_-]?timeseries$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    if atlas and atlas != "unknown":
        text = re.sub(rf"[_-]?{re.escape(atlas)}$", "", text, flags=re.IGNORECASE)
    return text.strip("_- ") or stem


def _expected_roi_counts(atlas: str) -> set[int]:
    if atlas == "AAL3":
        return {AAL3_EXPECTED_ROI}
    if atlas == "HCP":
        return set(HCP_EXPECTED_ROI)
    return set()


def detect_orientation(n_rows: int, n_cols: int, atlas: str) -> tuple[str, int | None, int | None]:
    """Return orientation plus normalized n_regions/n_timepoints when possible."""
    expected = _expected_roi_counts(atlas)
    if not expected:
        return "unknown", None, None
    row_match = n_rows in expected
    col_match = n_cols in expected
    if row_match and not col_match:
        return "ROI_by_time", n_rows, n_cols
    if col_match and not row_match:
        return "time_by_ROI", n_cols, n_rows
    if row_match and col_match:
        return "ambiguous", n_rows, n_cols
    return "shape_error", None, None


def _is_integer_index_row(row: "pd.Series[float]") -> bool:
    """Return True if the row looks like a 0-based integer index (0, 1, 2, …, N-1)."""
    vals = row.to_numpy(dtype=float)
    n = len(vals)
    if n < 2:
        return False
    return bool(np.allclose(vals, np.arange(n, dtype=float)))


def _read_numeric_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError(f"empty or non-numeric CSV: {path}")
    # Drop a leading integer-index row that some exporters add (e.g. 0, 1, 2, …, N-1).
    if len(numeric) > 1 and _is_integer_index_row(numeric.iloc[0]):
        logger.debug("Dropping integer-index header row in %s", path.name)
        numeric = numeric.iloc[1:].reset_index(drop=True)
    return numeric.reset_index(drop=True)


def _to_time_roi(df: pd.DataFrame, orientation: str) -> pd.DataFrame:
    if orientation == "ROI_by_time":
        out = df.T
    elif orientation in {"time_by_ROI", "ambiguous"}:
        out = df
    else:
        raise ValueError(f"cannot normalize orientation={orientation}")
    out = out.copy()
    out.columns = [f"roi_{i:03d}" for i in range(out.shape[1])]
    out.index = pd.RangeIndex(out.shape[0], name="time")
    return out


def _inventory_row(path: Path, group: str, atlas_filter: str) -> dict[str, Any]:
    atlas = detect_atlas(path.name)
    subject_id = parse_subject_id(path.name, atlas)
    if atlas_filter != "all" and atlas != atlas_filter:
        return {
            "file_path": str(path),
            "file_name": path.name,
            "group": group,
            "atlas": atlas,
            "subject_id": subject_id,
            "status": "filtered_atlas",
        }

    try:
        df = _read_numeric_csv(path)
        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        orientation, n_regions, n_timepoints = detect_orientation(n_rows, n_cols, atlas)
        arr = df.to_numpy(dtype=float, copy=False)
        if orientation in {"ROI_by_time", "ambiguous"}:
            roi_axis_arr = arr
        elif orientation == "time_by_ROI":
            roi_axis_arr = arr.T
        else:
            roi_axis_arr = np.empty((0, 0), dtype=float)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        finite_rows = np.isfinite(arr).all(axis=1)
        finite_cols = np.isfinite(arr).all(axis=0)
        n_zero_rows = int(np.sum(finite_rows & np.all(arr == 0.0, axis=1)))
        n_zero_cols = int(np.sum(finite_cols & np.all(arr == 0.0, axis=0)))
        if roi_axis_arr.size:
            constant = [
                bool(np.nanstd(row[np.isfinite(row)]) < 1e-12)
                if np.isfinite(row).any()
                else True
                for row in roi_axis_arr
            ]
            n_constant_regions = int(sum(constant))
        else:
            n_constant_regions = 0
        status = "ok" if orientation in {"ROI_by_time", "time_by_ROI", "ambiguous"} else orientation
        return {
            "file_path": str(path),
            "file_name": path.name,
            "group": group,
            "atlas": atlas,
            "subject_id": subject_id,
            "shape_raw": f"{n_rows}x{n_cols}",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "orientation": orientation,
            "n_regions": n_regions,
            "n_timepoints": n_timepoints,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "n_zero_rows": n_zero_rows,
            "n_zero_cols": n_zero_cols,
            "n_constant_regions": n_constant_regions,
            "status": status,
        }
    except Exception as exc:
        return {
            "file_path": str(path),
            "file_name": path.name,
            "group": group,
            "atlas": atlas,
            "subject_id": subject_id,
            "status": "load_error",
            "error": str(exc),
        }


def scan_inventory(hc_dir: str | Path, sz_dir: str | Path, *, atlas_filter: str = "all") -> pd.DataFrame:
    """Scan HC/SZ directories and summarize supported subject CSV files."""
    rows: list[dict[str, Any]] = []
    for group, directory in [("HC", hc_dir), ("SZ", sz_dir)]:
        root = Path(directory)
        files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        for path in files:
            rows.append(_inventory_row(path, group, atlas_filter))
    if not rows:
        raise FileNotFoundError("No CSV subject files found in HC/SZ directories")
    return pd.DataFrame(rows)


def load_valid_subjects(inventory: pd.DataFrame) -> list[RoiSubject]:
    subjects: list[RoiSubject] = []
    ok = inventory[inventory["status"].eq("ok")].copy()
    for _, row in ok.iterrows():
        path = Path(str(row["file_path"]))
        df = _read_numeric_csv(path)
        subjects.append(
            RoiSubject(
                group=str(row["group"]),
                subject_id=str(row["subject_id"]),
                atlas=str(row["atlas"]),
                path=path,
                data_time_roi=_to_time_roi(df, str(row["orientation"])),
                orientation=str(row["orientation"]),
                n_rows=int(row["n_rows"]),
                n_cols=int(row["n_cols"]),
            )
        )
    return subjects


def _mad(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    med = float(np.median(finite))
    return float(np.median(np.abs(finite - med)))


def _linear_trend(x: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    y = x[mask]
    t = np.arange(x.size, dtype=float)[mask]
    t = (t - t.mean()) / (t.std() + 1e-12)
    try:
        slope, intercept = np.polyfit(t, y, 1)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(slope), float(r2)


def build_roi_qc(subjects: list[RoiSubject], *, eps: float = 1e-12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    time_counts = pd.Series([s.data_time_roi.shape[0] for s in subjects])
    modal_time = int(time_counts.mode().iloc[0]) if not time_counts.empty else 0

    for subject in subjects:
        data = subject.data_time_roi
        for idx, col in enumerate(data.columns):
            values = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            std = float(np.nanstd(values)) if finite.size else float("nan")
            slope, r2 = _linear_trend(values)
            zero_flag = bool(finite.size > 0 and np.allclose(finite, 0.0))
            constant_flag = bool(finite.size == 0 or std < eps)
            rows.append(
                {
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "atlas": subject.atlas,
                    "roi_index_0based": idx,
                    "roi_index_1based": idx + 1,
                    "roi_id": col,
                    "mean": float(np.nanmean(values)) if finite.size else float("nan"),
                    "std": std,
                    "variance": float(np.nanvar(values)) if finite.size else float("nan"),
                    "min": float(np.nanmin(values)) if finite.size else float("nan"),
                    "max": float(np.nanmax(values)) if finite.size else float("nan"),
                    "median": float(np.nanmedian(values)) if finite.size else float("nan"),
                    "mad": _mad(values),
                    "fraction_zero": float(np.mean(finite == 0.0)) if finite.size else float("nan"),
                    "fraction_nan": float(np.isnan(values).mean()) if values.size else float("nan"),
                    "linear_trend_slope": slope,
                    "linear_trend_r2": r2,
                    "constant_flag": constant_flag,
                    "zero_flag": zero_flag,
                    "nan_flag": bool(np.isnan(values).any()),
                    "short_series_flag": bool(data.shape[0] != modal_time),
                }
            )
    qc = pd.DataFrame(rows)
    if qc.empty:
        return qc

    qc["amplitude"] = qc["max"] - qc["min"]
    qc["extreme_amplitude_flag"] = False
    for _atlas, idx in qc.groupby("atlas").groups.items():
        amp = qc.loc[idx, "amplitude"].to_numpy(dtype=float)
        med = np.nanmedian(amp)
        mad = np.nanmedian(np.abs(amp - med)) + 1e-12
        robust_z = np.abs((amp - med) / (1.4826 * mad))
        qc.loc[idx, "extreme_amplitude_flag"] = robust_z > 5.0
    return qc


def build_common_bad_rois(qc: pd.DataFrame, *, atlas: str) -> pd.DataFrame:
    subset = qc[qc["atlas"].eq(atlas)].copy()
    rows: list[dict[str, Any]] = []
    columns = [
        "atlas",
        "roi_index_0based",
        "roi_index_1based",
        "bad_subject_count",
        "n_subjects",
        "bad_frequency",
        "zero_subject_count",
        "constant_subject_count",
    ]
    if subset.empty:
        return pd.DataFrame(columns=columns)
    n_subjects = int(subset[["group", "subject_id"]].drop_duplicates().shape[0])
    grouped = subset.groupby(["roi_index_0based", "roi_index_1based"], sort=True)
    for (idx0, idx1), group in grouped:
        bad = group["zero_flag"].astype(bool) | group["constant_flag"].astype(bool)
        if bool(bad.any()):
            rows.append(
                {
                    "atlas": atlas,
                    "roi_index_0based": int(idx0),
                    "roi_index_1based": int(idx1),
                    "bad_subject_count": int(bad.sum()),
                    "n_subjects": n_subjects,
                    "bad_frequency": float(bad.sum() / max(1, n_subjects)),
                    "zero_subject_count": int(group["zero_flag"].astype(bool).sum()),
                    "constant_subject_count": int(group["constant_flag"].astype(bool).sum()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_threshold_bad_rois(
    qc: pd.DataFrame,
    *,
    atlas: str,
    threshold: float,
) -> pd.DataFrame:
    """Build bad ROI list using a minimum bad-subject frequency threshold."""
    baseline = build_common_bad_rois(qc, atlas=atlas)
    if baseline.empty:
        return baseline
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    return baseline[baseline["bad_frequency"] >= threshold].reset_index(drop=True)


def _acf_at_lag(x: np.ndarray, lag: int) -> float:
    finite = x[np.isfinite(x)]
    if finite.size <= lag + 2 or np.nanstd(finite) < 1e-12:
        return float("nan")
    a = finite[:-lag]
    b = finite[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pacf_lags(x: np.ndarray, lags: tuple[int, ...] = (1, 2, 3)) -> dict[int, float]:
    finite = x[np.isfinite(x)]
    out = {lag: float("nan") for lag in lags}
    if finite.size < max(lags) + 8 or np.nanstd(finite) < 1e-12:
        return out
    try:
        from statsmodels.tsa.stattools import pacf

        vals = pacf(finite, nlags=max(lags), method="yw")
        for lag in lags:
            out[lag] = float(vals[lag])
    except (ValueError, np.linalg.LinAlgError, ImportError):
        pass
    return out


def _ar2_coefficients(x: np.ndarray) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    if finite.size < 8 or np.nanstd(finite) < 1e-12:
        return float("nan"), float("nan")
    y = finite[2:]
    lag1 = finite[1:-1]
    lag2 = finite[:-2]
    design = np.column_stack([lag1, lag2])
    if y.size < 6 or np.linalg.matrix_rank(design) < 2:
        return float("nan"), float("nan")
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def build_temporal_qc(subjects: list[RoiSubject]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for idx, col in enumerate(subject.data_time_roi.columns):
            x = subject.data_time_roi[col].to_numpy(dtype=float)
            slope, r2 = _linear_trend(x)
            pacf_vals = _pacf_lags(x)
            ar2_1, ar2_2 = _ar2_coefficients(x)
            finite = x[np.isfinite(x)]
            half = finite.size // 2
            if half >= 3:
                first = finite[:half]
                second = finite[half:]
                mean_shift = float(np.nanmean(second) - np.nanmean(first))
                std_ratio = float(np.nanstd(second) / (np.nanstd(first) + 1e-12))
            else:
                mean_shift = float("nan")
                std_ratio = float("nan")
            rows.append(
                {
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "atlas": subject.atlas,
                    "roi_index_0based": idx,
                    "roi_index_1based": idx + 1,
                    "trend_slope": slope,
                    "trend_r2": r2,
                    "acf_lag_1": _acf_at_lag(x, 1),
                    "acf_lag_2": _acf_at_lag(x, 2),
                    "acf_lag_3": _acf_at_lag(x, 3),
                    "acf_lag_5": _acf_at_lag(x, 5),
                    "acf_lag_10": _acf_at_lag(x, 10),
                    "pacf_lag_1": pacf_vals[1],
                    "pacf_lag_2": pacf_vals[2],
                    "pacf_lag_3": pacf_vals[3],
                    "ar1_coeff": _acf_at_lag(x, 1),
                    "ar2_coeff_1": ar2_1,
                    "ar2_coeff_2": ar2_2,
                    "mean_shift_second_minus_first": mean_shift,
                    "std_ratio_second_to_first": std_ratio,
                    "stationarity_proxy": "trend_or_shift"
                    if (np.isfinite(slope) and abs(slope) > 0.5)
                    or (np.isfinite(mean_shift) and abs(mean_shift) > 0.5)
                    else "stable_proxy",
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()

    subject_summary = (
        long.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            median_ar1=("ar1_coeff", "median"),
            median_trend_magnitude=("trend_slope", lambda s: float(np.nanmedian(np.abs(s)))),
            median_acf_lag1=("acf_lag_1", "median"),
            fraction_high_autocorrelation_roi=("acf_lag_1", lambda s: float(np.nanmean(s > 0.5))),
        )
        .reset_index(drop=True)
    )
    group_summary = (
        subject_summary.groupby(["atlas", "group"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_ar1=("median_ar1", "median"),
            median_trend_magnitude=("median_trend_magnitude", "median"),
            median_acf_lag1=("median_acf_lag1", "median"),
            fraction_high_autocorrelation_roi=("fraction_high_autocorrelation_roi", "median"),
        )
        .reset_index(drop=True)
    )
    return long, group_summary


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    return (out - out.mean(axis=0)) / (out.std(axis=0, ddof=0).replace(0.0, np.nan) + 1e-12)


def _detrend_df(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)  # shape: (n_time, n_roi)
    t = np.arange(arr.shape[0], dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    out = arr.copy()
    for j in range(arr.shape[1]):
        x = arr[:, j]
        mask = np.isfinite(x)
        if int(mask.sum()) >= 3:
            slope, intercept = np.polyfit(t[mask], x[mask], 1)
            out[mask, j] = x[mask] - (slope * t[mask] + intercept)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _ar1_residualize_df(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)  # shape: (n_time, n_roi)
    n_time, n_roi = arr.shape
    out = np.full_like(arr, np.nan, dtype=float)
    for j in range(n_roi):
        x = arr[:, j]
        mask = np.isfinite(x[:-1]) & np.isfinite(x[1:])
        if int(mask.sum()) >= 8:
            a = x[:-1][mask]
            b = x[1:][mask]
            denom = float(np.dot(a, a))
            phi = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
            target_idx = np.arange(1, n_time)[mask]
            out[target_idx, j] = b - phi * a
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def preprocess_subject(subject: RoiSubject, bad_roi_indices: set[int], branch: str) -> pd.DataFrame:
    keep_cols = [
        col for idx, col in enumerate(subject.data_time_roi.columns) if idx not in bad_roi_indices
    ]
    data = subject.data_time_roi.loc[:, keep_cols].copy()
    if branch == "raw_cleaned":
        return _zscore(data)
    if branch == "detrended":
        return _zscore(_detrend_df(data))
    if branch == "ar1_residualized":
        return _zscore(_ar1_residualize_df(_detrend_df(data)))
    if branch == "roi_level_gsr":
        z = _zscore(data)
        global_signal = z.mean(axis=1, skipna=True).to_numpy(dtype=float)
        z_arr = z.to_numpy(dtype=float)  # (n_time, n_roi)
        out_arr = z_arr.copy()
        gs_finite = np.isfinite(global_signal)
        for j in range(z_arr.shape[1]):
            x = z_arr[:, j]
            mask = np.isfinite(x) & gs_finite
            if int(mask.sum()) >= 3:
                A = np.c_[np.ones(int(mask.sum())), global_signal[mask]]
                beta, *_ = np.linalg.lstsq(A, x[mask], rcond=None)
                out_arr[mask, j] = x[mask] - A @ beta
        out_df = pd.DataFrame(out_arr, index=z.index, columns=z.columns)
        return _zscore(out_df)
    raise ValueError(f"unknown preprocessing branch: {branch}")


def _subject_output_stem(subject: RoiSubject) -> str:
    stem = f"{subject.group}_{subject.subject_id}"
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    return stem.strip(" ._") or f"{subject.group}_subject"


def pearson_fisher_fc(df_time_roi: pd.DataFrame) -> np.ndarray:
    corr = df_time_roi.corr(method="pearson", min_periods=3).to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(corr, -0.999999, 0.999999, out=corr)
    z = np.arctanh(corr)
    np.fill_diagonal(z, 0.0)
    return z


def _edge_table_from_subjects(
    matrices: dict[str, np.ndarray],
    subjects: dict[str, RoiSubject],
    retained_indices: list[int],
) -> pd.DataFrame:
    edge_i, edge_j = np.triu_indices(len(retained_indices), k=1)
    rows: list[dict[str, Any]] = []
    for sid, matrix in matrices.items():
        subj = subjects[sid]
        vals = matrix[edge_i, edge_j]
        for k, value in enumerate(vals):
            i0 = retained_indices[int(edge_i[k])]
            j0 = retained_indices[int(edge_j[k])]
            rows.append(
                {
                    "group": subj.group,
                    "subject_id": subj.subject_id,
                    "edge_i": int(i0),
                    "edge_j": int(j0),
                    "roi_i_1based": int(i0 + 1),
                    "roi_j_1based": int(j0 + 1),
                    "fisher_z": float(value),
                }
            )
    return pd.DataFrame(rows)


def _write_fc_branch_outputs(
    *,
    atlas_subjects: list[RoiSubject],
    subject_by_key: dict[str, RoiSubject],
    atlas: str,
    branch: str,
    bad_indices: set[int],
    retained_indices: list[int],
    alpha: float,
    preprocessed_out: Path,
    matrix_out: Path,
    edge_out: Path,
    comparison_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrices: dict[str, np.ndarray] = {}
    preprocessed_out.mkdir(parents=True, exist_ok=True)
    matrix_out.mkdir(parents=True, exist_ok=True)
    edge_out.mkdir(parents=True, exist_ok=True)
    comparison_out.mkdir(parents=True, exist_ok=True)

    for subject in atlas_subjects:
        key = f"{subject.group}::{subject.subject_id}::{subject.atlas}"
        stem = _subject_output_stem(subject)
        pre = preprocess_subject(subject, bad_indices, branch)
        np.save(preprocessed_out / f"{stem}.npy", pre.to_numpy(dtype=float))
        matrix = pearson_fisher_fc(pre)
        matrices[key] = matrix
        np.save(matrix_out / f"{stem}_pearson_z.npy", matrix)

    edges = _edge_table_from_subjects(matrices, subject_by_key, retained_indices)
    edges.to_csv(edge_out / "fc_edges_long.csv", index=False)

    comparison = compare_fc_edges(edges, alpha=alpha)
    comparison.to_csv(comparison_out / "fc_group_comparison_edges.csv", index=False)

    subject_summary = build_subject_level_fc_summary(
        matrices,
        subject_by_key,
        atlas=atlas,
        branch=branch,
    )
    subject_comparison = compare_subject_level_fc(subject_summary, alpha=alpha)
    subject_summary.to_csv(comparison_out / "subject_level_fc_summary.csv", index=False)
    subject_comparison.to_csv(
        comparison_out / "subject_level_group_comparison.csv",
        index=False,
    )
    return edges, comparison, subject_summary, subject_comparison


def compare_fc_edges(edges: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (edge_i, edge_j), group in edges.groupby(["edge_i", "edge_j"], sort=True):
        hc = group[group["group"].eq("HC")]["fisher_z"].to_numpy(dtype=float)
        sz = group[group["group"].eq("SZ")]["fisher_z"].to_numpy(dtype=float)
        hc = hc[np.isfinite(hc)]
        sz = sz[np.isfinite(sz)]
        if hc.size == 0 or sz.size == 0:
            u_stat = float("nan")
            p_value = 1.0
            effect = float("nan")
        else:
            res = stats.mannwhitneyu(sz, hc, alternative="two-sided", method="auto")
            u_stat = float(res.statistic)
            p_value = float(res.pvalue)
            effect = float(2.0 * u_stat / (sz.size * hc.size) - 1.0)
        rows.append(
            {
                "edge_i": int(edge_i),
                "edge_j": int(edge_j),
                "roi_i_1based": int(edge_i + 1),
                "roi_j_1based": int(edge_j + 1),
                "n_HC": int(hc.size),
                "n_SZ": int(sz.size),
                "mean_HC": float(np.nanmean(hc)) if hc.size else float("nan"),
                "mean_SZ": float(np.nanmean(sz)) if sz.size else float("nan"),
                "median_HC": float(np.nanmedian(hc)) if hc.size else float("nan"),
                "median_SZ": float(np.nanmedian(sz)) if sz.size else float("nan"),
                "delta_mean": float(np.nanmean(sz) - np.nanmean(hc))
                if hc.size and sz.size
                else float("nan"),
                "u_stat": u_stat,
                "effect_size_rank_biserial_sz_gt_hc": effect,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    q, sig = _fdr_bh(out["p_value"].to_numpy(dtype=float), alpha=alpha)
    out["q_value_FDR"] = q
    out["significant"] = sig
    return out.sort_values(["q_value_FDR", "p_value"]).reset_index(drop=True)


def compare_fc_edges_ttest(edges: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    """Welch t-test sensitivity comparison for each FC edge."""
    rows: list[dict[str, Any]] = []
    for (edge_i, edge_j), group in edges.groupby(["edge_i", "edge_j"], sort=True):
        hc = group[group["group"].eq("HC")]["fisher_z"].to_numpy(dtype=float)
        sz = group[group["group"].eq("SZ")]["fisher_z"].to_numpy(dtype=float)
        hc = hc[np.isfinite(hc)]
        sz = sz[np.isfinite(sz)]
        if hc.size < 2 or sz.size < 2:
            t_stat = float("nan")
            p_value = 1.0
        else:
            res = stats.ttest_ind(sz, hc, equal_var=False, nan_policy="omit")
            t_stat = float(res.statistic) if np.isfinite(res.statistic) else float("nan")
            p_value = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
        rows.append(
            {
                "edge_i": int(edge_i),
                "edge_j": int(edge_j),
                "roi_i_1based": int(edge_i + 1),
                "roi_j_1based": int(edge_j + 1),
                "n_HC": int(hc.size),
                "n_SZ": int(sz.size),
                "mean_HC": float(np.nanmean(hc)) if hc.size else float("nan"),
                "mean_SZ": float(np.nanmean(sz)) if sz.size else float("nan"),
                "delta_mean": float(np.nanmean(sz) - np.nanmean(hc))
                if hc.size and sz.size
                else float("nan"),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    q, sig = _fdr_bh(out["p_value"].to_numpy(dtype=float), alpha=alpha)
    out["q_value_FDR"] = q
    out["significant"] = sig
    return out.sort_values(["q_value_FDR", "p_value"]).reset_index(drop=True)


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return np.asarray([], dtype=float)
    idx_i, idx_j = np.triu_indices(arr.shape[0], k=1)
    vals = arr[idx_i, idx_j]
    return vals[np.isfinite(vals)]


def summarize_subject_fc_matrix(
    matrix: np.ndarray,
    *,
    density_thresholds: tuple[float, ...] = SUBJECT_LEVEL_DENSITY_THRESHOLDS,
) -> dict[str, float | int]:
    """Compute subject-level FC summary metrics from one Fisher-z FC matrix."""
    vals = _upper_triangle_values(matrix)
    n_edges = int(vals.size)
    if n_edges == 0:
        out: dict[str, float | int] = {
            "n_edges": 0,
            "mean_fc": float("nan"),
            "mean_abs_fc": float("nan"),
            "fraction_positive_edges": float("nan"),
            "fraction_negative_edges": float("nan"),
            "fc_variance": float("nan"),
            "global_strength": float("nan"),
        }
    else:
        abs_vals = np.abs(vals)
        out = {
            "n_edges": n_edges,
            "mean_fc": float(np.mean(vals)),
            "mean_abs_fc": float(np.mean(abs_vals)),
            "fraction_positive_edges": float(np.mean(vals > 0.0)),
            "fraction_negative_edges": float(np.mean(vals < 0.0)),
            "fc_variance": float(np.var(vals)),
            "global_strength": float(np.sum(abs_vals)),
        }
    for threshold in density_thresholds:
        key = f"density_abs_z_ge_{str(threshold).replace('.', '_')}"
        out[key] = float(np.mean(np.abs(vals) >= float(threshold))) if n_edges else float("nan")
    return out


def build_subject_level_fc_summary(
    matrices: dict[str, np.ndarray],
    subjects: dict[str, RoiSubject],
    *,
    atlas: str,
    branch: str,
    density_thresholds: tuple[float, ...] = SUBJECT_LEVEL_DENSITY_THRESHOLDS,
) -> pd.DataFrame:
    """Build one row per subject with compact FC summary metrics."""
    rows: list[dict[str, Any]] = []
    for sid, matrix in matrices.items():
        subject = subjects[sid]
        rows.append(
            {
                "atlas": atlas,
                "branch": branch,
                "group": subject.group,
                "subject_id": subject.subject_id,
                **summarize_subject_fc_matrix(matrix, density_thresholds=density_thresholds),
            }
        )
    return pd.DataFrame(rows)


def compare_subject_level_fc(
    summary: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Compare HC vs SZ for each subject-level FC summary metric."""
    if summary.empty:
        return pd.DataFrame()
    metadata_cols = {"atlas", "branch", "group", "subject_id", "n_edges"}
    metric_cols = [
        col
        for col in summary.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(summary[col])
    ]
    rows: list[dict[str, Any]] = []
    for metric in metric_cols:
        hc = summary[summary["group"].eq("HC")][metric].to_numpy(dtype=float)
        sz = summary[summary["group"].eq("SZ")][metric].to_numpy(dtype=float)
        hc = hc[np.isfinite(hc)]
        sz = sz[np.isfinite(sz)]
        if hc.size == 0 or sz.size == 0:
            u_stat = float("nan")
            p_value = 1.0
            effect = float("nan")
        else:
            res = stats.mannwhitneyu(sz, hc, alternative="two-sided", method="auto")
            u_stat = float(res.statistic)
            p_value = float(res.pvalue)
            effect = float(2.0 * u_stat / (sz.size * hc.size) - 1.0)
        rows.append(
            {
                "atlas": str(summary["atlas"].iloc[0]) if "atlas" in summary else "",
                "branch": str(summary["branch"].iloc[0]) if "branch" in summary else "",
                "metric": metric,
                "n_HC": int(hc.size),
                "n_SZ": int(sz.size),
                "mean_HC": float(np.nanmean(hc)) if hc.size else float("nan"),
                "mean_SZ": float(np.nanmean(sz)) if sz.size else float("nan"),
                "median_HC": float(np.nanmedian(hc)) if hc.size else float("nan"),
                "median_SZ": float(np.nanmedian(sz)) if sz.size else float("nan"),
                "delta_mean": float(np.nanmean(sz) - np.nanmean(hc))
                if hc.size and sz.size
                else float("nan"),
                "u_stat": u_stat,
                "effect_size_rank_biserial_sz_gt_hc": effect,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    q, sig = _fdr_bh(out["p_value"].to_numpy(dtype=float), alpha=alpha)
    out["q_value_FDR"] = q
    out["significant"] = sig
    return out.sort_values(["q_value_FDR", "p_value", "metric"]).reset_index(drop=True)


def permutation_subject_level_fc(
    summary: pd.DataFrame,
    *,
    n_permutations: int = 1000,
    random_seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Permutation sensitivity for subject-level group mean differences."""
    if summary.empty:
        return pd.DataFrame()
    metadata_cols = {"atlas", "branch", "group", "subject_id", "n_edges"}
    metric_cols = [
        col
        for col in summary.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(summary[col])
    ]
    rng = np.random.default_rng(int(random_seed))
    n_perm = int(max(1, n_permutations))
    rows: list[dict[str, Any]] = []
    labels = summary["group"].to_numpy()
    n_sz = int(np.sum(labels == "SZ"))
    for metric in metric_cols:
        values = summary[metric].to_numpy(dtype=float)
        finite = np.isfinite(values)
        vals = values[finite]
        labs = labels[finite]
        if vals.size < 2 or np.sum(labs == "SZ") == 0 or np.sum(labs == "HC") == 0:
            observed = float("nan")
            p_value = 1.0
        else:
            observed = float(np.mean(vals[labs == "SZ"]) - np.mean(vals[labs == "HC"]))
            count = 0
            for _ in range(n_perm):
                perm = rng.permutation(labs)
                diff = float(np.mean(vals[perm == "SZ"]) - np.mean(vals[perm == "HC"]))
                if abs(diff) >= abs(observed):
                    count += 1
            p_value = float((count + 1) / (n_perm + 1))
        rows.append(
            {
                "atlas": str(summary["atlas"].iloc[0]) if "atlas" in summary else "",
                "branch": str(summary["branch"].iloc[0]) if "branch" in summary else "",
                "metric": metric,
                "n_permutations": n_perm,
                "random_seed": int(random_seed),
                "n_SZ": n_sz,
                "observed_delta_mean": observed,
                "p_value": p_value,
                "sensitivity_note": "exploratory subject-level permutation sensitivity",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    q, sig = _fdr_bh(out["p_value"].to_numpy(dtype=float), alpha=alpha)
    out["q_value_FDR"] = q
    out["significant"] = sig
    return out.sort_values(["q_value_FDR", "p_value", "metric"]).reset_index(drop=True)


def build_aal3_region_mapping(aal3_regions: str | Path | None, qc: pd.DataFrame) -> pd.DataFrame:
    aal_qc = qc[qc["atlas"].eq("AAL3")]
    if aal_qc.empty:
        return pd.DataFrame()
    if aal3_regions is None:
        n_regions = max(AAL3_EXPECTED_ROI, int(aal_qc["roi_index_0based"].max()) + 1)
        names = ["Background", *[f"unknown_region_{i}" for i in range(2, n_regions + 1)]]
    else:
        path = Path(aal3_regions)
        if not path.exists():
            raise FileNotFoundError(path)
        names: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            names.append(parts[1] if len(parts) > 1 and parts[0].isdigit() else line)
        required_n = max(AAL3_EXPECTED_ROI, int(aal_qc["roi_index_0based"].max()) + 1)
        if len(names) < required_n:
            names.extend(f"unknown_region_{i}" for i in range(len(names) + 1, required_n + 1))

    zero_global = set(
        aal_qc.groupby("roi_index_0based")
        .filter(lambda g: bool((g["zero_flag"].astype(bool) | g["constant_flag"].astype(bool)).any()))[
            "roi_index_0based"
        ]
        .astype(int)
        .tolist()
    )
    rows = []
    for idx0 in range(len(names)):
        rows.append(
            {
                "csv_region_index_0based": idx0,
                "csv_region_index_1based": idx0 + 1,
                "atlas_label_candidate": idx0 + 1,
                "region_name": names[idx0],
                "is_background": idx0 == 0 or names[idx0].lower() == "background",
                "present_in_timeseries": bool(idx0 in set(aal_qc["roi_index_0based"].astype(int))),
                "is_zero_region_global": bool(
                    idx0 in zero_global or idx0 in KNOWN_ZERO_AAL3_INDICES_0BASED
                ),
                "known_zero_example_index": bool(idx0 in KNOWN_ZERO_AAL3_INDICES_0BASED),
            }
        )
    return pd.DataFrame(rows)


def _count_components_for_region(
    coords: set[tuple[int, int, int]],
    offsets: list[tuple[int, int, int]],
) -> int:
    if not coords:
        return 0
    unseen = set(coords)
    n_components = 0
    while unseen:
        n_components += 1
        seed = unseen.pop()
        queue: deque[tuple[int, int, int]] = deque([seed])
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in offsets:
                nxt = (x + dx, y + dy, z + dz)
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
    return n_components


def _region_adjacency_metrics(
    *,
    coords: set[tuple[int, int, int]],
    region_id: str,
    coord_to_region: dict[tuple[int, int, int], str],
    offsets: list[tuple[int, int, int]],
) -> tuple[int, set[str], int, float]:
    neighbouring_regions: set[str] = set()
    boundary_voxels = 0
    for x, y, z in coords:
        is_boundary = False
        for dx, dy, dz in offsets:
            nxt = (x + dx, y + dy, z + dz)
            neighbour_region = coord_to_region.get(nxt)
            if neighbour_region != region_id:
                is_boundary = True
                if neighbour_region is not None and neighbour_region != "0":
                    neighbouring_regions.add(neighbour_region)
        if is_boundary:
            boundary_voxels += 1
    n_voxels = len(coords)
    proxy = float(boundary_voxels / n_voxels) if n_voxels else float("nan")
    return (
        _count_components_for_region(coords, offsets),
        neighbouring_regions,
        boundary_voxels,
        proxy,
    )


def build_hcp_mask_geometry_qc(
    voxel_map: str | Path,
    *,
    small_region_voxel_threshold: int = HCP_SMALL_REGION_VOXEL_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize HCP-MMP1 volume-space atlas geometry from a voxel map CSV.

    This is mask/atlas geometry QC only. It does not use BOLD time series and
    does not estimate functional homogeneity inside regions.
    """
    path = Path(voxel_map)
    df = pd.read_csv(path)
    required = {"x", "y", "z", "region_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HCP voxel map is missing required columns: {sorted(missing)}")

    work = df.copy()
    work["region_id"] = work["region_id"].astype(str)
    background = work["region_id"].isin({"0", "Background", "background"})
    if "region_name" in work.columns:
        background = background | work["region_name"].astype(str).str.lower().eq("background")

    total_voxels = int(len(work))
    non_background = work[~background].copy()
    bounds = {
        axis: (int(work[axis].min()), int(work[axis].max())) if total_voxels else (0, -1)
        for axis in ["x", "y", "z"]
    }
    geometry = pd.DataFrame(
        [
            {
                "total_voxels": total_voxels,
                "background_voxels": int(background.sum()),
                "non_background_voxels": int((~background).sum()),
                "n_regions": int(non_background["region_id"].nunique()),
                "x_min": bounds["x"][0],
                "x_max": bounds["x"][1],
                "y_min": bounds["y"][0],
                "y_max": bounds["y"][1],
                "z_min": bounds["z"][0],
                "z_max": bounds["z"][1],
                "implied_grid_shape": (
                    f"{bounds['x'][1] - bounds['x'][0] + 1}x"
                    f"{bounds['y'][1] - bounds['y'][0] + 1}x"
                    f"{bounds['z'][1] - bounds['z'][0] + 1}"
                ),
                "limitation": "volume-space atlas geometry only, not functional homogeneity QC",
            }
        ]
    )

    if non_background.empty:
        region_sizes = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "n_voxels",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
                "bbox_volume",
                "bbox_fill_fraction",
            ]
        )
        geometry["empty_regions"] = 0
        geometry["small_regions"] = 0
        geometry["small_region_voxel_threshold"] = int(max(1, small_region_voxel_threshold))
        return geometry, region_sizes, pd.DataFrame()

    rows: list[dict[str, Any]] = []
    region_coords: dict[str, set[tuple[int, int, int]]] = {}
    coord_to_region: dict[tuple[int, int, int], str] = {}
    for region_id, group in non_background.groupby("region_id", sort=True):
        x_min, x_max = int(group["x"].min()), int(group["x"].max())
        y_min, y_max = int(group["y"].min()), int(group["y"].max())
        z_min, z_max = int(group["z"].min()), int(group["z"].max())
        bbox_volume = int((x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1))
        n_voxels = int(len(group))
        region_name = (
            str(group["region_name"].iloc[0]) if "region_name" in group.columns else str(region_id)
        )
        coords = {
            (int(row.x), int(row.y), int(row.z))
            for row in group[["x", "y", "z"]].itertuples(index=False)
        }
        region_coords[str(region_id)] = coords
        for coord in coords:
            coord_to_region[coord] = str(region_id)
        rows.append(
            {
                "region_id": str(region_id),
                "region_name": region_name,
                "n_voxels": n_voxels,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
                "bbox_volume": bbox_volume,
                "bbox_fill_fraction": float(n_voxels / bbox_volume) if bbox_volume else float("nan"),
            }
        )
    region_sizes = pd.DataFrame(rows).sort_values("n_voxels").reset_index(drop=True)
    geometry["min_voxels_per_region"] = int(region_sizes["n_voxels"].min())
    geometry["max_voxels_per_region"] = int(region_sizes["n_voxels"].max())
    geometry["median_voxels_per_region"] = float(region_sizes["n_voxels"].median())
    threshold = int(max(1, small_region_voxel_threshold))
    geometry["empty_regions"] = 0
    geometry["small_regions"] = int((region_sizes["n_voxels"] < threshold).sum())
    geometry["small_region_voxel_threshold"] = threshold

    adjacency_rows: list[dict[str, Any]] = []
    region_name_lookup = dict(zip(region_sizes["region_id"], region_sizes["region_name"]))
    voxel_count_lookup = dict(zip(region_sizes["region_id"], region_sizes["n_voxels"]))
    for region_id, coords in sorted(region_coords.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {
            "region_id": region_id,
            "region_name": region_name_lookup.get(region_id, region_id),
            "n_voxels": int(voxel_count_lookup.get(region_id, len(coords))),
        }
        for connectivity in (6, 18, 26):
            components, neighbours, boundary_count, surface_proxy = _region_adjacency_metrics(
                coords=coords,
                region_id=region_id,
                coord_to_region=coord_to_region,
                offsets=neighbor_offsets(connectivity),
            )
            row[f"n_connected_components_{connectivity}"] = int(components)
            row[f"neighbouring_region_ids_{connectivity}"] = ";".join(sorted(neighbours))
            row[f"boundary_voxel_count_{connectivity}"] = int(boundary_count)
            row[f"surface_to_volume_proxy_{connectivity}"] = surface_proxy
        adjacency_rows.append(row)
    adjacency = pd.DataFrame(adjacency_rows)
    return geometry, region_sizes, adjacency


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(cols[i]), *(len(row[i]) for row in rows))
        for i in range(len(cols))
    ]
    header = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_temporal_figures(
    temporal_long: pd.DataFrame,
    *,
    atlas: str,
    out_dir: Path,
) -> list[str]:
    if temporal_long.empty:
        return []
    subset = temporal_long[temporal_long["atlas"].eq(atlas)].copy()
    if subset.empty:
        return []
    plt = _load_pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    acf_cols = ["acf_lag_1", "acf_lag_2", "acf_lag_3", "acf_lag_5", "acf_lag_10"]
    lags = [1, 2, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for group_name, group_df in subset.groupby("group"):
        means = [float(np.nanmean(group_df[col])) for col in acf_cols]
        ax.plot(lags, means, marker="o", label=str(group_name))
    ax.set_title(f"{atlas}: ACF profiles by group")
    ax.set_xlabel("lag")
    ax.set_ylabel("mean ACF")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / "acf_profiles_by_group.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    for col, filename, title, ylabel in [
        ("ar1_coeff", "ar1_distribution_HC_vs_SZ.png", "AR(1) by group", "AR(1)"),
        (
            "trend_slope",
            "trend_distribution_HC_vs_SZ.png",
            "Trend slope by group",
            "trend slope",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        groups = [g for g in ["HC", "SZ"] if g in set(subset["group"])]
        values = [
            subset[subset["group"].eq(group)][col].dropna().to_numpy(dtype=float) for group in groups
        ]
        if values and any(v.size for v in values):
            ax.boxplot(values, tick_labels=groups, showmeans=True)
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_xticks([])
        ax.set_title(f"{atlas}: {title}")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def _save_fc_figures(comparison: pd.DataFrame, *, atlas: str, branch: str, out_dir: Path) -> list[str]:
    if comparison.empty:
        return []
    plt = _load_pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    max_idx = int(max(comparison["edge_i"].max(), comparison["edge_j"].max()))
    matrix = np.full((max_idx + 1, max_idx + 1), np.nan, dtype=float)
    for row in comparison.itertuples(index=False):
        i = int(row.edge_i)
        j = int(row.edge_j)
        value = float(row.delta_mean) if np.isfinite(row.delta_mean) else np.nan
        matrix[i, j] = value
        matrix[j, i] = value
    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{atlas}/{branch}: delta mean Fisher z (SZ - HC)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    path = out_dir / "fc_delta_matrix_HC_vs_SZ.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    sig = (
        comparison[comparison["significant"].astype(bool)].head(80)
        if "significant" in comparison.columns
        else comparison.iloc[0:0]
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    if sig.empty:
        ax.text(0.5, 0.5, "No significant FDR edges", ha="center", va="center")
        ax.axis("off")
    else:
        nodes = sorted(set(sig["edge_i"].astype(int)) | set(sig["edge_j"].astype(int)))
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(nodes, angles)}
        for row in sig.itertuples(index=False):
            a = pos[int(row.edge_i)]
            b = pos[int(row.edge_j)]
            color = "tab:red" if float(row.delta_mean) > 0 else "tab:blue"
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, alpha=0.35, linewidth=1.0)
        xs = [pos[node][0] for node in nodes]
        ys = [pos[node][1] for node in nodes]
        ax.scatter(xs, ys, s=18, color="black", zorder=3)
        ax.set_title(f"{atlas}/{branch}: significant edges")
        ax.axis("equal")
        ax.axis("off")
    path = out_dir / "significant_edges_network.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    return written


def _save_hcp_geometry_figure(region_sizes: pd.DataFrame, *, out_dir: Path) -> list[str]:
    if region_sizes.empty or "n_voxels" not in region_sizes.columns:
        return []
    plt = _load_pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    values = region_sizes["n_voxels"].dropna().to_numpy(dtype=float)
    if values.size:
        ax.hist(values, bins=min(30, max(5, int(np.sqrt(values.size)))))
    else:
        ax.text(0.5, 0.5, "No region sizes", ha="center", va="center")
    ax.set_title("HCP region size distribution")
    ax.set_xlabel("voxels per region")
    ax.set_ylabel("region count")
    ax.grid(True, axis="y", alpha=0.25)
    path = out_dir / "hcp_region_size_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [str(path)]


def _write_inventory_md(inventory: pd.DataFrame, path: Path) -> None:
    counts = inventory.groupby(["group", "atlas", "status"]).size().reset_index(name="n_files")
    lines = [
        "# Data Inventory",
        "",
        FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE,
        "",
        "## Counts",
        "",
        _markdown_table(counts),
        "",
        "## Limitations",
        "",
        "- This is an audit of already-extracted ROI time series.",
        "- It does not validate voxel-wise functional homogeneity inside ROI.",
        "- It does not validate atlas-to-BOLD overlay quality.",
        "- It does not make clinical biomarker claims.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(
    path: Path,
    *,
    result: FmriRoiAuditResult,
    inventory: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    subject_level_comparison: pd.DataFrame | None = None,
    hcp_geometry_summary: pd.DataFrame | None = None,
    edge_comparison: pd.DataFrame | None = None,
    ttest_comparison: pd.DataFrame | None = None,
    permutation_summary: pd.DataFrame | None = None,
    figure_paths: list[str] | None = None,
    threshold_sensitivity_paths: list[str] | None = None,
) -> None:
    lines = [
        "# Final Pilot Report",
        "",
        FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE,
        "",
        "## Summary",
        "",
        f"- HC subjects: {result.n_hc}",
        f"- SZ subjects: {result.n_sz}",
        f"- Atlases: {', '.join(result.atlases) if result.atlases else 'none'}",
        f"- Branches: {', '.join(result.branches)}",
        "",
        "## Inventory",
        "",
        _markdown_table(
            inventory.groupby(["group", "atlas", "status"]).size().reset_index(name="n_files")
        ),
        "",
        "## Excluded ROI",
        "",
    ]
    for atlas, n_bad in result.n_bad_rois_by_atlas.items():
        lines.append(f"- {atlas}: {n_bad} common bad ROI excluded by conservative strategy")
    lines.extend(["", "## FC Group Comparison", ""])
    for key, n_edges in result.n_edges_by_atlas_branch.items():
        n_sig = result.n_significant_by_atlas_branch.get(key, 0)
        lines.append(f"- {key}: {n_sig} significant edges after FDR out of {n_edges}")
    if edge_comparison is not None and not edge_comparison.empty:
        lines.extend(["", "## Top FDR Edges", ""])
        top_cols = [
            "atlas",
            "branch",
            "edge_i",
            "edge_j",
            "delta_mean",
            "p_value",
            "q_value_FDR",
            "significant",
        ]
        lines.append(_markdown_table(edge_comparison[top_cols].head(20)))
    if ttest_comparison is not None and not ttest_comparison.empty:
        lines.extend(["", "## T-Test Sensitivity", ""])
        t_cols = ["atlas", "branch", "edge_i", "edge_j", "delta_mean", "p_value", "q_value_FDR"]
        lines.append(_markdown_table(ttest_comparison[t_cols].head(10)))
    if permutation_summary is not None and not permutation_summary.empty:
        lines.extend(["", "## Permutation Sensitivity", ""])
        p_cols = [
            "atlas",
            "branch",
            "metric",
            "observed_delta_mean",
            "p_value",
            "q_value_FDR",
            "n_permutations",
        ]
        lines.append(_markdown_table(permutation_summary[p_cols].head(10)))
    if subject_level_comparison is not None and not subject_level_comparison.empty:
        lines.extend(["", "## Subject-Level FC Summary", ""])
        compact = subject_level_comparison[
            [
                "atlas",
                "branch",
                "metric",
                "mean_HC",
                "mean_SZ",
                "delta_mean",
                "p_value",
                "q_value_FDR",
                "significant",
            ]
        ].head(20)
        lines.append(_markdown_table(compact))
    if hcp_geometry_summary is not None and not hcp_geometry_summary.empty:
        lines.extend(["", "## HCP Mask Geometry QC", ""])
        lines.append(_markdown_table(hcp_geometry_summary))
        lines.append("")
        lines.append(
            "HCP mask geometry QC is volume-space atlas geometry only; it is not "
            "surface-based cortical adjacency or functional homogeneity QC."
        )
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for fig_path in figure_paths:
            lines.append(f"- `{fig_path}`")
    if threshold_sensitivity_paths:
        lines.extend(["", "## Threshold Bad ROI Sensitivity", ""])
        for sensitivity_path in sorted(set(threshold_sensitivity_paths)):
            lines.append(f"- `{sensitivity_path}`")
        lines.append("")
        lines.append(
            "Threshold bad ROI sensitivity is exploratory and does not replace the "
            "conservative baseline FC path."
        )
    if not temporal_summary.empty:
        lines.extend(["", "## Temporal QC Summary", "", _markdown_table(temporal_summary)])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This analysis starts from extracted ROI time series and does not reconstruct upstream preprocessing.",
            "- Without voxel-level time series, voxel-wise functional homogeneity inside ROI cannot be tested.",
            "- ROI-level GSR, when enabled, is an approximation and not strict voxel-wise GSR.",
            "- These outputs are exploratory research diagnostics, not clinical biomarkers.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_fmri_roi_audit(
    hc_dir: str | Path,
    sz_dir: str | Path,
    output_dir: str | Path,
    *,
    aal3_regions: str | Path | None = None,
    atlas_filter: str = "all",
    bad_roi_strategy: str = "conservative",
    alpha: float = 0.05,
    branches: tuple[str, ...] = ("raw_cleaned",),
    include_hcp_mask_qc: bool = False,
    hcp_voxel_map: str | Path | None = None,
    make_figures: bool = True,
    bad_roi_thresholds: tuple[float, ...] = DEFAULT_BAD_ROI_THRESHOLDS,
    include_ttest: bool = False,
    include_permutation: bool = False,
    n_permutations: int = 1000,
    random_seed: int = 0,
) -> FmriRoiAuditResult:
    """Run the experimental MVP audit for HC/SZ ROI time-series CSV files."""
    if bad_roi_strategy != "conservative":
        raise ValueError("MVP supports only bad_roi_strategy='conservative'")
    atlas_filter = str(atlas_filter)
    if atlas_filter not in {"all", "AAL3", "HCP"}:
        raise ValueError("atlas_filter must be one of: all, AAL3, HCP")
    out = Path(output_dir)
    inventories_dir = out / "outputs" / "inventories"
    qc_dir = out / "outputs" / "qc"
    temporal_dir = out / "outputs" / "temporal"
    preprocessed_dir = out / "outputs" / "preprocessed"
    fc_matrices_dir = out / "outputs" / "fc_matrices"
    fc_edges_dir = out / "outputs" / "fc_edges"
    group_dir = out / "outputs" / "group_comparison"
    figures_dir = out / "outputs" / "figures"
    sensitivity_dir = out / "outputs" / "sensitivity"
    reports_dir = out / "reports"
    for directory in [
        inventories_dir,
        qc_dir,
        temporal_dir,
        preprocessed_dir,
        fc_matrices_dir,
        fc_edges_dir,
        group_dir,
        figures_dir,
        sensitivity_dir,
        reports_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    warnings = [FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE]
    hcp_geometry_summary = pd.DataFrame()
    figure_paths: list[str] = []
    edge_comparisons: list[pd.DataFrame] = []
    ttest_comparisons: list[pd.DataFrame] = []
    permutation_summaries: list[pd.DataFrame] = []
    threshold_sensitivity_paths: list[str] = []
    inventory = scan_inventory(hc_dir, sz_dir, atlas_filter=atlas_filter)
    inventory.to_csv(inventories_dir / "data_inventory.csv", index=False)
    _write_inventory_md(inventory, inventories_dir / "data_inventory.md")

    subjects = load_valid_subjects(inventory)
    if not subjects:
        raise RuntimeError("No valid ROI time-series files after inventory scan")
    qc = build_roi_qc(subjects)
    qc.to_csv(qc_dir / "roi_timeseries_qc_long.csv", index=False)
    if not qc.empty:
        qc.groupby(["atlas", "group", "subject_id"], as_index=False).agg(
            n_roi=("roi_id", "count"),
            n_zero=("zero_flag", "sum"),
            n_constant=("constant_flag", "sum"),
            n_nan=("nan_flag", "sum"),
            n_short=("short_series_flag", "sum"),
        ).to_csv(qc_dir / "roi_timeseries_qc_by_subject.csv", index=False)
        qc.groupby(["atlas", "roi_index_0based", "roi_index_1based"], as_index=False).agg(
            n_subjects=("subject_id", "count"),
            n_zero=("zero_flag", "sum"),
            n_constant=("constant_flag", "sum"),
            median_std=("std", "median"),
        ).to_csv(qc_dir / "roi_timeseries_qc_by_region.csv", index=False)

    mapping = build_aal3_region_mapping(aal3_regions, qc)
    if not mapping.empty:
        mapping.to_csv(inventories_dir / "aal3_region_mapping_report.csv", index=False)

    if include_hcp_mask_qc or hcp_voxel_map is not None:
        if hcp_voxel_map is None:
            warnings.append("HCP mask geometry QC requested but no hcp_voxel_map was provided")
        else:
            geometry, region_sizes, region_adjacency = build_hcp_mask_geometry_qc(hcp_voxel_map)
            geometry.to_csv(qc_dir / "hcp_mmp1_mask_geometry_report.csv", index=False)
            region_sizes.to_csv(qc_dir / "hcp_region_size_report.csv", index=False)
            region_adjacency.to_csv(qc_dir / "hcp_region_adjacency_report.csv", index=False)
            hcp_geometry_summary = geometry
            if make_figures:
                figure_paths.extend(
                    _save_hcp_geometry_figure(region_sizes, out_dir=figures_dir / "HCP_geometry")
                )
            warnings.append(
                "HCP mask geometry QC is volume-space atlas geometry only; "
                "it is not surface adjacency or functional homogeneity QC."
            )

    temporal_long, temporal_summary = build_temporal_qc(subjects)
    temporal_long.to_csv(temporal_dir / "temporal_qc_long.csv", index=False)
    temporal_summary.to_csv(temporal_dir / "temporal_qc_group_summary.csv", index=False)

    atlases = sorted({s.atlas for s in subjects if s.atlas in SUPPORTED_ATLASES})
    n_subjects_by_atlas: dict[str, int] = {}
    n_bad_rois_by_atlas: dict[str, int] = {}
    n_edges_by_atlas_branch: dict[str, int] = {}
    n_significant_by_atlas_branch: dict[str, int] = {}
    subject_level_comparisons: list[pd.DataFrame] = []

    branch_list = list(dict.fromkeys(branches))
    subject_by_key = {f"{s.group}::{s.subject_id}::{s.atlas}": s for s in subjects}

    for atlas in atlases:
        atlas_subjects = [s for s in subjects if s.atlas == atlas]
        groups = {s.group for s in atlas_subjects}
        if groups != {"HC", "SZ"}:
            warnings.append(f"{atlas}: skipped FC because both HC and SZ subjects are required")
            continue
        n_subjects_by_atlas[atlas] = len(atlas_subjects)
        bad_df = build_common_bad_rois(qc, atlas=atlas)
        bad_df.to_csv(qc_dir / f"common_bad_rois_{atlas}.csv", index=False)
        threshold_bad_by_name: dict[str, pd.DataFrame] = {}
        for threshold in bad_roi_thresholds:
            threshold_bad = build_threshold_bad_rois(qc, atlas=atlas, threshold=float(threshold))
            threshold_name = str(float(threshold)).replace(".", "_")
            threshold_bad_by_name[threshold_name] = threshold_bad
            threshold_out = sensitivity_dir / atlas / f"threshold_{threshold_name}"
            threshold_out.mkdir(parents=True, exist_ok=True)
            threshold_bad.to_csv(threshold_out / f"common_bad_rois_{atlas}.csv", index=False)
            threshold_sensitivity_paths.append(str(threshold_out))
            (threshold_out / "summary.md").write_text(
                "\n".join(
                    [
                        f"# Bad ROI Threshold Sensitivity: {atlas}",
                        "",
                        f"- Threshold: bad_frequency >= {float(threshold):.2f}",
                        f"- Bad ROI retained in exclusion set: {len(threshold_bad)}",
                        "- This is a sensitivity diagnostic only; conservative baseline remains primary.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        bad_indices = set(bad_df.get("roi_index_0based", pd.Series(dtype=int)).astype(int).tolist())
        n_bad_rois_by_atlas[atlas] = len(bad_indices)

        n_regions = atlas_subjects[0].data_time_roi.shape[1]
        retained_indices = [idx for idx in range(n_regions) if idx not in bad_indices]
        if len(retained_indices) < 2:
            warnings.append(f"{atlas}: skipped FC because fewer than 2 ROI remain after QC")
            continue

        for branch in branch_list:
            matrix_out = fc_matrices_dir / atlas / branch
            preprocessed_out = preprocessed_dir / atlas / branch
            edge_out = fc_edges_dir / atlas / branch
            cmp_out = group_dir / atlas / branch
            edges, comparison, subject_summary, subject_comparison = _write_fc_branch_outputs(
                atlas_subjects=atlas_subjects,
                subject_by_key=subject_by_key,
                atlas=atlas,
                branch=branch,
                bad_indices=bad_indices,
                retained_indices=retained_indices,
                alpha=alpha,
                preprocessed_out=preprocessed_out,
                matrix_out=matrix_out,
                edge_out=edge_out,
                comparison_out=cmp_out,
            )
            if not comparison.empty:
                report_comparison = comparison.copy()
                report_comparison.insert(0, "branch", branch)
                report_comparison.insert(0, "atlas", atlas)
                edge_comparisons.append(report_comparison)
            if include_ttest:
                ttest = compare_fc_edges_ttest(edges, alpha=alpha)
                ttest.to_csv(cmp_out / "fc_group_comparison_edges_ttest.csv", index=False)
                if not ttest.empty:
                    ttest_report = ttest.copy()
                    ttest_report.insert(0, "branch", branch)
                    ttest_report.insert(0, "atlas", atlas)
                    ttest_comparisons.append(ttest_report)
            if include_permutation:
                permutation = permutation_subject_level_fc(
                    subject_summary,
                    n_permutations=n_permutations,
                    random_seed=random_seed,
                    alpha=alpha,
                )
                permutation.to_csv(cmp_out / "permutation_summary.csv", index=False)
                if not permutation.empty:
                    permutation_summaries.append(permutation)
            if not subject_comparison.empty:
                subject_level_comparisons.append(subject_comparison)
            if make_figures:
                figure_dir = figures_dir / atlas / branch
                try:
                    figure_paths.extend(
                        _save_temporal_figures(temporal_long, atlas=atlas, out_dir=figure_dir)
                    )
                    figure_paths.extend(
                        _save_fc_figures(comparison, atlas=atlas, branch=branch, out_dir=figure_dir)
                    )
                except Exception as exc:
                    warnings.append(f"{atlas}/{branch}: figure generation failed: {exc}")
            n_edges = int(len(comparison))
            n_sig = int(comparison["significant"].sum()) if "significant" in comparison else 0
            key = f"{atlas}/{branch}"
            n_edges_by_atlas_branch[key] = n_edges
            n_significant_by_atlas_branch[key] = n_sig
            summary = [
                f"# FC Group Comparison: {atlas} / {branch}",
                "",
                FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE,
                "",
                f"- Edges tested: {n_edges}",
                f"- Significant after FDR alpha={alpha}: {n_sig}",
                f"- Bad ROI excluded: {len(bad_indices)}",
                "- Subject-level summary: subject_level_fc_summary.csv",
                "- Subject-level group comparison: subject_level_group_comparison.csv",
                f"- T-test sensitivity: {'fc_group_comparison_edges_ttest.csv' if include_ttest else 'not requested'}",
                f"- Permutation sensitivity: {'permutation_summary.csv' if include_permutation else 'not requested'}",
                "",
            ]
            if branch == "roi_level_gsr":
                summary.append("ROI-level global signal approximation, not voxel-wise GSR.\n")
            (cmp_out / "summary.md").write_text("\n".join(summary), encoding="utf-8")

            for threshold_name, threshold_bad in threshold_bad_by_name.items():
                threshold_bad_indices = set(
                    threshold_bad.get("roi_index_0based", pd.Series(dtype=int))
                    .astype(int)
                    .tolist()
                )
                threshold_retained = [
                    idx for idx in range(n_regions) if idx not in threshold_bad_indices
                ]
                threshold_branch_out = (
                    sensitivity_dir / atlas / f"threshold_{threshold_name}" / branch
                )
                threshold_branch_out.mkdir(parents=True, exist_ok=True)
                if len(threshold_retained) < 2:
                    (threshold_branch_out / "summary.md").write_text(
                        "\n".join(
                            [
                                f"# Threshold Sensitivity: {atlas} / {branch}",
                                "",
                                "- Skipped: fewer than 2 ROI remain after threshold bad ROI exclusion.",
                                "- Conservative baseline remains primary.",
                                "",
                            ]
                        ),
                        encoding="utf-8",
                    )
                    continue
                _, threshold_comparison, threshold_subject_summary, threshold_subject_comparison = (
                    _write_fc_branch_outputs(
                        atlas_subjects=atlas_subjects,
                        subject_by_key=subject_by_key,
                        atlas=atlas,
                        branch=branch,
                        bad_indices=threshold_bad_indices,
                        retained_indices=threshold_retained,
                        alpha=alpha,
                        preprocessed_out=threshold_branch_out / "preprocessed",
                        matrix_out=threshold_branch_out / "fc_matrices",
                        edge_out=threshold_branch_out,
                        comparison_out=threshold_branch_out,
                    )
                )
                threshold_n_sig = (
                    int(threshold_comparison["significant"].sum())
                    if "significant" in threshold_comparison
                    else 0
                )
                (threshold_branch_out / "summary.md").write_text(
                    "\n".join(
                        [
                            f"# Threshold Sensitivity: {atlas} / {branch}",
                            "",
                            FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE,
                            "",
                            f"- Threshold name: threshold_{threshold_name}",
                            f"- Bad ROI excluded: {len(threshold_bad_indices)}",
                            f"- Edges tested: {len(threshold_comparison)}",
                            f"- Significant after FDR alpha={alpha}: {threshold_n_sig}",
                            f"- Subjects summarized: {len(threshold_subject_summary)}",
                            f"- Subject-level tests: {len(threshold_subject_comparison)}",
                            "- This is exploratory threshold sensitivity and does not replace the conservative baseline.",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )

    result = FmriRoiAuditResult(
        n_hc=int(len({s.subject_id for s in subjects if s.group == "HC"})),
        n_sz=int(len({s.subject_id for s in subjects if s.group == "SZ"})),
        atlases=atlases,
        branches=branch_list,
        n_subjects_by_atlas=n_subjects_by_atlas,
        n_bad_rois_by_atlas=n_bad_rois_by_atlas,
        n_edges_by_atlas_branch=n_edges_by_atlas_branch,
        n_significant_by_atlas_branch=n_significant_by_atlas_branch,
        output_dir=str(out.resolve()),
        warnings=warnings,
    )
    _write_final_report(
        reports_dir / "final_pilot_report.md",
        result=result,
        inventory=inventory,
        temporal_summary=temporal_summary,
        subject_level_comparison=(
            pd.concat(subject_level_comparisons, ignore_index=True)
            if subject_level_comparisons
            else pd.DataFrame()
        ),
        hcp_geometry_summary=hcp_geometry_summary,
        edge_comparison=(
            pd.concat(edge_comparisons, ignore_index=True)
            if edge_comparisons
            else pd.DataFrame()
        ),
        ttest_comparison=(
            pd.concat(ttest_comparisons, ignore_index=True)
            if ttest_comparisons
            else pd.DataFrame()
        ),
        permutation_summary=(
            pd.concat(permutation_summaries, ignore_index=True)
            if permutation_summaries
            else pd.DataFrame()
        ),
        figure_paths=figure_paths,
        threshold_sensitivity_paths=threshold_sensitivity_paths,
    )
    return result


__all__ = [
    "FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE",
    "FmriRoiAuditResult",
    "DEFAULT_BAD_ROI_THRESHOLDS",
    "SUBJECT_LEVEL_DENSITY_THRESHOLDS",
    "build_aal3_region_mapping",
    "build_common_bad_rois",
    "build_hcp_mask_geometry_qc",
    "build_roi_qc",
    "build_subject_level_fc_summary",
    "build_temporal_qc",
    "build_threshold_bad_rois",
    "compare_fc_edges",
    "compare_fc_edges_ttest",
    "compare_subject_level_fc",
    "detect_atlas",
    "detect_orientation",
    "load_valid_subjects",
    "parse_subject_id",
    "pearson_fisher_fc",
    "permutation_subject_level_fc",
    "preprocess_subject",
    "run_fmri_roi_audit",
    "scan_inventory",
    "summarize_subject_fc_matrix",
]
