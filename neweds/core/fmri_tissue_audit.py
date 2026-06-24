"""Independent audit for GM/WM/CSF voxel time-series HDF5 files.

This module deliberately does not run ROI connectivity or reuse ROI audit
tables. Tissue HDF5 data have a separate evidence boundary and therefore a
separate output contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import stats

TISSUES = ("GM", "WM", "CSF")
TISSUE_AUDIT_TYPE = "tissue_gm_wm_csf"
TISSUE_AUDIT_LAYOUT_VERSION = 1


@dataclass(frozen=True, slots=True)
class FmriTissueAuditResult:
    """Compact result of one independent tissue audit."""

    output_dir: str
    files_discovered: int
    files_valid: int
    subjects_hc: int
    subjects_sz: int
    failures: int
    xyz_complete_files: int
    spatial_analysis_available: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _detect_group(path: Path, h5: h5py.File) -> str:
    attr_group = _attr_text(h5.attrs.get("group", "")).strip().upper()
    if attr_group in {"HC", "SZ"}:
        return attr_group
    text = f"{path.parent.name} {path.name}".upper()
    if "_HC" in text or "КОНТРОЛ" in text or "HEALTH" in text:
        return "HC"
    if "SZ" in text or "ШИЗОФР" in text:
        return "SZ"
    return "unknown"


def _root_shape(h5: h5py.File) -> tuple[int, ...]:
    value = h5.attrs.get("shape", h5.attrs.get("fMRI_shape", ()))
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return ()


def scan_tissue_h5_inventory(root_dir: str | Path) -> pd.DataFrame:
    """Inventory HDF5 files and validate the GM/WM/CSF schema without loading data."""
    root = Path(root_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.h5")):
        row: dict[str, Any] = {
            "file_path": str(path),
            "file_name": path.name,
            "file_size_bytes": int(path.stat().st_size),
            "status": "ok",
            "error": "",
        }
        try:
            with h5py.File(path, "r") as h5:
                row["group"] = _detect_group(path, h5)
                row["subject_id"] = _attr_text(
                    h5.attrs.get("id", path.stem.split("_")[0])
                )
                row["root_shape"] = "x".join(str(v) for v in _root_shape(h5))
                row["time_metadata_name"] = (
                    "T" if "T" in h5.attrs else "TR" if "TR" in h5.attrs else ""
                )
                row["time_metadata_value"] = _safe_float(
                    h5.attrs.get("T", h5.attrs.get("TR", np.nan))
                )
                missing_data: list[str] = []
                for tissue in TISSUES:
                    data_key = f"{tissue}/data"
                    xyz_key = f"{tissue}/xyz"
                    has_data = data_key in h5
                    has_xyz = xyz_key in h5
                    row[f"{tissue}_has_data"] = has_data
                    row[f"{tissue}_has_xyz"] = has_xyz
                    if not has_data:
                        missing_data.append(data_key)
                        continue
                    ds = h5[data_key]
                    row[f"{tissue}_shape"] = "x".join(str(v) for v in ds.shape)
                    row[f"{tissue}_dtype"] = str(ds.dtype)
                    row[f"{tissue}_n_voxels"] = int(ds.shape[0]) if ds.ndim == 2 else 0
                    row[f"{tissue}_n_timepoints"] = int(ds.shape[1]) if ds.ndim == 2 else 0
                    row[f"{tissue}_compression"] = str(ds.compression or "")
                    if ds.ndim != 2:
                        missing_data.append(f"{data_key}:expected_2d")
                    if has_xyz:
                        xyz = h5[xyz_key]
                        row[f"{tissue}_xyz_shape"] = "x".join(str(v) for v in xyz.shape)
                        row[f"{tissue}_xyz_matches_data"] = bool(
                            xyz.ndim == 2
                            and xyz.shape[0] == ds.shape[0]
                            and xyz.shape[1] == 3
                        )
                    else:
                        row[f"{tissue}_xyz_shape"] = ""
                        row[f"{tissue}_xyz_matches_data"] = False
                if missing_data:
                    row["status"] = "schema_error"
                    row["error"] = "; ".join(missing_data)
                timepoints = {
                    int(row.get(f"{tissue}_n_timepoints", 0)) for tissue in TISSUES
                }
                if row["status"] == "ok" and (len(timepoints) != 1 or 0 in timepoints):
                    row["status"] = "time_axis_mismatch"
                    row["error"] = f"tissue time axes: {sorted(timepoints)}"
        except Exception as exc:
            row["status"] = "read_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def _linear_trend(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 3 or not np.isfinite(x).all():
        return float("nan"), float("nan")
    t = np.arange(x.size, dtype=np.float64)
    slope, intercept = np.polyfit(t, x, 1)
    fitted = intercept + slope * t
    ss_total = float(np.sum((x - x.mean()) ** 2))
    ss_resid = float(np.sum((x - fitted) ** 2))
    r2 = 1.0 - ss_resid / ss_total if ss_total > 1e-12 else float("nan")
    return _safe_float(slope), _safe_float(r2)


def _ar1_residualize(values: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 8 or np.std(x) <= 1e-12:
        return np.full(max(0, x.size - 1), np.nan), float("nan")
    design = np.column_stack([np.ones(x.size - 1), x[:-1]])
    beta, *_ = np.linalg.lstsq(design, x[1:], rcond=None)
    residuals = x[1:] - design @ beta
    return residuals, _safe_float(beta[1])


def _lag_corr(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=np.float64)
    lag = int(max(0, lag))
    if lag == 0:
        return 1.0
    if x.size <= lag + 2:
        return float("nan")
    left = x[:-lag]
    right = x[lag:]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return _safe_float(np.corrcoef(left, right)[0, 1])


def _acf_pacf(values: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    nlags = int(max(1, min(max_lag, x.size // 2 - 1)))
    acf_values = np.asarray([_lag_corr(x, lag) for lag in range(nlags + 1)])
    try:
        from statsmodels.tsa.stattools import pacf

        pacf_values = np.asarray(pacf(x, nlags=nlags, method="ywmle"), dtype=float)
    except Exception:
        pacf_values = np.full(nlags + 1, np.nan)
        pacf_values[0] = 1.0
    return acf_values, pacf_values


def _adf_pvalue(values: np.ndarray) -> float:
    try:
        from statsmodels.tsa.stattools import adfuller

        return _safe_float(adfuller(np.asarray(values, dtype=float), autolag="AIC")[1])
    except Exception:
        return float("nan")


def _audit_dataset(
    dataset: h5py.Dataset,
    *,
    block_rows: int,
    constant_eps: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int]:
    n_voxels, n_timepoints = map(int, dataset.shape)
    all_sum = np.zeros(n_timepoints, dtype=np.float64)
    all_count = np.zeros(n_timepoints, dtype=np.int64)
    active_sum = np.zeros(n_timepoints, dtype=np.float64)
    active_count = 0
    nonfinite_values = 0
    zero_voxels = 0
    constant_voxels = 0
    nonzero_constant_voxels = 0
    voxel_std_sum = 0.0
    voxel_std_count = 0

    for start in range(0, n_voxels, int(max(1, block_rows))):
        stop = min(start + int(max(1, block_rows)), n_voxels)
        block = np.asarray(dataset[start:stop, :], dtype=np.float64)
        finite = np.isfinite(block)
        nonfinite_values += int((~finite).sum())
        safe = np.where(finite, block, 0.0)
        all_sum += safe.sum(axis=0)
        all_count += finite.sum(axis=0)

        complete = finite.all(axis=1)
        if not complete.any():
            continue
        complete_block = block[complete]
        minimum = complete_block.min(axis=1)
        maximum = complete_block.max(axis=1)
        standard_deviation = complete_block.std(axis=1)
        is_zero = np.all(complete_block == 0.0, axis=1)
        is_constant = (maximum - minimum) <= float(constant_eps)
        is_active = ~is_constant

        zero_voxels += int(is_zero.sum())
        constant_voxels += int(is_constant.sum())
        nonzero_constant_voxels += int((is_constant & ~is_zero).sum())
        if is_active.any():
            active_block = complete_block[is_active]
            active_sum += active_block.sum(axis=0)
            active_count += int(active_block.shape[0])
            voxel_std_sum += float(standard_deviation[is_active].sum())
            voxel_std_count += int(is_active.sum())

    mean_all = np.divide(
        all_sum,
        all_count,
        out=np.full(n_timepoints, np.nan),
        where=all_count > 0,
    )
    mean_active = (
        active_sum / active_count
        if active_count > 0
        else np.full(n_timepoints, np.nan)
    )
    summary = {
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "nonfinite_values": nonfinite_values,
        "zero_voxels": zero_voxels,
        "constant_voxels": constant_voxels,
        "nonzero_constant_voxels": nonzero_constant_voxels,
        "active_voxels": active_count,
        "zero_fraction": zero_voxels / max(1, n_voxels),
        "constant_fraction": constant_voxels / max(1, n_voxels),
        "active_fraction": active_count / max(1, n_voxels),
        "mean_active_voxel_std": (
            voxel_std_sum / voxel_std_count if voxel_std_count else float("nan")
        ),
    }
    return summary, mean_all, mean_active, active_count


def _temporal_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 8 or not np.isfinite(x).all():
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "trend_slope": float("nan"),
            "trend_r2": float("nan"),
            "acf_lag1": float("nan"),
            "pacf_lag1": float("nan"),
            "ar1_phi": float("nan"),
            "ar1_residual_acf_lag1": float("nan"),
            "adf_pvalue": float("nan"),
        }
    slope, r2 = _linear_trend(x)
    acf_values, pacf_values = _acf_pacf(x, 1)
    residuals, phi = _ar1_residualize(x)
    return {
        "mean": _safe_float(np.mean(x)),
        "std": _safe_float(np.std(x)),
        "trend_slope": slope,
        "trend_r2": r2,
        "acf_lag1": _safe_float(acf_values[1]) if acf_values.size > 1 else float("nan"),
        "pacf_lag1": (
            _safe_float(pacf_values[1]) if pacf_values.size > 1 else float("nan")
        ),
        "ar1_phi": phi,
        "ar1_residual_acf_lag1": _lag_corr(residuals, 1),
        "adf_pvalue": _adf_pvalue(x),
    }


def _fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    p = values[finite]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[finite] = restored
    return out


def _group_comparison(
    table: pd.DataFrame,
    *,
    id_columns: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = [
        col
        for col in table.columns
        if col not in id_columns and pd.api.types.is_numeric_dtype(table[col])
    ]
    tissue_values = sorted(table["tissue"].dropna().unique()) if "tissue" in table else [""]
    for tissue in tissue_values:
        part = table[table["tissue"].eq(tissue)] if tissue else table
        for feature in numeric:
            hc = pd.to_numeric(
                part[part["group"].eq("HC")][feature], errors="coerce"
            ).dropna()
            sz = pd.to_numeric(
                part[part["group"].eq("SZ")][feature], errors="coerce"
            ).dropna()
            if len(hc) >= 2 and len(sz) >= 2:
                try:
                    statistic, pvalue = stats.mannwhitneyu(
                        sz.to_numpy(float),
                        hc.to_numpy(float),
                        alternative="two-sided",
                    )
                    effect = 2.0 * float(statistic) / float(len(hc) * len(sz)) - 1.0
                except ValueError:
                    statistic = pvalue = effect = float("nan")
            else:
                statistic = pvalue = effect = float("nan")
            rows.append(
                {
                    "tissue": tissue,
                    "feature": feature,
                    "n_HC": int(len(hc)),
                    "n_SZ": int(len(sz)),
                    "HC_median": _safe_float(hc.median()) if len(hc) else float("nan"),
                    "SZ_median": _safe_float(sz.median()) if len(sz) else float("nan"),
                    "SZ_minus_HC": (
                        _safe_float(sz.median() - hc.median())
                        if len(hc) and len(sz)
                        else float("nan")
                    ),
                    "mannwhitney_u": statistic,
                    "rank_biserial_SZ_vs_HC": effect,
                    "p_value": pvalue,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_FDR"] = _fdr_bh(out["p_value"].to_numpy(float))
    return out


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    text = table.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join("---" for _ in text.columns) + " |",
    ]
    for _, row in text.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[col]).replace("|", "\\|") for col in text.columns)
            + " |"
        )
    return "\n".join(lines)


def _write_methodology_status(
    path: Path,
    *,
    inventory: pd.DataFrame,
) -> None:
    xyz_complete = int(
        inventory[
            [
                f"{tissue}_xyz_matches_data" for tissue in TISSUES
            ]
        ]
        .fillna(False)
        .all(axis=1)
        .sum()
    )
    rows = pd.DataFrame(
        [
            {
                "requirement": "GM/WM/CSF value and temporal QC",
                "status": "implemented",
                "evidence": "streamed data audit and tissue-mean temporal outputs",
            },
            {
                "requirement": "ACF and PACF before/after AR1",
                "status": "implemented",
                "evidence": "temporal/tissue_mean_acf_pacf.csv",
            },
            {
                "requirement": "global signal paired sensitivity",
                "status": "prepared_not_inferred",
                "evidence": "GLOBAL signal is exported; downstream with/without regression required",
            },
            {
                "requirement": "mask-constrained XYZ neighbourhood growth",
                "status": "blocked" if xyz_complete == 0 else "available",
                "evidence": f"complete xyz in {xyz_complete}/{len(inventory)} files",
            },
            {
                "requirement": "voxel-wise ROI homogeneity",
                "status": "blocked",
                "evidence": "requires voxel coordinates plus voxel-to-atlas/parcel mapping",
            },
            {
                "requirement": "mean vs sign-oriented PCA vs ICA regional signals",
                "status": "blocked",
                "evidence": "requires valid voxel-to-region membership",
            },
            {
                "requirement": "PCA/ICA sign ambiguity control",
                "status": "contract_defined",
                "evidence": "signed metrics require recorded orientation; r² is sign-invariant sensitivity",
            },
            {
                "requirement": "multiple-comparison correction",
                "status": "implemented",
                "evidence": "Benjamini-Hochberg FDR across tissue-feature group tests",
            },
            {
                "requirement": "lag x window x metric cube",
                "status": "deferred_exploratory",
                "evidence": "run only after node/signal validity and with bounded compute",
            },
        ]
    )
    lines = [
        "# Transcript-Derived Methodology Status",
        "",
        "This document records the requirements extracted from the supplied meeting transcript.",
        "It is a tissue-audit contract, not evidence that blocked spatial analyses were performed.",
        "",
        _markdown_table(rows),
        "",
        "## Non-negotiable guardrails",
        "",
        "- Do not divide folded cortex into unconstrained cubic volume bins.",
        "- Stop neighbourhood growth at invalid/background mask boundaries.",
        "- Atlas membership does not prove functional homogeneity.",
        "- Mean, PCA, ICA, and correlation-selected regional signals are competing constructions.",
        "- PCA/ICA sign must be oriented and recorded before signed FC comparison.",
        "- Report ACF/PACF and AR removal before/after, not only the chosen final branch.",
        "- Treat global signal regression as paired sensitivity: with and without.",
        "- Correct group-level multiple comparisons and keep subject-level individuality visible.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    path: Path,
    *,
    root_dir: Path,
    inventory: pd.DataFrame,
    qc: pd.DataFrame,
    comparisons: pd.DataFrame,
    tissue_correlations: pd.DataFrame,
) -> None:
    valid = inventory[inventory["status"].eq("ok")]
    group_counts = valid.groupby("group")["subject_id"].nunique().to_dict()
    count_rows = (
        qc.groupby(["tissue", "group"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_voxels=("n_voxels", "median"),
            median_active_voxels=("active_voxels", "median"),
            median_zero_fraction=("zero_fraction", "median"),
            median_tissue_mean_std=("active_mean_std", "median"),
            median_acf_lag1=("active_mean_acf_lag1", "median"),
        )
        .round(6)
    )
    key_features = {
        "n_voxels",
        "zero_fraction",
        "active_mean_std",
        "active_mean_acf_lag1",
        "active_mean_ar1_residual_acf_lag1",
    }
    comparison_view = comparisons[comparisons["feature"].isin(key_features)].copy()
    comparison_view = comparison_view.sort_values(["q_value_FDR", "tissue", "feature"])
    corr_summary = (
        tissue_correlations.groupby("group")[
            ["corr_GM_WM", "corr_GM_CSF", "corr_WM_CSF"]
        ]
        .median()
        .reset_index()
        .round(6)
        if not tissue_correlations.empty
        else pd.DataFrame()
    )
    xyz_complete = int(
        valid[[f"{tissue}_xyz_matches_data" for tissue in TISSUES]]
        .fillna(False)
        .all(axis=1)
        .sum()
    )
    lines = [
        "# Independent GM/WM/CSF Tissue Data Audit",
        "",
        "This report is intentionally separate from the whole-brain/ROI audit.",
        "It contains no ROI connectivity or atlas-level group conclusions.",
        "",
        "## Inputs",
        "",
        f"- Tissue HDF5 root: `{root_dir}`",
        f"- Files discovered: {len(inventory)}",
        f"- Valid files: {len(valid)}",
        f"- HC subjects: {int(group_counts.get('HC', 0))}",
        f"- SZ subjects: {int(group_counts.get('SZ', 0))}",
        f"- Files with complete matching GM/WM/CSF xyz: {xyz_complete}/{len(valid)}",
        "",
        "## Tissue QC summary",
        "",
        _markdown_table(count_rows),
        "",
        "## Exploratory HC vs SZ tissue-feature tests",
        "",
        _markdown_table(comparison_view.round(6)),
        "",
        "## Tissue-mean correlations",
        "",
        _markdown_table(corr_summary),
        "",
        "## Interpretation guardrails",
        "",
        "- Tissue voxel counts are QC/confound candidates, not automatic morphometric findings.",
        "- Missing xyz blocks voxel-to-atlas mapping, spatial neighbourhood growth, and ROI homogeneity.",
        "- `TR=600` in some files is treated as ambiguous metadata, not repetition time in seconds.",
        "- Global, WM, and CSF signals are exported for paired sensitivity analyses.",
        "- This audit does not diagnose schizophrenia and does not replace motion/site/demographic covariates.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_fmri_tissue_audit(
    root_dir: str | Path,
    output_dir: str | Path,
    *,
    max_lag: int = 20,
    block_rows: int = 8192,
    constant_eps: float = 1e-12,
) -> FmriTissueAuditResult:
    """Run and persist an independent streaming tissue audit."""
    root = Path(root_dir)
    output = Path(output_dir)
    inventory_dir = output / "inventories"
    qc_dir = output / "qc"
    temporal_dir = output / "temporal"
    comparison_dir = output / "group_comparison"
    reports_dir = output / "reports"
    for directory in (
        inventory_dir,
        qc_dir,
        temporal_dir,
        comparison_dir,
        reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    inventory = scan_tissue_h5_inventory(root)
    inventory.to_csv(
        inventory_dir / "tissue_hdf5_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qc_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    acf_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for record in inventory[inventory["status"].eq("ok")].to_dict("records"):
        path = Path(str(record["file_path"]))
        group = str(record["group"])
        subject_id = str(record["subject_id"])
        try:
            with h5py.File(path, "r") as h5:
                active_means: dict[str, np.ndarray] = {}
                active_sums: dict[str, np.ndarray] = {}
                active_counts: dict[str, int] = {}
                for tissue in TISSUES:
                    summary, mean_all, mean_active, active_count = _audit_dataset(
                        h5[f"{tissue}/data"],
                        block_rows=block_rows,
                        constant_eps=constant_eps,
                    )
                    active_means[tissue] = mean_active
                    active_sums[tissue] = mean_active * active_count
                    active_counts[tissue] = active_count
                    all_features = _temporal_features(mean_all)
                    active_features = _temporal_features(mean_active)
                    qc_rows.append(
                        {
                            "group": group,
                            "subject_id": subject_id,
                            "file_name": path.name,
                            "tissue": tissue,
                            **summary,
                            **{f"all_mean_{k}": v for k, v in all_features.items()},
                            **{
                                f"active_mean_{k}": v
                                for k, v in active_features.items()
                            },
                        }
                    )
                    residuals, _phi = _ar1_residualize(mean_active)
                    acf_before, pacf_before = _acf_pacf(mean_active, max_lag)
                    acf_after, pacf_after = _acf_pacf(residuals, max_lag)
                    for lag in range(max(len(acf_before), len(acf_after))):
                        acf_rows.append(
                            {
                                "group": group,
                                "subject_id": subject_id,
                                "tissue": tissue,
                                "lag": lag,
                                "acf_before": (
                                    _safe_float(acf_before[lag])
                                    if lag < len(acf_before)
                                    else float("nan")
                                ),
                                "pacf_before": (
                                    _safe_float(pacf_before[lag])
                                    if lag < len(pacf_before)
                                    else float("nan")
                                ),
                                "acf_after_AR1": (
                                    _safe_float(acf_after[lag])
                                    if lag < len(acf_after)
                                    else float("nan")
                                ),
                                "pacf_after_AR1": (
                                    _safe_float(pacf_after[lag])
                                    if lag < len(pacf_after)
                                    else float("nan")
                                ),
                            }
                        )
                total_active = sum(active_counts.values())
                global_signal = (
                    sum(active_sums.values()) / total_active
                    if total_active > 0
                    else np.full_like(next(iter(active_means.values())), np.nan)
                )
                signals = {**active_means, "GLOBAL": global_signal}
                for tissue, values in signals.items():
                    features = _temporal_features(values)
                    temporal_rows.append(
                        {
                            "group": group,
                            "subject_id": subject_id,
                            "tissue": tissue,
                            **features,
                        }
                    )
                for time_index in range(len(global_signal)):
                    timeseries_rows.append(
                        {
                            "group": group,
                            "subject_id": subject_id,
                            "time_index": time_index,
                            "GM": active_means["GM"][time_index],
                            "WM": active_means["WM"][time_index],
                            "CSF": active_means["CSF"][time_index],
                            "GLOBAL": global_signal[time_index],
                        }
                    )
                correlation_rows.append(
                    {
                        "group": group,
                        "subject_id": subject_id,
                        "corr_GM_WM": _safe_float(
                            np.corrcoef(active_means["GM"], active_means["WM"])[0, 1]
                        ),
                        "corr_GM_CSF": _safe_float(
                            np.corrcoef(active_means["GM"], active_means["CSF"])[0, 1]
                        ),
                        "corr_WM_CSF": _safe_float(
                            np.corrcoef(active_means["WM"], active_means["CSF"])[0, 1]
                        ),
                        "corr_GM_GLOBAL": _safe_float(
                            np.corrcoef(active_means["GM"], global_signal)[0, 1]
                        ),
                        "corr_WM_GLOBAL": _safe_float(
                            np.corrcoef(active_means["WM"], global_signal)[0, 1]
                        ),
                        "corr_CSF_GLOBAL": _safe_float(
                            np.corrcoef(active_means["CSF"], global_signal)[0, 1]
                        ),
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "group": group,
                    "subject_id": subject_id,
                    "file_name": path.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    qc = pd.DataFrame(qc_rows)
    temporal = pd.DataFrame(temporal_rows)
    acf_pacf = pd.DataFrame(acf_rows)
    timeseries = pd.DataFrame(timeseries_rows)
    tissue_correlations = pd.DataFrame(correlation_rows)
    failure_table = pd.DataFrame(
        failures,
        columns=["group", "subject_id", "file_name", "error"],
    )

    qc.to_csv(qc_dir / "tissue_dataset_qc.csv", index=False, encoding="utf-8-sig")
    failure_table.to_csv(
        qc_dir / "tissue_audit_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not qc.empty:
        counts = qc.pivot(
            index=["group", "subject_id"],
            columns="tissue",
            values="n_voxels",
        ).reset_index()
        counts["TOTAL_TISSUE"] = counts[list(TISSUES)].sum(axis=1)
        for tissue in TISSUES:
            counts[f"{tissue}_fraction"] = counts[tissue] / counts["TOTAL_TISSUE"]
        counts.to_csv(
            qc_dir / "tissue_voxel_counts_wide.csv",
            index=False,
            encoding="utf-8-sig",
        )

    temporal.to_csv(
        temporal_dir / "tissue_mean_temporal_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    acf_pacf.to_csv(
        temporal_dir / "tissue_mean_acf_pacf.csv",
        index=False,
        encoding="utf-8-sig",
    )
    timeseries.to_csv(
        temporal_dir / "tissue_mean_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tissue_correlations.to_csv(
        temporal_dir / "tissue_mean_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparisons = _group_comparison(
        qc,
        id_columns={"group", "subject_id", "file_name", "tissue"},
    )
    comparisons.to_csv(
        comparison_dir / "tissue_feature_group_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _write_report(
        reports_dir / "tissue_audit_report.md",
        root_dir=root,
        inventory=inventory,
        qc=qc,
        comparisons=comparisons,
        tissue_correlations=tissue_correlations,
    )
    _write_methodology_status(
        reports_dir / "transcript_methodology_status.md",
        inventory=inventory,
    )

    valid = inventory[inventory["status"].eq("ok")]
    xyz_complete_files = int(
        valid[[f"{tissue}_xyz_matches_data" for tissue in TISSUES]]
        .fillna(False)
        .all(axis=1)
        .sum()
    )
    manifest = {
        "audit_type": TISSUE_AUDIT_TYPE,
        "layout_version": TISSUE_AUDIT_LAYOUT_VERSION,
        "source_root": str(root),
        "output_root": str(output),
        "independent_from": "whole_brain_roi_audit",
        "contains_roi_connectivity": False,
        "contains_voxel_spatial_analysis": xyz_complete_files == len(valid) and len(valid) > 0,
        "xyz_complete_files": xyz_complete_files,
        "files_valid": int(len(valid)),
        "subjects_by_group": {
            str(key): int(value)
            for key, value in valid.groupby("group")["subject_id"].nunique().items()
        },
        "artifacts": {
            "inventory": "inventories/tissue_hdf5_inventory.csv",
            "dataset_qc": "qc/tissue_dataset_qc.csv",
            "temporal_qc": "temporal/tissue_mean_temporal_qc.csv",
            "acf_pacf": "temporal/tissue_mean_acf_pacf.csv",
            "mean_timeseries": "temporal/tissue_mean_timeseries.csv",
            "group_comparison": "group_comparison/tissue_feature_group_comparison.csv",
            "report": "reports/tissue_audit_report.md",
            "methodology_status": "reports/transcript_methodology_status.md",
        },
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return FmriTissueAuditResult(
        output_dir=str(output),
        files_discovered=int(len(inventory)),
        files_valid=int(len(valid)),
        subjects_hc=int(valid[valid["group"].eq("HC")]["subject_id"].nunique()),
        subjects_sz=int(valid[valid["group"].eq("SZ")]["subject_id"].nunique()),
        failures=int(len(failure_table)),
        xyz_complete_files=xyz_complete_files,
        spatial_analysis_available=bool(
            len(valid) > 0 and xyz_complete_files == len(valid)
        ),
    )


__all__ = [
    "FmriTissueAuditResult",
    "TISSUES",
    "TISSUE_AUDIT_LAYOUT_VERSION",
    "TISSUE_AUDIT_TYPE",
    "run_fmri_tissue_audit",
    "scan_tissue_h5_inventory",
]
