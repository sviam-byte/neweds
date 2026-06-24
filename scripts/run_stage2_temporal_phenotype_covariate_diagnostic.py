"""Temporal phenotype diagnostic with AR, Hurst, entropy, spectra, and covariate audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import signal, stats

from neweds.core.fmri_roi_audit import load_valid_subjects, scan_inventory
from scripts.run_fmri_stage2_sanity import _markdown_table, _select_roi_columns, _write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _interp(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    return pd.Series(x).interpolate(limit_direction="both").fillna(float(np.nanmean(x[finite]))).to_numpy(float)


def _zscore(values: np.ndarray) -> np.ndarray:
    x = _interp(values)
    s = float(np.std(x))
    return x - float(np.mean(x)) if s <= 1e-12 else (x - float(np.mean(x))) / s


def _ar_coefficients(values: np.ndarray, order: int) -> np.ndarray:
    y_full = _interp(values)
    if len(y_full) <= order + 5 or np.std(y_full) <= 1e-12:
        return np.full(order, np.nan)
    y = y_full[order:]
    design = np.column_stack([y_full[order - lag : -lag] for lag in range(1, order + 1)])
    design = np.column_stack([np.ones(design.shape[0]), design])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return np.asarray(beta[1:], dtype=float)
    except np.linalg.LinAlgError:
        return np.full(order, np.nan)


def _hurst_rs(values: np.ndarray, min_chunk: int = 10) -> float:
    y = _interp(values)
    if len(y) < 64 or np.std(y) <= 1e-12:
        return float("nan")
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk), np.log10(max(min_chunk + 1, len(y) // 2)), 10)).astype(int))
    xs, ys = [], []
    for size in sizes:
        if size < min_chunk or len(y) // size < 2:
            continue
        chunks = y[: (len(y) // size) * size].reshape(len(y) // size, size)
        vals = []
        for chunk in chunks:
            z = chunk - np.mean(chunk)
            cumulative = np.cumsum(z)
            r = float(np.max(cumulative) - np.min(cumulative))
            s = float(np.std(chunk, ddof=1))
            if r > 0 and s > 1e-12:
                vals.append(r / s)
        if vals:
            xs.append(size)
            ys.append(float(np.mean(vals)))
    if len(xs) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(xs), np.log(ys), 1)
    return _safe_float(slope)


def _hurst_dfa(values: np.ndarray, min_window: int = 8) -> float:
    y = _zscore(values)
    if len(y) < 64:
        return float("nan")
    profile = np.cumsum(y - np.mean(y))
    sizes = np.unique(np.floor(np.logspace(np.log10(min_window), np.log10(max(min_window + 1, len(y) // 4)), 10)).astype(int))
    xs, fs = [], []
    for size in sizes:
        if size < min_window or len(profile) // size < 2:
            continue
        chunks = profile[: (len(profile) // size) * size].reshape(len(profile) // size, size)
        rms = []
        t = np.arange(size)
        for chunk in chunks:
            coef = np.polyfit(t, chunk, 1)
            resid = chunk - np.polyval(coef, t)
            rms.append(float(np.sqrt(np.mean(resid**2))))
        val = float(np.mean(rms))
        if val > 0:
            xs.append(size)
            fs.append(val)
    if len(xs) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(xs), np.log(fs), 1)
    return _safe_float(slope)


def _hurst_wavelet_proxy(values: np.ndarray) -> float:
    y = _zscore(values)
    vars_, scales = [], []
    current = y.copy()
    scale = 1
    while len(current) >= 16:
        if len(current) % 2:
            current = current[:-1]
        approx = (current[0::2] + current[1::2]) / 2.0
        detail = (current[0::2] - current[1::2]) / 2.0
        v = float(np.var(detail))
        if v > 1e-12:
            vars_.append(v)
            scales.append(scale)
        current = approx
        scale *= 2
    if len(vars_) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log2(scales), np.log2(vars_), 1)
    # For fractional Gaussian noise-like signals, wavelet detail variance slope
    # is approximately 2H - 1. This is a rough screening proxy.
    return _safe_float((slope + 1.0) / 2.0)


def _spectral_features(values: np.ndarray) -> dict[str, float]:
    y = _zscore(values)
    if len(y) < 64 or np.std(y) <= 1e-12:
        return {"spectral_slope": np.nan, "alff": np.nan, "falff": np.nan}
    freqs, power = signal.welch(y, nperseg=min(256, len(y)))
    mask = (freqs > 0) & np.isfinite(power) & (power > 0)
    if mask.sum() >= 5:
        slope, _ = np.polyfit(np.log(freqs[mask]), np.log(power[mask]), 1)
    else:
        slope = np.nan
    # TR is unknown, so this is a normalized-frequency proxy: low band = 0.01..0.08 cycles/sample.
    low = (freqs >= 0.01) & (freqs <= 0.08)
    total = freqs > 0
    alff = float(np.sqrt(np.trapezoid(power[low], freqs[low]))) if low.any() else np.nan
    total_amp = float(np.sqrt(np.trapezoid(power[total], freqs[total]))) if total.any() else np.nan
    return {
        "spectral_slope": _safe_float(slope),
        "alff": _safe_float(alff),
        "falff": _safe_float(alff / total_amp) if total_amp and np.isfinite(total_amp) else np.nan,
    }


def _sample_entropy_approx(values: np.ndarray, m: int = 2, max_pairs: int = 5000) -> float:
    y = _zscore(values)
    n = len(y)
    if n <= m + 3:
        return float("nan")
    r = 0.2 * float(np.std(y))
    if r <= 1e-12:
        return float("nan")
    rng = np.random.default_rng(12345)
    max_i = n - m - 1
    pairs = rng.integers(0, max_i, size=(max_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return float("nan")
    cm = 0
    cm1 = 0
    for i, j in pairs:
        if np.max(np.abs(y[i : i + m] - y[j : j + m])) < r:
            cm += 1
            if abs(y[i + m] - y[j + m]) < r:
                cm1 += 1
    if cm == 0 or cm1 == 0:
        return float("nan")
    return _safe_float(-np.log(cm1 / cm))


def _coarse_grain(values: np.ndarray, scale: int) -> np.ndarray:
    y = _zscore(values)
    if scale <= 1:
        return y
    n = len(y) // scale
    return y[: n * scale].reshape(n, scale).mean(axis=1) if n > 0 else y


def _group_test(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    hc = df[df["group"].eq("HC")][feature].dropna().to_numpy(float)
    sz = df[df["group"].eq("SZ")][feature].dropna().to_numpy(float)
    if hc.size >= 2 and sz.size >= 2:
        try:
            u, p = stats.mannwhitneyu(sz, hc, alternative="two-sided")
            effect = 2.0 * float(u) / float(hc.size * sz.size) - 1.0
        except ValueError:
            p, effect = np.nan, np.nan
    else:
        p, effect = np.nan, np.nan
    return {
        "feature": feature,
        "HC_median": _safe_float(np.nanmedian(hc)) if hc.size else np.nan,
        "SZ_median": _safe_float(np.nanmedian(sz)) if sz.size else np.nan,
        "group_delta": (_safe_float(np.nanmedian(sz)) - _safe_float(np.nanmedian(hc))) if hc.size and sz.size else np.nan,
        "p_value": p,
        "effect_size": effect,
    }


def _fit_subject_model(subject: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    rows = []
    predictors = ["hurst_extreme10", "ar_phi1_median", *covariates]
    use = subject[["group_binary", *predictors]].dropna()
    row = {"model": "group ~ Hurst_extreme10 + AR_phi1_median + available_covariates", "n": len(use)}
    if len(use) < 10 or use["group_binary"].nunique() < 2:
        row["status"] = "insufficient_data"
        rows.append(row)
        return pd.DataFrame(rows)
    try:
        import statsmodels.api as sm

        X = sm.add_constant(use[predictors], has_constant="add")
        model = sm.Logit(use["group_binary"], X).fit(disp=False, maxiter=200)
        for name in X.columns:
            rows.append(
                {
                    **row,
                    "status": "fit",
                    "term": name,
                    "coef": _safe_float(model.params.get(name)),
                    "p_value": _safe_float(model.pvalues.get(name)),
                }
            )
    except Exception as exc:
        row["status"] = f"failed: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_diagnostic(
    *,
    hc_dir: Path,
    sz_dir: Path,
    decision_dir: Path,
    deep_ar_dir: Path,
    output_dir: Path,
    atlas: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(decision_dir / "roi_decision_layer_v2.csv")
    decisions = decisions[decisions["atlas"].eq(atlas)].copy()
    primary_cols = _select_roi_columns(decisions)
    region_map = decisions.set_index("roi_index_1based")["region_name"].to_dict()
    subjects = load_valid_subjects(scan_inventory(hc_dir, sz_dir, atlas_filter=atlas))

    rows = []
    subject_entropy_rows = []
    for subject in subjects:
        data = subject.data_time_roi[primary_cols]
        global_signal = data.mean(axis=1).to_numpy(float)
        global_feats = _spectral_features(global_signal)
        subject_entropy_rows.append(
            {
                "group": subject.group,
                "subject_id": subject.subject_id,
                "global_sampen": _sample_entropy_approx(global_signal, max_pairs=3000),
                "global_mse_scale2": _sample_entropy_approx(_coarse_grain(global_signal, 2), max_pairs=3000),
                "global_mse_scale4": _sample_entropy_approx(_coarse_grain(global_signal, 4), max_pairs=3000),
            }
        )
        for col in primary_cols:
            roi = int(col.replace("roi_", "")) + 1
            x = data[col].to_numpy(float)
            spec = _spectral_features(x)
            rows.append(
                {
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "roi": roi,
                    "region": region_map.get(roi, ""),
                    "AR1_phi1": _safe_float(_ar_coefficients(x, 1)[0]),
                    "AR2_phi_sum": _safe_float(np.nansum(_ar_coefficients(x, 2))),
                    "AR4_phi_sum": _safe_float(np.nansum(_ar_coefficients(x, 4))),
                    "hurst_rs": _hurst_rs(x),
                    "hurst_dfa": _hurst_dfa(x),
                    "hurst_wavelet_proxy": _hurst_wavelet_proxy(x),
                    "spectral_slope": spec["spectral_slope"],
                    "alff": spec["alff"],
                    "falff": spec["falff"],
                    "global_spectral_slope_proxy": global_feats["spectral_slope"],
                    "global_alff_proxy": global_feats["alff"],
                    "global_falff_proxy": global_feats["falff"],
                }
            )
    roi_subject = pd.DataFrame(rows)
    subject_entropy = pd.DataFrame(subject_entropy_rows)
    _write_csv(roi_subject, output_dir / "stage2_temporal_phenotype_subject_roi.csv")
    _write_csv(subject_entropy, output_dir / "stage2_temporal_entropy_subject_global.csv")
    hurst_extreme_threshold = _safe_float(roi_subject["hurst_dfa"].quantile(0.90))
    ar_extreme_threshold = _safe_float(roi_subject["AR1_phi1"].quantile(0.90))

    feature_cols = [
        "AR1_phi1",
        "AR2_phi_sum",
        "AR4_phi_sum",
        "hurst_rs",
        "hurst_dfa",
        "hurst_wavelet_proxy",
        "spectral_slope",
        "alff",
        "falff",
    ]
    group_rows = []
    for feature in feature_cols:
        group_rows.append(_group_test(roi_subject, feature))
    group_summary = pd.DataFrame(group_rows)
    _write_csv(group_summary, output_dir / "stage2_temporal_phenotype_group_tests_roi_level.csv")

    subject_rows = []
    for (group, subject_id), part in roi_subject.groupby(["group", "subject_id"], sort=False):
        h = part["hurst_dfa"]
        ar = part["AR1_phi1"]
        subject_rows.append(
            {
                "group": group,
                "group_binary": int(group == "SZ"),
                "subject_id": subject_id,
                "hurst_extreme10": _safe_float((h > hurst_extreme_threshold).mean()),
                "hurst_dfa_median": _safe_float(h.median()),
                "hurst_tail_top10_mean": _safe_float(h.nlargest(max(1, int(np.ceil(0.10 * len(h))))).mean()),
                "ar_phi1_median": _safe_float(ar.median()),
                "ar_extreme10": _safe_float((ar > ar_extreme_threshold).mean()),
                "ar_tail_top10_mean": _safe_float(ar.nlargest(max(1, int(np.ceil(0.10 * len(ar))))).mean()),
                "spectral_slope_median": _safe_float(part["spectral_slope"].median()),
                "alff_median": _safe_float(part["alff"].median()),
                "falff_median": _safe_float(part["falff"].median()),
                "global_spectral_slope_proxy": _safe_float(part["global_spectral_slope_proxy"].median()),
                "global_alff_proxy": _safe_float(part["global_alff_proxy"].median()),
                "global_falff_proxy": _safe_float(part["global_falff_proxy"].median()),
            }
        )
    subject = pd.DataFrame(subject_rows)
    subject = subject.merge(subject_entropy, on=["group", "subject_id"], how="left")
    _write_csv(subject, output_dir / "stage2_temporal_phenotype_subject_summary.csv")

    subject_tests = pd.DataFrame([_group_test(subject, col) for col in subject.columns if col not in {"group", "group_binary", "subject_id"}])
    _write_csv(subject_tests, output_dir / "stage2_temporal_phenotype_subject_group_tests.csv")

    covariate_audit = pd.DataFrame(
        [
            {"covariate": "motion", "status": "missing", "control_possible": False},
            {"covariate": "mean_FD", "status": "missing", "control_possible": False},
            {"covariate": "age", "status": "missing", "control_possible": False},
            {"covariate": "sex", "status": "missing", "control_possible": False},
            {"covariate": "site_scanner", "status": "missing", "control_possible": False},
            {"covariate": "ROI_size", "status": "missing_voxelwise_atlas_data", "control_possible": False},
            {"covariate": "ROI_SNR", "status": "time_series_proxy_only", "control_possible": "proxy_only"},
            {"covariate": "global_signal", "status": "ROI_mean_proxy_available", "control_possible": "proxy_only"},
            {"covariate": "WM_signal", "status": "missing", "control_possible": False},
            {"covariate": "CSF_signal", "status": "missing", "control_possible": False},
        ]
    )
    _write_csv(covariate_audit, output_dir / "stage2_temporal_covariate_availability.csv")

    model = _fit_subject_model(subject, covariates=[])
    _write_csv(model, output_dir / "stage2_subject_level_group_model.csv")

    edge = pd.read_csv(deep_ar_dir / "stage2_deep_ar_edge_diagnostic.csv")
    roi_summary = roi_subject.groupby("roi", as_index=False).agg(
        roi_hurst_dfa_median=("hurst_dfa", "median"),
        roi_hurst_rs_median=("hurst_rs", "median"),
        roi_ar_phi1_median=("AR1_phi1", "median"),
        roi_spectral_slope_median=("spectral_slope", "median"),
    )
    edge2 = edge.merge(roi_summary.add_suffix("_i"), left_on="roi_i", right_on="roi_i", how="left").merge(
        roi_summary.add_suffix("_j"), left_on="roi_j", right_on="roi_j", how="left"
    )
    edge2["endpoint_hurst_dfa_mean"] = edge2[["roi_hurst_dfa_median_i", "roi_hurst_dfa_median_j"]].mean(axis=1)
    edge2["endpoint_ar_phi1_mean"] = edge2[["roi_ar_phi1_median_i", "roi_ar_phi1_median_j"]].mean(axis=1)
    corr_rows = []
    for order, part in edge2.groupby("ar_order"):
        for feature in ["endpoint_hurst_dfa_mean", "endpoint_ar_phi1_mean"]:
            for target in ["edge_delta_baseline", "delta_shrinkage"]:
                mask = np.isfinite(part[feature]) & np.isfinite(part[target])
                r = float(np.corrcoef(part.loc[mask, feature], part.loc[mask, target])[0, 1]) if mask.sum() >= 3 else np.nan
                corr_rows.append({"ar_order": int(order), "feature": feature, "target": target, "pearson_r": _safe_float(r), "n_edges": int(mask.sum())})
    edge_corr = pd.DataFrame(corr_rows)
    _write_csv(edge_corr, output_dir / "stage2_temporal_phenotype_edge_shrinkage_links.csv")

    lines = [
        "# Stage 2 Temporal Phenotype Covariate Diagnostic",
        "",
        "## Covariate Availability",
        "",
        _markdown_table(covariate_audit),
        "",
        "## ROI-Level Temporal Phenotype Group Tests",
        "",
        _markdown_table(group_summary.round(5)),
        "",
        "## Subject-Level Group Tests",
        "",
        _markdown_table(subject_tests.round(5)),
        "",
        "## Subject-Level Model",
        "",
        "Requested model: `group ~ Hurst_extreme10 + AR_phi1_median + motion + covariates`.",
        "Only temporal predictors are available locally; motion/FD/age/sex/site/WM/CSF/ROI-size are missing.",
        "",
        _markdown_table(model.round(5)),
        "",
        "## Does Temporal Phenotype Explain Baseline FC / AR Shrinkage?",
        "",
        _markdown_table(edge_corr.round(5)),
        "",
        "## Interpretation",
        "",
        "This run can test temporal phenotype links and proxy global signal summaries, but it cannot perform the requested covariate-controlled inference until motion/FD, demographics, site/scanner, tissue signals, and voxelwise ROI metadata are supplied.",
    ]
    (output_dir / "stage2_temporal_phenotype_covariate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--deep-ar-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas", default="AAL3")
    args = parser.parse_args()
    run_diagnostic(
        hc_dir=Path(args.hc_dir),
        sz_dir=Path(args.sz_dir),
        decision_dir=Path(args.decision_dir),
        deep_ar_dir=Path(args.deep_ar_dir),
        output_dir=Path(args.output_dir),
        atlas=str(args.atlas),
    )


if __name__ == "__main__":
    main()
