"""Deep AR(p) and Hurst diagnostic for Stage 2 correlation_full."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import signal, stats

from neweds.core.fmri_roi_audit import load_valid_subjects, scan_inventory
from neweds.core.group_pipeline import _fdr_bh
from neweds.core.metric_runner import compute_metric
from scripts.run_fmri_stage2_sanity import (
    _ar_residualize_array,
    _markdown_table,
    _select_roi_columns,
    _write_csv,
)
from scripts.run_fmri_stage2_exploratory_full_grid import _edge_records


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _split_int_csv(text: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    return vals or (1, 2, 3, 4)


def _ar_coefficients(values: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() <= order + 5 or np.nanstd(x[finite]) <= 1e-12:
        return np.full(order, np.nan)
    y_full = pd.Series(x).interpolate(limit_direction="both").fillna(float(np.nanmean(x[finite]))).to_numpy()
    y = y_full[order:]
    design = np.column_stack([y_full[order - lag : -lag] for lag in range(1, order + 1)])
    design = np.column_stack([np.ones(design.shape[0]), design])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return np.asarray(beta[1:], dtype=float)
    except np.linalg.LinAlgError:
        return np.full(order, np.nan)


def _hurst_rs(values: np.ndarray, min_chunk: int = 10) -> float:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 64 or np.nanstd(x[finite]) <= 1e-12:
        return float("nan")
    y = pd.Series(x).interpolate(limit_direction="both").fillna(float(np.nanmean(x[finite]))).to_numpy()
    n = len(y)
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk), np.log10(max(min_chunk + 1, n // 2)), 10)).astype(int))
    rs_vals: list[float] = []
    used_sizes: list[int] = []
    for size in sizes:
        if size < min_chunk or n // size < 2:
            continue
        chunks = y[: (n // size) * size].reshape(n // size, size)
        chunk_rs = []
        for chunk in chunks:
            z = chunk - np.mean(chunk)
            cumulative = np.cumsum(z)
            r = float(np.max(cumulative) - np.min(cumulative))
            s = float(np.std(chunk, ddof=1))
            if s > 1e-12 and r > 0:
                chunk_rs.append(r / s)
        if chunk_rs:
            rs_vals.append(float(np.mean(chunk_rs)))
            used_sizes.append(int(size))
    if len(rs_vals) < 3:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(used_sizes), np.log(rs_vals), 1)
    return _safe_float(slope)


def _compare_edge_groups(edge_values: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (edge, roi_i, roi_j), group in edge_values.groupby(["edge", "roi_i", "roi_j"], sort=False):
        hc = pd.to_numeric(group[group["group"].eq("HC")]["value"], errors="coerce").to_numpy(float)
        sz = pd.to_numeric(group[group["group"].eq("SZ")]["value"], errors="coerce").to_numpy(float)
        hc = hc[np.isfinite(hc)]
        sz = sz[np.isfinite(sz)]
        if hc.size >= 2 and sz.size >= 2:
            try:
                u_stat, p_value = stats.mannwhitneyu(sz, hc, alternative="two-sided")
                effect = 2.0 * float(u_stat) / float(hc.size * sz.size) - 1.0
            except ValueError:
                u_stat, p_value, effect = float("nan"), float("nan"), float("nan")
        else:
            u_stat, p_value, effect = float("nan"), float("nan"), float("nan")
        rows.append(
            {
                "edge": edge,
                "roi_i": int(roi_i),
                "roi_j": int(roi_j),
                "HC_value": _safe_float(np.nanmedian(hc)) if hc.size else float("nan"),
                "SZ_value": _safe_float(np.nanmedian(sz)) if sz.size else float("nan"),
                "group_delta": (_safe_float(np.nanmedian(sz)) - _safe_float(np.nanmedian(hc))) if hc.size and sz.size else float("nan"),
                "u_stat": u_stat,
                "effect_size": effect,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    q, sig = _fdr_bh(out["p_value"].fillna(1.0).to_numpy(float), alpha=alpha)
    out["q_value_FDR"] = q
    out["significant"] = sig
    return out


def _compute_ar_branch_edges(subjects, primary_cols: list[str], order: int, alpha: float, region_map: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        arr = subject.data_time_roi[primary_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        residualized = pd.DataFrame(
            {col: _ar_residualize_array(arr[col].to_numpy(dtype=float), order=order) for col in arr.columns},
            index=arr.index,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = compute_metric(residualized.reset_index(drop=True), "correlation_full", lag=1)
        for rec in _edge_records(matrix, primary_cols, directed=False, pvalue_based=False):
            rows.append({**rec, "group": subject.group, "subject_id": subject.subject_id})
    compared = _compare_edge_groups(pd.DataFrame(rows), alpha=alpha)
    compared["branch"] = f"AR{order}_residualized"
    compared["ar_order"] = int(order)
    compared["region_i"] = compared["roi_i"].map(region_map)
    compared["region_j"] = compared["roi_j"].map(region_map)
    return compared


def _roi_temporal_table(subjects, primary_cols: list[str], decisions: pd.DataFrame, orders: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_map = decisions.set_index("roi_index_1based")["region_name"].to_dict()
    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for col in primary_cols:
            roi = int(col.replace("roi_", "")) + 1
            x = subject.data_time_roi[col].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "group": subject.group,
                "subject_id": subject.subject_id,
                "roi": roi,
                "region": region_map.get(roi, ""),
                "hurst_rs": _hurst_rs(x),
            }
            for order in orders:
                coefs = _ar_coefficients(x, order=order)
                row[f"AR{order}_phi1"] = _safe_float(coefs[0]) if coefs.size else float("nan")
                row[f"AR{order}_phi_sum"] = _safe_float(np.nansum(coefs)) if np.isfinite(coefs).any() else float("nan")
            subject_rows.append(row)
    per_subject = pd.DataFrame(subject_rows)
    summary_rows: list[dict[str, Any]] = []
    features = ["hurst_rs"] + [f"AR{order}_phi1" for order in orders] + [f"AR{order}_phi_sum" for order in orders]
    for (roi, region), group in per_subject.groupby(["roi", "region"], sort=True):
        out: dict[str, Any] = {"roi": int(roi), "region": region}
        for feature in features:
            hc = group[group["group"].eq("HC")][feature].dropna().to_numpy(float)
            sz = group[group["group"].eq("SZ")][feature].dropna().to_numpy(float)
            out[f"{feature}_HC_median"] = _safe_float(np.nanmedian(hc)) if hc.size else float("nan")
            out[f"{feature}_SZ_median"] = _safe_float(np.nanmedian(sz)) if sz.size else float("nan")
            out[f"{feature}_group_delta"] = (
                _safe_float(np.nanmedian(sz)) - _safe_float(np.nanmedian(hc)) if hc.size and sz.size else float("nan")
            )
            if hc.size >= 2 and sz.size >= 2:
                try:
                    _u, p_value = stats.mannwhitneyu(sz, hc, alternative="two-sided")
                except ValueError:
                    p_value = float("nan")
            else:
                p_value = float("nan")
            out[f"{feature}_p_value"] = p_value
        summary_rows.append(out)
    return per_subject, pd.DataFrame(summary_rows)


def _zone_label(region_i: str, region_j: str) -> str:
    text = f"{region_i} {region_j}".lower()
    labels = []
    if any(x in text for x in ("front", "precentral", "supp_motor", "paracentral", "rectus")):
        labels.append("frontal_motor")
    if "thalam" in text:
        labels.append("thalamus")
    if any(x in text for x in ("cerebel", "vermis")):
        labels.append("cerebellum")
    if any(x in text for x in ("temporal", "heschl")):
        labels.append("temporal")
    return ",".join(labels) if labels else "other"


def run_deep_ar_hurst(
    *,
    hc_dir: Path,
    sz_dir: Path,
    decision_dir: Path,
    baseline_stage2_dir: Path,
    output_dir: Path,
    atlas: str,
    orders: tuple[int, ...],
    alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(decision_dir / "roi_decision_layer_v2.csv")
    decisions = decisions[decisions["atlas"].eq(atlas)].copy()
    primary_cols = _select_roi_columns(decisions)
    region_map = decisions.set_index("roi_index_1based")["region_name"].to_dict()
    subjects = load_valid_subjects(scan_inventory(hc_dir, sz_dir, atlas_filter=atlas))

    baseline_edges = pd.read_csv(baseline_stage2_dir / "stage2_correlation_full_all_primary_roi_edges.csv")
    baseline = baseline_edges[baseline_edges["branch"].eq("baseline")].copy()
    baseline = baseline.rename(
        columns={
            "group_delta": "edge_delta_baseline",
            "p_value": "p_baseline",
            "q_value_FDR": "q_baseline",
            "significant": "baseline_FDR",
        }
    )
    branch_parts = []
    diagnostic_parts = []
    for order in orders:
        ar_edges = _compute_ar_branch_edges(subjects, primary_cols, order, alpha, region_map)
        branch_parts.append(ar_edges)
        ar = ar_edges.rename(
            columns={
                "group_delta": f"edge_delta_AR{order}",
                "p_value": f"p_AR{order}",
                "q_value_FDR": f"q_AR{order}",
                "significant": f"AR{order}_FDR",
            }
        )
        merged = baseline[
            ["edge", "roi_i", "roi_j", "region_i", "region_j", "edge_delta_baseline", "p_baseline", "q_baseline", "baseline_FDR"]
        ].merge(
            ar[["edge", f"edge_delta_AR{order}", f"p_AR{order}", f"q_AR{order}", f"AR{order}_FDR"]],
            on="edge",
            how="inner",
        )
        merged["ar_order"] = int(order)
        merged["delta_shrinkage"] = merged[f"edge_delta_AR{order}"].abs() / merged["edge_delta_baseline"].abs().replace(0, np.nan)
        merged["baseline_FDR_AR_p_lt_0_05"] = merged["baseline_FDR"].astype(bool) & merged[f"p_AR{order}"].lt(alpha)
        merged["zone_label"] = [
            _zone_label(a, b) for a, b in zip(merged["region_i"].astype(str), merged["region_j"].astype(str))
        ]
        diagnostic_parts.append(merged)
    branch_edges = pd.concat(branch_parts, ignore_index=True)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    _write_csv(branch_edges, output_dir / "stage2_deep_ar_branch_edges.csv")
    _write_csv(diagnostics, output_dir / "stage2_deep_ar_edge_diagnostic.csv")

    per_subject_temporal, roi_temporal = _roi_temporal_table(subjects, primary_cols, decisions, orders)
    _write_csv(per_subject_temporal, output_dir / "stage2_deep_ar_hurst_subject_roi.csv")
    _write_csv(roi_temporal, output_dir / "stage2_deep_ar_hurst_roi_group_differences.csv")

    summary_rows = []
    for order in orders:
        part = diagnostics[diagnostics["ar_order"].eq(order)].copy()
        base_fdr = part[part["baseline_FDR"].astype(bool)]
        summary_rows.append(
            {
                "ar_order": int(order),
                "baseline_FDR_edges": int(part["baseline_FDR"].astype(bool).sum()),
                "AR_FDR_edges": int(part[f"AR{order}_FDR"].astype(bool).sum()),
                "baseline_FDR_edges_with_AR_p_lt_0_05": int(base_fdr[f"p_AR{order}"].lt(alpha).sum()),
                "any_AR_p_lt_0_05_edges": int(part[f"p_AR{order}"].lt(alpha).sum()),
                "min_AR_p": _safe_float(part[f"p_AR{order}"].min()),
                "min_AR_q": _safe_float(part[f"q_AR{order}"].min()),
                "median_delta_shrinkage_baseline_FDR": _safe_float(base_fdr["delta_shrinkage"].median()) if len(base_fdr) else float("nan"),
                "fraction_baseline_FDR_SZ_lt_HC_after_AR": _safe_float((base_fdr[f"edge_delta_AR{order}"] < 0).mean()) if len(base_fdr) else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    _write_csv(summary, output_dir / "stage2_deep_ar_order_summary.csv")

    zone_loss = (
        diagnostics[diagnostics["baseline_FDR"].astype(bool)]
        .groupby(["ar_order", "zone_label"], as_index=False)
        .agg(
            baseline_FDR_edges=("edge", "count"),
            retained_AR_FDR=("edge", lambda s: 0),
            retained_AR_p_lt_0_05=("baseline_FDR_AR_p_lt_0_05", lambda s: int(s.astype(bool).sum())),
            median_delta_shrinkage=("delta_shrinkage", "median"),
        )
    )
    for idx, row in zone_loss.iterrows():
        order = int(row["ar_order"])
        subset = diagnostics[
            diagnostics["baseline_FDR"].astype(bool)
            & diagnostics["ar_order"].eq(order)
            & diagnostics["zone_label"].eq(row["zone_label"])
        ]
        zone_loss.loc[idx, "retained_AR_FDR"] = int(subset[f"AR{order}_FDR"].astype(bool).sum())
    _write_csv(zone_loss, output_dir / "stage2_deep_ar_zone_loss.csv")

    hurst_cols = ["roi", "region", "hurst_rs_HC_median", "hurst_rs_SZ_median", "hurst_rs_group_delta", "hurst_rs_p_value"]
    hurst_top = roi_temporal.reindex(columns=hurst_cols).copy()
    hurst_sig = int(hurst_top["hurst_rs_p_value"].lt(alpha).sum())

    lines = [
        "# Stage 2 Deep AR(p) and Hurst Diagnostic",
        "",
        "This diagnostic extends AR1 sensitivity to AR orders 1..4 and adds ROI-level Hurst exponent screening.",
        "",
        "## Scope",
        "",
        "- Metric: `correlation_full`",
        "- ROI: Stage 1.5 v2 primary ROI, all 159",
        "- Subjects: all HC/SZ subjects",
        "- Window: full",
        "- AR residualization orders: " + ", ".join(f"AR{order}" for order in orders),
        "- Hurst estimator: rescaled-range slope over chunk sizes",
        "",
        "## AR Order Summary",
        "",
        _markdown_table(summary.round(5)),
        "",
        "## Zone Loss By AR Order",
        "",
        _markdown_table(zone_loss.round(5)),
        "",
        "## Hurst Group Differences",
        "",
        f"- ROI with Hurst HC/SZ p<0.05: {hurst_sig} / {len(hurst_top)}",
        "",
        "Top absolute Hurst HC/SZ differences:",
        "",
        _markdown_table(
            hurst_top.assign(abs_hurst_delta=hurst_top["hurst_rs_group_delta"].abs())
            .sort_values("abs_hurst_delta", ascending=False)
            .head(30)
            .round(5)
        ),
        "",
        "## Interpretation Guardrail",
        "",
        "If AR2..AR4 show the same pattern as AR1, the baseline FC pattern is broadly AR-sensitive, not just sensitive to one specific AR1 implementation. Hurst results should be treated as temporal-phenotype diagnostics, not as evidence of neural mechanism by themselves.",
    ]
    (output_dir / "stage2_deep_ar_hurst_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--baseline-stage2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas", default="AAL3")
    parser.add_argument("--orders", default="1,2,3,4")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_deep_ar_hurst(
        hc_dir=Path(args.hc_dir),
        sz_dir=Path(args.sz_dir),
        decision_dir=Path(args.decision_dir),
        baseline_stage2_dir=Path(args.baseline_stage2_dir),
        output_dir=Path(args.output_dir),
        atlas=str(args.atlas),
        orders=_split_int_csv(args.orders),
        alpha=float(args.alpha),
    )


if __name__ == "__main__":
    main()
