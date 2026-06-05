"""Stage 2 AR1 sensitivity diagnostic for correlation_full all-primary ROI."""

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
from scipy import stats

from neweds.core.fmri_roi_audit import load_valid_subjects, scan_inventory
from scripts.run_fmri_stage2_sanity import _markdown_table, _select_roi_columns, _write_csv


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _ar1_coef(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        return float("nan")
    s = pd.Series(x).interpolate(limit_direction="both")
    s = s.fillna(float(np.nanmean(x[finite])))
    y = s.to_numpy(dtype=float)
    a = y[:-1]
    b = y[1:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return _safe_float(np.corrcoef(a, b)[0, 1])


def _roi_ar1_table(hc_dir: Path, sz_dir: Path, decisions: pd.DataFrame, atlas: str) -> pd.DataFrame:
    primary_cols = _select_roi_columns(decisions)
    region_map = decisions.set_index("roi_index_1based")["region_name"].to_dict()
    subjects = load_valid_subjects(scan_inventory(hc_dir, sz_dir, atlas_filter=atlas))
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for col in primary_cols:
            roi = int(col.replace("roi_", "")) + 1
            rows.append(
                {
                    "atlas": atlas,
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "roi": roi,
                    "region": region_map.get(roi, ""),
                    "AR1": _ar1_coef(subject.data_time_roi[col].to_numpy(dtype=float)),
                }
            )
    per_subject = pd.DataFrame(rows)
    out_rows: list[dict[str, Any]] = []
    for (roi, region), group in per_subject.groupby(["roi", "region"], sort=True):
        hc = group[group["group"].eq("HC")]["AR1"].dropna().to_numpy(float)
        sz = group[group["group"].eq("SZ")]["AR1"].dropna().to_numpy(float)
        if hc.size >= 2 and sz.size >= 2:
            try:
                u_stat, p_value = stats.mannwhitneyu(sz, hc, alternative="two-sided")
                effect = 2.0 * float(u_stat) / float(hc.size * sz.size) - 1.0
            except ValueError:
                p_value = float("nan")
                effect = float("nan")
        else:
            p_value = float("nan")
            effect = float("nan")
        out_rows.append(
            {
                "roi": int(roi),
                "region": region,
                "AR1_HC_by_ROI": _safe_float(np.nanmedian(hc)) if hc.size else float("nan"),
                "AR1_SZ_by_ROI": _safe_float(np.nanmedian(sz)) if sz.size else float("nan"),
                "AR1_group_difference_by_ROI": (
                    _safe_float(np.nanmedian(sz)) - _safe_float(np.nanmedian(hc))
                    if hc.size and sz.size
                    else float("nan")
                ),
                "AR1_group_p_value": p_value,
                "AR1_group_effect_size": effect,
                "n_HC": int(hc.size),
                "n_SZ": int(sz.size),
            }
        )
    return pd.DataFrame(out_rows)


def _zone_label(region_i: str, region_j: str) -> str:
    text = f"{region_i} {region_j}".lower()
    labels = []
    if any(x in text for x in ("front", "precentral", "supp_motor", "paracentral", "rectus")):
        labels.append("frontal_motor")
    if any(x in text for x in ("thalam",)):
        labels.append("thalamus")
    if any(x in text for x in ("cerebel", "vermis")):
        labels.append("cerebellum")
    if any(x in text for x in ("temporal", "heschl")):
        labels.append("temporal")
    return ",".join(labels) if labels else "other"


def build_diagnostic(
    *,
    hc_dir: Path,
    sz_dir: Path,
    decision_dir: Path,
    stage2_dir: Path,
    output_dir: Path,
    atlas: str,
    alpha: float,
    metric_name: str,
    edges_file: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(decision_dir / "roi_decision_layer_v2.csv")
    decisions = decisions[decisions["atlas"].eq(atlas)].copy()
    region_map = decisions.set_index("roi_index_1based")["region_name"].to_dict()

    edges_path = stage2_dir / edges_file
    edges = pd.read_csv(edges_path)
    baseline = edges[edges["branch"].eq("baseline")].copy()
    ar1 = edges[edges["branch"].eq("AR1_residualized")].copy()
    keep_cols = ["edge", "roi_i", "roi_j", "region_i", "region_j", "group_delta", "p_value", "q_value_FDR", "significant"]
    merged = baseline[keep_cols].merge(
        ar1[keep_cols],
        on=["edge", "roi_i", "roi_j", "region_i", "region_j"],
        suffixes=("_baseline", "_AR1"),
    )
    merged = merged.rename(
        columns={
            "group_delta_baseline": "edge_delta_baseline",
            "group_delta_AR1": "edge_delta_AR1",
            "p_value_baseline": "p_baseline",
            "p_value_AR1": "p_AR1",
            "q_value_FDR_baseline": "q_baseline",
            "q_value_FDR_AR1": "q_AR1",
        }
    )
    merged["delta_shrinkage"] = (
        merged["edge_delta_AR1"].abs() / merged["edge_delta_baseline"].abs().replace(0, np.nan)
    )
    merged["baseline_FDR"] = merged["significant_baseline"].astype(bool)
    merged["AR1_FDR"] = merged["significant_AR1"].astype(bool)
    merged["baseline_FDR_AR1_p_lt_0_05"] = merged["baseline_FDR"] & merged["p_AR1"].lt(alpha)
    merged["zone_label"] = [
        _zone_label(a, b) for a, b in zip(merged["region_i"].astype(str), merged["region_j"].astype(str))
    ]
    _write_csv(merged, output_dir / f"stage2_ar1_edge_diagnostic_{metric_name}.csv")

    ar1_roi = _roi_ar1_table(hc_dir, sz_dir, decisions, atlas)
    _write_csv(ar1_roi, output_dir / f"stage2_ar1_roi_group_differences_{metric_name}.csv")

    roi_loss_rows = []
    base_fdr = merged[merged["baseline_FDR"]].copy()
    for _, row in base_fdr.iterrows():
        shrink = _safe_float(row["delta_shrinkage"])
        for side in ("i", "j"):
            roi = int(row[f"roi_{side}"])
            roi_loss_rows.append(
                {
                    "roi": roi,
                    "region": region_map.get(roi, row[f"region_{side}"]),
                    "baseline_FDR_edge_count": 1,
                    "lost_after_AR1_FDR": int(not bool(row["AR1_FDR"])),
                    "AR1_p_lt_0_05_after_baseline_FDR": int(bool(row["baseline_FDR_AR1_p_lt_0_05"])),
                    "median_delta_shrinkage": shrink,
                }
            )
    roi_loss = (
        pd.DataFrame(roi_loss_rows)
        .groupby(["roi", "region"], as_index=False)
        .agg(
            baseline_FDR_edge_count=("baseline_FDR_edge_count", "sum"),
            lost_after_AR1_FDR=("lost_after_AR1_FDR", "sum"),
            AR1_p_lt_0_05_after_baseline_FDR=("AR1_p_lt_0_05_after_baseline_FDR", "sum"),
            median_delta_shrinkage=("median_delta_shrinkage", "median"),
        )
        .sort_values(["lost_after_AR1_FDR", "baseline_FDR_edge_count"], ascending=False)
        if roi_loss_rows
        else pd.DataFrame()
    )
    _write_csv(roi_loss, output_dir / f"stage2_ar1_roi_effect_loss_{metric_name}.csv")

    near_miss = merged[merged["baseline_FDR"] & merged["p_AR1"].lt(alpha) & ~merged["AR1_FDR"]].copy()
    near_miss = near_miss.sort_values(["p_AR1", "q_AR1"]).head(500)
    _write_csv(near_miss, output_dir / f"stage2_ar1_near_miss_edges_{metric_name}.csv")

    base_fdr_count = int(merged["baseline_FDR"].sum())
    ar1_fdr_count = int(merged["AR1_FDR"].sum())
    base_fdr_ar1_p = int(merged["baseline_FDR_AR1_p_lt_0_05"].sum())
    ar1_p_any = int(merged["p_AR1"].lt(alpha).sum())
    q_min_ar1 = _safe_float(merged["q_AR1"].min())
    p_min_ar1 = _safe_float(merged["p_AR1"].min())
    shrink_median_base_fdr = _safe_float(base_fdr["delta_shrinkage"].median()) if len(base_fdr) else float("nan")
    shrink_q25 = _safe_float(base_fdr["delta_shrinkage"].quantile(0.25)) if len(base_fdr) else float("nan")
    shrink_q75 = _safe_float(base_fdr["delta_shrinkage"].quantile(0.75)) if len(base_fdr) else float("nan")

    endpoint_ar1 = ar1_roi.copy()
    endpoint_ar1["abs_AR1_group_difference"] = endpoint_ar1["AR1_group_difference_by_ROI"].abs()
    ar1_sig = endpoint_ar1[endpoint_ar1["AR1_group_p_value"].lt(alpha)].copy()
    edge_roi_ar1 = merged.merge(
        endpoint_ar1[["roi", "AR1_HC_by_ROI", "AR1_SZ_by_ROI", "AR1_group_difference_by_ROI"]].rename(
            columns={
                "roi": "roi_i",
                "AR1_HC_by_ROI": "AR1_HC_i",
                "AR1_SZ_by_ROI": "AR1_SZ_i",
                "AR1_group_difference_by_ROI": "AR1_group_difference_i",
            }
        ),
        on="roi_i",
        how="left",
    ).merge(
        endpoint_ar1[["roi", "AR1_HC_by_ROI", "AR1_SZ_by_ROI", "AR1_group_difference_by_ROI"]].rename(
            columns={
                "roi": "roi_j",
                "AR1_HC_by_ROI": "AR1_HC_j",
                "AR1_SZ_by_ROI": "AR1_SZ_j",
                "AR1_group_difference_by_ROI": "AR1_group_difference_j",
            }
        ),
        on="roi_j",
        how="left",
    )
    edge_roi_ar1["edge_mean_abs_AR1_group_difference"] = edge_roi_ar1[
        ["AR1_group_difference_i", "AR1_group_difference_j"]
    ].abs().mean(axis=1)
    edge_roi_ar1["neg_log10_p_baseline"] = -np.log10(edge_roi_ar1["p_baseline"].clip(lower=1e-300))
    mask = np.isfinite(edge_roi_ar1["edge_mean_abs_AR1_group_difference"]) & np.isfinite(
        edge_roi_ar1["neg_log10_p_baseline"]
    )
    corr_ar1_sig = (
        float(
            np.corrcoef(
                edge_roi_ar1.loc[mask, "edge_mean_abs_AR1_group_difference"],
                edge_roi_ar1.loc[mask, "neg_log10_p_baseline"],
            )[0, 1]
        )
        if mask.sum() >= 3
        else float("nan")
    )

    zone_loss = (
        base_fdr.groupby("zone_label", as_index=False)
        .agg(
            baseline_FDR_edges=("edge", "count"),
            retained_AR1_FDR=("AR1_FDR", lambda s: int(s.astype(bool).sum())),
            retained_AR1_p_lt_0_05=("baseline_FDR_AR1_p_lt_0_05", lambda s: int(s.astype(bool).sum())),
            median_delta_shrinkage=("delta_shrinkage", "median"),
        )
        .sort_values("baseline_FDR_edges", ascending=False)
        if len(base_fdr)
        else pd.DataFrame()
    )
    _write_csv(zone_loss, output_dir / f"stage2_ar1_zone_effect_loss_{metric_name}.csv")

    lines = [
        f"# Stage 2 AR1 Sensitivity Diagnostic: {metric_name}",
        "",
        "This diagnostic asks why `AR1_residualized` removes the FDR-surviving `correlation_full` effects observed in baseline/detrended branches.",
        "",
        "## Scope",
        "",
        f"- Metric: `{metric_name}`",
        "- ROI: Stage 1.5 v2 primary ROI only",
        "- Window: full",
        "- Lag metadata: 1",
        "- Compared branches: `baseline` vs `AR1_residualized`",
        "",
        "## Main Counts",
        "",
        f"- Baseline FDR edges: {base_fdr_count}",
        f"- AR1 FDR edges: {ar1_fdr_count}",
        f"- Baseline-FDR edges that remain AR1 p<0.05 before FDR: {base_fdr_ar1_p}",
        f"- Any AR1 p<0.05 edges: {ar1_p_any}",
        f"- Minimum AR1 p-value: {p_min_ar1:.4g}",
        f"- Minimum AR1 q-value: {q_min_ar1:.4g}",
        "",
        "## Delta Shrinkage",
        "",
        f"- Median `abs(delta_AR1) / abs(delta_baseline)` among baseline-FDR edges: {shrink_median_base_fdr:.4f}",
        f"- IQR: {shrink_q25:.4f} to {shrink_q75:.4f}",
        "",
        "## ROI-Level AR1 Group Differences",
        "",
        f"- ROI with AR1 HC/SZ p<0.05: {len(ar1_sig)} / {len(ar1_roi)}",
        f"- Correlation between edge endpoint AR1 group-difference magnitude and baseline edge significance strength: {corr_ar1_sig:.4f}",
        "",
        "Top absolute ROI-level AR1 HC/SZ differences:",
        "",
        _markdown_table(endpoint_ar1.sort_values("abs_AR1_group_difference", ascending=False).head(20).round(4)),
        "",
        "## Zones Losing Baseline-FDR Effects After AR1",
        "",
        _markdown_table(zone_loss.round(4)),
        "",
        "## ROI Losing Baseline-FDR Effects After AR1",
        "",
        _markdown_table(roi_loss.head(30).round(4)),
        "",
        "## AR1 Near-Miss Edges",
        "",
        "These edges were baseline-FDR and remain `p_AR1 < 0.05`, but do not survive AR1 FDR.",
        "",
        _markdown_table(near_miss[["edge", "region_i", "region_j", "edge_delta_baseline", "edge_delta_AR1", "delta_shrinkage", "p_AR1", "q_AR1"]].head(30).round(5)),
        "",
        "## Methodological Decision",
        "",
        "For v0.1, `correlation_full` should be treated as a secondary/sensitivity metric rather than a standalone primary metric. The baseline signal is not random noise, because AR1 retains some nominal p<0.05 structure, but the FDR-surviving baseline core is highly AR1-sensitive.",
        "",
        "## Upstream Guardrail",
        "",
        "This diagnostic still uses already-extracted ROI mean time series. It does not validate gray-matter overlap, atlas mapping, voxel-wise ROI homogeneity, the source of systemic zero ROI, or whether mean ROI signal is trustworthy. Because the current FC result is AR1-sensitive, upstream extraction/ROI homogeneity checks remain a blocking methodological issue before promoting baseline FC as biological evidence.",
    ]
    (output_dir / f"stage2_ar1_sensitivity_diagnostic_report_{metric_name}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas", default="AAL3")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--metric-name", default="correlation_full")
    parser.add_argument("--edges-file", default="stage2_correlation_full_all_primary_roi_edges.csv")
    args = parser.parse_args()
    build_diagnostic(
        hc_dir=Path(args.hc_dir),
        sz_dir=Path(args.sz_dir),
        decision_dir=Path(args.decision_dir),
        stage2_dir=Path(args.stage2_dir),
        output_dir=Path(args.output_dir),
        atlas=str(args.atlas),
        alpha=float(args.alpha),
        metric_name=str(args.metric_name),
        edges_file=str(args.edges_file),
    )


if __name__ == "__main__":
    main()
