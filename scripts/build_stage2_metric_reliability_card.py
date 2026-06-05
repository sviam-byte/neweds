"""Build a Stage 2 metric reliability card from exploratory and AR diagnostics."""

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
from pandas.errors import EmptyDataError

from scripts.run_fmri_stage2_sanity import _markdown_table


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _format_float(value: Any, digits: int = 4) -> str:
    val = _safe_float(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _read_csv_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def _status_from_evidence(
    *,
    baseline_fdr: int,
    ar1_fdr: int,
    ar1_plus_fdr: int,
    baseline_ar1_survived: int,
    branch_ar_similarity: float,
    ar1_q_min: float,
    ar1_roi_qc_corr: float,
) -> str:
    if ar1_fdr > 0 or ar1_plus_fdr > 0 or baseline_ar1_survived > 0:
        if branch_ar_similarity >= 0.85 and abs(ar1_roi_qc_corr) < 0.25:
            return "primary_candidate"
        return "secondary_candidate"
    if baseline_fdr > 0 and np.isfinite(ar1_q_min) and ar1_q_min < 0.10:
        return "secondary_candidate"
    if baseline_fdr > 0:
        return "sensitivity_only"
    return "do_not_trust_yet"


def build_card(
    *,
    metric_name: str,
    metric_family: str,
    stage2_dir: Path,
    ar1_diagnostic_dir: Path,
    output_dir: Path,
    edges_file: str,
    branch_stability_file: str,
    lag_policy: str,
    multiple_comparison_policy: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(stage2_dir / edges_file)
    branch_stability = pd.read_csv(stage2_dir / branch_stability_file)
    ar1_edges = pd.read_csv(ar1_diagnostic_dir / f"stage2_ar1_edge_diagnostic_{metric_name}.csv")
    ar1_roi = pd.read_csv(ar1_diagnostic_dir / f"stage2_ar1_roi_group_differences_{metric_name}.csv")
    near_miss = _read_csv_or_empty(ar1_diagnostic_dir / f"stage2_ar1_near_miss_edges_{metric_name}.csv")
    zone_loss = _read_csv_or_empty(ar1_diagnostic_dir / f"stage2_ar1_zone_effect_loss_{metric_name}.csv")

    branch_summary = (
        edges.groupby("branch", as_index=False)
        .agg(
            n_edges=("edge", "count"),
            n_p_lt_005=("p_value", lambda s: int(pd.to_numeric(s, errors="coerce").lt(0.05).sum())),
            n_FDR_edges=("significant", lambda s: int(s.astype(bool).sum())),
            median_delta=("group_delta", "median"),
            median_effect_size=("effect_size", "median"),
        )
        .sort_values("branch")
    )
    branch_summary["fraction_SZ_lt_HC"] = branch_summary["branch"].map(
        lambda branch: _safe_float((edges.loc[edges["branch"].eq(branch), "group_delta"] < 0).mean())
    )

    baseline_fdr = int(branch_summary.loc[branch_summary["branch"].eq("baseline"), "n_FDR_edges"].iloc[0])
    ar1_fdr = int(branch_summary.loc[branch_summary["branch"].eq("AR1_residualized"), "n_FDR_edges"].iloc[0])
    ar1_plus_fdr = int(
        branch_summary.loc[branch_summary["branch"].eq("AR1_plus_detrended"), "n_FDR_edges"].iloc[0]
    )
    baseline_ar1_row = branch_stability[branch_stability["branch_pair"].eq("baseline_vs_AR1_residualized")]
    baseline_ar1_survived = int(baseline_ar1_row["n_edges_survived"].iloc[0]) if len(baseline_ar1_row) else 0
    branch_ar_similarity = _safe_float(baseline_ar1_row["matrix_similarity"].iloc[0]) if len(baseline_ar1_row) else np.nan

    base_fdr_edges = ar1_edges[ar1_edges["baseline_FDR"].astype(bool)]
    baseline_fdr_ar1_p = int(base_fdr_edges["p_AR1"].lt(0.05).sum())
    shrinkage_median = _safe_float(base_fdr_edges["delta_shrinkage"].median())
    shrinkage_q25 = _safe_float(base_fdr_edges["delta_shrinkage"].quantile(0.25))
    shrinkage_q75 = _safe_float(base_fdr_edges["delta_shrinkage"].quantile(0.75))
    min_ar1_p = _safe_float(ar1_edges["p_AR1"].min())
    min_ar1_q = _safe_float(ar1_edges["q_AR1"].min())
    any_ar1_p = int(ar1_edges["p_AR1"].lt(0.05).sum())

    ar1_roi_sig = int(ar1_roi["AR1_group_p_value"].lt(0.05).sum())
    endpoint_ar1 = ar1_roi.copy()
    endpoint_ar1["abs_AR1_group_difference"] = endpoint_ar1["AR1_group_difference_by_ROI"].abs()
    edge_endpoint = ar1_edges.merge(
        endpoint_ar1[["roi", "AR1_group_difference_by_ROI"]].rename(
            columns={"roi": "roi_i", "AR1_group_difference_by_ROI": "ar1_diff_i"}
        ),
        on="roi_i",
        how="left",
    ).merge(
        endpoint_ar1[["roi", "AR1_group_difference_by_ROI"]].rename(
            columns={"roi": "roi_j", "AR1_group_difference_by_ROI": "ar1_diff_j"}
        ),
        on="roi_j",
        how="left",
    )
    edge_endpoint["mean_abs_endpoint_ar1_diff"] = edge_endpoint[["ar1_diff_i", "ar1_diff_j"]].abs().mean(axis=1)
    edge_endpoint["neg_log10_p_baseline"] = -np.log10(edge_endpoint["p_baseline"].clip(lower=1e-300))
    mask = np.isfinite(edge_endpoint["mean_abs_endpoint_ar1_diff"]) & np.isfinite(edge_endpoint["neg_log10_p_baseline"])
    ar1_qc_corr = (
        float(
            np.corrcoef(
                edge_endpoint.loc[mask, "mean_abs_endpoint_ar1_diff"],
                edge_endpoint.loc[mask, "neg_log10_p_baseline"],
            )[0, 1]
        )
        if mask.sum() >= 3
        else float("nan")
    )

    recommendation = _status_from_evidence(
        baseline_fdr=baseline_fdr,
        ar1_fdr=ar1_fdr,
        ar1_plus_fdr=ar1_plus_fdr,
        baseline_ar1_survived=baseline_ar1_survived,
        branch_ar_similarity=branch_ar_similarity,
        ar1_q_min=min_ar1_q,
        ar1_roi_qc_corr=ar1_qc_corr,
    )

    card_row = pd.DataFrame(
        [
            {
                "metric_name": metric_name,
                "metric_family": metric_family,
                "input_ROI_set": "Stage 1.5 v2 primary ROI: keep + qc_flag_keep",
                "subject_policy": "all subjects; subject warnings reported, not excluded",
                "branches": "baseline,detrended,AR1_residualized,AR1_plus_detrended",
                "lag_policy": lag_policy,
                "window_policy": "full 600-point series",
                "statistical_test": "HC vs SZ Mann-Whitney U per edge",
                "multiple_comparison_policy": multiple_comparison_policy,
                "branch_stability": f"baseline_vs_AR1 matrix_similarity={_format_float(branch_ar_similarity)}; AR1-surviving FDR edges={baseline_ar1_survived}",
                "AR_sensitivity": f"baseline_FDR={baseline_fdr}; AR1_FDR={ar1_fdr}; baseline_FDR_with_AR1_p_lt_0.05={baseline_fdr_ar1_p}; median_shrinkage={_format_float(shrinkage_median)}; min_AR1_q={_format_float(min_ar1_q)}",
                "QC_sensitivity": f"ROI AR1 group differences p<0.05 in {ar1_roi_sig}/{len(ar1_roi)} ROI; endpoint AR1 difference vs baseline significance r={_format_float(ar1_qc_corr)}",
                "subject_sensitivity": "not yet stress-tested; primary policy keeps all subjects",
                "ROI_sensitivity": "primary excludes Background/systemic-zero and sensitivity_only ROI; include_sensitivity_roi not yet re-run for this card",
                "recommendation": recommendation,
            }
        ]
    )
    card_row.to_csv(output_dir / f"metric_reliability_card_{metric_name}.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"# Metric Reliability Card: {metric_name}",
        "",
        "## Identity",
        "",
        f"- `metric_name`: `{metric_name}`",
        f"- `metric_family`: `{metric_family}`",
        "- `input_ROI_set`: Stage 1.5 v2 primary ROI = `keep + qc_flag_keep`",
        "- `subject_policy`: all 39 subjects, subject warnings preserved",
        "- `branches`: `baseline`, `detrended`, `AR1_residualized`, `AR1_plus_detrended`",
        f"- `lag_policy`: {lag_policy}",
        "- `window_policy`: full series",
        "- `statistical_test`: HC vs SZ Mann-Whitney U per edge",
        f"- `multiple_comparison_policy`: {multiple_comparison_policy}",
        "",
        "## Effect Existence",
        "",
        _markdown_table(branch_summary.round(4)),
        "",
        "## Branch Stability",
        "",
        _markdown_table(branch_stability.round(4)),
        "",
        "## AR Sensitivity",
        "",
        f"- Baseline FDR edges: `{baseline_fdr}`",
        f"- AR1 FDR edges: `{ar1_fdr}`",
        f"- AR1+detrended FDR edges: `{ar1_plus_fdr}`",
        f"- Baseline-FDR edges with AR1 `p < 0.05`: `{baseline_fdr_ar1_p}`",
        f"- Any AR1 edges with `p < 0.05`: `{any_ar1_p}`",
        f"- Median delta shrinkage among baseline-FDR edges: `{_format_float(shrinkage_median)}`",
        f"- Delta shrinkage IQR: `{_format_float(shrinkage_q25)}` to `{_format_float(shrinkage_q75)}`",
        f"- Minimum AR1 p-value: `{min_ar1_p:.4g}`",
        f"- Minimum AR1 q-value: `{min_ar1_q:.4g}`",
        "",
        "## QC / Temporal Phenotype Sensitivity",
        "",
        f"- ROI-level AR1 HC/SZ differences at p<0.05: `{ar1_roi_sig} / {len(ar1_roi)}`",
        f"- Endpoint AR1 group-difference magnitude vs baseline edge significance strength: `r = {_format_float(ar1_qc_corr)}`",
        "",
        "Top ROI-level AR1 group differences:",
        "",
        _markdown_table(endpoint_ar1.sort_values("abs_AR1_group_difference", ascending=False).head(15).round(4)),
        "",
        "## Zones Most Affected By AR1",
        "",
        _markdown_table(zone_loss.round(4)),
        "",
        "## AR1 Near-Miss Edges",
        "",
        f"`{len(near_miss)}` baseline-FDR edges remain `p_AR1 < 0.05` but do not survive AR1 FDR. Top rows:",
        "",
        _markdown_table(
            near_miss.reindex(
                columns=[
                    "edge",
                    "region_i",
                    "region_j",
                    "edge_delta_baseline",
                    "edge_delta_AR1",
                    "delta_shrinkage",
                    "p_AR1",
                    "q_AR1",
                ]
            )[
                [
                    "edge",
                    "region_i",
                    "region_j",
                    "edge_delta_baseline",
                    "edge_delta_AR1",
                    "delta_shrinkage",
                    "p_AR1",
                    "q_AR1",
                ]
            ]
            .head(25)
            .round(5)
        ),
        "",
        "## Recommendation",
        "",
        f"**`{recommendation}`**",
        "",
        f"`{metric_name}` is evaluated as a metric-level reliability unit, not just a source of significant edges. The recommendation above reflects FDR evidence, branch survival, AR sensitivity, and the association with ROI-level temporal phenotype.",
        "",
        "## Upstream Constraint",
        "",
        "This card still operates on already-extracted mean ROI time series. It does not validate gray-matter overlap, atlas mapping, systemic-zero ROI causes, voxel-wise ROI homogeneity, or the trustworthiness of mean ROI signal.",
    ]
    (output_dir / f"metric_reliability_card_{metric_name}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-name", default="correlation_full")
    parser.add_argument("--metric-family", default="undirected_functional_connectivity")
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--ar1-diagnostic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edges-file", default="stage2_correlation_full_all_primary_roi_edges.csv")
    parser.add_argument("--branch-stability-file", default="stage2_correlation_full_all_primary_roi_branch_stability.csv")
    parser.add_argument("--lag-policy", default="fixed lag=1 metadata; undirected zero-lag correlation")
    parser.add_argument(
        "--multiple-comparison-policy",
        default="Benjamini-Hochberg FDR per branch over the tested primary edge family",
    )
    args = parser.parse_args()
    build_card(
        metric_name=str(args.metric_name),
        metric_family=str(args.metric_family),
        stage2_dir=Path(args.stage2_dir),
        ar1_diagnostic_dir=Path(args.ar1_diagnostic_dir),
        output_dir=Path(args.output_dir),
        edges_file=str(args.edges_file),
        branch_stability_file=str(args.branch_stability_file),
        lag_policy=str(args.lag_policy),
        multiple_comparison_policy=str(args.multiple_comparison_policy),
    )


if __name__ == "__main__":
    main()
