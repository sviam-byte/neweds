"""Summarize Stage 2 v0.1 correlation_full all-primary-ROI outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    small = df.head(max_rows).copy()
    cols = list(small.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in small.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = str(text).lower()
    return any(needle in lower for needle in needles)


def build_report(output_dir: Path) -> None:
    edges_path = output_dir / "stage2_correlation_full_all_primary_roi_edges.csv"
    stability_path = output_dir / "stage2_correlation_full_all_primary_roi_branch_stability.csv"
    subnet_path = output_dir / "stage2_correlation_full_all_primary_roi_candidate_subnetworks.md"
    report_path = output_dir / "stage2_correlation_full_all_primary_roi_report.md"

    edges = pd.read_csv(edges_path)
    stability = pd.read_csv(stability_path)

    sig = edges[edges["significant"].astype(bool)].copy()
    stable = edges[edges["stability_score"].ge(0.75)].copy()
    stable_sig = stable[stable["significant"].astype(bool)].copy()

    branch_summary = (
        edges.groupby("branch", as_index=False)
        .agg(
            n_edges=("edge", "count"),
            n_p_lt_005=("p_value", lambda s: int(pd.to_numeric(s, errors="coerce").lt(0.05).sum())),
            n_fdr=("significant", lambda s: int(s.astype(bool).sum())),
            median_delta=("group_delta", "median"),
        )
        .sort_values("branch")
    )
    branch_summary["fraction_delta_sz_lt_hc"] = branch_summary["branch"].map(
        lambda branch: float((edges.loc[edges["branch"].eq(branch), "group_delta"] < 0).mean())
    )
    branch_summary["fraction_sig_delta_sz_lt_hc"] = branch_summary["branch"].map(
        lambda branch: float(
            (sig.loc[sig["branch"].eq(branch), "group_delta"] < 0).mean()
            if len(sig.loc[sig["branch"].eq(branch)]) > 0
            else np.nan
        )
    )

    survival_by_count = (
        edges.drop_duplicates(["edge", "branch_survival_count"])
        .groupby("branch_survival_count", as_index=False)["edge"]
        .nunique()
        .rename(columns={"edge": "n_unique_edges"})
        .sort_values("branch_survival_count")
    )
    stable_unique = stable.drop_duplicates("edge").copy()
    stable_unique["direction"] = np.where(stable_unique["group_delta"] < 0, "SZ < HC", "SZ >= HC")
    direction_summary = stable_unique["direction"].value_counts(dropna=False).rename_axis("direction").reset_index(name="n_edges")

    roi_rows = []
    for _, row in stable_unique.iterrows():
        for side in ("i", "j"):
            roi_rows.append(
                {
                    "roi": int(row[f"roi_{side}"]),
                    "region": row[f"region_{side}"],
                    "n_stable_edges": 1,
                    "n_sz_lt_hc": int(float(row["group_delta"]) < 0),
                }
            )
    roi_table = (
        pd.DataFrame(roi_rows)
        .groupby(["roi", "region"], as_index=False)
        .agg(n_stable_edges=("n_stable_edges", "sum"), n_sz_lt_hc=("n_sz_lt_hc", "sum"))
        .sort_values(["n_stable_edges", "n_sz_lt_hc"], ascending=False)
        if roi_rows
        else pd.DataFrame(columns=["roi", "region", "n_stable_edges", "n_sz_lt_hc"])
    )

    zone_needles = {
        "frontal": ("front", "precentral", "supp_motor", "paracentral", "rectus"),
        "motor": ("precentral", "postcentral", "supp_motor", "paracentral"),
        "thalamus": ("thalam",),
        "cerebellum": ("cerebel", "vermis"),
        "temporal": ("temporal", "heschl"),
    }
    zone_rows = []
    for zone, needles in zone_needles.items():
        mask = roi_table["region"].map(lambda value: _contains_any(value, needles)) if not roi_table.empty else []
        part = roi_table[mask] if len(roi_table) else roi_table
        zone_rows.append(
            {
                "zone": zone,
                "n_roi_involved": int(len(part)),
                "stable_edge_involvement": int(part["n_stable_edges"].sum()) if len(part) else 0,
                "top_regions": ", ".join(part.head(8)["region"].astype(str).tolist()) if len(part) else "",
            }
        )
    zone_table = pd.DataFrame(zone_rows)

    ar1_row = branch_summary[branch_summary["branch"].eq("AR1_residualized")]
    baseline_row = branch_summary[branch_summary["branch"].eq("baseline")]
    baseline_fdr = int(baseline_row["n_fdr"].iloc[0]) if len(baseline_row) else 0
    ar1_fdr = int(ar1_row["n_fdr"].iloc[0]) if len(ar1_row) else 0
    ar1_ratio = ar1_fdr / baseline_fdr if baseline_fdr else np.nan

    lines = [
        "# Stage 2 v0.1 Correlation Full All Primary ROI",
        "",
        "This run extends the previous 20-ROI smoke check to all Stage 1.5 v2 primary AAL3 ROI, while keeping the analysis otherwise fixed.",
        "",
        "## Run Configuration",
        "",
        "- Metric: `correlation_full`",
        "- Branches: `baseline`, `detrended`, `AR1_residualized`, `AR1_plus_detrended`",
        "- Lag: `1`",
        "- Window: `full`",
        "- ROI: all primary ROI from `roi_decision_layer_v2.csv`",
        "- Subjects: all HC/SZ subjects",
        "",
        "## Four Answers",
        "",
        f"1. **Does AR1 again cut effects from hundreds to a small core?** Baseline has {baseline_fdr} FDR-surviving edge rows; AR1 has {ar1_fdr}. AR1/baseline ratio: {ar1_ratio:.3f}.",
        f"2. **Are stable effects mostly SZ < HC?** Among unique stable edges, {int((stable_unique['group_delta'] < 0).sum())} / {len(stable_unique)} are SZ < HC ({_pct(float((stable_unique['group_delta'] < 0).mean())) if len(stable_unique) else 'NA'}).",
        "3. **Does the frontal-motor pattern remain main?** See the zone and ROI involvement tables below; this is now assessed over all primary ROI, not the first 20.",
        "4. **Do thalamus/cerebellum/temporal zones appear?** See the zone involvement table below.",
        "",
        "## Branch Summary",
        "",
        _markdown_table(branch_summary.round(4)),
        "",
        "## Branch Stability",
        "",
        _markdown_table(stability.round(4)),
        "",
        "## Unique Edge Survival",
        "",
        _markdown_table(survival_by_count),
        "",
        "## Stable Direction Summary",
        "",
        _markdown_table(direction_summary),
        "",
        "## Zone Involvement Among Stable Edges",
        "",
        _markdown_table(zone_table),
        "",
        "## Top ROI Involvement Among Stable Edges",
        "",
        _markdown_table(roi_table.head(30)),
        "",
        "## Candidate Subnetwork Artifact",
        "",
        f"See `{subnet_path.name}` for the companion candidate-subnetwork table.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build_report(Path(args.output_dir))


if __name__ == "__main__":
    main()
