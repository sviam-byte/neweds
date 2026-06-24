"""Stage 2 exploratory full-grid v0 for already-extracted fMRI ROI data."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from neweds.core.fmri_roi_audit import load_valid_subjects, scan_inventory
from neweds.core.group_pipeline import _fdr_bh
from neweds.core.metric_runner import compute_metric
from scripts.run_fmri_stage2_sanity import (
    PRIMARY_BRANCHES,
    _compare_matrices,
    _markdown_table,
    _metric_properties,
    _preprocess_branch,
    _select_granger_columns,
    _select_roi_columns,
    _split_int_csv,
    _write_csv,
)

DEFAULT_METRICS = (
    "correlation_full",
    "correlation_partial",
    "wavelet_full",
    "correlation_directed",
    "granger_full",
)
BRANCH_LABELS = {
    "baseline": "survives_baseline",
    "detrended": "survives_detrended",
    "AR1_residualized": "survives_AR1",
    "AR1_plus_detrended": "survives_AR1_plus_detrended",
}


def _parse_windows(text: str) -> tuple[str, ...]:
    vals = tuple(part.strip() for part in str(text).split(",") if part.strip())
    return vals or ("120", "240", "full")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _window_specs(n_time: int, windows: tuple[str, ...]) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in windows:
        if item.lower() == "full":
            key = ("full", 0)
            if key not in seen:
                specs.append(("full", 0, int(n_time)))
                seen.add(key)
            continue
        size = max(8, min(int(item), int(n_time)))
        stride = max(1, size // 2)
        for start in range(0, max(1, n_time - size + 1), stride):
            key = (str(size), int(start))
            if key not in seen:
                specs.append((str(size), int(start), int(start + size)))
                seen.add(key)
        if n_time - size > 0:
            key = (str(size), int(n_time - size))
            if key not in seen:
                specs.append((str(size), int(n_time - size), int(n_time)))
                seen.add(key)
    return specs


def _edge_records(matrix: np.ndarray, cols: list[str], *, directed: bool, pvalue_based: bool) -> list[dict[str, Any]]:
    arr = np.asarray(matrix, dtype=float)
    records: list[dict[str, Any]] = []
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return records
    n = min(arr.shape[0], len(cols))
    if directed:
        iterator = ((i, j) for i in range(n) for j in range(n) if i != j)
    else:
        iterator = ((i, j) for i in range(n) for j in range(i + 1, n))
    for i, j in iterator:
        value = _safe_float(arr[i, j])
        if not np.isfinite(value):
            continue
        records.append(
            {
                "edge": f"{cols[i]}->{cols[j]}" if directed else f"{cols[i]}--{cols[j]}",
                "roi_i": int(cols[i].replace("roi_", "")) + 1,
                "roi_j": int(cols[j].replace("roi_", "")) + 1,
                "value": value,
                "pvalue_based": bool(pvalue_based),
            }
        )
    return records


def _compare_edge_groups(edge_values: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (edge, roi_i, roi_j), group in edge_values.groupby(["edge", "roi_i", "roi_j"], sort=False):
        hc = pd.to_numeric(group[group["group"].eq("HC")]["value"], errors="coerce").to_numpy(float)
        sz = pd.to_numeric(group[group["group"].eq("SZ")]["value"], errors="coerce").to_numpy(float)
        hc = hc[np.isfinite(hc)]
        sz = sz[np.isfinite(sz)]
        if hc.size < 2 or sz.size < 2:
            p_value = float("nan")
            u_stat = float("nan")
            effect = float("nan")
        else:
            try:
                u_stat, p_value = stats.mannwhitneyu(sz, hc, alternative="two-sided")
                effect = 2.0 * float(u_stat) / float(hc.size * sz.size) - 1.0
            except ValueError:
                p_value = float("nan")
                u_stat = float("nan")
                effect = float("nan")
        rows.append(
            {
                "edge": edge,
                "roi_i": int(roi_i),
                "roi_j": int(roi_j),
                "HC_value": _safe_float(np.nanmedian(hc)) if hc.size else float("nan"),
                "SZ_value": _safe_float(np.nanmedian(sz)) if sz.size else float("nan"),
                "group_delta": (_safe_float(np.nanmedian(sz)) - _safe_float(np.nanmedian(hc)))
                if hc.size and sz.size
                else float("nan"),
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


def _attach_survival(edge_results: pd.DataFrame) -> pd.DataFrame:
    if edge_results.empty:
        return edge_results
    keys = ["edge", "metric", "lag", "window_size", "window_start"]
    survival = {}
    for key, group in edge_results.groupby(keys, sort=False):
        by_branch = group.set_index("branch")
        signs = []
        count = 0
        flags: dict[str, bool] = {}
        for branch, col in BRANCH_LABELS.items():
            ok = bool(branch in by_branch.index and bool(by_branch.loc[branch, "significant"]))
            flags[col] = ok
            if ok:
                count += 1
                signs.append(np.sign(float(by_branch.loc[branch, "group_delta"])))
        direction_consistent = bool(signs and len({int(s) for s in signs if s != 0}) <= 1)
        score = float(count / len(BRANCH_LABELS))
        if direction_consistent and count > 1:
            score = min(1.0, score + 0.25)
        survival[key] = {
            **flags,
            "branch_survival_count": count,
            "effect_direction_consistent": direction_consistent,
            "stability_score": score,
        }
    rows = []
    for _, row in edge_results.iterrows():
        key = tuple(row[k] for k in keys)
        rows.append({**row.to_dict(), **survival[key]})
    return pd.DataFrame(rows)


def _branch_label(row: pd.Series) -> str:
    corr = _safe_float(row.get("matrix_similarity"))
    sign_change = _safe_float(row.get("fraction_edges_changing_sign"))
    pair = str(row.get("branch_pair", ""))
    if np.isfinite(corr) and corr >= 0.9 and np.isfinite(sign_change) and sign_change <= 0.1:
        return "stable"
    if "AR1" in pair:
        return "AR1_sensitive"
    if "detrended" in pair:
        return "detrend_sensitive"
    return "branch_sensitive"


def _metric_reliability(edge_results: pd.DataFrame, branch_stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, group in edge_results.groupby("metric", sort=True):
        sig = group[group["significant"].astype(bool)]
        branch_stability_score = _safe_float(
            branch_stability[branch_stability["metric"].eq(metric)]["matrix_similarity"].median()
        )
        lag_sens = _safe_float(group.groupby("lag")["group_delta"].median().std())
        win_sens = _safe_float(group.groupby("window_size")["group_delta"].median().std())
        direction_consistency = _safe_float(group["effect_direction_consistent"].astype(bool).mean())
        stable_edges = group[group["stability_score"].ge(0.75)]
        if branch_stability_score >= 0.85 and direction_consistency >= 0.75:
            rec = "primary_candidate"
        elif branch_stability_score >= 0.65:
            rec = "secondary_candidate"
        elif len(stable_edges) > 0:
            rec = "sensitivity_only"
        else:
            rec = "do_not_trust_yet"
        rows.append(
            {
                "metric": metric,
                "n_significant_edges": int(group["p_value"].lt(0.05).sum()),
                "n_FDR_significant_edges": int(group["significant"].astype(bool).sum()),
                "branch_stability": branch_stability_score,
                "QC_sensitivity": float("nan"),
                "lag_sensitivity": lag_sens,
                "window_sensitivity": win_sens,
                "effect_direction_consistency": direction_consistency,
                "n_stable_edges": int(len(stable_edges)),
                "recommendation": rec,
            }
        )
    return pd.DataFrame(rows)


def _candidate_subnetworks(edge_results: pd.DataFrame, decisions: pd.DataFrame) -> str:
    if edge_results.empty or "stability_score" not in edge_results.columns:
        return "\n".join(
            [
                "# Stage 2 Candidate Subnetworks",
                "",
                "No candidate subnetworks were available because no valid edge-level results were produced.",
            ]
        )
    stable = edge_results[edge_results["stability_score"].ge(0.75)].copy()
    if stable.empty:
        stable = edge_results.sort_values(["stability_score", "q_value_FDR"], ascending=[False, True]).head(100)
    region_map = decisions.set_index("roi_index_1based")["region_name"].to_dict()
    roi_counts = defaultdict(int)
    for _, row in stable.iterrows():
        roi_counts[int(row["roi_i"])] += 1
        roi_counts[int(row["roi_j"])] += 1
    roi_table = pd.DataFrame(
        [
            {"roi": roi, "region": region_map.get(roi, ""), "involvement_score": count}
            for roi, count in sorted(roi_counts.items(), key=lambda item: item[1], reverse=True)
        ]
    )
    module_counts = (
        roi_table.assign(module=roi_table["region"].astype(str).str.split("_").str[0])
        .groupby("module", as_index=False)["involvement_score"]
        .sum()
        .sort_values("involvement_score", ascending=False)
        if not roi_table.empty
        else pd.DataFrame(columns=["module", "involvement_score"])
    )
    lines = [
        "# Stage 2 Candidate Subnetworks",
        "",
        "This exploratory summary reports repeated ROI involvement among stable or highest-stability effects.",
        "",
        "## ROI Involvement",
        "",
        _markdown_table(roi_table.head(30)),
        "",
        "## Module-Level Involvement",
        "",
        _markdown_table(module_counts.head(20)),
        "",
        "## Hub Disruption Candidates",
        "",
        _markdown_table(roi_table.head(10)),
    ]
    return "\n".join(lines)


def run_full_grid(
    *,
    hc_dir: Path,
    sz_dir: Path,
    decision_dir: Path,
    output_dir: Path,
    atlas: str,
    lags: tuple[int, ...],
    windows: tuple[str, ...],
    n_jobs: int,
    granger_max_roi: int,
    alpha: float,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    max_primary_roi: int | None = None,
) -> None:
    os.environ["TS_TOOL_N_JOBS"] = str(max(1, int(n_jobs)))
    os.environ.setdefault("TS_TOOL_PARALLEL_BACKEND", "threading")
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(decision_dir / "roi_decision_layer_v2.csv")
    decisions = decisions[decisions["atlas"].eq(atlas)].copy()
    primary_cols = _select_roi_columns(decisions)
    if max_primary_roi and max_primary_roi > 0:
        primary_cols = primary_cols[: int(max_primary_roi)]
    granger_cols = _select_granger_columns(decisions, granger_max_roi)
    subjects = load_valid_subjects(scan_inventory(hc_dir, sz_dir, atlas_filter=atlas))

    edge_result_parts: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    group_delta_vectors: dict[tuple[str, int, str, int, str], pd.DataFrame] = {}
    mean_matrices: dict[tuple[str, int, str, int, str, str], np.ndarray] = {}
    preprocessed_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = {}

    metrics = tuple(metrics) or DEFAULT_METRICS
    for branch in PRIMARY_BRANCHES:
        for metric in metrics:
            directed, pvalue_based = _metric_properties(metric)
            use_lags = lags if directed else (1,)
            for lag in use_lags:
                metric_cols = granger_cols if metric == "granger_full" else primary_cols
                by_window_values: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
                matrix_accumulator: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
                for subject in subjects:
                    cols_key = tuple(metric_cols)
                    cache_key = (subject.subject_id, branch, cols_key)
                    preprocessed = preprocessed_cache.get(cache_key)
                    if preprocessed is None:
                        source = subject.data_time_roi[metric_cols]
                        preprocessed = _preprocess_branch(source, branch)
                        preprocessed_cache[cache_key] = preprocessed
                    for window_size, window_start, window_end in _window_specs(len(preprocessed), windows):
                        chunk = preprocessed.iloc[window_start:window_end].reset_index(drop=True)
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                matrix = compute_metric(chunk, metric, lag=int(lag))
                            matrix_accumulator[(window_size, window_start, subject.group)].append(matrix)
                            for rec in _edge_records(
                                matrix,
                                metric_cols,
                                directed=directed,
                                pvalue_based=pvalue_based,
                            ):
                                by_window_values[(window_size, window_start)].append(
                                    {
                                        **rec,
                                        "group": subject.group,
                                        "subject_id": subject.subject_id,
                                    }
                                )
                        except Exception as exc:
                            failures.append(
                                {
                                    "atlas": atlas,
                                    "group": subject.group,
                                    "subject_id": subject.subject_id,
                                    "branch": branch,
                                    "metric": metric,
                                    "lag": int(lag),
                                    "window_size": window_size,
                                    "window_start": int(window_start),
                                    "error": str(exc),
                                }
                            )
                for (window_size, window_start), rows in by_window_values.items():
                    compared = _compare_edge_groups(pd.DataFrame(rows), alpha=alpha)
                    if compared.empty:
                        continue
                    compared.insert(0, "atlas", atlas)
                    compared.insert(1, "branch", branch)
                    compared.insert(2, "metric", metric)
                    compared.insert(3, "lag", int(lag))
                    compared.insert(4, "window_size", window_size)
                    compared.insert(5, "window_start", int(window_start))
                    compared["region_i"] = compared["roi_i"].map(
                        decisions.set_index("roi_index_1based")["region_name"].to_dict()
                    )
                    compared["region_j"] = compared["roi_j"].map(
                        decisions.set_index("roi_index_1based")["region_name"].to_dict()
                    )
                    edge_result_parts.append(compared)
                    group_delta_vectors[(metric, int(lag), window_size, int(window_start), branch)] = compared[
                        ["edge", "group_delta", "significant"]
                    ].copy()
                for (window_size, window_start, group), mats in matrix_accumulator.items():
                    if mats:
                        mean_matrices[(metric, int(lag), window_size, int(window_start), branch, group)] = np.nanmean(
                            np.asarray(mats, dtype=float), axis=0
                        )

    edge_results = pd.concat(edge_result_parts, ignore_index=True) if edge_result_parts else pd.DataFrame()
    edge_results = _attach_survival(edge_results)
    _write_csv(edge_results, output_dir / "stage2_full_edge_results.csv")

    branch_rows: list[dict[str, Any]] = []
    branch_pairs = [
        ("baseline", "AR1_residualized"),
        ("baseline", "detrended"),
        ("baseline", "AR1_plus_detrended"),
    ]
    for metric in metrics:
        directed, _p = _metric_properties(metric)
        use_lags = lags if directed else (1,)
        for lag in use_lags:
            for window_size, window_start, _end in _window_specs(600, windows):
                for left, right in branch_pairs:
                    deltas_left = group_delta_vectors.get((metric, int(lag), window_size, int(window_start), left))
                    deltas_right = group_delta_vectors.get((metric, int(lag), window_size, int(window_start), right))
                    if deltas_left is None or deltas_right is None:
                        continue
                    joined = deltas_left.merge(deltas_right, on="edge", suffixes=("_left", "_right"))
                    if joined.empty:
                        continue
                    a = joined["group_delta_left"].to_numpy(float)
                    b = joined["group_delta_right"].to_numpy(float)
                    mask = np.isfinite(a) & np.isfinite(b)
                    group_delta_similarity = (
                        float(np.corrcoef(a[mask], b[mask])[0, 1])
                        if mask.sum() >= 3 and np.std(a[mask]) > 1e-12 and np.std(b[mask]) > 1e-12
                        else float("nan")
                    )
                    mats_left = [
                        mean_matrices.get((metric, int(lag), window_size, int(window_start), left, group))
                        for group in ("HC", "SZ")
                    ]
                    mats_right = [
                        mean_matrices.get((metric, int(lag), window_size, int(window_start), right, group))
                        for group in ("HC", "SZ")
                    ]
                    sims = [
                        _compare_matrices(ml, mr, directed=directed)["matrix_correlation"]
                        for ml, mr in zip(mats_left, mats_right)
                        if ml is not None and mr is not None
                    ]
                    row = {
                        "branch_pair": f"{left}_vs_{right}",
                        "metric": metric,
                        "lag": int(lag),
                        "window_size": window_size,
                        "window_start": int(window_start),
                        "matrix_similarity": _safe_float(np.nanmedian(sims)) if sims else float("nan"),
                        "edge_rank_correlation": group_delta_similarity,
                        "group_delta_similarity": group_delta_similarity,
                        "n_edges_survived": int((joined["significant_left"] & joined["significant_right"]).sum()),
                        "fraction_edges_changing_sign": _safe_float(np.mean(np.sign(a[mask]) != np.sign(b[mask])))
                        if mask.any()
                        else float("nan"),
                    }
                    row["interpretation"] = _branch_label(pd.Series(row))
                    branch_rows.append(row)
    branch_stability = pd.DataFrame(branch_rows)
    _write_csv(branch_stability, output_dir / "stage2_branch_stability.csv")

    metric_reliability = _metric_reliability(edge_results, branch_stability) if not edge_results.empty else pd.DataFrame()
    _write_csv(metric_reliability, output_dir / "stage2_metric_reliability.csv")
    failure_columns = [
        "atlas",
        "group",
        "subject_id",
        "branch",
        "metric",
        "lag",
        "window_size",
        "window_start",
        "error",
    ]
    _write_csv(pd.DataFrame(failures, columns=failure_columns), output_dir / "stage2_full_failures.csv")

    (output_dir / "stage2_candidate_subnetworks.md").write_text(
        _candidate_subnetworks(edge_results, decisions), encoding="utf-8"
    )
    report_lines = [
        "# Stage 2 Exploratory Full Grid v0",
        "",
        "This is an exploratory stability map, not final Stage 2 evidence.",
        "",
        f"- Atlas: `{atlas}`",
        f"- Subjects: {len(subjects)}",
        f"- Primary ROI: {len(primary_cols)}",
        f"- Granger ROI: {len(granger_cols)}",
        f"- Branches: {', '.join(PRIMARY_BRANCHES)}",
        f"- Metrics: {', '.join(metrics)}",
        f"- Lags: {', '.join(str(lag) for lag in lags)}",
        f"- Windows: {', '.join(windows)}",
        "",
        "## Output Rows",
        "",
        f"- Edge result rows: {len(edge_results)}",
        f"- Branch stability rows: {len(branch_stability)}",
        f"- Metric reliability rows: {len(metric_reliability)}",
        f"- Failure rows: {len(failures)}",
        "",
        "## Metric Reliability",
        "",
        _markdown_table(metric_reliability),
        "",
        "## Branch Stability Snapshot",
        "",
        _markdown_table(branch_stability.head(40)),
        "",
        "## Interpretation Guardrail",
        "",
        "Effects that vanish after AR1/detrending are risk flags, not promoted discoveries.",
    ]
    (output_dir / "stage2_exploratory_full_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas", default="AAL3")
    parser.add_argument("--lags", default="1,2,3,4,5")
    parser.add_argument("--windows", default="120,240,full")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--granger-max-roi", type=int, default=30)
    parser.add_argument("--max-primary-roi", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_full_grid(
        hc_dir=Path(args.hc_dir),
        sz_dir=Path(args.sz_dir),
        decision_dir=Path(args.decision_dir),
        output_dir=Path(args.output_dir),
        atlas=str(args.atlas),
        lags=_split_int_csv(args.lags),
        windows=_parse_windows(args.windows),
        n_jobs=int(args.n_jobs),
        granger_max_roi=int(args.granger_max_roi),
        alpha=float(args.alpha),
        metrics=tuple(part.strip() for part in str(args.metrics).split(",") if part.strip()),
        max_primary_roi=int(args.max_primary_roi) or None,
    )


if __name__ == "__main__":
    main()
