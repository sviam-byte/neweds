"""Run a lightweight Stage 2 sanity check on fMRI ROI time-series data."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from neweds.core.fmri_roi_audit import load_valid_subjects, scan_inventory
from neweds.core.metric_runner import compute_metric

PRIMARY_BRANCHES = ("baseline", "detrended", "AR1_residualized", "AR1_plus_detrended")
SANITY_METRICS = (
    "correlation_full",
    "correlation_partial",
    "wavelet_full",
    "correlation_directed",
    "granger_full",
)


def _split_int_csv(text: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    return vals or (1, 2, 3)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in view.columns) + " |")
    return "\n".join(lines)


def _ar_residualize_array(values: np.ndarray, order: int = 1) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)
    if int(finite.sum()) <= order + 3 or np.nanstd(x[finite]) <= 1e-12:
        return np.nan_to_num(x - np.nanmean(x), nan=0.0)
    y_full = pd.Series(x).interpolate(limit_direction="both").fillna(float(np.nanmean(x))).to_numpy()
    y = y_full[order:]
    design = np.column_stack([y_full[order - lag : -lag] for lag in range(1, order + 1)])
    design = np.column_stack([np.ones(design.shape[0]), design])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
    except np.linalg.LinAlgError:
        resid = y - np.nanmean(y)
    out[:order] = 0.0
    out[order:] = resid
    return np.nan_to_num(out, nan=0.0)


def _preprocess_branch(data: pd.DataFrame, branch: str) -> pd.DataFrame:
    arr = data.apply(pd.to_numeric, errors="coerce").astype(float)
    if branch in {"detrended", "AR1_plus_detrended"}:
        arr = pd.DataFrame(
            {
                col: signal.detrend(arr[col].interpolate(limit_direction="both").fillna(arr[col].mean()))
                for col in arr.columns
            },
            index=arr.index,
        )
    if branch in {"AR1_residualized", "AR1_plus_detrended"}:
        arr = pd.DataFrame(
            {col: _ar_residualize_array(arr[col].to_numpy(dtype=float), order=1) for col in arr.columns},
            index=arr.index,
        )
    return arr.reset_index(drop=True)


def _matrix_values(matrix: np.ndarray, *, directed: bool, pvalue_based: bool) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray([], dtype=float)
    mask = np.isfinite(arr)
    if arr.shape[0] == arr.shape[1]:
        np.fill_diagonal(mask, False)
        if not directed:
            mask &= np.triu(np.ones(arr.shape, dtype=bool), 1)
    vals = arr[mask]
    if pvalue_based:
        vals = vals[(vals >= 0.0) & (vals <= 1.0)]
    return vals[np.isfinite(vals)]


def _matrix_summary(
    matrix: np.ndarray,
    *,
    metric: str,
    directed: bool,
    pvalue_based: bool,
) -> dict[str, float | int | str]:
    vals = _matrix_values(matrix, directed=directed, pvalue_based=pvalue_based)
    arr = np.asarray(matrix, dtype=float)
    finite_fraction = _safe_float(np.isfinite(arr).mean()) if arr.size else float("nan")
    row: dict[str, float | int | str] = {
        "metric": metric,
        "n_values": int(vals.size),
        "finite_fraction": finite_fraction,
        "mean_abs": _safe_float(np.nanmean(np.abs(vals))) if vals.size else float("nan"),
        "median": _safe_float(np.nanmedian(vals)) if vals.size else float("nan"),
        "std": _safe_float(np.nanstd(vals)) if vals.size else float("nan"),
        "positive_fraction": _safe_float(np.nanmean(vals > 0.0)) if vals.size else float("nan"),
        "negative_fraction": _safe_float(np.nanmean(vals < 0.0)) if vals.size else float("nan"),
        "p_below_0_05_fraction": float("nan"),
        "p_median": float("nan"),
    }
    if pvalue_based:
        row["p_below_0_05_fraction"] = _safe_float(np.nanmean(vals < 0.05)) if vals.size else float("nan")
        row["p_median"] = _safe_float(np.nanmedian(vals)) if vals.size else float("nan")
    return row


def _vector_for_compare(matrix: np.ndarray, *, directed: bool) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return arr[np.isfinite(arr)]
    mask = ~np.eye(arr.shape[0], dtype=bool)
    if not directed:
        mask &= np.triu(np.ones(arr.shape, dtype=bool), 1)
    return arr[mask]


def _rank_overlap(a: np.ndarray, b: np.ndarray, top_k: int = 100) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    aa = np.asarray(a[mask], dtype=float)
    bb = np.asarray(b[mask], dtype=float)
    if aa.size == 0:
        return float("nan")
    k = min(int(top_k), aa.size)
    top_a = set(np.argsort(np.abs(aa))[-k:])
    top_b = set(np.argsort(np.abs(bb))[-k:])
    return float(len(top_a & top_b) / max(1, k))


def _compare_matrices(
    a: np.ndarray,
    b: np.ndarray,
    *,
    directed: bool,
) -> dict[str, float]:
    va = _vector_for_compare(a, directed=directed)
    vb = _vector_for_compare(b, directed=directed)
    n = min(va.size, vb.size)
    va = va[:n]
    vb = vb[:n]
    mask = np.isfinite(va) & np.isfinite(vb)
    if not np.any(mask):
        return {
            "matrix_correlation": float("nan"),
            "mean_abs_delta": float("nan"),
            "fraction_edges_changing_sign": float("nan"),
            "edge_rank_overlap_top100": float("nan"),
        }
    aa = va[mask]
    bb = vb[mask]
    corr = float(np.corrcoef(aa, bb)[0, 1]) if aa.size >= 3 and np.std(aa) > 1e-12 and np.std(bb) > 1e-12 else float("nan")
    return {
        "matrix_correlation": corr,
        "mean_abs_delta": _safe_float(np.mean(np.abs(aa - bb))),
        "fraction_edges_changing_sign": _safe_float(np.mean(np.sign(aa) != np.sign(bb))),
        "edge_rank_overlap_top100": _rank_overlap(aa, bb, top_k=100),
    }


def _metric_properties(metric: str) -> tuple[bool, bool]:
    if metric in {"correlation_directed", "granger_full"}:
        return True, metric == "granger_full"
    return False, False


def _select_roi_columns(decisions: pd.DataFrame, *, include_sensitivity: bool = False) -> list[str]:
    include_col = "include_review_roi_include" if include_sensitivity else "primary_stage2_include"
    selected = decisions[decisions[include_col].astype(bool)].copy()
    selected = selected.sort_values("roi_index_0based")
    return [f"roi_{int(idx):03d}" for idx in selected["roi_index_0based"]]


def _select_granger_columns(decisions: pd.DataFrame, max_roi: int) -> list[str]:
    primary = decisions[decisions["primary_stage2_include"].astype(bool)].copy()
    primary["warning_frequency"] = (
        pd.to_numeric(primary["high_acf_frequency"], errors="coerce").fillna(0)
        + pd.to_numeric(primary["spectral_warning_frequency"], errors="coerce").fillna(0)
        + pd.to_numeric(primary["stationarity_review_frequency"], errors="coerce").fillna(0)
        + pd.to_numeric(primary["extreme_amplitude_frequency"], errors="coerce").fillna(0)
    )
    primary = primary.sort_values(["warning_frequency", "roi_index_0based"]).head(int(max_roi))
    return [f"roi_{int(idx):03d}" for idx in primary["roi_index_0based"]]


def run_sanity(
    *,
    hc_dir: Path,
    sz_dir: Path,
    characterization_dir: Path,
    decision_dir: Path,
    output_dir: Path,
    atlas: str,
    lags: tuple[int, ...],
    n_jobs: int,
    granger_max_roi: int,
) -> None:
    os.environ["TS_TOOL_N_JOBS"] = str(max(1, int(n_jobs)))
    os.environ.setdefault("TS_TOOL_PARALLEL_BACKEND", "threading")
    matrices_dir = output_dir / "matrices"
    summaries_dir = output_dir / "summaries"
    reports_dir = output_dir / "reports"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_csv(decision_dir / "roi_decision_layer_v2.csv")
    decisions = decisions[decisions["atlas"].eq(atlas)].copy()
    primary_cols = _select_roi_columns(decisions)
    sensitivity_cols = _select_roi_columns(decisions, include_sensitivity=True)
    granger_cols = _select_granger_columns(decisions, granger_max_roi)
    inventory = scan_inventory(hc_dir, sz_dir, atlas_filter=atlas)
    subjects = load_valid_subjects(inventory)

    summary_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    matrix_cache: dict[tuple[str, str, str, str, int], np.ndarray] = {}

    for subject in subjects:
        base_all = subject.data_time_roi
        missing = [col for col in primary_cols if col not in base_all.columns]
        if missing:
            failure_rows.append(
                {
                    "group": subject.group,
                    "subject_id": subject.subject_id,
                    "metric": "all",
                    "branch": "all",
                    "lag": "",
                    "error": f"missing primary ROI columns: {missing[:5]}",
                }
            )
            continue
        branch_data = {
            branch: _preprocess_branch(base_all[primary_cols], branch) for branch in PRIMARY_BRANCHES
        }
        sensitivity_data = {
            branch: _preprocess_branch(base_all[sensitivity_cols], branch)
            for branch in ("baseline",)
            if sensitivity_cols
        }
        granger_data = {
            branch: _preprocess_branch(base_all[granger_cols], branch)
            for branch in PRIMARY_BRANCHES
            if granger_cols
        }
        for branch in PRIMARY_BRANCHES:
            for metric in SANITY_METRICS:
                directed, pvalue_based = _metric_properties(metric)
                use_lags = lags if directed else (1,)
                for lag in use_lags:
                    data = granger_data[branch] if metric == "granger_full" else branch_data[branch]
                    key = (subject.group, subject.subject_id, branch, metric, int(lag))
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            matrix = compute_metric(data, metric, lag=int(lag))
                        matrix_cache[key] = matrix
                        out_path = (
                            matrices_dir
                            / atlas
                            / branch
                            / metric
                            / f"{subject.group}_{subject.subject_id}_lag{lag}.npy"
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(out_path, matrix)
                        row = {
                            "atlas": atlas,
                            "group": subject.group,
                            "subject_id": subject.subject_id,
                            "branch": branch,
                            "metric": metric,
                            "lag": int(lag),
                            "n_roi": int(data.shape[1]),
                            "matrix_path": str(out_path),
                            "n_warnings": len(caught),
                            **_matrix_summary(
                                matrix,
                                metric=metric,
                                directed=directed,
                                pvalue_based=pvalue_based,
                            ),
                        }
                        summary_rows.append(row)
                    except Exception as exc:
                        failure_rows.append(
                            {
                                "atlas": atlas,
                                "group": subject.group,
                                "subject_id": subject.subject_id,
                                "branch": branch,
                                "metric": metric,
                                "lag": int(lag),
                                "error": str(exc),
                            }
                        )
            if branch == "baseline" and sensitivity_data:
                for metric in ("correlation_full", "correlation_directed"):
                    directed, pvalue_based = _metric_properties(metric)
                    use_lags = lags if directed else (1,)
                    for lag in use_lags:
                        try:
                            matrix = compute_metric(sensitivity_data["baseline"], metric, lag=int(lag))
                            summary_rows.append(
                                {
                                    "atlas": atlas,
                                    "group": subject.group,
                                    "subject_id": subject.subject_id,
                                    "branch": "include_review_roi",
                                    "metric": metric,
                                    "lag": int(lag),
                                    "n_roi": int(sensitivity_data["baseline"].shape[1]),
                                    "matrix_path": "",
                                    "n_warnings": 0,
                                    **_matrix_summary(
                                        matrix,
                                        metric=metric,
                                        directed=directed,
                                        pvalue_based=pvalue_based,
                                    ),
                                }
                            )
                        except Exception as exc:
                            failure_rows.append(
                                {
                                    "atlas": atlas,
                                    "group": subject.group,
                                    "subject_id": subject.subject_id,
                                    "branch": "include_review_roi",
                                    "metric": metric,
                                    "lag": int(lag),
                                    "error": str(exc),
                                }
                            )

        comparisons = [
            ("baseline", "AR1_residualized"),
            ("baseline", "detrended"),
            ("baseline", "AR1_plus_detrended"),
        ]
        for metric in SANITY_METRICS:
            directed, _pvalue_based = _metric_properties(metric)
            use_lags = lags if directed else (1,)
            for lag in use_lags:
                for left, right in comparisons:
                    k1 = (subject.group, subject.subject_id, left, metric, int(lag))
                    k2 = (subject.group, subject.subject_id, right, metric, int(lag))
                    if k1 not in matrix_cache or k2 not in matrix_cache:
                        continue
                    stability_rows.append(
                        {
                            "atlas": atlas,
                            "group": subject.group,
                            "subject_id": subject.subject_id,
                            "metric": metric,
                            "lag": int(lag),
                            "left_branch": left,
                            "right_branch": right,
                            **_compare_matrices(matrix_cache[k1], matrix_cache[k2], directed=directed),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    stability = pd.DataFrame(
        stability_rows,
        columns=[
            "atlas",
            "group",
            "subject_id",
            "metric",
            "lag",
            "left_branch",
            "right_branch",
            "matrix_correlation",
            "mean_abs_delta",
            "fraction_edges_changing_sign",
            "edge_rank_overlap_top100",
        ],
    )
    failures = pd.DataFrame(
        failure_rows,
        columns=["atlas", "group", "subject_id", "branch", "metric", "lag", "error"],
    )
    _write_csv(summary, output_dir / "stage2_sanity_summary.csv")
    _write_csv(stability, summaries_dir / "stage2_sanity_stability.csv")
    _write_csv(failures, summaries_dir / "stage2_sanity_failures.csv")
    if not summary.empty:
        by_group = (
            summary.groupby(["atlas", "branch", "metric", "lag", "group"], as_index=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                median_mean_abs=("mean_abs", "median"),
                median_finite_fraction=("finite_fraction", "median"),
                median_p_below_0_05_fraction=("p_below_0_05_fraction", "median"),
            )
            .reset_index(drop=True)
        )
        _write_csv(by_group, summaries_dir / "stage2_sanity_group_summary.csv")
    else:
        by_group = pd.DataFrame()

    report = [
        "# Stage 2 Sanity Stability Report",
        "",
        "This is a small sanity run, not the full metric x lag x window grid.",
        "",
        f"- Atlas: `{atlas}`",
        f"- Primary ROI count: {len(primary_cols)}",
        f"- Include-review ROI count: {len(sensitivity_cols)}",
        f"- Granger-lite ROI count: {len(granger_cols)}",
        f"- Subjects loaded: {len(subjects)}",
        f"- Lags: {', '.join(str(lag) for lag in lags)}",
        "",
        "## Summary Rows",
        "",
        f"- Matrix summary rows: {len(summary)}",
        f"- Stability rows: {len(stability)}",
        f"- Failure rows: {len(failures)}",
        "",
        "## Group Summary",
        "",
        _markdown_table(by_group.head(80) if not by_group.empty else by_group),
        "",
        "## Stability Snapshot",
        "",
        _markdown_table(stability.head(80) if not stability.empty else stability),
        "",
        "## Failures",
        "",
        _markdown_table(failures.head(80) if not failures.empty else failures),
        "",
        "## Interpretation",
        "",
        "Use this report only to decide whether full Stage 2 is safe to open. Strong branch deltas, many sign changes, or Granger failures mean the full grid should be run cautiously and reported as sensitivity-heavy.",
    ]
    (reports_dir / "stage2_sanity_stability_report.md").write_text("\n".join(report), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--characterization-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas", default="AAL3")
    parser.add_argument("--aal3-regions", default=None)
    parser.add_argument("--lags", default="1,2,3")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--granger-max-roi", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_sanity(
        hc_dir=Path(args.hc_dir),
        sz_dir=Path(args.sz_dir),
        characterization_dir=Path(args.characterization_dir),
        decision_dir=Path(args.decision_dir),
        output_dir=Path(args.output_dir),
        atlas=str(args.atlas),
        lags=_split_int_csv(args.lags),
        n_jobs=int(args.n_jobs),
        granger_max_roi=int(args.granger_max_roi),
    )


if __name__ == "__main__":
    main()
