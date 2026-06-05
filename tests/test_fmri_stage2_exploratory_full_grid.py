from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_fmri_stage2_exploratory_full_grid as full_grid
from scripts.run_fmri_stage2_exploratory_full_grid import run_full_grid
from scripts.run_fmri_stage2_sanity import _select_roi_columns


def _write_roi_csv(path: Path, seed: int, *, n_time: int = 40) -> None:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(167, n_time))
    t = np.linspace(-1, 1, n_time)
    data[1] = t + rng.normal(scale=0.05, size=n_time)
    data[2] = -t + rng.normal(scale=0.05, size=n_time)
    for idx in [34, 35, 80, 81]:
        data[idx] = 0.0
    pd.DataFrame(data).to_csv(path, header=False, index=False)


def _decisions(path: Path) -> pd.DataFrame:
    rows = []
    for roi in range(1, 168):
        decision = "keep"
        primary = True
        review = True
        if roi in {1, 35, 36, 81, 82}:
            decision = "exclude_conservative"
            primary = False
            review = False
        elif roi in {133, 134, 167}:
            decision = "sensitivity_only"
            primary = False
            review = True
        elif roi in {10, 11, 106, 111, 160}:
            decision = "qc_flag_keep"
        rows.append(
            {
                "atlas": "AAL3",
                "roi_index_0based": roi - 1,
                "roi_index_1based": roi,
                "region_name": "Background" if roi == 1 else f"Region_{roi}",
                "decision": decision,
                "primary_stage2_include": primary,
                "include_review_roi_include": review,
                "high_acf_frequency": 0.0,
                "spectral_warning_frequency": 0.0,
                "stationarity_review_frequency": 0.0,
                "extreme_amplitude_frequency": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def test_roi_selection_primary_and_sensitivity_policy(tmp_path: Path) -> None:
    decisions = _decisions(tmp_path / "decisions" / "roi_decision_layer_v2.csv")
    primary = _select_roi_columns(decisions)
    sensitivity = _select_roi_columns(decisions, include_sensitivity=True)

    assert "roi_000" not in primary
    assert "roi_034" not in primary
    assert "roi_132" not in primary
    assert "roi_132" in sensitivity
    assert "roi_000" not in sensitivity
    assert "roi_009" in primary
    assert "roi_159" in primary


def test_exploratory_full_grid_writes_required_outputs(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", 1)
    _write_roi_csv(hc / "hc02_AAL3_timeseries.csv", 2)
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", 3)
    _write_roi_csv(sz / "sz02_AAL3_timeseries.csv", 4)
    decision_dir = tmp_path / "decisions"
    _decisions(decision_dir / "roi_decision_layer_v2.csv")
    out = tmp_path / "full_grid"

    run_full_grid(
        hc_dir=hc,
        sz_dir=sz,
        decision_dir=decision_dir,
        output_dir=out,
        atlas="AAL3",
        lags=(1,),
        windows=("full",),
        n_jobs=1,
        granger_max_roi=3,
        alpha=0.05,
        metrics=("correlation_full",),
        max_primary_roi=6,
    )

    required = [
        "stage2_full_edge_results.csv",
        "stage2_branch_stability.csv",
        "stage2_metric_reliability.csv",
        "stage2_candidate_subnetworks.md",
        "stage2_exploratory_full_report.md",
        "stage2_full_failures.csv",
    ]
    for name in required:
        assert (out / name).exists()

    edges = pd.read_csv(out / "stage2_full_edge_results.csv")
    stability = pd.read_csv(out / "stage2_branch_stability.csv")
    reliability = pd.read_csv(out / "stage2_metric_reliability.csv")

    assert {
        "survives_baseline",
        "survives_detrended",
        "survives_AR1",
        "survives_AR1_plus_detrended",
        "stability_score",
    }.issubset(edges.columns)
    assert {
        "baseline_vs_AR1_residualized",
        "baseline_vs_detrended",
        "baseline_vs_AR1_plus_detrended",
    }.issubset(set(stability["branch_pair"]))
    assert "recommendation" in reliability.columns
    assert set(reliability["recommendation"]).issubset(
        {"primary_candidate", "secondary_candidate", "sensitivity_only", "do_not_trust_yet"}
    )


def test_exploratory_full_grid_records_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", 1, n_time=10)
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", 2, n_time=10)
    decision_dir = tmp_path / "decisions"
    _decisions(decision_dir / "roi_decision_layer_v2.csv")
    out = tmp_path / "full_grid_failures"

    def _raise_metric(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("synthetic metric failure")

    monkeypatch.setattr(full_grid, "compute_metric", _raise_metric)
    run_full_grid(
        hc_dir=hc,
        sz_dir=sz,
        decision_dir=decision_dir,
        output_dir=out,
        atlas="AAL3",
        lags=(8,),
        windows=("full",),
        n_jobs=1,
        granger_max_roi=3,
        alpha=0.05,
        metrics=("correlation_full",),
        max_primary_roi=6,
    )

    failures = pd.read_csv(out / "stage2_full_failures.csv")
    assert failures.columns.tolist() == [
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
    assert (out / "stage2_candidate_subnetworks.md").exists()
