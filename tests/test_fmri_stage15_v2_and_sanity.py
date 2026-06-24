from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_fmri_stage15_decisions import (
    build_roi_decisions_v2,
    build_subject_decisions_v2,
    write_branch_recommendations_v2,
)
from scripts.run_fmri_stage2_sanity import run_sanity


def _features(subjects: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for group, subject_id in subjects:
        for roi in range(1, 168):
            row = {
                "atlas": "AAL3",
                "group": group,
                "subject_id": subject_id,
                "roi_index_0based": roi - 1,
                "roi_index_1based": roi,
                "zero_flag": False,
                "constant_flag": False,
                "extreme_amplitude_flag": False,
                "nan_flag": False,
                "acf_lag_1": 0.1,
                "linear_trend_slope": 0.01,
                "linear_trend_r2": 0.001,
                "mean_shift_second_minus_first": 0.0,
                "std_ratio_second_to_first": 1.0,
                "low_high_power_ratio": 0.5,
                "spectral_entropy": 0.9,
                "spectral_slope": -0.5,
                "amplitude": float(rng.uniform(1.0, 2.0)),
                "std": 1.0,
            }
            if roi in {35, 36, 81, 82}:
                row["zero_flag"] = True
                row["constant_flag"] = True
            if roi in {133, 134, 167}:
                row["extreme_amplitude_flag"] = True
                row["amplitude"] = 100.0
            if roi == 160:
                row["extreme_amplitude_flag"] = group == "HC" and subject_id == "hc01"
                row["amplitude"] = 50.0
            if roi == 10:
                row["acf_lag_1"] = 0.9
            if roi == 11:
                row["spectral_entropy"] = 0.3
            if group == "SZ" and subject_id == "1177_Ivanov_S_A" and roi in {106, 111}:
                row["zero_flag"] = True
                row["constant_flag"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def _regions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "roi_index_1based": list(range(1, 168)),
            "region_name": ["Background", *[f"Region_{idx}" for idx in range(2, 168)]],
        }
    )


def _subject_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "atlas": "AAL3",
                "group": "HC",
                "subject_id": "hc01",
                "n_zero_roi": 4,
                "n_constant_roi": 4,
                "n_extreme_amplitude_roi": 4,
                "warning_count": 10,
                "median_ar1": 0.75,
            },
            {
                "atlas": "AAL3",
                "group": "SZ",
                "subject_id": "1177_Ivanov_S_A",
                "n_zero_roi": 6,
                "n_constant_roi": 6,
                "n_extreme_amplitude_roi": 5,
                "warning_count": 17,
                "median_ar1": 0.6,
            },
        ]
    )


def test_stage15_v2_roi_decision_semantics() -> None:
    features = _features([("HC", "hc01"), ("SZ", "1177_Ivanov_S_A")])
    roi, _diagnostic = build_roi_decisions_v2(features, _regions())
    by_roi = roi.set_index("roi_index_1based")

    assert by_roi.loc[1, "region_name"] == "Background"
    assert by_roi.loc[1, "decision"] == "exclude_conservative"
    assert "background_node" in by_roi.loc[1, "reason_category"]

    for roi_id in [35, 36, 81, 82]:
        assert by_roi.loc[roi_id, "decision"] == "exclude_conservative"

    for roi_id in [133, 134, 167]:
        assert by_roi.loc[roi_id, "decision"] == "sensitivity_only"
        assert not bool(by_roi.loc[roi_id, "primary_stage2_include"])
        assert bool(by_roi.loc[roi_id, "include_review_roi_include"])

    assert by_roi.loc[160, "decision"] == "qc_flag_keep"
    assert bool(by_roi.loc[160, "primary_stage2_include"])
    assert by_roi.loc[106, "decision"] == "qc_flag_keep"
    assert by_roi.loc[111, "decision"] == "qc_flag_keep"
    assert by_roi.loc[10, "decision"] == "qc_flag_keep"
    assert by_roi.loc[11, "decision"] == "qc_flag_keep"


def test_stage15_v2_subjects_and_branch_policy(tmp_path: Path) -> None:
    features = _features([("HC", "hc01"), ("SZ", "1177_Ivanov_S_A")])
    subjects = build_subject_decisions_v2(features, _subject_summary())
    by_subject = subjects.set_index("subject_id")

    assert by_subject.loc["1177_Ivanov_S_A", "decision"] == "sensitivity_review"
    assert "subject_specific_zero_constant" in by_subject.loc["1177_Ivanov_S_A", "reason_category"]
    assert bool(by_subject.loc["1177_Ivanov_S_A", "primary_stage2_include"])
    assert by_subject.loc["hc01", "decision"] == "qc_flag_keep"

    policy_path = tmp_path / "preprocessing_branch_recommendations_v2.csv"
    write_branch_recommendations_v2(policy_path)
    policy = pd.read_csv(policy_path).set_index("branch")
    assert "keep + qc_flag_keep" in policy.loc["baseline", "roi_policy"]
    assert "sensitivity_only" in policy.loc["include_review_roi", "roi_policy"]
    assert "exclude_conservative" in policy.loc["include_review_roi", "roi_policy"]


def _write_roi_csv(path: Path, seed: int, *, n_time: int = 48) -> None:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(167, n_time))
    data[34] = 0.0
    data[35] = 0.0
    data[80] = 0.0
    data[81] = 0.0
    pd.DataFrame(data).to_csv(path, header=False, index=False)


def _write_decisions(path: Path) -> None:
    rows = []
    for roi in range(1, 168):
        decision = "keep"
        primary = True
        include_review = True
        if roi in {1, 35, 36, 81, 82}:
            decision = "exclude_conservative"
            primary = False
            include_review = False
        elif roi in {133, 134, 167}:
            decision = "sensitivity_only"
            primary = False
            include_review = True
        elif roi in {10, 11, 106, 111, 160}:
            decision = "qc_flag_keep"
        rows.append(
            {
                "atlas": "AAL3",
                "roi_index_0based": roi - 1,
                "roi_index_1based": roi,
                "decision": decision,
                "primary_stage2_include": primary,
                "include_review_roi_include": include_review,
                "high_acf_frequency": 0.0,
                "spectral_warning_frequency": 0.0,
                "stationarity_review_frequency": 0.0,
                "extreme_amplitude_frequency": 0.0,
            }
        )
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stage2_sanity_smoke_continues_with_granger_failures(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", 1)
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", 2)
    decision_dir = tmp_path / "decisions"
    _write_decisions(decision_dir / "roi_decision_layer_v2.csv")

    out = tmp_path / "stage2_sanity"
    run_sanity(
        hc_dir=hc,
        sz_dir=sz,
        characterization_dir=tmp_path,
        decision_dir=decision_dir,
        output_dir=out,
        atlas="AAL3",
        lags=(1,),
        n_jobs=1,
        granger_max_roi=4,
    )

    summary = pd.read_csv(out / "stage2_sanity_summary.csv")
    failures = pd.read_csv(out / "summaries" / "stage2_sanity_failures.csv")
    assert {
        "correlation_full",
        "correlation_partial",
        "wavelet_full",
        "correlation_directed",
    }.issubset(set(summary["metric"]))
    assert (out / "summaries" / "stage2_sanity_stability.csv").exists()
    assert (out / "reports" / "stage2_sanity_stability_report.md").exists()
    assert "granger_full" in set(summary["metric"]) or "granger_full" in set(failures["metric"])
