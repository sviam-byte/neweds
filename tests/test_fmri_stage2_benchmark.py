from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neweds.core.fmri_stage2_benchmark import (
    EXPECTED_STAGE2_METRICS,
    FmriStage2Config,
    PreprocessingBranchConfig,
    discover_hcp_inputs,
    matrix_to_features,
    nested_loocv_predictions,
    paired_representation_comparison,
    preprocess_regional_timeseries,
    registry_snapshot,
    run_metric_stage,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_hcp_result(tmp_path: Path, n_subjects: int = 6) -> Path:
    result = tmp_path / "hcp"
    subjects = result / "subjects"
    whole = result / "whole_brain"
    subjects.mkdir(parents=True)
    whole.mkdir()
    node_order = [1, 2, 3, 4]
    status_rows = []
    paired_rows = []
    rng = np.random.default_rng(123)
    n_time = 600
    for idx in range(n_subjects):
        sid = f"{1000 + idx}"
        group = "HC" if idx < n_subjects // 2 else "SZ"
        base = rng.normal(size=(4, n_time)).astype(np.float32)
        if group == "SZ":
            base[0] += base[1] * 0.8
        gm_path = subjects / f"{sid}_HCP-MMP1-360_gm_signals.npz"
        np.savez_compressed(
            gm_path,
            active_mean_z=base,
            pca_pc1_oriented_z=base,
            ica_1_oriented_z=base,
            correlation_core_z=base,
        )
        wb_path = whole / f"{sid}_HCP-MMP1-360_whole_brain.npz"
        np.savez_compressed(wb_path, z=base + rng.normal(scale=0.01, size=base.shape))
        status_rows.append(
            {
                "subject_id": sid,
                "group": group,
                "atlas_id": "HCP-MMP1-360",
                "status": "ok",
                "signal_npz": str(gm_path),
                "node_order": node_order,
            }
        )
        paired_rows.append(
            {
                "subject_id": sid,
                "group": group,
                "atlas_id": "HCP-MMP1-360",
                "status": "ok",
                "output_npz": str(wb_path),
                "node_order": node_order,
            }
        )
    status_rows.append(
        {
            "subject_id": "1186",
            "group": "HC",
            "atlas_id": "HCP-MMP1-360",
            "status": "blocked_recovery",
            "signal_npz": "",
            "node_order": node_order,
        }
    )
    _write_jsonl(result / "subject_status.jsonl", status_rows)
    _write_jsonl(result / "paired_input_manifest.jsonl", paired_rows)
    pd.DataFrame({"region_id": node_order, "region_name": [f"r{i}" for i in node_order]}).to_csv(
        result / "node_table.csv", index=False
    )
    return result


def test_discover_hcp_inputs_requires_paired_node_order_and_skips_blocked(tmp_path: Path) -> None:
    result = _make_hcp_result(tmp_path, n_subjects=2)
    manifests, alignment = discover_hcp_inputs(
        result, representations=("gm_active_mean", "whole_brain")
    )

    assert {item.subject_id for item in manifests} == {"1000", "1001"}
    assert len(manifests) == 4
    assert not bool(alignment.loc[alignment["subject_id"].eq("1186"), "paired_stage2_eligible"].item())
    for item in manifests:
        assert item.node_order == (1, 2, 3, 4)


def test_preprocessing_preserves_shape_and_gsr_is_subject_local() -> None:
    t = np.linspace(0, 4, 20)
    values = np.vstack(
        [
            np.sin(t) + 0.1 * t,
            np.cos(t * 0.7) + 0.05 * t,
            np.sin(t * 1.3 + 0.2),
        ]
    )
    branch = PreprocessingBranchConfig("AR1_plus_detrended_with_GSR", detrend=True, ar_order=1, gsr_mode="representation_global")

    processed = preprocess_regional_timeseries(values, branch)

    assert processed.shape == (20, 3)
    assert np.isnan(processed[0]).all()
    assert np.isfinite(processed[2:]).any()


def test_matrix_to_features_symmetric_and_directed_keep_nan() -> None:
    matrix = np.array([[1.0, 0.2, np.nan], [0.2, 1.0, -0.1], [np.nan, -0.1, 1.0]])

    sym, sym_ids = matrix_to_features(matrix, [10, 20, 30], directed=False)
    directed, directed_ids = matrix_to_features(matrix, [10, 20, 30], directed=True)

    assert sym_ids == ["10--20", "10--30", "20--30"]
    assert np.isnan(sym[1])
    assert len(directed) == 6
    assert "10->20" in directed_ids


def test_registry_snapshot_contains_expected_26_metrics() -> None:
    snapshot = registry_snapshot()

    assert set(EXPECTED_STAGE2_METRICS).issubset(set(snapshot["metric"]))
    assert len(EXPECTED_STAGE2_METRICS) == 26
    assert {"wavelet_full", "wavelet_partial"}.issubset(set(snapshot["metric"]))


def test_nested_loocv_handles_nan_features_inside_pipeline() -> None:
    X = np.array(
        [
            [0.0, np.nan, 1.0],
            [0.1, 1.0, 1.1],
            [0.2, 1.1, 0.9],
            [2.0, 1.2, 0.0],
            [2.1, np.nan, 0.1],
            [2.2, 1.3, -0.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    pred, score, stability = nested_loocv_predictions(
        X, y, model="l1_logistic", feature_ids=["a", "b", "c"], seed=7
    )

    assert pred.shape == y.shape
    assert np.isfinite(score).all()
    assert isinstance(stability, list)


def test_paired_representation_comparison_uses_matched_subjects_only() -> None:
    oof = pd.DataFrame(
        [
            {"subject_id": "1", "true_group": "HC", "representation": "gm_active_mean", "branch": "baseline_without_GSR", "metric": "correlation_full", "model": "l1_logistic", "lag": 1, "oof_score_SZ": 0.1, "oof_predicted_group": "HC"},
            {"subject_id": "2", "true_group": "SZ", "representation": "gm_active_mean", "branch": "baseline_without_GSR", "metric": "correlation_full", "model": "l1_logistic", "lag": 1, "oof_score_SZ": 0.9, "oof_predicted_group": "SZ"},
            {"subject_id": "1", "true_group": "HC", "representation": "whole_brain", "branch": "baseline_without_GSR", "metric": "correlation_full", "model": "l1_logistic", "lag": 1, "oof_score_SZ": 0.2, "oof_predicted_group": "HC"},
            {"subject_id": "2", "true_group": "SZ", "representation": "whole_brain", "branch": "baseline_without_GSR", "metric": "correlation_full", "model": "l1_logistic", "lag": 1, "oof_score_SZ": 0.8, "oof_predicted_group": "SZ"},
            {"subject_id": "3", "true_group": "SZ", "representation": "whole_brain", "branch": "baseline_without_GSR", "metric": "correlation_full", "model": "l1_logistic", "lag": 1, "oof_score_SZ": 0.7, "oof_predicted_group": "SZ"},
        ]
    )

    paired = paired_representation_comparison(oof, pd.DataFrame(), bootstraps=2, seed=1)

    assert paired.empty

    performance = pd.DataFrame({"dummy": [1]})
    paired = paired_representation_comparison(oof, performance, bootstraps=2, seed=1)
    assert paired.loc[0, "n_subjects"] == 2


def test_run_metric_stage_smoke_writes_manifest_and_status(tmp_path: Path) -> None:
    hcp = _make_hcp_result(tmp_path, n_subjects=6)
    metrics_dir = tmp_path / "metrics"
    class_dir = tmp_path / "class"
    cfg = FmriStage2Config(
        gm_hcp_result=str(hcp),
        whole_brain_hcp_inputs=str(hcp),
        new_results_root=str(tmp_path),
        metrics_result_dir=str(metrics_dir),
        classification_result_dir=str(class_dir),
        representations=("gm_active_mean", "whole_brain"),
        metrics=("correlation_full",),
        branches=("baseline_without_GSR",),
        permutations=0,
        bootstraps=0,
        smoke=False,
    )

    status, features, temporal = run_metric_stage(cfg)

    assert not status.empty
    assert set(status["status"]) == {"ok"}
    assert (metrics_dir / "metrics" / "feature_manifest.parquet").is_file()
    assert len(features) == 12
    assert len(temporal) == 12
