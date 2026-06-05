from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import neweds.cli_fmri_audit as cli_fmri_audit
from neweds.core.fmri_roi_audit import (
    AAL3_EXPECTED_ROI,
    build_aal3_region_mapping,
    build_common_bad_rois,
    build_hcp_mask_geometry_qc,
    build_roi_qc,
    build_subject_level_fc_summary,
    build_temporal_qc,
    build_threshold_bad_rois,
    compare_fc_edges,
    compare_fc_edges_ttest,
    compare_subject_level_fc,
    detect_atlas,
    detect_orientation,
    load_valid_subjects,
    parse_subject_id,
    pearson_fisher_fc,
    permutation_subject_level_fc,
    run_fmri_roi_audit,
    scan_inventory,
    summarize_subject_fc_matrix,
)


def _write_roi_csv(path: Path, data: np.ndarray) -> None:
    pd.DataFrame(data).to_csv(path, header=False, index=False)


def _aal3_data(seed: int, *, n_time: int = 32, zero_rows: tuple[int, ...] = ()) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(AAL3_EXPECTED_ROI, n_time))
    trend = np.linspace(-1.0, 1.0, n_time)
    data[0] = trend + rng.normal(scale=0.01, size=n_time)
    data[1] = -trend + rng.normal(scale=0.01, size=n_time)
    for idx in zero_rows:
        data[idx] = 0.0
    return data


def _compact_aal3_data(seed: int, *, n_time: int = 40) -> np.ndarray:
    data = _aal3_data(seed, n_time=n_time, zero_rows=tuple(range(12, AAL3_EXPECTED_ROI)))
    return data


def test_filename_parsing_atlas_detection_and_orientation() -> None:
    assert detect_atlas("1097_Shulgina_Yu_E_AAL3_timeseries.csv") == "AAL3"
    assert detect_atlas("sub01_HCP_timeseries.csv") == "HCP"
    assert detect_atlas("sub01_unknown.csv") == "unknown"

    assert parse_subject_id("1097_Shulgina_Yu_E_AAL3_timeseries.csv", "AAL3") == "1097_Shulgina_Yu_E"
    assert detect_orientation(167, 600, "AAL3") == ("ROI_by_time", 167, 600)
    assert detect_orientation(600, 167, "AAL3") == ("time_by_ROI", 167, 600)
    assert detect_orientation(10, 20, "AAL3")[0] == "shape_error"


def test_scan_inventory_detects_shape_nan_zero_and_constant_regions(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    data = _aal3_data(1, zero_rows=(34,))
    data[2] = 5.0
    data[3, 0] = np.nan
    data[4, 1] = np.inf
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", data)
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", data.T)
    _write_roi_csv(hc / "bad_AAL3_timeseries.csv", np.ones((10, 20)))

    inv = scan_inventory(hc, sz, atlas_filter="all")
    rows = inv.set_index("file_name")

    assert rows.loc["hc01_AAL3_timeseries.csv", "orientation"] == "ROI_by_time"
    assert rows.loc["sz01_AAL3_timeseries.csv", "orientation"] == "time_by_ROI"
    assert rows.loc["bad_AAL3_timeseries.csv", "status"] == "shape_error"
    assert int(rows.loc["hc01_AAL3_timeseries.csv", "n_nan"]) == 1
    assert int(rows.loc["hc01_AAL3_timeseries.csv", "n_inf"]) == 1
    assert int(rows.loc["hc01_AAL3_timeseries.csv", "n_zero_rows"]) == 1
    assert int(rows.loc["hc01_AAL3_timeseries.csv", "n_constant_regions"]) >= 2


def test_aal3_mapping_marks_background_and_known_zero_indices(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", _aal3_data(1, zero_rows=(34,)))
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", _aal3_data(2))
    regions = tmp_path / "aal3_regions.txt"
    regions.write_text(
        "\n".join(["1 Background", *[f"{i} Region_{i}" for i in range(2, 168)]]),
        encoding="utf-8",
    )

    subjects = load_valid_subjects(scan_inventory(hc, sz))
    qc = build_roi_qc(subjects)
    mapping = build_aal3_region_mapping(regions, qc)

    assert bool(mapping.loc[0, "is_background"])
    assert bool(mapping.loc[34, "is_zero_region_global"])
    assert bool(mapping.loc[34, "known_zero_example_index"])


def test_hcp_mask_geometry_qc_from_voxel_map(tmp_path: Path) -> None:
    voxel_map = tmp_path / "HCP-MMP1_atlas_voxel_map_from_xml.csv"
    pd.DataFrame(
        {
            "N": list(range(8)),
            "x": [0, 1, 1, 1, 2, 3, 4, 5],
            "y": [0, 0, 0, 1, 0, 1, 0, 0],
            "z": [0, 0, 1, 0, 0, 0, 0, 0],
            "region_id": [0, 1, 1, 2, 2, 3, 3, 4],
            "region_name": ["Background", "R1", "R1", "R2", "R2", "R3", "R3", "R4"],
        }
    ).to_csv(voxel_map, index=False)

    geometry, sizes, adjacency = build_hcp_mask_geometry_qc(voxel_map)

    assert int(geometry.loc[0, "total_voxels"]) == 8
    assert int(geometry.loc[0, "background_voxels"]) == 1
    assert int(geometry.loc[0, "non_background_voxels"]) == 7
    assert int(geometry.loc[0, "n_regions"]) == 4
    assert str(geometry.loc[0, "implied_grid_shape"]) == "6x2x2"
    assert int(geometry.loc[0, "small_regions"]) == 4
    assert set(sizes["region_id"]) == {"1", "2", "3", "4"}

    by_region = adjacency.set_index("region_id")
    assert int(by_region.loc["3", "n_connected_components_6"]) == 2
    assert int(by_region.loc["3", "n_connected_components_26"]) == 1
    assert "2" in str(by_region.loc["1", "neighbouring_region_ids_6"]).split(";")
    assert "3" not in str(by_region.loc["2", "neighbouring_region_ids_6"]).split(";")
    assert "3" in str(by_region.loc["2", "neighbouring_region_ids_26"]).split(";")
    assert int(by_region.loc["4", "boundary_voxel_count_6"]) == 1
    assert np.isclose(float(by_region.loc["4", "surface_to_volume_proxy_6"]), 1.0)


def test_common_bad_rois_temporal_qc_and_fc_group_comparison(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", _aal3_data(1, zero_rows=(34,)))
    _write_roi_csv(hc / "hc02_AAL3_timeseries.csv", _aal3_data(2, zero_rows=(34,)))
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", _aal3_data(3, zero_rows=(34,)))
    _write_roi_csv(sz / "sz02_AAL3_timeseries.csv", _aal3_data(4, zero_rows=(34,)))

    subjects = load_valid_subjects(scan_inventory(hc, sz))
    qc = build_roi_qc(subjects)
    bad = build_common_bad_rois(qc, atlas="AAL3")
    temporal_long, temporal_summary = build_temporal_qc(subjects)

    assert 34 in set(bad["roi_index_0based"].astype(int))
    assert {"acf_lag_1", "pacf_lag_1", "ar1_coeff"}.issubset(temporal_long.columns)
    assert set(temporal_summary["group"]) == {"HC", "SZ"}
    assert {"ar2_coeff_1", "ar2_coeff_2"}.issubset(temporal_long.columns)

    subject = subjects[0]
    kept = subject.data_time_roi.drop(columns=["roi_034"])
    matrix = pearson_fisher_fc(kept)
    assert matrix.shape == (166, 166)
    assert np.isfinite(matrix).all()

    edges = pd.DataFrame(
        {
            "group": ["HC", "HC", "SZ", "SZ"],
            "edge_i": [0, 0, 0, 0],
            "edge_j": [1, 1, 1, 1],
            "fisher_z": [0.1, 0.2, 0.8, 0.9],
        }
    )
    comparison = compare_fc_edges(edges, alpha=0.05)
    assert "q_value_FDR" in comparison.columns
    assert comparison.loc[0, "effect_size_rank_biserial_sz_gt_hc"] > 0


def test_temporal_qc_ar2_coefficients_are_finite(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    hc.mkdir()
    sz.mkdir()
    n_time = 48
    x = np.zeros(n_time)
    x[:2] = [0.2, -0.1]
    for t in range(2, n_time):
        x[t] = 0.55 * x[t - 1] - 0.20 * x[t - 2] + 0.01 * np.sin(t)
    data = np.zeros((AAL3_EXPECTED_ROI, n_time))
    data[0] = x
    data[1] = np.linspace(-1.0, 1.0, n_time)
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", data)
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", data)

    subjects = load_valid_subjects(scan_inventory(hc, sz))
    temporal_long, _ = build_temporal_qc(subjects)
    roi0 = temporal_long[
        temporal_long["subject_id"].eq("hc01") & temporal_long["roi_index_0based"].eq(0)
    ].iloc[0]

    assert np.isfinite(float(roi0["ar2_coeff_1"]))
    assert np.isfinite(float(roi0["ar2_coeff_2"]))


def test_threshold_bad_roi_sensitivity_and_ttest() -> None:
    qc = pd.DataFrame(
        {
            "atlas": ["AAL3"] * 6,
            "group": ["HC", "HC", "SZ", "HC", "HC", "SZ"],
            "subject_id": ["s1", "s2", "s3", "s1", "s2", "s3"],
            "roi_index_0based": [1, 1, 1, 2, 2, 2],
            "roi_index_1based": [2, 2, 2, 3, 3, 3],
            "zero_flag": [True, False, False, True, True, False],
            "constant_flag": [False, False, False, False, False, False],
        }
    )

    bad_05 = build_threshold_bad_rois(qc, atlas="AAL3", threshold=0.05)
    bad_20 = build_threshold_bad_rois(qc, atlas="AAL3", threshold=0.20)
    bad_70 = build_threshold_bad_rois(qc, atlas="AAL3", threshold=0.70)

    assert set(bad_05["roi_index_0based"]) == {1, 2}
    assert set(bad_20["roi_index_0based"]) == {1, 2}
    assert set(bad_70["roi_index_0based"]) == set()

    edges = pd.DataFrame(
        {
            "group": ["HC", "HC", "HC", "SZ", "SZ", "SZ"],
            "edge_i": [0, 0, 0, 0, 0, 0],
            "edge_j": [1, 1, 1, 1, 1, 1],
            "fisher_z": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )
    ttest = compare_fc_edges_ttest(edges, alpha=0.05)

    assert {"t_stat", "p_value", "q_value_FDR", "significant"}.issubset(ttest.columns)
    assert ttest.loc[0, "delta_mean"] > 0
    assert 0.0 <= ttest.loc[0, "q_value_FDR"] <= 1.0


def test_permutation_subject_level_fc_is_deterministic() -> None:
    summary = pd.DataFrame(
        {
            "atlas": ["AAL3", "AAL3", "AAL3", "AAL3"],
            "branch": ["raw_cleaned"] * 4,
            "group": ["HC", "HC", "SZ", "SZ"],
            "subject_id": ["h1", "h2", "s1", "s2"],
            "n_edges": [3, 3, 3, 3],
            "mean_fc": [0.1, 0.2, 0.8, 0.9],
            "global_strength": [1.0, 1.1, 2.0, 2.1],
        }
    )

    first = permutation_subject_level_fc(
        summary,
        n_permutations=25,
        random_seed=7,
        alpha=0.05,
    )
    second = permutation_subject_level_fc(
        summary,
        n_permutations=25,
        random_seed=7,
        alpha=0.05,
    )

    pd.testing.assert_frame_equal(first, second)
    assert {"mean_fc", "global_strength"} <= set(first["metric"])
    assert first["p_value"].between(0.0, 1.0).all()
    assert first["q_value_FDR"].between(0.0, 1.0).all()


def test_subject_level_fc_summary_and_group_comparison() -> None:
    matrix = np.array(
        [
            [0.0, 0.5, -0.3],
            [0.5, 0.0, 0.1],
            [-0.3, 0.1, 0.0],
        ],
        dtype=float,
    )

    summary = summarize_subject_fc_matrix(matrix)

    assert summary["n_edges"] == 3
    assert np.isclose(summary["mean_fc"], 0.1)
    assert np.isclose(summary["mean_abs_fc"], 0.3)
    assert np.isclose(summary["fraction_positive_edges"], 2 / 3)
    assert np.isclose(summary["fraction_negative_edges"], 1 / 3)
    assert np.isclose(summary["global_strength"], 0.9)
    assert np.isclose(summary["density_abs_z_ge_0_2"], 2 / 3)
    assert np.isclose(summary["density_abs_z_ge_0_4"], 1 / 3)
    assert np.isclose(summary["density_abs_z_ge_0_6"], 0.0)

    class _Subject:
        def __init__(self, group: str, subject_id: str) -> None:
            self.group = group
            self.subject_id = subject_id

    matrices = {
        "HC::s1::AAL3": matrix * 0.5,
        "HC::s2::AAL3": matrix * 0.6,
        "SZ::s1::AAL3": matrix * 1.5,
        "SZ::s2::AAL3": matrix * 1.6,
    }
    subjects = {
        "HC::s1::AAL3": _Subject("HC", "s1"),
        "HC::s2::AAL3": _Subject("HC", "s2"),
        "SZ::s1::AAL3": _Subject("SZ", "s1"),
        "SZ::s2::AAL3": _Subject("SZ", "s2"),
    }

    subject_summary = build_subject_level_fc_summary(
        matrices,
        subjects,
        atlas="AAL3",
        branch="raw_cleaned",
    )
    comparison = compare_subject_level_fc(subject_summary, alpha=0.05)

    assert {"mean_fc", "mean_abs_fc", "global_strength"}.issubset(subject_summary.columns)
    assert "q_value_FDR" in comparison.columns
    assert "n_edges" not in set(comparison["metric"])
    assert comparison.loc[comparison["metric"].eq("global_strength"), "delta_mean"].iloc[0] > 0


def test_run_fmri_roi_audit_writes_mvp_outputs(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    out = tmp_path / "audit_out"
    hc.mkdir()
    sz.mkdir()
    voxel_map = tmp_path / "HCP-MMP1_atlas_voxel_map_from_xml.csv"
    pd.DataFrame(
        {
            "N": [0, 1, 2],
            "x": [0, 1, 2],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "region_id": [0, 1, 2],
            "region_name": ["Background", "R1", "R2"],
        }
    ).to_csv(voxel_map, index=False)
    _write_roi_csv(hc / "sub01_AAL3_timeseries.csv", _compact_aal3_data(1))
    _write_roi_csv(hc / "hc02_AAL3_timeseries.csv", _compact_aal3_data(2))
    _write_roi_csv(sz / "sub01_AAL3_timeseries.csv", _compact_aal3_data(3))
    _write_roi_csv(sz / "sz02_AAL3_timeseries.csv", _compact_aal3_data(4))

    result = run_fmri_roi_audit(
        hc,
        sz,
        out,
        branches=("raw_cleaned",),
        hcp_voxel_map=voxel_map,
        make_figures=True,
        include_ttest=True,
        include_permutation=True,
        n_permutations=25,
        random_seed=11,
    )

    assert result.experimental is True
    assert result.n_hc == 2
    assert result.n_sz == 2
    assert (out / "outputs" / "inventories" / "data_inventory.csv").exists()
    assert (out / "outputs" / "inventories" / "aal3_region_mapping_report.csv").exists()
    assert (out / "outputs" / "qc" / "roi_timeseries_qc_long.csv").exists()
    assert (out / "outputs" / "qc" / "common_bad_rois_AAL3.csv").exists()
    assert (out / "outputs" / "qc" / "hcp_mmp1_mask_geometry_report.csv").exists()
    assert (out / "outputs" / "qc" / "hcp_region_size_report.csv").exists()
    assert (out / "outputs" / "qc" / "hcp_region_adjacency_report.csv").exists()
    assert (out / "outputs" / "temporal" / "temporal_qc_group_summary.csv").exists()
    assert (
        out
        / "outputs"
        / "preprocessed"
        / "AAL3"
        / "raw_cleaned"
        / "HC_sub01.npy"
    ).exists()
    assert (
        out
        / "outputs"
        / "preprocessed"
        / "AAL3"
        / "raw_cleaned"
        / "SZ_sub01.npy"
    ).exists()
    assert (
        out
        / "outputs"
        / "fc_matrices"
        / "AAL3"
        / "raw_cleaned"
        / "HC_sub01_pearson_z.npy"
    ).exists()
    assert (
        out
        / "outputs"
        / "fc_matrices"
        / "AAL3"
        / "raw_cleaned"
        / "SZ_sub01_pearson_z.npy"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "fc_group_comparison_edges.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "subject_level_fc_summary.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "subject_level_group_comparison.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "fc_group_comparison_edges_ttest.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "permutation_summary.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "sensitivity"
        / "AAL3"
        / "threshold_0_05"
        / "common_bad_rois_AAL3.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "sensitivity"
        / "AAL3"
        / "threshold_0_05"
        / "raw_cleaned"
        / "fc_group_comparison_edges.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "sensitivity"
        / "AAL3"
        / "threshold_0_05"
        / "raw_cleaned"
        / "subject_level_fc_summary.csv"
    ).exists()
    for figure in [
        out / "outputs" / "figures" / "AAL3" / "raw_cleaned" / "acf_profiles_by_group.png",
        out / "outputs" / "figures" / "AAL3" / "raw_cleaned" / "fc_delta_matrix_HC_vs_SZ.png",
        out / "outputs" / "figures" / "HCP_geometry" / "hcp_region_size_distribution.png",
    ]:
        assert figure.exists()
        assert figure.stat().st_size > 0
    assert (out / "reports" / "final_pilot_report.md").exists()


def test_run_fmri_roi_audit_full_10_hc_10_sz_synthetic(tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    out = tmp_path / "audit_10x10"
    hc.mkdir()
    sz.mkdir()
    for idx in range(10):
        _write_roi_csv(hc / f"sub{idx:02d}_AAL3_timeseries.csv", _compact_aal3_data(100 + idx))
        _write_roi_csv(sz / f"sub{idx:02d}_AAL3_timeseries.csv", _compact_aal3_data(200 + idx))

    result = run_fmri_roi_audit(
        hc,
        sz,
        out,
        atlas_filter="AAL3",
        branches=("raw_cleaned", "detrended", "ar1_residualized", "roi_level_gsr"),
        make_figures=True,
        include_ttest=True,
        include_permutation=True,
        n_permutations=50,
        random_seed=123,
        bad_roi_thresholds=(0.05,),
    )

    assert result.experimental is True
    assert result.n_hc == 10
    assert result.n_sz == 10
    assert (
        out
        / "outputs"
        / "preprocessed"
        / "AAL3"
        / "roi_level_gsr"
        / "HC_sub00.npy"
    ).exists()
    assert (
        out
        / "outputs"
        / "fc_matrices"
        / "AAL3"
        / "raw_cleaned"
        / "SZ_sub00_pearson_z.npy"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "fc_group_comparison_edges_ttest.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "group_comparison"
        / "AAL3"
        / "raw_cleaned"
        / "permutation_summary.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "sensitivity"
        / "AAL3"
        / "threshold_0_05"
        / "roi_level_gsr"
        / "fc_group_comparison_edges.csv"
    ).exists()
    assert (
        out
        / "outputs"
        / "sensitivity"
        / "AAL3"
        / "threshold_0_05"
        / "roi_level_gsr"
        / "subject_level_group_comparison.csv"
    ).exists()
    assert (
        out / "outputs" / "figures" / "AAL3" / "raw_cleaned" / "fc_delta_matrix_HC_vs_SZ.png"
    ).exists()
    assert (out / "reports" / "final_pilot_report.md").exists()


def test_cli_fmri_audit_runs_on_synthetic_directories(monkeypatch, tmp_path: Path) -> None:
    hc = tmp_path / "Group_HC"
    sz = tmp_path / "Group_SZ"
    out = tmp_path / "cli_out"
    hc.mkdir()
    sz.mkdir()
    _write_roi_csv(hc / "hc01_AAL3_timeseries.csv", _compact_aal3_data(1))
    _write_roi_csv(sz / "sz01_AAL3_timeseries.csv", _compact_aal3_data(2))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neweds-fmri-audit",
            "--hc-dir",
            str(hc),
            "--sz-dir",
            str(sz),
            "--output-dir",
            str(out),
            "--atlas",
            "AAL3",
        ],
    )

    cli_fmri_audit.main()

    assert (out / "outputs" / "inventories" / "data_inventory.csv").exists()
    assert (out / "reports" / "final_pilot_report.md").exists()
