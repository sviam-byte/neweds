from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from neweds.core.fmri_tissue_audit import (
    run_fmri_tissue_audit,
    scan_tissue_h5_inventory,
)


def _write_tissue_h5(
    path: Path,
    *,
    subject_id: str,
    group: str,
    seed: int,
    include_xyz: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["id"] = subject_id
        h5.attrs["group"] = group
        h5.attrs["T"] = 48
        h5.attrs["shape"] = [5, 5, 5, 48]
        shared = np.sin(np.linspace(0.0, 4.0 * np.pi, 48))
        for tissue, n_voxels in [("GM", 24), ("WM", 18), ("CSF", 12)]:
            group_h5 = h5.create_group(tissue)
            data = rng.normal(scale=0.2, size=(n_voxels, 48)) + shared
            data[0] = 0.0
            group_h5.create_dataset("data", data=data.astype(np.float32))
            if include_xyz:
                xyz = np.column_stack(
                    [
                        np.arange(n_voxels) % 5,
                        (np.arange(n_voxels) // 5) % 5,
                        (np.arange(n_voxels) // 25) % 5,
                    ]
                )
                group_h5.create_dataset("xyz", data=xyz.astype(np.int16))


def test_tissue_inventory_marks_missing_xyz_without_rejecting_data(tmp_path: Path) -> None:
    root = tmp_path / "tissues"
    _write_tissue_h5(
        root / "Контроль" / "1001_HC.h5",
        subject_id="1001",
        group="HC",
        seed=1,
    )

    inventory = scan_tissue_h5_inventory(root)

    assert len(inventory) == 1
    assert inventory.loc[0, "status"] == "ok"
    assert not bool(inventory.loc[0, "GM_has_xyz"])
    assert not bool(inventory.loc[0, "GM_xyz_matches_data"])
    assert int(inventory.loc[0, "GM_n_timepoints"]) == 48


def test_tissue_audit_writes_independent_output_contract(tmp_path: Path) -> None:
    root = tmp_path / "tissues"
    _write_tissue_h5(
        root / "Контроль" / "1001_HC.h5",
        subject_id="1001",
        group="HC",
        seed=1,
    )
    _write_tissue_h5(
        root / "Контроль" / "1002_HC.h5",
        subject_id="1002",
        group="HC",
        seed=2,
    )
    _write_tissue_h5(
        root / "Шизофрения" / "2001_tissues.h5",
        subject_id="2001",
        group="SZ",
        seed=3,
    )
    _write_tissue_h5(
        root / "Шизофрения" / "2002_tissues.h5",
        subject_id="2002",
        group="SZ",
        seed=4,
    )
    output = tmp_path / "tissue_gm_wm_csf_audit"

    result = run_fmri_tissue_audit(root, output, max_lag=5, block_rows=8)

    assert result.files_valid == 4
    assert result.subjects_hc == 2
    assert result.subjects_sz == 2
    assert not result.spatial_analysis_available
    required = [
        "audit_manifest.json",
        "inventories/tissue_hdf5_inventory.csv",
        "qc/tissue_dataset_qc.csv",
        "qc/tissue_voxel_counts_wide.csv",
        "temporal/tissue_mean_timeseries.csv",
        "temporal/tissue_mean_temporal_qc.csv",
        "temporal/tissue_mean_acf_pacf.csv",
        "temporal/tissue_mean_correlations.csv",
        "group_comparison/tissue_feature_group_comparison.csv",
        "reports/tissue_audit_report.md",
        "reports/transcript_methodology_status.md",
    ]
    for relative in required:
        assert (output / relative).exists(), relative

    manifest = json.loads((output / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_type"] == "tissue_gm_wm_csf"
    assert manifest["contains_roi_connectivity"] is False
    assert manifest["independent_from"] == "whole_brain_roi_audit"

    qc = pd.read_csv(output / "qc" / "tissue_dataset_qc.csv")
    assert set(qc["tissue"]) == {"GM", "WM", "CSF"}
    assert (qc["zero_voxels"] >= 1).all()
    assert (qc["active_voxels"] < qc["n_voxels"]).all()


def test_tissue_audit_enables_spatial_status_only_with_matching_xyz(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tissues"
    _write_tissue_h5(
        root / "Контроль" / "1001_HC.h5",
        subject_id="1001",
        group="HC",
        seed=1,
        include_xyz=True,
    )
    output = tmp_path / "audit"

    result = run_fmri_tissue_audit(root, output, max_lag=3, block_rows=8)

    assert result.xyz_complete_files == 1
    assert result.spatial_analysis_available
