from __future__ import annotations

import numpy as np

from neweds.analysis.mask_qc import mask_coverage_summary, summarize_region_mask_qc


def test_mask_coverage_summary_counts_candidate_inside_and_outside_mask() -> None:
    valid = np.zeros((2, 2, 2), dtype=bool)
    valid[0, :, :] = True
    candidate = np.zeros_like(valid)
    candidate[0, 0, 0] = True
    candidate[1, 0, 0] = True

    summary = mask_coverage_summary(valid, candidate)

    assert summary["n_total_voxels"] == 8
    assert summary["n_valid_voxels"] == 4
    assert summary["n_candidate_voxels"] == 2
    assert summary["n_candidate_in_mask"] == 1
    assert summary["n_candidate_outside_mask"] == 1
    assert summary["candidate_mask_fraction"] == 0.5


def test_region_mask_qc_flags_regions_crossing_mask_boundary() -> None:
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0:2, 0:2, 0:2] = 1
    labels[2, 2, 2] = 2

    valid = np.zeros_like(labels, dtype=bool)
    valid[0, 0:2, 0:2] = True
    valid[2, 2, 2] = True

    qc = summarize_region_mask_qc(labels, valid, min_valid_voxels=2)
    region_1 = qc.loc[qc["region_id"] == "1"].iloc[0]
    region_2 = qc.loc[qc["region_id"] == "2"].iloc[0]

    assert bool(region_1["crosses_mask_boundary"]) is True
    assert region_1["mask_fraction"] == 0.5
    assert region_1["recommended_status"] == "weak"
    assert region_2["recommended_status"] == "bad"
