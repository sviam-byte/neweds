from __future__ import annotations

import numpy as np
import pandas as pd

from neweds.analysis.roi_qc import summarize_region_qc, within_region_homogeneity


def test_homogeneous_region_is_low_risk() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 80)
    base = np.sin(t)
    block = pd.DataFrame(
        {
            "v1": base,
            "v2": base * 1.01 + 0.01,
            "v3": base * 0.99 - 0.01,
        }
    )

    qc = within_region_homogeneity(block)

    assert qc["n_series"] == 3
    assert qc["n_active"] == 3
    assert qc["aggregation_risk"] == "ok"
    assert qc["median_pairwise_corr"] > 0.99
    assert qc["frac_negative_corr"] == 0.0
    assert qc["mean_signal_validity"] > 0.99


def test_opposite_sign_mixture_is_flagged() -> None:
    t = np.linspace(0.0, 4.0 * np.pi, 100)
    base = np.sin(t)
    block = pd.DataFrame(
        {
            "pos_1": base,
            "pos_2": base + 0.02 * np.cos(t),
            "neg_1": -base,
            "neg_2": -base + 0.02 * np.cos(t),
        }
    )

    qc = within_region_homogeneity(block)

    assert qc["n_active"] == 4
    assert qc["frac_negative_corr"] > 0.5
    assert qc["aggregation_risk"] == "bad"


def test_constant_and_nan_region_is_safe_failure() -> None:
    block = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            "nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    qc = within_region_homogeneity(block)

    assert qc["n_series"] == 2
    assert qc["n_active"] == 0
    assert qc["aggregation_risk"] == "bad"
    assert np.isnan(qc["median_pairwise_corr"])


def test_summarize_region_qc_accepts_region_mapping() -> None:
    t = np.linspace(0.0, 1.0, 12)
    data = pd.DataFrame({"a": t, "b": t + 0.1, "c": -t})
    out = summarize_region_qc(data, {"r1": ["a", "b"], "r2": ["c", "missing"]})

    assert list(out["region"]) == ["r1", "r2"]
    assert out.loc[out["region"] == "r1", "aggregation_risk"].iloc[0] == "ok"
    assert out.loc[out["region"] == "r2", "aggregation_risk"].iloc[0] == "bad"
