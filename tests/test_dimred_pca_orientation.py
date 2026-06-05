from __future__ import annotations

import numpy as np
import pandas as pd

from neweds.analysis.dimred import _orient_pca_sign, _standardize_matrix, apply_dimred


def test_pca_orientation_helper_is_stable_under_global_sign_flip() -> None:
    pc1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = np.column_stack([pc1, pc1 * 0.5])
    scores = pc1.reshape(-1, 1)
    comps = np.array([[0.9, 0.4]])

    scores_a, comps_a, meta_a = _orient_pca_sign(x, scores, comps, "mean_corr")
    scores_b, comps_b, meta_b = _orient_pca_sign(x, -scores, -comps, "mean_corr")

    assert np.allclose(scores_a, scores_b)
    assert np.allclose(comps_a, comps_b)
    assert meta_a["corr_pc1_mean"] >= 0.0
    assert meta_b["corr_pc1_mean"] >= 0.0
    assert meta_a["sign_flip"] is False
    assert meta_b["sign_flip"] is True


def test_apply_dimred_can_orient_pc1_to_mean_signal() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 60)
    base = np.sin(t)
    data = pd.DataFrame(
        {
            "a": base,
            "b": base * 1.5 + 0.05,
            "c": base * 0.5 - 0.05,
        }
    )

    result = apply_dimred(
        data,
        method="pca",
        target_n=1,
        seed=7,
        pca_orient_sign="mean_corr",
    )
    x, _ = _standardize_matrix(data)
    mean_signal = np.nanmean(x, axis=1)
    pc1 = result.reduced["pc_0001"].to_numpy(dtype=float)
    corr = float(np.corrcoef(pc1, mean_signal)[0, 1])

    assert result.meta["pca_orient_sign"] == "mean_corr"
    assert result.meta["orientation_rule"] == "mean_corr"
    assert result.meta["corr_pc1_mean"] is not None
    assert corr >= 0.0
