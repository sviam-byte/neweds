"""Mask and region-definition QC for voxel-like arrays.

These helpers operate in volume space. They do not model cortical surface
geometry, sulcal banks, atlas registration quality, or grey/white tissue
segmentation quality. Their purpose is to make mask assumptions auditable before
regional signal extraction and connectivity analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_bool_mask(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.asarray([], dtype=bool)
    arr = np.asarray(mask)
    if arr.size == 0:
        return np.asarray([], dtype=bool)
    return np.isfinite(arr) & (arr.astype(bool))


def _status_for_region(
    *,
    valid_voxels: int,
    mask_fraction: float,
    min_valid_voxels: int,
    elongated_ratio: float,
    fill_fraction: float,
) -> str:
    if valid_voxels <= 0 or valid_voxels < min_valid_voxels:
        return "bad"
    if mask_fraction < 0.50:
        return "bad"
    if mask_fraction < 0.90:
        return "weak"
    if np.isfinite(elongated_ratio) and elongated_ratio > 6.0:
        return "weak"
    if np.isfinite(fill_fraction) and fill_fraction < 0.20:
        return "weak"
    return "ok"


def mask_coverage_summary(
    valid_mask: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize how many voxels are allowed by a brain/GM/cortical mask.

    Parameters
    ----------
    valid_mask:
        Boolean-like mask of acceptable tissue. In neuroimaging use this should
        normally be a brain, grey-matter, cortical, or atlas-validity mask.
    candidate_mask:
        Optional boolean-like mask for voxels proposed by a loader, atlas, grid,
        or preprocessing step.
    """
    valid = _as_bool_mask(valid_mask)
    n_total = int(valid.size)
    n_valid = int(np.sum(valid))
    out: dict[str, Any] = {
        "n_total_voxels": n_total,
        "n_valid_voxels": n_valid,
        "valid_fraction": float(n_valid / n_total) if n_total else float("nan"),
    }

    if candidate_mask is None:
        return out

    candidate = _as_bool_mask(candidate_mask)
    if candidate.shape != valid.shape:
        raise ValueError(
            f"candidate_mask shape {candidate.shape} does not match valid_mask shape {valid.shape}"
        )
    n_candidate = int(np.sum(candidate))
    n_candidate_in_mask = int(np.sum(candidate & valid))
    out.update(
        {
            "n_candidate_voxels": n_candidate,
            "n_candidate_in_mask": n_candidate_in_mask,
            "n_candidate_outside_mask": int(n_candidate - n_candidate_in_mask),
            "candidate_mask_fraction": (
                float(n_candidate_in_mask / n_candidate) if n_candidate else float("nan")
            ),
        }
    )
    return out


def summarize_region_mask_qc(
    region_labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    background: int | str | None = 0,
    min_valid_voxels: int = 3,
) -> pd.DataFrame:
    """Evaluate whether candidate regions stay inside a valid tissue mask.

    ``region_labels`` and ``valid_mask`` must have the same shape. Each non-
    background label is treated as a candidate region/bin/parcel.
    """
    labels = np.asarray(region_labels)
    valid = _as_bool_mask(valid_mask)
    if labels.shape != valid.shape:
        raise ValueError(f"region_labels shape {labels.shape} != valid_mask shape {valid.shape}")
    if labels.ndim != 3:
        raise ValueError("region mask QC expects 3D volume-like labels")

    flat_labels = labels.reshape(-1)
    if background is None:
        unique_labels = pd.unique(flat_labels)
    else:
        unique_labels = [v for v in pd.unique(flat_labels) if v != background]

    rows: list[dict[str, Any]] = []
    valid_flat = valid.reshape(-1)
    for label in unique_labels:
        region_flat = flat_labels == label
        total_voxels = int(np.sum(region_flat))
        if total_voxels == 0:
            continue
        valid_voxels = int(np.sum(region_flat & valid_flat))
        outside_voxels = int(total_voxels - valid_voxels)
        mask_fraction = float(valid_voxels / total_voxels) if total_voxels else float("nan")

        coords = np.argwhere(labels == label)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        spans = (maxs - mins + 1).astype(int)
        bbox_volume = int(np.prod(spans))
        fill_fraction = float(total_voxels / bbox_volume) if bbox_volume else float("nan")
        positive_spans = spans[spans > 0]
        elongated_ratio = (
            float(np.max(positive_spans) / np.min(positive_spans))
            if positive_spans.size
            else float("nan")
        )
        crosses_mask_boundary = bool(0 < valid_voxels < total_voxels)

        rows.append(
            {
                "region_id": str(label),
                "n_voxels": total_voxels,
                "n_valid_voxels": valid_voxels,
                "n_outside_mask_voxels": outside_voxels,
                "mask_fraction": mask_fraction,
                "crosses_mask_boundary": crosses_mask_boundary,
                "bbox_x": int(spans[0]),
                "bbox_y": int(spans[1]),
                "bbox_z": int(spans[2]),
                "bbox_volume": bbox_volume,
                "bbox_fill_fraction": fill_fraction,
                "elongated_ratio": elongated_ratio,
                "recommended_status": _status_for_region(
                    valid_voxels=valid_voxels,
                    mask_fraction=mask_fraction,
                    min_valid_voxels=int(max(1, min_valid_voxels)),
                    elongated_ratio=elongated_ratio,
                    fill_fraction=fill_fraction,
                ),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["mask_coverage_summary", "summarize_region_mask_qc"]
