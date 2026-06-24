"""Prepare independent whole-brain/ROI and GM/WM/CSF tissue audit directories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neweds.core.fmri_tissue_audit import run_fmri_tissue_audit  # noqa: E402

WHOLE_BRAIN_ALLOWED_DIRECTORIES = (
    "inventories",
    "qc",
    "distributions",
    "temporal",
    "spectral",
    "reports",
    "decisions",
)
WHOLE_BRAIN_ALLOWED_FILES = ("roi_signal_characterization_all_features.csv",)


def _copy_whole_brain_audit(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for name in WHOLE_BRAIN_ALLOWED_DIRECTORIES:
        src = source / name
        dst = target / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied.append(name)
    for name in WHOLE_BRAIN_ALLOWED_FILES:
        src = source / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, target / name)
        copied.append(name)

    report_dir = target / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    methodology = [
        "# Transcript-Derived Methodology Status: Whole-Brain / ROI Audit",
        "",
        "This status belongs only to the whole-brain/ROI audit.",
        "",
        "| Requirement | Status | Evidence / next action |",
        "| --- | --- | --- |",
        "| ROI value, trend, temporal, and spectral QC | implemented | copied Stage 1 artifacts |",
        "| AAL3 row-to-label mapping | unresolved | observed CSV shape is 168 × 600 while aal3_regions.txt contains 167 labels |",
        "| ACF/PACF and AR1 before/after | partial | ACF/AR summaries exist; PACF/explicit before-after plots must be retained in the next rerun |",
        "| Atlas membership as candidate node only | contract_defined | atlas membership does not prove homogeneity |",
        "| Voxel-wise ROI homogeneity | blocked | extracted ROI means do not contain voxel membership |",
        "| Mean vs sign-oriented PCA vs ICA | blocked | requires voxel-to-region series |",
        "| PCA/ICA sign orientation | contract_defined | signed metrics require recorded orientation |",
        "| Mask-constrained cortical neighbourhood growth | blocked | requires coordinates/masks/surface information |",
        "| Global signal with/without | sensitivity_required | current ROI-level GSR is an approximation |",
        "| Multiple-comparison correction | implemented | FDR is required for group-level scans |",
        "| Lag × window × metric cube | downstream_exploratory | not part of this data audit |",
        "",
        "## Separation rule",
        "",
        "Tissue GM/WM/CSF QC and nuisance-regressor results are stored in a sibling",
        "`tissue_gm_wm_csf_audit` directory and are not inserted into these tables.",
    ]
    (report_dir / "transcript_methodology_status.md").write_text(
        "\n".join(methodology),
        encoding="utf-8",
    )

    manifest = {
        "audit_type": "whole_brain_roi",
        "layout_version": 1,
        "source_root": str(source),
        "output_root": str(target),
        "independent_from": "tissue_gm_wm_csf_audit",
        "contains_tissue_hdf5_qc": False,
        "contains_stage2_connectivity_grid": False,
        "known_mapping_issue": (
            "Observed AAL3 CSV files contain 168 rows while aal3_regions.txt "
            "contains 167 labels; row-to-region mapping requires upstream clarification."
        ),
        "copy_policy": {
            "allowed_directories": list(WHOLE_BRAIN_ALLOWED_DIRECTORIES),
            "allowed_files": list(WHOLE_BRAIN_ALLOWED_FILES),
            "copied": copied,
            "missing": missing,
        },
        "reports": {
            "stage1_report": "reports/data_characterization_report.md",
            "methodology_status": "reports/transcript_methodology_status.md",
        },
    }
    (target / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare_separate_audits(
    *,
    whole_brain_source: Path,
    tissue_source: Path,
    output_root: Path,
    max_lag: int,
    block_rows: int,
) -> None:
    whole_brain_target = output_root / "whole_brain_roi_audit"
    tissue_target = output_root / "tissue_gm_wm_csf_audit"
    output_root.mkdir(parents=True, exist_ok=True)

    _copy_whole_brain_audit(whole_brain_source, whole_brain_target)
    tissue_result = run_fmri_tissue_audit(
        tissue_source,
        tissue_target,
        max_lag=max_lag,
        block_rows=block_rows,
    )
    collection_manifest = {
        "collection_type": "separate_fmri_data_audits",
        "layout_version": 1,
        "whole_brain_roi_audit": {
            "path": str(whole_brain_target),
            "manifest": "whole_brain_roi_audit/audit_manifest.json",
        },
        "tissue_gm_wm_csf_audit": {
            "path": str(tissue_target),
            "manifest": "tissue_gm_wm_csf_audit/audit_manifest.json",
            "result": tissue_result.as_dict(),
        },
        "cross_dataset_checks": {
            "path": str(output_root / "cross_dataset_checks"),
            "status": "not_run",
        },
        "separation_guarantee": (
            "Whole-brain/ROI and tissue HDF5 audit tables are stored in sibling "
            "directories. Cross-source comparisons require a third explicit layer."
        ),
    }
    (output_root / "audit_collection_manifest.json").write_text(
        json.dumps(collection_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-brain-source", required=True)
    parser.add_argument("--tissue-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--block-rows", type=int, default=8192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_separate_audits(
        whole_brain_source=Path(args.whole_brain_source),
        tissue_source=Path(args.tissue_source),
        output_root=Path(args.output_root),
        max_lag=int(args.max_lag),
        block_rows=int(args.block_rows),
    )


if __name__ == "__main__":
    main()
