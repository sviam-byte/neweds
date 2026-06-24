"""CLI for exact GM coordinate recovery and regional signal generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core.fmri_gm_signals import (
    VoxelRecoveryConfig,
    align_whole_brain_input,
    build_regional_signals,
    load_aal3_atlas,
    load_hcp_atlas,
    recover_voxel_coordinates,
    validate_atlas_against_ready_roi,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neweds-fmri-gm-signals")
    parser.add_argument("--manifest", required=True, help="CSV: subject_id,group,h5,voxel_csv")
    parser.add_argument("--new-results", required=True)
    parser.add_argument("--hcp-map")
    parser.add_argument("--aal-nifti")
    parser.add_argument("--aal-lut")
    parser.add_argument("--aal-local-regions")
    parser.add_argument(
        "--aal-validation-manifest",
        help="CSV with voxel_csv,ready_roi_csv; required before AAL signals",
    )
    parser.add_argument("--hcp-ready-manifest", help="CSV: subject_id,ready_roi_csv")
    parser.add_argument("--aal-ready-manifest", help="CSV: subject_id,ready_roi_csv")
    parser.add_argument("--atlas", choices=("hcp", "aal", "both"), default="both")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recovery-only", action="store_true")
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--random-seed", type=int, default=1729)
    return parser


def _ready_lookup(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    frame = pd.read_csv(path, dtype={"subject_id": str})
    return dict(zip(frame["subject_id"], frame["ready_roi_csv"]))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.new_results)
    recovery_dir = root / "voxel-coordinate-recovery" / "mappings"
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    if args.subjects:
        manifest = manifest[manifest["subject_id"].isin(set(args.subjects))]

    recovery_results = []
    for row in manifest.itertuples(index=False):
        result = recover_voxel_coordinates(
            VoxelRecoveryConfig(
                subject_id=str(row.subject_id),
                group=str(row.group),
                tissue_h5=str(row.h5),
                voxel_csv=str(row.voxel_csv),
                output_parquet=str(recovery_dir / f"{row.subject_id}_gm_voxel_mapping.parquet"),
                resume=args.resume,
            )
        )
        recovery_results.append(result.as_dict())
        print(json.dumps(result.as_dict(), ensure_ascii=False))
    recovery_table = pd.DataFrame(recovery_results)
    recovery_table.to_csv(
        root / "voxel-coordinate-recovery" / "subject_recovery_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if args.recovery_only:
        return int(not recovery_table["status"].eq("ok").all())

    atlases = []
    if args.atlas in {"hcp", "both"}:
        if not args.hcp_map:
            raise SystemExit("--hcp-map is required for HCP")
        atlases.append(load_hcp_atlas(args.hcp_map))
    if args.atlas in {"aal", "both"}:
        if not args.aal_nifti or not args.aal_lut:
            raise SystemExit("--aal-nifti and --aal-lut are required for AAL")
        aal = load_aal3_atlas(
            args.aal_nifti,
            args.aal_lut,
            local_regions_path=args.aal_local_regions,
        )
        if args.aal_validation_manifest:
            validation = pd.read_csv(args.aal_validation_manifest)
            cases = list(zip(validation["voxel_csv"], validation["ready_roi_csv"]))
            aal = validate_atlas_against_ready_roi(aal, cases)
        else:
            aal.validation_status = "blocked_atlas_validation"
            aal.validation_details = {"reason": "validation manifest not supplied"}
        atlases.append(aal)

    hcp_ready = _ready_lookup(args.hcp_ready_manifest)
    aal_ready = _ready_lookup(args.aal_ready_manifest)
    for atlas in atlases:
        slug = "gm-regional-signals-HCP360" if atlas.atlas_id.startswith("HCP") else (
            "gm-regional-signals-AAL3v2-167"
        )
        result_dir = root / slug
        result_dir.mkdir(parents=True, exist_ok=True)
        atlas.node_table.to_csv(result_dir / "node_table.csv", index=False, encoding="utf-8-sig")
        (result_dir / "atlas_validation.json").write_text(
            json.dumps(
                {
                    "status": atlas.validation_status,
                    "details": atlas.validation_details,
                    "source_sha256": atlas.source_sha256,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        regional = []
        paired = []
        ready_lookup = hcp_ready if atlas.atlas_id.startswith("HCP") else aal_ready
        for row in manifest.itertuples(index=False):
            mapping = recovery_dir / f"{row.subject_id}_gm_voxel_mapping.parquet"
            result = build_regional_signals(
                subject_id=str(row.subject_id),
                group=str(row.group),
                tissue_h5=str(row.h5),
                mapping_parquet=mapping,
                atlas=atlas,
                output_dir=result_dir / "subjects",
                random_seed=args.random_seed,
            )
            regional.append(result.as_dict())
            ready = ready_lookup.get(str(row.subject_id))
            if ready and result.status == "ok":
                paired.append(
                    align_whole_brain_input(
                        subject_id=str(row.subject_id),
                        group=str(row.group),
                        atlas=atlas,
                        ready_roi_csv=ready,
                        output_dir=result_dir / "whole_brain",
                    )
                )
        pd.DataFrame(regional).to_json(
            result_dir / "subject_status.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        pd.DataFrame(paired).to_json(
            result_dir / "paired_input_manifest.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
