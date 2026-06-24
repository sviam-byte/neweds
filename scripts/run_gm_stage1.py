"""Run and package stage 1 GM geometry/signal results for the 39-subject cohort."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from neweds.core.fmri_gm_signals import (  # noqa: E402
    VoxelRecoveryConfig,
    align_whole_brain_input,
    build_regional_signals,
    load_aal3_atlas,
    load_hcp_atlas,
    read_whole_brain_roi_csv,
    recover_voxel_coordinates,
    validate_atlas_against_ready_roi,
)
from scripts.write_result_provenance import write_result_provenance  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=r"D:\Данные по сознанию-")
    parser.add_argument("--tissue-root", default=r"D:\Данные по сознанию- (2)\Серое, белое и СМЖ")
    parser.add_argument("--new-results", default=r"D:\Данные по сознанию- (2)\new_results")
    parser.add_argument("--result-date", default=date.today().isoformat())
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--recovery-only", action="store_true")
    parser.add_argument("--skip-aal", action="store_true")
    parser.add_argument("--skip-hcp", action="store_true")
    parser.add_argument("--skip-provenance", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=1729)
    return parser


def _subject_id(path: Path) -> str:
    return path.name.split("_", 1)[0]


def _discover_inputs(source_root: Path, tissue_root: Path) -> pd.DataFrame:
    h5_paths = {_subject_id(path): path for path in tissue_root.rglob("*.h5")}
    voxel_paths: dict[str, Path] = {}
    for directory in ("Здоровые (600 точек)", "Шизофрения (600 точек)"):
        for path in (source_root / directory).glob("*voxels_xyz_timeseries.csv"):
            voxel_paths[_subject_id(path)] = path
    rows = []
    for subject_id in sorted(h5_paths, key=int):
        h5 = h5_paths[subject_id]
        group = "HC" if h5.stem.upper().endswith("_HC") else "SZ"
        voxel = voxel_paths.get(subject_id)
        if voxel is None:
            raise FileNotFoundError(f"voxel CSV not found for {subject_id}")
        rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "h5": str(h5),
                "voxel_csv": str(voxel),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 39 or frame["group"].value_counts().to_dict() != {"SZ": 24, "HC": 15}:
        raise RuntimeError(
            f"expected 39 subjects (15 HC, 24 SZ), got {len(frame)} "
            f"{frame['group'].value_counts().to_dict()}"
        )
    return frame


def _discover_aal_ready(source_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for directory in ("Group_HC", "Group_SZ"):
        for path in (source_root / directory).glob("*_AAL3_timeseries.csv"):
            found[_subject_id(path)] = path
    return found


def _extract_hcp_ready(source_root: Path, cache_root: Path) -> dict[str, Path]:
    destination = cache_root / "hcp_ready"
    marker = destination / ".extracted"
    if not marker.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_root / "atlas_HCP.zip") as archive:
            archive.extractall(destination)
        marker.write_text("atlas_HCP.zip extracted by run_gm_stage1.py\n", encoding="utf-8")
    return {
        _subject_id(path): path
        for path in destination.rglob("*_HCP_MMP1_timeseries.csv")
    }


def _write_aal_header_correction(
    *,
    result_dir: Path,
    ready: dict[str, Path],
    repository: Path,
    package: bool,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_id, path in sorted(ready.items(), key=lambda item: int(item[0])):
        array = read_whole_brain_roi_csv(path, expected_nodes=167)
        rows.append(
            {
                "subject_id": subject_id,
                "path": str(path),
                "rows_with_numeric_header_interpreted_as_data": 168,
                "correct_rows": int(array.shape[0]),
                "timepoints": int(array.shape[1]),
                "status": "corrected_167x600",
            }
        )
    pd.DataFrame(rows).to_csv(
        result_dir / "aal3_shape_correction.csv", index=False, encoding="utf-8-sig"
    )
    (result_dir / "correction_statement.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "correct_shape": [167, 600],
                "affected_files": len(rows),
                "refuted_warning": "AAL3 files have 168 data rows",
                "reason": "the first CSV row 0..599 is a numeric header, not ROI data",
                "historical_audit_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if package:
        write_result_provenance(
            result_dir=result_dir,
            result_id="aal3-numeric-header-correction",
            title="AAL3 numeric-header shape correction",
            result_type="data_audit_correction",
            status="completed_correction",
            execution_mode="fresh_validation",
            summary="All 39 AAL3 CSV files are confirmed as 167×600; the apparent 168th row is the numeric 0…599 header.",
            meaning="This correction removes a false shape warning without rewriting the historical audit.",
            command="python scripts/run_gm_stage1.py",
            inputs=[str(path) for path in ready.values()],
            code_files=[
                repository / "neweds/core/fmri_gm_signals.py",
                repository / "scripts/run_gm_stage1.py",
            ],
            repository=repository,
            findings=[
                "39/39 AAL3 files parse to 167×600.",
                "The historical 168-row warning is explicitly refuted.",
            ],
            limitations=["This result corrects shape interpretation only; it does not validate atlas geometry."],
        )


def _package_result(
    *,
    result_dir: Path,
    result_id: str,
    title: str,
    result_type: str,
    status: str,
    summary: str,
    meaning: str,
    inputs: list[str],
    findings: list[str],
    limitations: list[str],
) -> None:
    write_result_provenance(
        result_dir=result_dir,
        result_id=result_id,
        title=title,
        result_type=result_type,
        status=status,
        execution_mode="fresh_run_with_resume",
        summary=summary,
        meaning=meaning,
        command="python scripts/run_gm_stage1.py",
        inputs=inputs,
        code_files=[
            REPOSITORY / "neweds/core/fmri_gm_signals.py",
            REPOSITORY / "neweds/cli_fmri_gm_signals.py",
            REPOSITORY / "scripts/run_gm_stage1.py",
            REPOSITORY / "tests/test_fmri_gm_signals.py",
            REPOSITORY / "pyproject.toml",
        ],
        repository=REPOSITORY,
        findings=findings,
        limitations=limitations,
    )


def _update_catalog(
    new_results: Path,
    entries: list[dict[str, Any]],
    correction_path: str,
) -> None:
    catalog_path = new_results / "results_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    by_id = {item["result_id"]: item for item in catalog.get("results", [])}
    for entry in entries:
        by_id[entry["result_id"]] = entry
    catalog["results"] = list(by_id.values())
    catalog["result_count"] = len(catalog["results"])
    catalog["updated_at"] = pd.Timestamp.now(tz="Europe/Moscow").isoformat()
    catalog.setdefault("superseded_warnings", []).append(
        {
            "claim": "AAL3 ROI CSV data shape is 168x600",
            "status": "refuted",
            "correction_result": correction_path,
            "correct_shape": [167, 600],
        }
    )
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    source_root = Path(args.source_root)
    tissue_root = Path(args.tissue_root)
    new_results = Path(args.new_results)
    prefix = args.result_date
    recovery_dir = new_results / f"{prefix}_voxel-coordinate-recovery"
    hcp_dir = new_results / f"{prefix}_gm-regional-signals-HCP360"
    aal_dir = new_results / f"{prefix}_gm-regional-signals-AAL3v2-167"
    correction_dir = new_results / f"{prefix}_aal3-numeric-header-correction"
    cache_root = new_results / "_atlas_cache"

    manifest = _discover_inputs(source_root, tissue_root)
    full_cohort = not args.subjects
    if args.subjects:
        manifest = manifest[manifest["subject_id"].isin(set(args.subjects))].copy()
    recovery_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(recovery_dir / "input_manifest.csv", index=False, encoding="utf-8-sig")

    recovery_rows = []
    for row in manifest.itertuples(index=False):
        result = recover_voxel_coordinates(
            VoxelRecoveryConfig(
                subject_id=row.subject_id,
                group=row.group,
                tissue_h5=row.h5,
                voxel_csv=row.voxel_csv,
                output_parquet=str(recovery_dir / "mappings" / f"{row.subject_id}.parquet"),
                resume=args.resume,
            )
        )
        recovery_rows.append(result.as_dict())
        print(
            f"recovery {row.subject_id} {result.status} "
            f"{result.n_matched_rows}/{result.n_nonzero_rows} zero={result.n_zero_rows}",
            flush=True,
        )
    recovery_table = pd.DataFrame(recovery_rows)
    recovery_table.to_csv(
        recovery_dir / "subject_recovery_status.csv", index=False, encoding="utf-8-sig"
    )
    failures = recovery_table.loc[recovery_table["status"] != "ok"]
    failures.to_csv(recovery_dir / "failure_table.csv", index=False, encoding="utf-8-sig")

    if full_cohort and not args.skip_provenance:
        _package_result(
            result_dir=recovery_dir,
            result_id="voxel-coordinate-recovery",
            title="Exact GM voxel-coordinate recovery",
            result_type="voxel_coordinate_recovery",
            status="completed" if failures.empty else "completed_with_blocked_subjects",
            summary=(
                f"Exact float32/BLAKE2b GM coordinate recovery for {len(recovery_table)} subjects; "
                f"{len(failures)} subjects blocked."
            ),
            meaning="The sidecars restore voxel geometry without assigning coordinates to zero signals.",
            inputs=[*manifest["h5"].tolist(), *manifest["voxel_csv"].tolist()],
            findings=[
                f"{int(recovery_table['n_matched_rows'].sum())} non-zero GM rows matched.",
                f"{int(recovery_table['n_zero_rows'].sum())} all-zero GM rows retained as unresolved QC.",
            ],
            limitations=["Only GM coordinates are recovered in this stage."],
        )
    if args.recovery_only:
        return int(not failures.empty)

    aal_ready = _discover_aal_ready(source_root)
    if full_cohort:
        _write_aal_header_correction(
            result_dir=correction_dir,
            ready=aal_ready,
            repository=REPOSITORY,
            package=not args.skip_provenance,
        )

    result_entries: list[dict[str, Any]] = []
    if not args.skip_hcp:
        hcp = load_hcp_atlas(source_root / "HCP-MMP1_atlas_voxel_map_from_xml.csv")
        hcp_ready = _extract_hcp_ready(source_root, cache_root)
        hcp_dir.mkdir(parents=True, exist_ok=True)
        hcp.node_table.to_csv(hcp_dir / "node_table.csv", index=False, encoding="utf-8-sig")
        regional_rows = []
        paired_rows = []
        for row in manifest.itertuples(index=False):
            result = build_regional_signals(
                subject_id=row.subject_id,
                group=row.group,
                tissue_h5=row.h5,
                mapping_parquet=recovery_dir / "mappings" / f"{row.subject_id}.parquet",
                atlas=hcp,
                output_dir=hcp_dir / "subjects",
                random_seed=args.random_seed,
            )
            regional_rows.append(result.as_dict())
            if result.status == "ok":
                paired_rows.append(
                    align_whole_brain_input(
                        subject_id=row.subject_id,
                        group=row.group,
                        atlas=hcp,
                        ready_roi_csv=hcp_ready[row.subject_id],
                        output_dir=hcp_dir / "whole_brain",
                    )
                )
            print(f"HCP {row.subject_id} {result.status}", flush=True)
        pd.DataFrame(regional_rows).to_json(
            hcp_dir / "subject_status.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        pd.DataFrame(paired_rows).to_json(
            hcp_dir / "paired_input_manifest.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        homogeneity = pd.concat(
            [pd.read_parquet(path) for path in sorted((hcp_dir / "subjects").glob("*homogeneity.parquet"))],
            ignore_index=True,
        )
        methods = pd.concat(
            [pd.read_parquet(path) for path in sorted((hcp_dir / "subjects").glob("*method_status.parquet"))],
            ignore_index=True,
        )
        homogeneity.to_parquet(hcp_dir / "region_homogeneity.parquet", compression="zstd", index=False)
        methods.to_parquet(hcp_dir / "method_status.parquet", compression="zstd", index=False)
        hcp_failures = sum(row["status"] != "ok" for row in regional_rows)
        if full_cohort and not args.skip_provenance:
            _package_result(
                result_dir=hcp_dir,
                result_id="gm-regional-signals-HCP360",
                title="GM-only regional signals — HCP-MMP1 360",
                result_type="gm_regional_signals",
                status="completed" if hcp_failures == 0 else "completed_with_blocked_subjects",
                summary=f"Four GM-only signals and spatial/homogeneity QC for HCP360; {hcp_failures} subjects blocked.",
                meaning="This is the node-aligned GM-only half of the next-stage GM versus whole-brain comparison.",
                inputs=[str(source_root / "HCP-MMP1_atlas_voxel_map_from_xml.csv")],
                findings=[
                    f"{len(regional_rows)} subject status records.",
                    "Signals: active mean, oriented PC1, oriented one-component ICA, correlation core.",
                    "No connectivity or HC/SZ classification was run.",
                ],
                limitations=["Descriptive QC flags are not optimized against HC/SZ labels."],
            )
        result_entries.append(
            {
                "result_id": "gm-regional-signals-HCP360",
                "title": "GM-only regional signals — HCP-MMP1 360",
                "path": hcp_dir.name,
                "status": "completed" if hcp_failures == 0 else "completed_with_blocked_subjects",
                "execution_mode": "fresh_run_with_resume",
                "summary": "Four GM-only HCP360 signal variants with spatial and homogeneity QC.",
            }
        )

    if not args.skip_aal:
        aal_dir.mkdir(parents=True, exist_ok=True)
        aal_nifti = cache_root / "aal3v2" / "AAL3v1.nii"
        aal_xml = cache_root / "aal3v2" / "AAL3v1.xml"
        aal_status = "blocked_atlas_validation"
        reason = "Exact Nilearn AAL3v2 NIfTI/XML is unavailable or failed ready-ROI reconstruction."
        if aal_nifti.is_file() and aal_xml.is_file():
            try:
                aal = load_aal3_atlas(
                    aal_nifti,
                    aal_xml,
                    local_regions_path=source_root / "aal3_regions.txt",
                )
                cases = []
                for subject_id in ("1185", "1186", "1097", "1103"):
                    row = manifest.loc[manifest["subject_id"] == subject_id]
                    if row.empty:
                        full_manifest = _discover_inputs(source_root, tissue_root)
                        row = full_manifest.loc[full_manifest["subject_id"] == subject_id]
                    cases.append((row.iloc[0]["voxel_csv"], aal_ready[subject_id]))
                aal = validate_atlas_against_ready_roi(aal, cases)
                aal_status = aal.validation_status
                reason = json.dumps(aal.validation_details, ensure_ascii=False)
            except Exception as exc:
                aal_status = "blocked_atlas_validation"
                reason = f"{type(exc).__name__}: {exc}"
        pd.DataFrame(
            [
                {
                    "subject_id": row.subject_id,
                    "group": row.group,
                    "atlas_id": "AAL3v2-167",
                    "status": aal_status,
                    "message": reason,
                }
                for row in manifest.itertuples(index=False)
            ]
        ).to_csv(aal_dir / "subject_status.csv", index=False, encoding="utf-8-sig")
        (aal_dir / "atlas_validation.json").write_text(
            json.dumps({"status": aal_status, "reason": reason}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if full_cohort and not args.skip_provenance:
            _package_result(
                result_dir=aal_dir,
                result_id="gm-regional-signals-AAL3v2-167",
                title="GM-only regional signals — AAL3v2 167",
                result_type="gm_regional_signals",
                status=aal_status,
                summary="AAL3v2 branch was evaluated fail-closed against exact atlas/version and ready-ROI reconstruction.",
                meaning="AAL outputs are withheld unless geometry and node order are proven exactly.",
                inputs=[
                    str(source_root / "aal3_regions.txt"),
                    *[str(path) for path in aal_ready.values()],
                ],
                findings=[f"Branch status: {aal_status}.", reason],
                limitations=["No approximate or AAL3v1 fallback mapping is emitted."],
            )
        result_entries.append(
            {
                "result_id": "gm-regional-signals-AAL3v2-167",
                "title": "GM-only regional signals — AAL3v2 167",
                "path": aal_dir.name,
                "status": aal_status,
                "execution_mode": "fail_closed_atlas_validation",
                "summary": "AAL3v2 branch is withheld unless exact atlas reconstruction succeeds.",
            }
        )

    if full_cohort and not args.skip_provenance:
        result_entries.extend(
            [
                {
                    "result_id": "voxel-coordinate-recovery",
                    "title": "Exact GM voxel-coordinate recovery",
                    "path": recovery_dir.name,
                    "status": "completed" if failures.empty else "completed_with_blocked_subjects",
                    "execution_mode": "fresh_run_with_resume",
                    "summary": "Full float32/BLAKE2b recovery with zero-signal rows retained as unresolved.",
                },
                {
                    "result_id": "aal3-numeric-header-correction",
                    "title": "AAL3 numeric-header shape correction",
                    "path": correction_dir.name,
                    "status": "completed_correction",
                    "execution_mode": "fresh_validation",
                    "summary": "All 39 AAL3 files are 167×600; the numeric first line is a header.",
                },
            ]
        )
        _update_catalog(new_results, result_entries, correction_dir.name)
    return int(not failures.empty)


if __name__ == "__main__":
    raise SystemExit(main())
