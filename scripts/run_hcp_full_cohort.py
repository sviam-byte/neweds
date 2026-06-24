"""Assemble the immutable full-cohort HCP360 Stage 1 input result.

This script does not overwrite the earlier two-subject smoke result.  It creates
``<date>_gm-regional-signals-HCP360-full-cohort`` and resumes subject-by-subject:
existing full-cohort outputs are reused, the smoke outputs for 1097/1185 may be
copied, and remaining recovery-OK subjects are computed from the HDF5 + mapping
sidecars.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neweds.core.fmri_gm_signals import (  # noqa: E402
    align_whole_brain_input,
    build_regional_signals,
    load_hcp_atlas,
    sha256_file,
)
from scripts.write_result_provenance import write_result_provenance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=r"D:\Данные по сознанию-")
    parser.add_argument("--new-results", default=r"D:\Данные по сознанию- (2)\new_results")
    parser.add_argument("--recovery-result", default="")
    parser.add_argument("--smoke-result", default="")
    parser.add_argument("--result-date", default=date.today().isoformat())
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--skip-provenance", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1729)
    return parser


def _subject_id(path: Path) -> str:
    return path.name.split("_", 1)[0]


def _extract_hcp_ready(source_root: Path, cache_root: Path) -> dict[str, Path]:
    destination = cache_root / "hcp_ready"
    marker = destination / ".extracted"
    if not marker.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_root / "atlas_HCP.zip") as archive:
            archive.extractall(destination)
        marker.write_text("atlas_HCP.zip extracted by run_hcp_full_cohort.py\n", encoding="utf-8")
    return {_subject_id(path): path for path in destination.rglob("*_HCP_MMP1_timeseries.csv")}


def _copy_smoke_subject(smoke: Path, target: Path, subject_id: str, atlas_id: str) -> bool:
    copied = False
    for suffix in ("gm_signals.npz", "homogeneity.parquet", "method_status.parquet"):
        src = smoke / "subjects" / f"{subject_id}_{atlas_id}_{suffix}"
        dst = target / "subjects" / src.name
        if src.is_file() and not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied = True
    return copied


def _status_from_files(
    *,
    subject_id: str,
    group: str,
    atlas_id: str,
    result_dir: Path,
    recovery_row: pd.Series,
    node_order: list[int],
) -> dict[str, Any] | None:
    signal = result_dir / "subjects" / f"{subject_id}_{atlas_id}_gm_signals.npz"
    homogeneity = result_dir / "subjects" / f"{subject_id}_{atlas_id}_homogeneity.parquet"
    methods = result_dir / "subjects" / f"{subject_id}_{atlas_id}_method_status.parquet"
    if not (signal.is_file() and homogeneity.is_file() and methods.is_file()):
        return None
    hom = pd.read_parquet(homogeneity)
    coverage = float((hom["active_gm_voxels"] > 0).mean()) if not hom.empty else 0.0
    return {
        "subject_id": str(subject_id),
        "group": str(group),
        "atlas_id": atlas_id,
        "status": "ok",
        "n_nodes": len(node_order),
        "n_timepoints": 600,
        "signal_npz": str(signal),
        "homogeneity_table": str(homogeneity),
        "method_status_table": str(methods),
        "coverage": coverage,
        "unresolved_rows": int(recovery_row.get("n_zero_rows", 0)),
        "node_order": node_order,
        "method_metadata": {
            "methods": [
                "active_mean",
                "pca_pc1_oriented",
                "ica_1_oriented",
                "correlation_core",
            ],
            "resume_source": "existing_or_smoke_copy",
        },
        "input_sha256": {
            str(signal): sha256_file(signal),
            str(homogeneity): sha256_file(homogeneity),
            str(methods): sha256_file(methods),
        },
        "message": "",
    }


def _write_catalog(root: Path, result_dir: Path, status: str) -> None:
    catalog_path = root / "results_catalog.json"
    if not catalog_path.is_file():
        return
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    entry = {
        "result_id": "gm-regional-signals-HCP360-full-cohort",
        "title": "GM-only regional signals — HCP-MMP1 360 full cohort",
        "path": result_dir.name,
        "status": status,
        "execution_mode": "fresh_run_with_resume",
        "summary": "Full HCP360 GM-only regional-signal assembly and paired whole-brain inputs for recovery-eligible subjects.",
    }
    by_id = {item["result_id"]: item for item in catalog.get("results", [])}
    by_id[entry["result_id"]] = entry
    catalog["results"] = list(by_id.values())
    catalog["result_count"] = len(catalog["results"])
    catalog["updated_at"] = pd.Timestamp.now(tz="Europe/Moscow").isoformat()
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root)
    root = Path(args.new_results)
    recovery = Path(args.recovery_result) if args.recovery_result else root / f"{args.result_date}_voxel-coordinate-recovery"
    smoke = Path(args.smoke_result) if args.smoke_result else root / f"{args.result_date}_gm-regional-signals-HCP360"
    result = root / f"{args.result_date}_gm-regional-signals-HCP360-full-cohort"
    result.mkdir(parents=True, exist_ok=True)
    (result / "subjects").mkdir(parents=True, exist_ok=True)
    (result / "whole_brain").mkdir(parents=True, exist_ok=True)
    (result / "cohort").mkdir(parents=True, exist_ok=True)

    recovery_table = pd.read_csv(recovery / "subject_recovery_status.csv")
    if args.subjects:
        wanted = set(str(subject) for subject in args.subjects)
        recovery_table = recovery_table[recovery_table["subject_id"].astype(str).isin(wanted)].copy()
    atlas = load_hcp_atlas(source_root / "HCP-MMP1_atlas_voxel_map_from_xml.csv")
    atlas.node_table.to_csv(result / "node_table.csv", index=False, encoding="utf-8-sig")
    node_order = atlas.node_table["region_id"].astype(int).tolist()
    hcp_ready = _extract_hcp_ready(source_root, root / "_atlas_cache")

    regional_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for _, row in recovery_table.sort_values("subject_id").iterrows():
        subject_id = str(row["subject_id"])
        group = str(row["group"])
        copied_from_smoke = False
        if row["status"] == "ok":
            existing = _status_from_files(
                subject_id=subject_id,
                group=group,
                atlas_id=atlas.atlas_id,
                result_dir=result,
                recovery_row=row,
                node_order=node_order,
            )
            if existing is None and smoke.is_dir():
                copied_from_smoke = _copy_smoke_subject(smoke, result, subject_id, atlas.atlas_id)
                existing = _status_from_files(
                    subject_id=subject_id,
                    group=group,
                    atlas_id=atlas.atlas_id,
                    result_dir=result,
                    recovery_row=row,
                    node_order=node_order,
                )
            if existing is None:
                built = build_regional_signals(
                    subject_id=subject_id,
                    group=group,
                    tissue_h5=row["tissue_h5"],
                    mapping_parquet=recovery / "mappings" / f"{subject_id}.parquet",
                    atlas=atlas,
                    output_dir=result / "subjects",
                    random_seed=int(args.random_seed),
                )
                existing = built.as_dict()
            if existing["status"] == "ok" and subject_id in hcp_ready:
                paired_rows.append(
                    align_whole_brain_input(
                        subject_id=subject_id,
                        group=group,
                        atlas=atlas,
                        ready_roi_csv=hcp_ready[subject_id],
                        output_dir=result / "whole_brain",
                    )
                )
            regional_rows.append(existing)
            print(f"HCP-full {subject_id} {existing['status']} copied_smoke={copied_from_smoke}", flush=True)
        else:
            blocked = {
                "subject_id": subject_id,
                "group": group,
                "atlas_id": atlas.atlas_id,
                "status": "blocked_recovery",
                "n_nodes": atlas.n_nodes,
                "n_timepoints": 600,
                "signal_npz": "",
                "homogeneity_table": "",
                "method_status_table": "",
                "coverage": 0.0,
                "unresolved_rows": int(row.get("n_unmatched_rows", 0)) + int(row.get("n_zero_rows", 0)),
                "node_order": node_order,
                "method_metadata": {},
                "input_sha256": {},
                "message": str(row.get("message", "")),
            }
            regional_rows.append(blocked)
            print(f"HCP-full {subject_id} blocked_recovery", flush=True)
        alignment_rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "recovery_status": str(row["status"]),
                "gm_status": regional_rows[-1]["status"],
                "whole_brain_available": subject_id in hcp_ready,
                "paired_stage2_eligible": bool(regional_rows[-1]["status"] == "ok" and subject_id in hcp_ready),
            }
        )

    pd.DataFrame(regional_rows).to_json(
        result / "subject_status.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    pd.DataFrame(paired_rows).to_json(
        result / "paired_input_manifest.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    pd.DataFrame(alignment_rows).to_csv(
        result / "cohort" / "subject_alignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hom_paths = sorted((result / "subjects").glob("*_homogeneity.parquet"))
    method_paths = sorted((result / "subjects").glob("*_method_status.parquet"))
    if hom_paths:
        pd.concat([pd.read_parquet(path) for path in hom_paths], ignore_index=True).to_parquet(
            result / "region_homogeneity.parquet", compression="zstd", index=False
        )
    if method_paths:
        pd.concat([pd.read_parquet(path) for path in method_paths], ignore_index=True).to_parquet(
            result / "method_status.parquet", compression="zstd", index=False
        )
    n_ok = sum(row["status"] == "ok" for row in regional_rows)
    n_blocked = len(regional_rows) - n_ok
    status = "completed" if n_blocked == 0 else "completed_with_blocked_subjects"
    (result / "scope_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "planned_subjects": int(len(recovery_table)),
                "gm_ok_subjects": int(n_ok),
                "blocked_subjects": int(n_blocked),
                "paired_stage2_eligible_subjects": int(len(paired_rows)),
                "blocked_reason": "Subject 1186 remains blocked at exact coordinate recovery" if n_blocked else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.skip_provenance:
        write_result_provenance(
            result_dir=result,
            result_id="gm-regional-signals-HCP360-full-cohort",
            title="GM-only regional signals — HCP-MMP1 360 full cohort",
            result_type="gm_regional_signals_full_cohort",
            status=status,
            execution_mode="fresh_run_with_resume",
            summary=f"HCP360 full-cohort assembly produced {n_ok} GM-only subject inputs and {len(paired_rows)} paired whole-brain inputs; {n_blocked} subjects blocked.",
            meaning="This result is the validated HCP-first input set for the Stage2 GM-only versus whole-brain benchmark.",
            command="python scripts/run_hcp_full_cohort.py",
            inputs=[
                str(recovery),
                str(source_root / "HCP-MMP1_atlas_voxel_map_from_xml.csv"),
                str(source_root / "atlas_HCP.zip"),
            ],
            code_files=[
                REPO / "neweds/core/fmri_gm_signals.py",
                REPO / "scripts/run_hcp_full_cohort.py",
                REPO / "scripts/write_result_provenance.py",
            ],
            repository=REPO,
            findings=[
                f"GM-only HCP360 ok subjects: {n_ok}.",
                f"Paired GM-only ↔ whole-brain subjects: {len(paired_rows)}.",
                f"Blocked subjects: {n_blocked}.",
            ],
            limitations=[
                "AAL3v2 remains blocked and is not approximated here.",
                "Blocked recovery subjects are not assigned invented coordinates.",
                "No connectivity or classification is computed in this result.",
            ],
        )
        _write_catalog(root, result, status)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
