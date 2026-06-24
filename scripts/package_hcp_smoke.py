"""Package the completed two-subject HCP real-data smoke without overstating scope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neweds.core.fmri_gm_signals import align_whole_brain_input, load_hcp_atlas  # noqa: E402
from scripts.write_result_provenance import write_result_provenance  # noqa: E402


def main() -> None:
    source = Path(r"D:\Данные по сознанию-")
    root = Path(r"D:\Данные по сознанию- (2)\new_results")
    result = root / "2026-06-23_gm-regional-signals-HCP360"
    atlas = load_hcp_atlas(source / "HCP-MMP1_atlas_voxel_map_from_xml.csv")
    ready = {
        path.name.split("_", 1)[0]: path
        for path in (root / "_atlas_cache" / "hcp_ready").rglob("*_HCP_MMP1_timeseries.csv")
    }
    groups = {"1097": "SZ", "1185": "HC"}
    paired = []
    statuses = []
    for subject_id, group in groups.items():
        paired.append(
            align_whole_brain_input(
                subject_id=subject_id,
                group=group,
                atlas=atlas,
                ready_roi_csv=ready[subject_id],
                output_dir=result / "whole_brain",
            )
        )
        statuses.append(
            {
                "subject_id": subject_id,
                "group": group,
                "atlas_id": atlas.atlas_id,
                "status": "ok_real_data_smoke",
                "shape": [360, 600],
                "gm_signal_npz": str(
                    result / "subjects" / f"{subject_id}_{atlas.atlas_id}_gm_signals.npz"
                ),
            }
        )
    pd.DataFrame(statuses).to_json(
        result / "subject_status.jsonl", orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame(paired).to_json(
        result / "paired_input_manifest.jsonl", orient="records", lines=True, force_ascii=False
    )
    (result / "scope_status.json").write_text(
        json.dumps(
            {
                "status": "partial_real_data_smoke",
                "completed_subjects": ["1097", "1185"],
                "planned_subjects": 39,
                "full_cohort_complete": False,
                "reason": "Full regional signal run is computationally pending; recovery is complete separately.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_result_provenance(
        result_dir=result,
        result_id="gm-regional-signals-HCP360",
        title="GM-only regional signals — HCP-MMP1 360",
        result_type="gm_regional_signals_real_data_smoke",
        status="partial_real_data_smoke",
        execution_mode="fresh_run_with_resume",
        summary="The full HCP360 pipeline was validated on real HC 1185 and SZ 1097; full-cohort regional signals remain pending.",
        meaning="The smoke proves node order, four GM signal outputs, QC tables, and paired whole-brain alignment without claiming 39-subject completion.",
        command="python scripts/run_gm_stage1.py --subjects 1185 1097 --skip-aal --skip-provenance",
        inputs=[
            str(source / "HCP-MMP1_atlas_voxel_map_from_xml.csv"),
            str(source / "atlas_HCP.zip"),
        ],
        code_files=[
            REPO / "neweds/core/fmri_gm_signals.py",
            REPO / "neweds/cli_fmri_gm_signals.py",
            REPO / "scripts/run_gm_stage1.py",
            REPO / "scripts/package_hcp_smoke.py",
            REPO / "tests/test_fmri_gm_signals.py",
        ],
        repository=REPO,
        findings=[
            "HC subject 1185 and SZ subject 1097 produced 360×600 outputs.",
            "All four signal methods and spatial/homogeneity QC were emitted.",
            "Whole-brain HCP inputs were aligned to the identical node table.",
        ],
        limitations=[
            "Only 2/39 subjects have HCP regional signals in this result.",
            "Subject 1186 is independently blocked at coordinate recovery.",
            "No connectivity or HC/SZ classification was run.",
        ],
    )
    catalog_path = root / "results_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    entry = {
        "result_id": "gm-regional-signals-HCP360",
        "title": "GM-only regional signals — HCP-MMP1 360",
        "path": result.name,
        "status": "partial_real_data_smoke",
        "execution_mode": "fresh_run_with_resume",
        "summary": "Real-data smoke on 1185 and 1097; full-cohort HCP signals remain pending.",
    }
    by_id = {item["result_id"]: item for item in catalog["results"]}
    by_id[entry["result_id"]] = entry
    catalog["results"] = list(by_id.values())
    catalog["result_count"] = len(catalog["results"])
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
