"""CLI for the HCP-first fMRI Stage 2 benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from neweds.core.fmri_stage2_benchmark import (
    DEFAULT_REPRESENTATIONS,
    FmriStage2Config,
    run_stage2_benchmark,
)


def _split_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gm-hcp-result", required=True)
    parser.add_argument("--whole-brain-hcp-inputs", default="")
    parser.add_argument("--new-results-root", required=True)
    parser.add_argument("--result-date", default=date.today().isoformat())
    parser.add_argument(
        "--representations",
        default=",".join(DEFAULT_REPRESENTATIONS),
        help="Comma-separated representation list.",
    )
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--branches", default="")
    parser.add_argument("--primary-lag", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-provenance", action="store_true")
    return parser


def _update_catalog(root: Path, entries: list[dict]) -> None:
    catalog_path = root / "results_catalog.json"
    if not catalog_path.is_file():
        return
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    by_id = {item["result_id"]: item for item in catalog.get("results", [])}
    for entry in entries:
        by_id[entry["result_id"]] = entry
    catalog["results"] = list(by_id.values())
    catalog["result_count"] = len(catalog["results"])
    catalog["updated_at"] = pd.Timestamp.now(tz="Europe/Moscow").isoformat()
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_provenance(config: FmriStage2Config) -> None:
    try:
        from scripts.write_result_provenance import write_result_provenance
    except Exception:
        return
    repo = Path(__file__).resolve().parents[1]
    metrics_dir = Path(config.metrics_result_dir)
    class_dir = Path(config.classification_result_dir)
    common_code = [
        repo / "neweds/core/fmri_stage2_benchmark.py",
        repo / "neweds/cli_fmri_stage2_benchmark.py",
        repo / "pyproject.toml",
    ]
    write_result_provenance(
        result_dir=metrics_dir,
        result_id="fmri-HCP360-temporal-qc-and-metric-matrices",
        title="HCP360 temporal QC and metric matrices",
        result_type="fmri_stage2_metric_matrices",
        status="completed_smoke" if config.smoke else "completed_or_partial_with_status_table",
        execution_mode="fresh_run",
        summary="HCP360 regional inputs were preprocessed into Stage2 branches and NewEDS metric matrices/features were emitted with per-subject status.",
        meaning="This result is the auditable feature ledger used by the downstream HC/SZ nested LOOCV benchmark.",
        command="neweds-fmri-stage2-benchmark",
        inputs=[config.gm_hcp_result, config.whole_brain_hcp_inputs],
        code_files=common_code,
        repository=repo,
        findings=[
            "Registry snapshot, temporal QC, global-signal QC, metric status, matrices and feature manifest were written.",
            "Invalid values remain NaN and are not replaced by zeros.",
        ],
        limitations=[
            "This is a method-comparison feature ledger, not diagnostic evidence.",
            "AAL3v2 remains outside this HCP-first result.",
        ],
    )
    write_result_provenance(
        result_dir=class_dir,
        result_id="fmri-HCP360-HC-SZ-nested-loocv",
        title="HCP360 HC/SZ nested LOOCV benchmark",
        result_type="fmri_stage2_nested_loocv",
        status="completed_smoke" if config.smoke else "completed_or_partial_with_status_table",
        execution_mode="fresh_run",
        summary="Nested LOOCV HC/SZ benchmark was run from the Stage2 HCP360 feature ledger.",
        meaning="Each subject receives out-of-fold predictions; results are exploratory method comparisons, not a clinical classifier.",
        command="neweds-fmri-stage2-benchmark",
        inputs=[config.metrics_result_dir, config.gm_hcp_result],
        code_files=common_code,
        repository=repo,
        findings=[
            "OOF predictions, metric performance, permutation p-values, bootstrap CIs, FDR, paired GM-vs-whole comparisons and feature stability were written.",
        ],
        limitations=[
            "Small-n exploratory benchmark; no diagnostic or clinical validation claim.",
            "Model/preprocessing choices are not selected by held-out performance.",
        ],
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.new_results_root)
    metrics_dir = root / f"{args.result_date}_fmri-HCP360-temporal-qc-and-metric-matrices"
    class_dir = root / f"{args.result_date}_fmri-HCP360-HC-SZ-nested-loocv"
    branches = _split_csv(args.branches) if args.branches else ()
    config = FmriStage2Config(
        gm_hcp_result=str(Path(args.gm_hcp_result)),
        whole_brain_hcp_inputs=str(Path(args.whole_brain_hcp_inputs or args.gm_hcp_result)),
        new_results_root=str(root),
        metrics_result_dir=str(metrics_dir),
        classification_result_dir=str(class_dir),
        representations=_split_csv(args.representations),
        metrics=_split_csv(args.metrics),
        branches=branches or FmriStage2Config.__dataclass_fields__["branches"].default,
        primary_lag=int(args.primary_lag),
        permutations=int(args.permutations),
        bootstraps=int(args.bootstraps),
        smoke=bool(args.smoke),
        random_seed=int(args.random_seed),
    )
    _features, performance = run_stage2_benchmark(config)
    if not args.skip_provenance:
        _write_provenance(config)
        _update_catalog(
            root,
            [
                {
                    "result_id": "fmri-HCP360-temporal-qc-and-metric-matrices",
                    "title": "HCP360 temporal QC and metric matrices",
                    "path": metrics_dir.name,
                    "status": "completed_smoke" if config.smoke else "completed_or_partial_with_status_table",
                    "execution_mode": "fresh_run",
                    "summary": "HCP360 Stage2 preprocessing, temporal QC, metric matrices and feature manifest.",
                },
                {
                    "result_id": "fmri-HCP360-HC-SZ-nested-loocv",
                    "title": "HCP360 HC/SZ nested LOOCV benchmark",
                    "path": class_dir.name,
                    "status": "completed_smoke" if config.smoke else "completed_or_partial_with_status_table",
                    "execution_mode": "fresh_run",
                    "summary": "Exploratory nested LOOCV benchmark comparing HCP360 GM-only and whole-brain representations.",
                },
            ],
        )
    if performance.empty:
        print("Stage2 completed but no classification performance rows were produced.", file=sys.stderr)
        return 0 if config.smoke else 1
    print(f"Stage2 wrote {len(performance)} performance rows to {class_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
