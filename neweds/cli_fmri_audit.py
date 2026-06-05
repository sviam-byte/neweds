"""CLI for the experimental fMRI ROI time-series audit pipeline."""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_float_csv(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    try:
        parsed = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    for item in parsed:
        if not 0.0 <= item <= 1.0:
            raise argparse.ArgumentTypeError("thresholds must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neweds-fmri-audit",
        description=(
            "Experimental audit for already-extracted fMRI ROI time series in Group_HC/Group_SZ."
        ),
    )
    parser.add_argument("--hc-dir", required=True, help="Directory with HC/control subject CSV files.")
    parser.add_argument("--sz-dir", required=True, help="Directory with SZ/case subject CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated audit outputs.")
    parser.add_argument(
        "--aal3-regions",
        default=None,
        help="Optional aal3_regions.txt file for AAL3 mapping diagnostics.",
    )
    parser.add_argument(
        "--atlas",
        default="all",
        choices=["all", "AAL3", "HCP"],
        help="Atlas family to include. Default: all.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR alpha for edge-wise HC vs SZ comparison. Default: 0.05.",
    )
    parser.add_argument(
        "--bad-roi-strategy",
        default="conservative",
        choices=["conservative"],
        help="Bad ROI exclusion strategy. MVP supports only conservative.",
    )
    parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Also run detrended, AR1 residualized, and ROI-level GSR sensitivity branches.",
    )
    parser.add_argument(
        "--hcp-voxel-map",
        default=None,
        help="Reserved for HCP mask geometry QC; accepted for interface stability.",
    )
    parser.add_argument(
        "--make-figures",
        dest="make_figures",
        action="store_true",
        default=True,
        help="Generate static PNG figures when matplotlib is available. Default: enabled.",
    )
    parser.add_argument(
        "--no-make-figures",
        dest="make_figures",
        action="store_false",
        help="Disable static PNG figure generation.",
    )
    parser.add_argument(
        "--bad-roi-thresholds",
        type=_parse_float_csv,
        default=(0.05, 0.10, 0.20),
        help="Comma-separated threshold sensitivity values. Default: 0.05,0.10,0.20.",
    )
    parser.add_argument(
        "--include-ttest",
        action="store_true",
        help="Write Welch t-test edge-wise sensitivity outputs.",
    )
    parser.add_argument(
        "--include-permutation",
        action="store_true",
        help="Write exploratory permutation sensitivity for subject-level FC metrics.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Permutation count for exploratory subject-level sensitivity. Default: 1000.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for deterministic permutation sensitivity. Default: 0.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    from neweds.core.fmri_roi_audit import (
        FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE,
        run_fmri_roi_audit,
    )

    print(FMRI_ROI_AUDIT_EXPERIMENTAL_NOTICE, file=sys.stderr)
    branches = ("raw_cleaned",)
    if args.include_sensitivity:
        branches = ("raw_cleaned", "detrended", "ar1_residualized", "roi_level_gsr")

    try:
        result = run_fmri_roi_audit(
            args.hc_dir,
            args.sz_dir,
            args.output_dir,
            aal3_regions=args.aal3_regions,
            atlas_filter=args.atlas,
            bad_roi_strategy=args.bad_roi_strategy,
            alpha=args.alpha,
            branches=branches,
            include_hcp_mask_qc=bool(args.hcp_voxel_map),
            hcp_voxel_map=args.hcp_voxel_map,
            make_figures=args.make_figures,
            bad_roi_thresholds=args.bad_roi_thresholds,
            include_ttest=args.include_ttest,
            include_permutation=args.include_permutation,
            n_permutations=args.n_permutations,
            random_seed=args.random_seed,
        )
    except Exception as exc:
        logging.error("fMRI ROI audit failed: %s", exc, exc_info=args.verbose)
        sys.exit(1)

    summary = result.as_dict()
    print("\n=== fMRI ROI Audit Result ===")
    col_w = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"  {key:<{col_w}} : {value}")
    print(f"\nResults saved in: {result.output_dir}")


if __name__ == "__main__":
    main()
