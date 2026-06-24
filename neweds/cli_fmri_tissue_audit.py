"""CLI for the independent GM/WM/CSF HDF5 audit."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neweds-fmri-tissue-audit",
        description=(
            "Independent streaming audit for HDF5 files with "
            "GM/data, WM/data, and CSF/data."
        ),
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--block-rows", type=int, default=8192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from neweds.core.fmri_tissue_audit import run_fmri_tissue_audit

    try:
        result = run_fmri_tissue_audit(
            args.input_dir,
            args.output_dir,
            max_lag=int(args.max_lag),
            block_rows=int(args.block_rows),
        )
    except Exception as exc:
        print(f"tissue audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("=== Independent GM/WM/CSF Tissue Audit ===")
    for key, value in result.as_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
