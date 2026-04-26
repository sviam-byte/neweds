"""Modern CLI for NewEDS time-series connectivity analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.batch_pipeline import (
    iter_supported_input_files,
    make_batch_zip,
    run_batch,
    run_one_analysis,
    safe_slug,
    write_batch_manifest,
)

# Backward-compatible aliases for older tests/imports; implementation lives in core.
_iter_supported_input_files = iter_supported_input_files
_safe_slug = safe_slug
_write_batch_manifest = write_batch_manifest
_make_batch_zip = make_batch_zip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neweds",
        description="Compute connectivity metrics for multivariate time-series data.",
    )
    p.add_argument("input_file", nargs="?", default="demo.csv")
    p.add_argument("--variants", default="correlation_full,dcor_full,ordinal_full")
    p.add_argument("--output-dir", default="outputs/demo")
    p.add_argument("--lags", type=int, default=1)
    p.add_argument("--lag-selection", choices=["fixed", "optimize"], default="fixed")
    p.add_argument("--graph-threshold", type=float, default=0.2)
    p.add_argument("--p-alpha", type=float, default=0.05)
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--batch-zip", action="store_true")
    return p


def _run_one(input_path: str, out_dir: str, args: argparse.Namespace) -> dict[str, str]:
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    return run_one_analysis(
        input_path,
        out_dir,
        variants=variants,
        lags=int(args.lags),
        lag_selection=str(args.lag_selection),
        graph_threshold=float(args.graph_threshold),
        p_alpha=float(args.p_alpha),
        write_html=not args.no_html,
        write_excel=not args.no_excel,
    )


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input_file)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
        _rows, manifest, zip_path = run_batch(
            str(input_path),
            str(output_root),
            variants=variants,
            recursive=bool(args.recursive),
            lags=int(args.lags),
            lag_selection=str(args.lag_selection),
            graph_threshold=float(args.graph_threshold),
            p_alpha=float(args.p_alpha),
            write_html=not args.no_html,
            write_excel=not args.no_excel,
            create_zip=bool(args.batch_zip),
        )
        if zip_path:
            print(f"ZIP: {zip_path}")

        print(f"Manifest: {manifest}")
        return

    try:
        row = _run_one(str(input_path), str(output_root), args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Analysis complete.")
    print(f"Output directory: {row['run_dir']}")
    if row["html_path"]:
        print(f"HTML: {row['html_path']}")
    if row["excel_path"]:
        print(f"Excel: {row['excel_path']}")


if __name__ == "__main__":
    main()
