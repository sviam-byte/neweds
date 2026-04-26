"""Modern CLI for NewEDS time-series connectivity analysis."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AnalysisConfig
from src.core.pipeline import run_analysis
from src.reporting.excel_writer import write_excel_report
from src.reporting.html_generator import write_html_report

_SUPPORTED_EXTS = (".csv", ".xlsx", ".xls", ".parquet")


def _iter_supported_input_files(folder: str, recursive: bool = False) -> list[str]:
    """Return supported input files from a directory."""

    root = Path(folder)
    if recursive:
        files = [str(pp) for pp in root.rglob("*") if pp.is_file() and pp.suffix.lower() in _SUPPORTED_EXTS]
    else:
        files = [str(pp) for pp in root.iterdir() if pp.is_file() and pp.suffix.lower() in _SUPPORTED_EXTS]
    return sorted(files)


def _safe_slug(text: str) -> str:
    """Normalize a path fragment into a safe directory slug."""

    slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (text or "item"))
    slug = slug.strip("._")
    return slug or "item"


def _write_batch_manifest(rows: list[dict], manifest_csv: str) -> None:
    """Write a stable CSV manifest for batch runs."""

    fields = ["input_file", "status", "run_dir", "excel_path", "html_path", "error"]
    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _make_batch_zip(out_root: str) -> str:
    """Create a ZIP archive with the whole batch output directory."""

    out_root_p = Path(out_root)
    zip_path = str(out_root_p.with_suffix(".zip"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pp in sorted(out_root_p.rglob("*")):
            if pp.is_file():
                zf.write(pp, arcname=pp.relative_to(out_root_p.parent))
    return zip_path


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

    cfg = AnalysisConfig(
        max_lag=int(args.lags),
        lag_selection=str(args.lag_selection),
        p_value_alpha=float(args.p_alpha),
        graph_threshold=float(args.graph_threshold),
        variants=variants,
    )

    result = run_analysis(input_path, cfg)

    html_path = ""
    excel_path = ""

    if not args.no_html:
        html_path = write_html_report(
            result,
            out_dir,
            graph_threshold=float(args.graph_threshold),
            p_alpha=float(args.p_alpha),
        )

    if not args.no_excel:
        excel_path = write_excel_report(
            result,
            out_dir,
            threshold=float(args.graph_threshold),
            p_value_alpha=float(args.p_alpha),
        )

    return {
        "input_file": input_path,
        "status": "ok",
        "run_dir": out_dir,
        "excel_path": excel_path,
        "html_path": html_path,
        "error": "",
    }


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input_file)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        rows = []
        files = _iter_supported_input_files(str(input_path), recursive=args.recursive)

        for fp in files:
            run_dir = output_root / _safe_slug(Path(fp).stem)
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                rows.append(_run_one(fp, str(run_dir), args))
            except Exception as exc:
                rows.append(
                    {
                        "input_file": fp,
                        "status": "error",
                        "run_dir": str(run_dir),
                        "excel_path": "",
                        "html_path": "",
                        "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    }
                )

        manifest = output_root / "batch_manifest.csv"
        _write_batch_manifest(rows, str(manifest))

        if args.batch_zip:
            zip_path = _make_batch_zip(str(output_root))
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
