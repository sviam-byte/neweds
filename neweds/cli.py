"""CLI публичного пайплайна NewEDS (time-series connectivity)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neweds.core.batch_pipeline import run_batch, run_one_analysis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neweds",
        description="Считает метрики связности для многомерных временных рядов.",
    )
    p.add_argument("input_file", nargs="?", default="examples/demo_timeseries.csv")
    p.add_argument(
        "--variants",
        default="correlation_full,dcor_full,ordinal_full",
        help="Список метрик через запятую.",
    )
    p.add_argument("--output-dir", default="outputs/demo")
    p.add_argument("--lags", type=int, default=1)
    p.add_argument(
        "--lag-selection",
        choices=["fixed", "optimize"],
        default="fixed",
        help="'fixed' — использовать --lags как есть; 'optimize' — перебрать лаги до --lags.",
    )
    p.add_argument(
        "--controls",
        default="",
        help="Контрольные колонки через запятую (для *_partial метрик).",
    )
    p.add_argument("--graph-threshold", type=float, default=0.2)
    p.add_argument("--p-alpha", type=float, default=0.05)
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--batch-zip", action="store_true")
    return p


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _run_one(input_path: str, out_dir: str, args: argparse.Namespace) -> dict[str, str]:
    return run_one_analysis(
        input_path,
        out_dir,
        variants=_split_csv(args.variants),
        controls=_split_csv(args.controls) or None,
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
        _rows, manifest, zip_path = run_batch(
            str(input_path),
            str(output_root),
            variants=_split_csv(args.variants),
            controls=_split_csv(args.controls) or None,
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

    print("Готово.")
    print(f"Каталог результатов: {row['run_dir']}")
    if row["html_path"]:
        print(f"HTML: {row['html_path']}")
    if row["excel_path"]:
        print(f"Excel: {row['excel_path']}")


if __name__ == "__main__":
    main()
