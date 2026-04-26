"""CLI группового сравнения fMRI connectivity (case vs control)."""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neweds-group",
        description=(
            "Group fMRI connectivity comparison. Inputs are subject-wise CSV/Excel/Parquet "
            "after spatial binning; HDF5 group input is not supported yet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Пример:
  neweds-group \\
      --case-dir data/case \\
      --control-dir data/control \\
      --output-dir results/group \\
      --spatial-grid-size 10 \\
      --strategy intersection \\
      --alpha 0.05
""",
    )
    p.add_argument(
        "--case-dir",
        required=True,
        help="Каталог с CSV — по одному файлу на пациента (case-группа).",
    )
    p.add_argument(
        "--control-dir",
        required=True,
        help="Каталог с CSV — по одному файлу на контрольного субъекта.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Каталог для результатов (будет создан, если нет).",
    )
    p.add_argument(
        "--method",
        default="correlation",
        choices=["correlation"],
        help="Метод connectivity (по умолчанию: correlation).",
    )
    p.add_argument(
        "--spatial-grid-size",
        type=int,
        default=10,
        metavar="N",
        help="Размер пространственного бина: bin_key = floor(coord / N) (по умолчанию: 10).",
    )
    p.add_argument(
        "--strategy",
        default="intersection",
        choices=["intersection", "union"],
        help="Стратегия canonical voxel space (по умолчанию: intersection).",
    )
    p.add_argument(
        "--canonical-reference",
        default="all",
        choices=["case", "control", "schiz", "healthy", "all"],
        help="По какой группе строится canonical space (по умолчанию: all).",
    )
    p.add_argument(
        "--min-bin-coverage",
        type=float,
        default=0.8,
        metavar="F",
        help="Минимальное покрытие бина субъектами, доля в [0, 1] (по умолчанию: 0.8).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="A",
        help="Уровень значимости для FDR Бенджамини–Хохберга (по умолчанию: 0.05).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=32768,
        metavar="ROWS",
        help="Размер чанка при потоковом чтении CSV (по умолчанию: 32768).",
    )
    p.add_argument(
        "--no-save-features",
        action="store_true",
        help="Не сохранять матрицы признаков по субъектам (экономит диск).",
    )
    p.add_argument(
        "--allow-skip-subjects",
        action="store_true",
        help="Allow failed subject files to be skipped; default is fail-fast.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Подробный DEBUG-лог.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    from neweds.core.group_pipeline import run_group_pipeline

    try:
        summary = run_group_pipeline(
            schiz_dir=args.case_dir,
            healthy_dir=args.control_dir,
            output_dir=args.output_dir,
            method=args.method,
            spatial_grid_size=args.spatial_grid_size,
            strategy=args.strategy,
            alpha=args.alpha,
            canonical_reference=args.canonical_reference,
            min_bin_coverage=args.min_bin_coverage,
            csv_chunk_rows=args.chunk_size,
            save_feature_matrix=not args.no_save_features,
            allow_skip=bool(args.allow_skip_subjects),
        )
    except Exception as exc:
        logging.error("Группой пайплайн упал: %s", exc, exc_info=args.verbose)
        sys.exit(1)

    print("\n=== Результат ===")
    col_w = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"  {k:<{col_w}} : {v}")
    print(f"\nРезультаты сохранены в: {summary['output_dir']}")


if __name__ == "__main__":
    main()
