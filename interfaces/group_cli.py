"""CLI точка входа: групповое сравнение fMRI connectivity (шизофрения vs здоровые)."""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="group_compare",
        description="Групповое сравнение fMRI connectivity: шизофрения vs здоровые.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Примеры:
  python -m interfaces.group_cli \
      --schiz-dir "D:\Шизофрения" \
      --healthy-dir "D:\Здоровые" \
      --output-dir results\group_compare

  python -m interfaces.group_cli \
      --schiz-dir data/schiz \
      --healthy-dir data/healthy \
      --output-dir results/ \
      --spatial-grid-size 10 \
      --method correlation \
      --strategy intersection \
      --alpha 0.05 \
      --no-save-features
""",
    )
    p.add_argument(
        "--schiz-dir", required=True,
        help="Директория с CSV-файлами группы шизофрения (по одному файлу на субъекта)",
    )
    p.add_argument(
        "--healthy-dir", required=True,
        help="Директория с CSV-файлами здоровых (по одному файлу на субъекта)",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Директория для результатов (создаётся если не существует)",
    )
    p.add_argument(
        "--method", default="correlation",
        choices=["correlation"],
        help="Метод connectivity (default: correlation)",
    )
    p.add_argument(
        "--spatial-grid-size", type=int, default=10, metavar="N",
        help=(
            "Размер пространственного бина: bin_key = floor(coord / N). "
            "N=10 даёт ~800-1400 бинов для стандартного mfRI-пространства (default: 10)"
        ),
    )
    p.add_argument(
        "--strategy", default="intersection",
        choices=["intersection", "union"],
        help=(
            "Стратегия canonical voxel space:\n"
            "  intersection — только общие бины (рекомендуется, default)\n"
            "  union        — все бины, отсутствующие заполняются NaN"
        ),
    )
    p.add_argument(
        "--alpha", type=float, default=0.05, metavar="A",
        help="Уровень значимости FDR Benjamini-Hochberg (default: 0.05)",
    )
    p.add_argument(
        "--chunk-size", type=int, default=32768, metavar="ROWS",
        help="Размер чанка при потоковой загрузке CSV (default: 32768)",
    )
    p.add_argument(
        "--no-save-features", action="store_true",
        help="Не сохранять матрицы признаков .npy (экономия места на диске)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Подробный лог уровня DEBUG",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    from src.core.group_pipeline import run_group_pipeline

    try:
        summary = run_group_pipeline(
            schiz_dir=args.schiz_dir,
            healthy_dir=args.healthy_dir,
            output_dir=args.output_dir,
            method=args.method,
            spatial_grid_size=args.spatial_grid_size,
            strategy=args.strategy,
            alpha=args.alpha,
            csv_chunk_rows=args.chunk_size,
            save_feature_matrix=not args.no_save_features,
        )
    except Exception as exc:
        logging.error("Pipeline завершился с ошибкой: %s", exc, exc_info=args.verbose)
        sys.exit(1)

    print("\n=== Результат ===")
    col_w = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"  {k:<{col_w}} : {v}")
    print(f"\nРезультаты сохранены в: {summary['output_dir']}")


if __name__ == "__main__":
    main()
