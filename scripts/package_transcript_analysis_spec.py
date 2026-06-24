"""Package the transcript-derived analysis specification into new_results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neweds.metrics import list_metrics  # noqa: E402
from scripts.write_result_provenance import write_result_provenance  # noqa: E402

TASKS = [
    ("T01", "Сверить субъекты ROI и tissue", "completed", "39/39: 15 HC, 24 SZ"),
    (
        "T02",
        "Разрешить соответствие 168 ROI-строк 167 atlas labels",
        "blocked_external",
        "Нужна однозначная row-to-label таблица",
    ),
    (
        "T03",
        "Получить GM xyz или точное восстановление voxel coordinates",
        "blocked_external",
        "GM/xyz отсутствует во всех 39 HDF5",
    ),
    (
        "T04",
        "Получить atlas/parcellation mask и spatial transform",
        "blocked_external",
        "Нельзя использовать кубическое или порядковое разбиение",
    ),
    (
        "T05",
        "Построить mask-constrained voxel-to-parcel mapping",
        "pending_after_T03_T04",
        "Остановка роста на границе GM-маски",
    ),
    (
        "T06",
        "Оценить внутрозонную voxel-wise однородность",
        "pending_after_T05",
        "Распределения корреляций и доверие к зоне",
    ),
    (
        "T07",
        "Сравнить mean, oriented PCA, ICA и correlation-core",
        "pending_after_T05",
        "Выбор только по label-blind QC",
    ),
    (
        "T08",
        "Проверить trend, stationarity, ACF, PACF и Ljung-Box",
        "partially_completed",
        "Tissue means готовы; региональные ряды ещё нет",
    ),
    (
        "T09",
        "Сравнить baseline и AR(1), при необходимости AR(2)",
        "partially_completed",
        "AR(1) для tissue means готов; требуется региональный уровень",
    ),
    (
        "T10",
        "Выполнить парные without-GSR и with-GSR ветки",
        "pending",
        "GSR не выбирается по классификационной AUC",
    ),
    (
        "T11",
        "Рассчитать все зарегистрированные метрики для всех субъектов",
        "pending",
        "26 метрик; failures и NaN сохраняются явно",
    ),
    (
        "T12",
        "Переработать HC/SZ benchmark без leakage",
        "pending",
        "Nested LOOCV, все метрики, no NaN-to-zero",
    ),
    (
        "T13",
        "Получить OOF-прогноз каждого из 39 субъектов",
        "pending_after_T11_T12",
        "Отдельно GM-only и whole-brain",
    ),
    (
        "T14",
        "Permutation tests, CI и BH-FDR",
        "pending_after_T13",
        "Основная семья: 26 метрик × 2 представления",
    ),
    (
        "T15",
        "Парно сравнить GM-only и whole-brain",
        "pending_after_T13",
        "Paired bootstrap/permutation на тех же 39 субъектах",
    ),
    (
        "T16",
        "Исследовать lag × window × metric cube",
        "deferred_exploratory",
        "Только после завершения основного benchmark",
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cohort_alignment(roi_inventory: Path, tissue_inventory: Path) -> pd.DataFrame:
    roi = pd.read_csv(roi_inventory, dtype={"subject_id": str})
    tissue = pd.read_csv(tissue_inventory, dtype={"subject_id": str})
    roi_pairs = {
        (str(row.group), str(row.subject_id)): str(row.file_path)
        for row in roi.itertuples()
    }
    tissue_pairs = {
        (str(row.group), str(row.subject_id)): str(row.file_path)
        for row in tissue.itertuples()
    }
    pairs = sorted(set(roi_pairs) | set(tissue_pairs))
    return pd.DataFrame(
        [
            {
                "group": group,
                "subject_id": subject_id,
                "roi_present": (group, subject_id) in roi_pairs,
                "tissue_present": (group, subject_id) in tissue_pairs,
                "in_primary_intersection": (group, subject_id) in roi_pairs
                and (group, subject_id) in tissue_pairs,
                "roi_file": roi_pairs.get((group, subject_id), ""),
                "tissue_file": tissue_pairs.get((group, subject_id), ""),
            }
            for group, subject_id in pairs
        ]
    )


def _write_registered_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "name": metric.name,
            "category": metric.category,
            "directed": bool(metric.directed),
            "partial_mode": str(metric.partial_mode),
            "experimental": bool(metric.experimental),
        }
        for metric in list_metrics()
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return rows


def _update_catalog(
    output_root: Path,
    *,
    result_id: str,
    title: str,
    result_path: str,
    status: str,
    execution_mode: str,
    summary: str,
) -> None:
    catalog_path = output_root / "results_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["results"] = [
        result
        for result in catalog.get("results", [])
        if result.get("result_id") != result_id
    ]
    catalog["results"].append(
        {
            "result_id": result_id,
            "title": title,
            "path": result_path,
            "status": status,
            "execution_mode": execution_mode,
            "summary": summary,
        }
    )
    catalog["result_count"] = len(catalog["results"])
    catalog["updated_at"] = datetime.now().astimezone().isoformat()
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_path = output_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    heading = f"### [{title}]({result_path}/ABOUT_THIS_RESULT.md)"
    if heading not in readme:
        section = "\n".join(
            [
                "",
                heading,
                "",
                f"- ID: `{result_id}`",
                f"- Статус: `{status}`",
                f"- Режим: `{execution_mode}`",
                f"- Кратко: {summary}",
                "",
            ]
        )
        marker = "## Общие материалы"
        readme = readme.replace(marker, section + "\n" + marker)
        readme_path.write_text(readme, encoding="utf-8")


def _write_workspace_inventory(output_root: Path) -> None:
    inventory_path = output_root / "workspace_file_inventory.csv"
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != inventory_path:
                writer.writerow(
                    {
                        "path": path.relative_to(output_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )


def package_spec(
    *,
    output_root: Path,
    result_dir: Path,
    transcript: Path,
    specification: Path,
    roi_inventory: Path,
    tissue_inventory: Path,
    repository: Path,
) -> None:
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Result directory already exists: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(transcript, result_dir / "source_transcript.txt")
    shutil.copy2(specification, result_dir / "ANALYSIS_REQUIREMENTS.md")

    cohort_dir = result_dir / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    alignment = _cohort_alignment(roi_inventory, tissue_inventory)
    alignment.to_csv(
        cohort_dir / "subject_alignment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    task_dir = result_dir / "planning"
    task_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        TASKS,
        columns=["task_id", "task", "status", "acceptance_or_blocker"],
    ).to_csv(task_dir / "analysis_task_matrix.csv", index=False, encoding="utf-8-sig")

    metrics_dir = result_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = _write_registered_metrics(
        metrics_dir / "registered_metrics_snapshot.csv"
    )

    intersection = alignment["in_primary_intersection"].astype(bool)
    group_counts = (
        alignment[intersection].groupby("group").size().astype(int).to_dict()
    )
    summary_data = {
        "subjects_total": int(intersection.sum()),
        "group_counts": group_counts,
        "roi_only": int((alignment["roi_present"] & ~alignment["tissue_present"]).sum()),
        "tissue_only": int(
            (alignment["tissue_present"] & ~alignment["roi_present"]).sum()
        ),
        "registered_metric_count": len(metrics),
        "primary_representations": ["GM-only", "whole-brain"],
        "primary_classification_family": "26 metrics x 2 representations",
        "blocking_items": [
            "GM voxel coordinates / exact row-to-coordinate mapping",
            "parcellation mask and spatial transform",
            "resolution of 168 ROI rows versus 167 atlas labels",
        ],
    }
    (result_dir / "planning" / "analysis_scope.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = (
        "python scripts/package_transcript_analysis_spec.py "
        f'--output-root "{output_root}" --result-dir "{result_dir}" '
        f'--transcript "{transcript}" --specification "{specification}" '
        f'--roi-inventory "{roi_inventory}" '
        f'--tissue-inventory "{tissue_inventory}" '
        f'--repository "{repository}"'
    )
    metadata = write_result_provenance(
        result_dir=result_dir,
        result_id="transcript-derived-hc-sz-analysis-spec-2026-06-23",
        title="Transcript-derived HC/SZ analysis specification",
        result_type="analysis_specification",
        status="ready_with_spatial_blockers",
        execution_mode="requirements_extraction",
        summary=(
            "Разговор преобразован в исполнимую спецификацию GM-only и "
            "whole-brain benchmark по всем 26 метрикам для 39 субъектов."
        ),
        meaning=(
            "Спецификация отделяет обязательный HC/SZ benchmark от spatial, "
            "signal-QC и preprocessing предпосылок и запрещает leakage."
        ),
        command=command,
        inputs=[
            str(transcript),
            str(roi_inventory),
            str(tissue_inventory),
        ],
        code_files=[
            specification,
            repository / "scripts" / "package_transcript_analysis_spec.py",
            repository / "scripts" / "write_result_provenance.py",
            repository / "metric_benchmark.py",
            repository / "docs" / "fmri_signal_qc_protocol.md",
        ],
        repository=repository,
        findings=[
            "ROI и tissue наборы совпадают: 39/39 субъектов, 15 HC и 24 SZ.",
            "Основной benchmark зафиксирован для GM-only и whole-brain.",
            f"В текущем registry зафиксировано {len(metrics)} метрик.",
            "Для каждого субъекта требуется out-of-fold prediction.",
            "Основная статистическая семья: 26 metrics × 2 representations.",
            "Полноценная GM-only connectivity заблокирована отсутствием xyz.",
            "Whole-brain ветка требует разрешения 168 ROI rows ↔ 167 labels.",
        ],
        limitations=[
            "Это спецификация работ, а не выполненная классификация.",
            "В разговоре есть ошибки автоматической транскрипции.",
            "GM voxel-to-parcel mapping пока невозможно воспроизвести.",
            "Физический TR и часть upstream preprocessing metadata неизвестны.",
            "При n=39 результаты останутся exploratory и потребуют внешней выборки.",
        ],
        source_time_start=datetime.fromtimestamp(transcript.stat().st_mtime)
        .astimezone()
        .isoformat(),
        source_time_end=datetime.now().astimezone().isoformat(),
        notes=[
            "Все требования проверены против текущего registry и существующего benchmark prototype.",
            "Существующий metric_benchmark.py требует переработки перед запуском.",
        ],
    )

    _update_catalog(
        output_root,
        result_id=metadata["result_id"],
        title=metadata["title"],
        result_path=result_dir.name,
        status=metadata["status"],
        execution_mode=metadata["execution_mode"],
        summary=metadata["summary"],
    )
    _write_workspace_inventory(output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--specification", required=True)
    parser.add_argument("--roi-inventory", required=True)
    parser.add_argument("--tissue-inventory", required=True)
    parser.add_argument("--repository", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    package_spec(
        output_root=Path(args.output_root),
        result_dir=Path(args.result_dir),
        transcript=Path(args.transcript),
        specification=Path(args.specification),
        roi_inventory=Path(args.roi_inventory),
        tissue_inventory=Path(args.tissue_inventory),
        repository=Path(args.repository),
    )


if __name__ == "__main__":
    main()
