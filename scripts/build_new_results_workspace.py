"""Build the canonical new_results workspace with per-result provenance."""

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

from neweds.core.fmri_tissue_audit import run_fmri_tissue_audit  # noqa: E402
from scripts.prepare_separate_fmri_audits import _copy_whole_brain_audit  # noqa: E402
from scripts.write_result_provenance import write_result_provenance  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _csv_shape(path: Path) -> tuple[int, int]:
    frame = pd.read_csv(path, header=None)
    return int(frame.shape[0]), int(frame.shape[1])


def _current_roi_input_inventory(
    hc_dir: Path,
    sz_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, directory in (("HC", hc_dir), ("SZ", sz_dir)):
        for path in sorted(directory.glob("*AAL3_timeseries.csv")):
            n_rows, n_cols = _csv_shape(path)
            rows.append(
                {
                    "group": group,
                    "subject_id": path.name.split("_")[0],
                    "file_name": path.name,
                    "file_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_local": datetime.fromtimestamp(path.stat().st_mtime)
                    .astimezone()
                    .isoformat(),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "sha256": _sha256(path),
                }
            )
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table


def _source_artifact_times(source: Path) -> tuple[str | None, str | None]:
    allowed_dirs = (
        "inventories",
        "qc",
        "distributions",
        "temporal",
        "spectral",
        "reports",
        "decisions",
    )
    files: list[Path] = []
    for name in allowed_dirs:
        directory = source / name
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    feature_file = source / "roi_signal_characterization_all_features.csv"
    if feature_file.is_file():
        files.append(feature_file)
    if not files:
        return None, None
    earliest = min(path.stat().st_mtime for path in files)
    latest = max(path.stat().st_mtime for path in files)
    return (
        datetime.fromtimestamp(earliest).astimezone().isoformat(),
        datetime.fromtimestamp(latest).astimezone().isoformat(),
    )


def _write_compatibility_note(
    result_dir: Path,
    historical_inventory: Path,
    current_inventory: pd.DataFrame,
) -> dict[str, Any]:
    historical = pd.read_csv(historical_inventory)
    historical_shapes = (
        historical.groupby(["n_rows", "n_cols", "status"]).size().reset_index(name="files")
    )
    current_shapes = (
        current_inventory.groupby(["n_rows", "n_cols"]).size().reset_index(name="files")
    )
    historical_shape_text = ", ".join(
        f"{int(row.n_rows)}×{int(row.n_cols)} ({int(row.files)} files)"
        for row in historical_shapes.itertuples()
    )
    current_shape_text = ", ".join(
        f"{int(row.n_rows)}×{int(row.n_cols)} ({int(row.files)} files)"
        for row in current_shapes.itertuples()
    )
    changed = historical_shape_text != current_shape_text
    lines = [
        "# Input Snapshot Compatibility",
        "",
        "## Historical audit input recorded in the saved inventory",
        "",
        f"- {historical_shape_text}",
        "",
        "## Current files now present in Group_HC / Group_SZ",
        "",
        f"- {current_shape_text}",
        "",
        "## Verdict",
        "",
        (
            "**Not the same input snapshot.** The historical audit must not be "
            "presented as a rerun of the current files."
            if changed
            else "The recorded and current input shapes agree."
        ),
        "",
        "The current inventory with paths, timestamps, sizes, and SHA-256 is saved as",
        "`input_state/current_aal3_input_inventory.csv`.",
    ]
    note_path = result_dir / "input_state" / "INPUT_COMPATIBILITY.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "historical_shapes": historical_shape_text,
        "current_shapes": current_shape_text,
        "same_snapshot_by_shape": not changed,
        "note": str(note_path.relative_to(result_dir)),
    }


def _copy_reference_materials(
    output_root: Path,
    transcript: Path,
    skill_dir: Path,
) -> dict[str, Any]:
    references = output_root / "references"
    references.mkdir(parents=True, exist_ok=True)
    transcript_target = references / "meeting_transcript_fmri_methodology.txt"
    shutil.copy2(transcript, transcript_target)
    skill_target = references / "research-result-provenance-skill"
    shutil.copytree(skill_dir, skill_target, dirs_exist_ok=True)
    return {
        "transcript": {
            "source": str(transcript),
            "path": str(transcript_target.relative_to(output_root)),
            "sha256": _sha256(transcript_target),
        },
        "skill": {
            "source": str(skill_dir),
            "path": str(skill_target.relative_to(output_root)),
        },
    }


def _write_root_files(
    output_root: Path,
    results: list[dict[str, Any]],
    references: dict[str, Any],
) -> None:
    catalog = {
        "schema": "new-results-catalog/v1",
        "workspace": str(output_root),
        "created_at": datetime.now().astimezone().isoformat(),
        "canonical": True,
        "result_count": len(results),
        "results": results,
        "references": references,
        "required_per_result": [
            "ABOUT_THIS_RESULT.md",
            "run_metadata.json",
            "output_inventory.csv",
            "code_snapshot/",
        ],
    }
    (output_root / "results_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# NewEDS new_results",
        "",
        "Это канонический каталог исследовательских результатов.",
        "Каждый результат хранится отдельно и содержит собственную карточку",
        "происхождения, версию кода, dirty diff, среду, входы, смысл и ограничения.",
        "",
        "## Результаты",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### [{result['title']}]({result['path']}/ABOUT_THIS_RESULT.md)",
                "",
                f"- ID: `{result['result_id']}`",
                f"- Статус: `{result['status']}`",
                f"- Режим: `{result['execution_mode']}`",
                f"- Кратко: {result['summary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Общие материалы",
            "",
            "- `references/meeting_transcript_fmri_methodology.txt` — исходный текст встречи.",
            "- `references/research-result-provenance-skill/` — снимок Codex skill.",
            "- `results_catalog.json` — машинный каталог.",
            "",
            "## Правило",
            "",
            "Новые запуски не перезаписывают существующие результаты. Для них создаётся",
            "новая папка с датой и собственной provenance-карточкой.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_workspace(
    *,
    output_root: Path,
    whole_brain_source: Path,
    hc_dir: Path,
    sz_dir: Path,
    tissue_source: Path,
    transcript: Path,
    skill_dir: Path,
    repository: Path,
    max_lag: int,
    block_rows: int,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Canonical output directory already exists and is not empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    whole_dir = output_root / "2026-06-04_whole-brain-roi-audit_legacy"
    tissue_dir = output_root / "2026-06-23_tissue-gm-wm-csf-audit"

    _copy_whole_brain_audit(whole_brain_source, whole_dir)
    current_input = _current_roi_input_inventory(
        hc_dir,
        sz_dir,
        whole_dir / "input_state" / "current_aal3_input_inventory.csv",
    )
    compatibility = _write_compatibility_note(
        whole_dir,
        whole_dir / "inventories" / "data_inventory.csv",
        current_input,
    )
    whole_start, whole_end = _source_artifact_times(whole_brain_source)
    whole_command = (
        "Original execution command: not recorded. "
        "Packaging command: python scripts/build_new_results_workspace.py "
        f'--whole-brain-source "{whole_brain_source}" '
        f'--hc-dir "{hc_dir}" --sz-dir "{sz_dir}" '
        f'--output-root "{output_root}"'
    )
    whole_meta = write_result_provenance(
        result_dir=whole_dir,
        result_id="whole-brain-roi-audit-2026-06-04-legacy",
        title="Whole-brain / AAL3 ROI data audit — historical snapshot",
        result_type="whole_brain_roi_data_audit",
        status="legacy_snapshot_with_input_mismatch",
        execution_mode="legacy_snapshot_packaged",
        summary=(
            "Упакован Stage 1 аудит 39 ROI-наборов, выполненный 4 июня 2026 года "
            "на снимке входов 167×600."
        ),
        meaning=(
            "Результат сохраняет исторический QC ROI, temporal и spectral summaries, "
            "но не является пересчётом текущих файлов."
        ),
        command=whole_command,
        inputs=[str(hc_dir), str(sz_dir), str(whole_brain_source)],
        code_files=[
            repository / "scripts" / "characterize_fmri_roi_data.py",
            repository / "scripts" / "build_fmri_stage15_decisions.py",
            repository / "neweds" / "core" / "fmri_roi_audit.py",
            repository / "scripts" / "prepare_separate_fmri_audits.py",
            repository / "scripts" / "build_new_results_workspace.py",
            repository / "scripts" / "write_result_provenance.py",
        ],
        repository=repository,
        findings=[
            "Сохранены inventory, ROI QC, distributions, temporal, spectral и decision artifacts.",
            "Исторический inventory содержит 39 файлов формы 167×600.",
            "Текущие файлы на диске имеют форму 168×600 и отдельные SHA-256.",
            "Stage 2 connectivity grids в этот аудит не включены.",
        ],
        limitations=[
            "Точная исходная команда и Git revision запуска 4 июня не были записаны.",
            "Исторический snapshot не соответствует текущим AAL3 CSV по числу строк.",
            "Сопоставление 168 строк с 167 atlas labels остаётся нерешённым.",
            "Из ROI means нельзя проверить voxel-wise однородность или atlas overlay.",
        ],
        source_time_start=whole_start,
        source_time_end=whole_end,
        notes=[
            json.dumps(compatibility, ensure_ascii=False),
            "Текущий Git snapshot относится к упаковке результата, а не доказывает исходную версию запуска.",
        ],
    )

    tissue_started = datetime.now().astimezone()
    tissue_command = (
        "python -m neweds.cli_fmri_tissue_audit "
        f'--input-dir "{tissue_source}" --output-dir "{tissue_dir}" '
        f"--max-lag {max_lag} --block-rows {block_rows}"
    )
    tissue_result = run_fmri_tissue_audit(
        tissue_source,
        tissue_dir,
        max_lag=max_lag,
        block_rows=block_rows,
    )
    tissue_finished = datetime.now().astimezone()
    tissue_meta = write_result_provenance(
        result_dir=tissue_dir,
        result_id="tissue-gm-wm-csf-audit-2026-06-23",
        title="GM / WM / CSF tissue HDF5 audit",
        result_type="tissue_gm_wm_csf_data_audit",
        status="completed_exploratory_audit",
        execution_mode="fresh_run",
        summary=(
            "Потоково проверены 39 HDF5: 15 HC и 24 SZ, отдельно по GM, WM и CSF."
        ),
        meaning=(
            "Аудит описывает качество tissue masks и временную структуру тканевых "
            "сигналов и подготавливает WM/CSF/GLOBAL sensitivity branches."
        ),
        command=tissue_command,
        inputs=[str(tissue_source)],
        code_files=[
            repository / "neweds" / "core" / "fmri_tissue_audit.py",
            repository / "neweds" / "cli_fmri_tissue_audit.py",
            repository / "scripts" / "build_new_results_workspace.py",
            repository / "scripts" / "write_result_provenance.py",
            repository / "docs" / "fmri_tissue_audit.md",
            repository / "docs" / "fmri_audit_separation_contract.md",
        ],
        repository=repository,
        findings=[
            f"Валидно обработано файлов: {tissue_result.files_valid}; ошибок: {tissue_result.failures}.",
            "Сохранены tissue voxel counts, zero/constant QC и active tissue means.",
            "Сохранены ACF/PACF до и после AR1, GM/WM/CSF/GLOBAL time series.",
            "HC/SZ tissue-feature tests скорректированы Benjamini-Hochberg FDR.",
            "GM/xyz, WM/xyz и CSF/xyz отсутствуют во всех 39 файлах.",
        ],
        limitations=[
            "Tissue voxel counts являются QC/confound-кандидатами, а не доказанной морфометрией.",
            "Без xyz нельзя выполнять voxel-to-atlas mapping и spatial homogeneity.",
            "Атрибут TR=600 неоднозначен; физический TR в секундах неизвестен.",
            "Нет motion/FD, age, sex, site/scanner и независимой выборки.",
            "Групповые тесты exploratory и не являются диагностическим результатом.",
        ],
        source_time_start=tissue_started.isoformat(),
        source_time_end=tissue_finished.isoformat(),
        notes=[
            "Вычисление выполнено непосредственно в каноническую папку new_results.",
            "Результат не содержит ROI connectivity и хранится отдельно от whole-brain аудита.",
        ],
    )

    references = _copy_reference_materials(output_root, transcript, skill_dir)
    results = [
        {
            "result_id": whole_meta["result_id"],
            "title": whole_meta["title"],
            "path": whole_dir.name,
            "status": whole_meta["status"],
            "execution_mode": whole_meta["execution_mode"],
            "summary": whole_meta["summary"],
        },
        {
            "result_id": tissue_meta["result_id"],
            "title": tissue_meta["title"],
            "path": tissue_dir.name,
            "status": tissue_meta["status"],
            "execution_mode": tissue_meta["execution_mode"],
            "summary": tissue_meta["summary"],
        },
    ]
    _write_root_files(output_root, results, references)

    with (output_root / "workspace_file_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path.name != "workspace_file_inventory.csv":
                writer.writerow(
                    {
                        "path": path.relative_to(output_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--whole-brain-source", required=True)
    parser.add_argument("--hc-dir", required=True)
    parser.add_argument("--sz-dir", required=True)
    parser.add_argument("--tissue-source", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--block-rows", type=int, default=8192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_workspace(
        output_root=Path(args.output_root),
        whole_brain_source=Path(args.whole_brain_source),
        hc_dir=Path(args.hc_dir),
        sz_dir=Path(args.sz_dir),
        tissue_source=Path(args.tissue_source),
        transcript=Path(args.transcript),
        skill_dir=Path(args.skill_dir),
        repository=Path(args.repository),
        max_lag=int(args.max_lag),
        block_rows=int(args.block_rows),
    )


if __name__ == "__main__":
    main()
