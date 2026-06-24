"""Write human and machine provenance files for one research result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROVENANCE_NAMES = {"ABOUT_THIS_RESULT.md", "run_metadata.json", "output_inventory.csv"}


def _command(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return 127, ""
    return result.returncode, result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def _git_snapshot(result_dir: Path, repository: Path | None) -> dict[str, Any]:
    empty = {
        "available": False,
        "repository": str(repository) if repository else None,
        "head": None,
        "branch": None,
        "describe": None,
        "dirty": None,
    }
    if repository is None:
        return empty
    code, root_text = _command(["git", "rev-parse", "--show-toplevel"], repository)
    if code != 0:
        return empty
    root = Path(root_text)
    _, head = _command(["git", "rev-parse", "HEAD"], root)
    _, branch = _command(["git", "branch", "--show-current"], root)
    _, describe = _command(["git", "describe", "--always", "--dirty", "--tags"], root)
    _, status = _command(["git", "status", "--short"], root)
    _, diff = _command(["git", "diff", "--binary", "--no-ext-diff"], root)
    _, log_text = _command(
        ["git", "log", "-10", "--date=iso-strict", "--pretty=fuller"],
        root,
    )
    snapshot = result_dir / "code_snapshot"
    _write_text(snapshot / "git_status.txt", status)
    _write_text(snapshot / "git_diff.patch", diff)
    _write_text(snapshot / "git_log_last_10.txt", log_text)
    return {
        "available": True,
        "repository": str(root),
        "head": head,
        "branch": branch,
        "describe": describe,
        "dirty": bool(status),
        "status": status,
        "diff_sha256": _sha256(snapshot / "git_diff.patch"),
    }


def _copy_code(
    result_dir: Path,
    code_files: list[Path],
    repository: Path | None,
) -> list[dict[str, Any]]:
    destination = result_dir / "code_snapshot" / "files"
    destination.mkdir(parents=True, exist_ok=True)
    repo = repository.resolve() if repository else None
    records: list[dict[str, Any]] = []
    for source in code_files:
        source = source.resolve()
        if not source.is_file():
            records.append({"source": str(source), "copied": False, "error": "missing"})
            continue
        try:
            relative = source.relative_to(repo) if repo else Path(source.name)
        except ValueError:
            relative = Path(source.name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source),
                "snapshot_path": target.relative_to(result_dir).as_posix(),
                "copied": True,
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    return records


def _environment(result_dir: Path) -> dict[str, Any]:
    packages = sorted(
        {
            f"{dist.metadata.get('Name', 'unknown')}=={dist.version}"
            for dist in importlib.metadata.distributions()
        },
        key=str.casefold,
    )
    _write_text(result_dir / "code_snapshot" / "python_packages.txt", "\n".join(packages))
    data = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "working_directory": os.getcwd(),
        "package_count": len(packages),
    }
    _write_text(
        result_dir / "code_snapshot" / "environment.json",
        json.dumps(data, ensure_ascii=False, indent=2),
    )
    return data


def _inventory_rows(
    result_dir: Path,
    *,
    excluded_names: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(result_dir).as_posix()
        if relative in excluded_names:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "modified_local": datetime.fromtimestamp(path.stat().st_mtime)
                .astimezone()
                .isoformat(),
                "sha256": _sha256(path),
            }
        )
    return rows


def _write_inventory(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (result_dir / "output_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "size_bytes", "modified_local", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _bullets(items: list[str], fallback: str = "не указано") -> list[str]:
    return [f"- {item}" for item in items] if items else [f"- {fallback}"]


def write_result_provenance(
    *,
    result_dir: Path,
    result_id: str,
    title: str,
    result_type: str,
    status: str,
    execution_mode: str,
    summary: str,
    meaning: str,
    command: str,
    inputs: list[str],
    code_files: list[Path],
    repository: Path | None,
    findings: list[str],
    limitations: list[str],
    source_time_start: str | None = None,
    source_time_end: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    git = _git_snapshot(result_dir, repository)
    code = _copy_code(result_dir, code_files, repository)
    environment = _environment(result_dir)
    artifact_inventory = _inventory_rows(
        result_dir,
        excluded_names=PROVENANCE_NAMES,
    )
    integrity_manifest_file_count = len(artifact_inventory) + 2
    metadata = {
        "schema": "research-result-provenance/v1",
        "result_id": result_id,
        "title": title,
        "result_type": result_type,
        "status": status,
        "execution_mode": execution_mode,
        "recorded_at_local": now.isoformat(),
        "recorded_at_utc": now.astimezone(UTC).isoformat(),
        "source_result_time_start": source_time_start,
        "source_result_time_end": source_time_end,
        "summary": summary,
        "meaning": meaning,
        "command": command,
        "inputs": inputs,
        "findings": findings,
        "limitations": limitations,
        "notes": notes or [],
        "git": git,
        "environment": environment,
        "code_files": code,
        "output_file_count_excluding_provenance": len(artifact_inventory),
        "output_total_bytes_excluding_provenance": sum(
            int(row["size_bytes"]) for row in artifact_inventory
        ),
        "integrity_manifest_file_count": integrity_manifest_file_count,
    }
    _write_text(
        result_dir / "run_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    version = (
        f"`{git.get('describe')}` / `{git.get('head')}`"
        if git.get("available")
        else "Git metadata unavailable"
    )
    lines = [
        f"# {title}",
        "",
        "## Коротко",
        "",
        summary,
        "",
        "## В чём смысл",
        "",
        meaning,
        "",
        "## Статус и время",
        "",
        f"- ID: `{result_id}`",
        f"- Тип: `{result_type}`",
        f"- Статус: `{status}`",
        f"- Режим получения: `{execution_mode}`",
        f"- Карточка записана: `{now.isoformat()}`",
        f"- Исходный результат: `{source_time_start or 'неизвестно'}` — "
        f"`{source_time_end or 'неизвестно'}`",
        "",
        "## Что сделано",
        "",
        *_bullets(findings),
        "",
        "## Входные данные",
        "",
        *_bullets([f"`{item}`" for item in inputs]),
        "",
        "## Как запускалось",
        "",
        "```text",
        command or "точная исходная команда не была сохранена",
        "```",
        "",
        "## Каким кодом",
        "",
        f"- Git version: {version}",
        f"- Ветка: `{git.get('branch')}`",
        f"- Dirty worktree: `{git.get('dirty')}`",
        "- Dirty diff: `code_snapshot/git_diff.patch`",
        "- Git status: `code_snapshot/git_status.txt`",
        "- Копии исходников: `code_snapshot/files/`",
        "- SHA-256 исходников: `run_metadata.json`",
        "",
        "## Среда",
        "",
        f"- Python: `{environment['python_version']}`",
        f"- Python executable: `{environment['python_executable']}`",
        f"- ОС: `{environment['platform']}`",
        "- Зависимости: `code_snapshot/python_packages.txt`",
        "",
        "## Ограничения",
        "",
        *_bullets(limitations),
        "",
        "## Примечания",
        "",
        *_bullets(notes or []),
        "",
        "## Проверка целостности",
        "",
        "- Машинное описание: `run_metadata.json`",
        "- Inventory и SHA-256 всех файлов, кроме самого inventory: `output_inventory.csv`",
        f"- Учтено файлов: `{integrity_manifest_file_count}`",
    ]
    _write_text(result_dir / "ABOUT_THIS_RESULT.md", "\n".join(lines))
    integrity_inventory = _inventory_rows(
        result_dir,
        excluded_names={"output_inventory.csv"},
    )
    if len(integrity_inventory) != integrity_manifest_file_count:
        raise RuntimeError(
            "Result contents changed while the integrity inventory was being written"
        )
    _write_inventory(result_dir, integrity_inventory)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "result-dir",
        "result-id",
        "title",
        "result-type",
        "status",
        "execution-mode",
        "summary",
        "meaning",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--command", default="")
    parser.add_argument("--repository")
    parser.add_argument("--source-time-start")
    parser.add_argument("--source-time-end")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--code-file", action="append", default=[])
    parser.add_argument("--finding", action="append", default=[])
    parser.add_argument("--limitation", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_result_provenance(
        result_dir=Path(args.result_dir),
        result_id=args.result_id,
        title=args.title,
        result_type=args.result_type,
        status=args.status,
        execution_mode=args.execution_mode,
        summary=args.summary,
        meaning=args.meaning,
        command=args.command,
        inputs=args.input,
        code_files=[Path(path) for path in args.code_file],
        repository=Path(args.repository) if args.repository else None,
        findings=args.finding,
        limitations=args.limitation,
        source_time_start=args.source_time_start,
        source_time_end=args.source_time_end,
        notes=args.note,
    )


if __name__ == "__main__":
    main()
