"""Каноническое пространство вокселей для выравнивания субъектов."""

from __future__ import annotations

import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VoxelStrategy(str, Enum):
    """Стратегии построения канонического пространства."""

    INTERSECTION = "intersection"
    UNION = "union"
    TEMPLATE = "template"


class CanonicalVoxelSpace:
    """Фиксированное пространство вокселей для согласования колонок между субъектами."""

    def __init__(
        self,
        voxel_ids: list[str],
        strategy: VoxelStrategy,
        fill_value: float = float("nan"),
        source_info: dict | None = None,
    ):
        if not voxel_ids:
            raise ValueError("CanonicalVoxelSpace: список вокселей пуст.")

        self.voxel_ids: list[str] = sorted(voxel_ids, key=_voxel_id_sort_key)
        self._id_set: frozenset[str] = frozenset(self.voxel_ids)
        self.strategy = VoxelStrategy(strategy)
        self.fill_value = float(fill_value)
        self.source_info: dict = source_info or {}
        self.n_voxels = len(self.voxel_ids)

    @classmethod
    def from_dataframes(
        cls,
        dfs: Iterable[pd.DataFrame],
        strategy: str | VoxelStrategy = VoxelStrategy.INTERSECTION,
        fill_value: float = float("nan"),
    ) -> "CanonicalVoxelSpace":
        """Строит пространство по набору DataFrame time×voxel_id."""
        strategy = VoxelStrategy(strategy)
        voxel_sets: list[set[str]] = []
        n_subjects = 0

        for df in dfs:
            voxel_sets.append(set(df.columns))
            n_subjects += 1
            logger.info("subject %d: %d вокселей", n_subjects, len(voxel_sets[-1]))

        if not voxel_sets:
            raise ValueError("from_dataframes: не передано ни одного DataFrame.")

        if strategy == VoxelStrategy.INTERSECTION:
            canonical = voxel_sets[0]
            for s in voxel_sets[1:]:
                canonical &= s
        elif strategy == VoxelStrategy.UNION:
            canonical = set()
            for s in voxel_sets:
                canonical |= s
        else:
            raise ValueError("from_dataframes поддерживает только intersection/union.")

        if not canonical:
            raise RuntimeError("Пустое canonical space: intersection/union дал 0 колонок.")

        return cls(
            voxel_ids=list(canonical),
            strategy=strategy,
            fill_value=fill_value,
            source_info={"n_subjects_used": n_subjects},
        )

    @classmethod
    def from_voxel_ids(
        cls,
        voxel_ids: list[str],
        fill_value: float = float("nan"),
    ) -> "CanonicalVoxelSpace":
        """Создаёт пространство из внешнего шаблона voxel_id."""
        return cls(
            voxel_ids=voxel_ids,
            strategy=VoxelStrategy.TEMPLATE,
            fill_value=fill_value,
            source_info={"source": "external_template"},
        )

    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит DataFrame субъекта к каноническому набору колонок."""
        present = [v for v in self.voxel_ids if v in df.columns]
        missing = [v for v in self.voxel_ids if v not in df.columns]
        extra = [v for v in df.columns if v not in self._id_set]

        coverage = len(present) / self.n_voxels
        if coverage < 0.5:
            logger.warning(
                "align: покрытие canonical space %.1f%% (%d/%d).",
                coverage * 100,
                len(present),
                self.n_voxels,
            )
        if extra:
            logger.debug("align: %d колонок вне canonical space отброшены.", len(extra))

        result = pd.DataFrame(index=df.index, columns=self.voxel_ids, dtype=np.float32)
        result[:] = self.fill_value
        if present:
            result[present] = df[present].values.astype(np.float32)

        result.attrs = {
            "canonical_n_voxels": self.n_voxels,
            "canonical_n_present": len(present),
            "canonical_n_missing": len(missing),
            "canonical_coverage": coverage,
            "canonical_strategy": self.strategy.value,
            "format": "voxel_wide",
        }
        return result

    def coverage_report(self, df: pd.DataFrame) -> dict:
        """Возвращает метрики покрытия canonical space для субъекта."""
        present = sum(1 for v in self.voxel_ids if v in df.columns)
        return {
            "n_canonical": self.n_voxels,
            "n_present": present,
            "n_missing": self.n_voxels - present,
            "coverage": present / self.n_voxels,
            "n_extra": sum(1 for v in df.columns if v not in self._id_set),
        }

    def save(self, path: str | Path) -> None:
        """Сохраняет canonical space в JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strategy": self.strategy.value,
            "n_voxels": self.n_voxels,
            "fill_value": self.fill_value if np.isfinite(self.fill_value) else None,
            "voxel_ids": self.voxel_ids,
            "source_info": self.source_info,
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "created_at": pd.Timestamp.now().isoformat(),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalVoxelSpace":
        """Загружает canonical space из JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        fv = data.get("fill_value")
        fill_value = float("nan") if fv is None else float(fv)
        return cls(
            voxel_ids=data["voxel_ids"],
            strategy=VoxelStrategy(data["strategy"]),
            fill_value=fill_value,
            source_info=data.get("source_info", {}),
        )

    def __repr__(self) -> str:
        return (
            f"CanonicalVoxelSpace(n_voxels={self.n_voxels}, "
            f"strategy={self.strategy.value}, fill_value={self.fill_value})"
        )


def _voxel_id_sort_key(voxel_id: str) -> tuple[int, int, int]:
    """Ключ сортировки для voxel_id формата x{X}_y{Y}_z{Z}."""
    try:
        parts = voxel_id.split("_")
        return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])
    except Exception:
        return 0, 0, 0


def align_subjects(
    dfs: dict[str, pd.DataFrame],
    space: CanonicalVoxelSpace,
) -> dict[str, pd.DataFrame]:
    """Выравнивает словарь субъектов к canonical space."""
    result: dict[str, pd.DataFrame] = {}
    for sid, df in dfs.items():
        aligned = space.align(df)
        report = space.coverage_report(df)
        logger.info(
            "%s: coverage=%.1f%% (%d/%d, missing=%d)",
            sid,
            report["coverage"] * 100,
            report["n_present"],
            report["n_canonical"],
            report["n_missing"],
        )
        result[sid] = aligned
    return result
