"""Конфиги и общие константы для NewEDS."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

# Дефолтные параметры анализа.
DEFAULT_MAX_LAG = 5
DEFAULT_K_MI = 5
DEFAULT_BINS = 8
DEFAULT_OUTLIER_Z = 5
DEFAULT_REGULARIZATION = 1e-8
DEFAULT_EMBED_DIM = 3
DEFAULT_EMBED_TAU = 1
DEFAULT_PVALUE_ALPHA = 0.05
DEFAULT_EDGE_THRESHOLD = 0.2

# Регуляризация для partial-correlation методов.
REG_ALPHA = 1e-5

# Признак доступности опционального pyinform — даёт более точный TE.
# Если нет, ядро откатывается на numpy-fallback.
PYINFORM_AVAILABLE = importlib.util.find_spec("pyinform") is not None


@dataclass(slots=True)
class ComputationContract:
    """Контракт одного вычисления: что считали и как.

    Привязывается к каждому результату, чтобы потом можно было воспроизвести
    цифры или объяснить, откуда они взялись.
    """

    variant: str = ""
    input_channels: int = 0
    input_T: int = 0
    input_missing_frac: float = 0.0

    preprocess_steps: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    control_strategy: str = "none"

    directed: bool = False
    directed_lag: int = 1
    lag_selection: str = "fixed"

    validity_warnings: list[str] = field(default_factory=list)

    output_shape: tuple = (0, 0)
    output_type: str = "matrix_NxN"

    seed: int | None = None
    config_hash: str = ""

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "input": {
                "channels": self.input_channels,
                "T": self.input_T,
                "missing_frac": self.input_missing_frac,
            },
            "preprocess": list(self.preprocess_steps),
            "controls": {
                "strategy": self.control_strategy,
                "variables": list(self.controls),
            },
            "directed": {
                "is_directed": self.directed,
                "lag": self.directed_lag,
                "lag_selection": self.lag_selection,
            },
            "validity": list(self.validity_warnings),
            "output": {
                "shape": list(self.output_shape),
                "type": self.output_type,
            },
            "repro": {"seed": self.seed, "config_hash": self.config_hash},
        }

    def summary_text(self) -> str:
        lines = [
            f"Метрика: {self.variant}",
            f"Вход: {self.input_channels} каналов × {self.input_T} точек "
            f"(пропуски: {self.input_missing_frac:.1%})",
            "Препроцессинг: "
            + (", ".join(self.preprocess_steps) if self.preprocess_steps else "нет"),
            f"Контроль: {self.control_strategy}"
            + (f" ({', '.join(self.controls)})" if self.controls else ""),
        ]
        if self.directed:
            lines.append(f"Направление: lag={self.directed_lag}, выбор={self.lag_selection}")
        if self.validity_warnings:
            lines.append("Предупреждения: " + "; ".join(self.validity_warnings))
        lines.append(f"Выход: {self.output_type} {self.output_shape}")
        return "\n".join(lines)


@dataclass(slots=True)
class AnalysisConfig:
    """Конфиг публичного пайплайна анализа."""

    max_lag: int = DEFAULT_MAX_LAG
    p_value_alpha: float = DEFAULT_PVALUE_ALPHA
    graph_threshold: float = DEFAULT_EDGE_THRESHOLD
    enable_experimental: bool = False
    auto_difference: bool = False
    pvalue_correction: str = "none"

    window_sizes: list[int] | None = None
    window_stride: int | None = None
    window_policy: str = "best"

    lag_selection: str = "fixed"
    master_seed: int = 12345

    # Пространственная агрегация каналов после загрузки (time × channels → time × bins).
    spatial_bin_size: int = 1
    spatial_bin_method: str = "mean"

    # Агрегация 4D fMRI (X, Y, Z, T) на этапе HDF5-загрузчика.
    spatial_grid_size: int = 10
    spatial_grid_method: str = "mean"
    lazy_spatial_bin: bool = False
    time_chunk: int = 50

    # Явный список метрик. Если None — пайплайн возьмёт стабильный дефолт.
    variants: list[str] | None = None

    # Контрольные колонки для *_partial метрик.
    controls: list[str] | None = None


# Реэкспорт: исторически из neweds.config импортировали и таблицы методов,
# чтобы не ломать старый код, отдаём их и отсюда.
from neweds.methods import (  # noqa: E402  (намеренный реэкспорт в конце модуля)
    DIRECTED_METHODS,
    EXPERIMENTAL_METHODS,
    EXPERIMENTAL_METHODS_BASE,
    METHOD_INFO,
    PVAL_METHODS,
    STABLE_METHODS,
    is_control_sensitive_method,
    is_directed_method,
    is_pvalue_method,
)

__all__ = [
    "AnalysisConfig",
    "ComputationContract",
    "DEFAULT_BINS",
    "DEFAULT_EDGE_THRESHOLD",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_EMBED_TAU",
    "DEFAULT_K_MI",
    "DEFAULT_MAX_LAG",
    "DEFAULT_OUTLIER_Z",
    "DEFAULT_PVALUE_ALPHA",
    "DEFAULT_REGULARIZATION",
    "DIRECTED_METHODS",
    "EXPERIMENTAL_METHODS",
    "EXPERIMENTAL_METHODS_BASE",
    "METHOD_INFO",
    "PVAL_METHODS",
    "PYINFORM_AVAILABLE",
    "REG_ALPHA",
    "STABLE_METHODS",
    "is_control_sensitive_method",
    "is_directed_method",
    "is_pvalue_method",
]
