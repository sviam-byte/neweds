"""Запуск сценариев валидации и проверка ожиданий.

Основная точка входа: ``run_validation(scenarios, metrics, ...)``.
Возвращает структурированный отчёт, пригодный для Streamlit-отображения.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..metrics.registry import METRICS_REGISTRY, get_metric_func

from .scenarios import (
    ALL_SCENARIOS,
    DEFAULT_SCENARIOS,
    QUICK_SCENARIOS,
    Expectation,
    Scenario,
)


@dataclass
class CheckResult:
    """Результат одной проверки."""

    scenario: str
    metric: str
    pair: tuple[int, int]
    check: str
    expected: str
    actual_value: float
    passed: bool
    message: str = ""


@dataclass
class MetricResult:
    """Результат расчёта одной метрики на одном сценарии."""

    scenario: str
    metric: str
    matrix: Optional[np.ndarray]
    elapsed_sec: float
    error: Optional[str] = None


@dataclass
class ValidationReport:
    """Полный отчёт валидации."""

    metric_results: list[MetricResult] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    elapsed_total_sec: float = 0.0

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def n_total(self) -> int:
        return len(self.checks)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(1, self.n_total)

    def summary_df(self) -> pd.DataFrame:
        """Сводная таблица по всем проверкам."""

        rows = []
        for c in self.checks:
            rows.append(
                {
                    "scenario": c.scenario,
                    "metric": c.metric,
                    "pair": f"{c.pair[0]}→{c.pair[1]}",
                    "check": c.check,
                    "value": c.actual_value,
                    "passed": "PASS" if c.passed else "FAIL",
                    "message": c.message,
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def failures_df(self) -> pd.DataFrame:
        """Только провалившиеся проверки."""

        df = self.summary_df()
        return df[df["passed"] == "FAIL"] if not df.empty else df

    def by_scenario(self) -> dict[str, dict]:
        """Группировка: scenario → {passed, failed, total, checks}."""

        out: dict[str, dict] = {}
        for c in self.checks:
            s = out.setdefault(c.scenario, {"passed": 0, "failed": 0, "total": 0, "checks": []})
            s["total"] += 1
            if c.passed:
                s["passed"] += 1
            else:
                s["failed"] += 1
            s["checks"].append(c)
        return out

    def by_metric(self) -> dict[str, dict]:
        """Группировка: metric → {passed, failed, total}."""

        out: dict[str, dict] = {}
        for c in self.checks:
            m = out.setdefault(c.metric, {"passed": 0, "failed": 0, "total": 0})
            m["total"] += 1
            if c.passed:
                m["passed"] += 1
            else:
                m["failed"] += 1
        return out


def _check_expectation(
    exp: Expectation,
    mat: np.ndarray,
    scenario_name: str,
) -> CheckResult:
    """Проверяет одно ожидание на рассчитанной матрице."""

    i, j = exp.pair
    n = mat.shape[0] if mat is not None and mat.ndim == 2 else 0

    if mat is None or i >= n or j >= n:
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected=exp.description,
            actual_value=float("nan"),
            passed=False,
            message="Матрица None или индекс вне диапазона",
        )

    val = float(mat[i, j])

    if exp.check == "near_zero":
        ok = np.isfinite(val) and abs(val) <= exp.threshold + exp.tolerance
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected=f"|value| < {exp.threshold + exp.tolerance:.3f}",
            actual_value=val,
            passed=ok,
            message="" if ok else f"|{val:.4f}| > {exp.threshold + exp.tolerance:.3f}",
        )

    if exp.check == "significantly_positive":
        ok = np.isfinite(val) and abs(val) > exp.threshold
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected=f"|value| > {exp.threshold:.3f}",
            actual_value=val,
            passed=ok,
            message="" if ok else f"|{val:.4f}| <= {exp.threshold:.3f}",
        )

    if exp.check == "pvalue_high":
        ok = np.isfinite(val) and val > exp.threshold
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected=f"p > {exp.threshold:.3f}",
            actual_value=val,
            passed=ok,
            message="" if ok else f"p = {val:.4f} <= {exp.threshold:.3f}",
        )

    if exp.check == "pvalue_low":
        ok = np.isfinite(val) and val < exp.threshold
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected=f"p < {exp.threshold:.3f}",
            actual_value=val,
            passed=ok,
            message="" if ok else f"p = {val:.4f} >= {exp.threshold:.3f}",
        )

    if exp.check == "greater_than" and exp.compare_pair is not None:
        ci, cj = exp.compare_pair
        val2 = float(mat[ci, cj]) if ci < n and cj < n else float("nan")
        ok = np.isfinite(val) and np.isfinite(val2) and abs(val) > abs(val2)
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected=f"|M[{i},{j}]| > |M[{ci},{cj}]|",
            actual_value=val,
            passed=ok,
            message="" if ok else f"|{val:.4f}| <= |{val2:.4f}| (compare [{ci},{cj}])",
        )

    if exp.check == "finite":
        ok = np.isfinite(val)
        return CheckResult(
            scenario=scenario_name,
            metric=exp.metric_group,
            pair=exp.pair,
            check=exp.check,
            expected="finite value",
            actual_value=val,
            passed=ok,
            message="" if ok else f"Value is {val}",
        )

    return CheckResult(
        scenario=scenario_name,
        metric=exp.metric_group,
        pair=exp.pair,
        check=exp.check,
        expected=exp.description,
        actual_value=val,
        passed=False,
        message=f"Unknown check type: {exp.check}",
    )


def _compute_metric_safe(
    metric_name: str,
    data: pd.DataFrame,
    lag: int = 3,
) -> tuple[np.ndarray | None, float, str | None]:
    """Считает одну метрику, ловит ошибки, возвращает (matrix, elapsed, error)."""

    t0 = time.monotonic()
    try:
        func = get_metric_func(metric_name)
        mat = func(data, lag=lag, control=None)
        elapsed = time.monotonic() - t0
        if mat is None:
            return None, elapsed, "returned None"
        mat = np.asarray(mat, dtype=float)
        if mat.ndim != 2:
            return None, elapsed, f"ndim={mat.ndim}, expected 2"
        return mat, elapsed, None
    except Exception as e:
        elapsed = time.monotonic() - t0
        return None, elapsed, str(e)


def run_validation(
    scenario_names: list[str] | None = None,
    metric_names: list[str] | None = None,
    lag: int = 3,
    progress_callback=None,
) -> ValidationReport:
    """Запуск полного набора валидационных сценариев.

    Args:
        scenario_names: список сценариев (по умолчанию DEFAULT_SCENARIOS)
        metric_names: список метрик (по умолчанию все из METRICS_REGISTRY)
        lag: лаг для directed метрик
        progress_callback: fn(stage: str, progress: float) для UI

    Returns:
        ValidationReport со всеми результатами и проверками.
    """

    if scenario_names is None:
        scenario_names = list(DEFAULT_SCENARIOS)
    if metric_names is None:
        metric_names = list(METRICS_REGISTRY.keys())

    # Фильтруем неизвестные
    scenario_names = [s for s in scenario_names if s in ALL_SCENARIOS]
    metric_names = [m for m in metric_names if m in METRICS_REGISTRY]

    report = ValidationReport()
    t_total_start = time.monotonic()

    n_total_steps = len(scenario_names) * len(metric_names)
    step = 0

    for s_name in scenario_names:
        scenario_factory = ALL_SCENARIOS[s_name]
        scenario: Scenario = scenario_factory()

        if progress_callback:
            try:
                progress_callback(
                    f"Сценарий: {scenario.name}",
                    step / max(1, n_total_steps),
                )
            except Exception:
                pass

        # Считаем все метрики для этого сценария
        matrices: dict[str, np.ndarray | None] = {}
        for m_name in metric_names:
            mat, elapsed, error = _compute_metric_safe(m_name, scenario.data, lag=lag)
            matrices[m_name] = mat
            report.metric_results.append(
                MetricResult(
                    scenario=scenario.name,
                    metric=m_name,
                    matrix=mat,
                    elapsed_sec=elapsed,
                    error=error,
                )
            )
            step += 1
            if progress_callback and step % 3 == 0:
                try:
                    progress_callback(
                        f"{scenario.name} / {m_name}",
                        step / max(1, n_total_steps),
                    )
                except Exception:
                    pass

        # Проверяем ожидания
        for exp in scenario.expectations:
            mat = matrices.get(exp.metric_group)
            if mat is None:
                # Метрика не рассчитана (не в списке или ошибка)
                if exp.metric_group not in metric_names:
                    continue  # пропускаем, если метрика не выбрана
                report.checks.append(
                    CheckResult(
                        scenario=scenario.name,
                        metric=exp.metric_group,
                        pair=exp.pair,
                        check=exp.check,
                        expected=exp.description,
                        actual_value=float("nan"),
                        passed=False,
                        message=f"Метрика {exp.metric_group} вернула ошибку или None",
                    )
                )
                continue

            result = _check_expectation(exp, mat, scenario.name)
            report.checks.append(result)

    report.elapsed_total_sec = time.monotonic() - t_total_start

    if progress_callback:
        try:
            progress_callback("Готово", 1.0)
        except Exception:
            pass

    return report


def run_quick_validation(
    lag: int = 3,
    progress_callback=None,
) -> ValidationReport:
    """Быстрый набор (4 сценария × стабильные метрики)."""

    stable = [
        "correlation_full", "correlation_partial", "correlation_directed",
        "h2_full", "h2_directed",
        "mutinf_full", "dcor_full",
        "coherence_full",
        "granger_full",
        "te_full",
        "ordinal_full",
    ]
    available = [m for m in stable if m in METRICS_REGISTRY]
    return run_validation(
        scenario_names=QUICK_SCENARIOS,
        metric_names=available,
        lag=lag,
        progress_callback=progress_callback,
    )


def run_full_validation(
    lag: int = 3,
    progress_callback=None,
) -> ValidationReport:
    """Полный набор: все сценарии × все метрики."""

    return run_validation(
        scenario_names=list(ALL_SCENARIOS.keys()),
        metric_names=list(METRICS_REGISTRY.keys()),
        lag=lag,
        progress_callback=progress_callback,
    )
