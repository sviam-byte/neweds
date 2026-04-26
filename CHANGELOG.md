# Changelog

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версионирование по [SemVer](https://semver.org/lang/ru/).

## [0.2.0] — 2026-04-26

### Добавлено
- Plugin-style реестр метрик на основе декоратора `@register_metric(...)` и `@dataclass(frozen=True, slots=True)` `Metric`. Все метрики снабжены категорией (`correlation` / `information` / `spectral` / `ordinal` / `causal`) и описанием.
- `controls` параметр у `run_analysis(...)` и CLI (`--controls`) для `*_partial` метрик.
- Реальная lag-optimization для направленных метрик (`config.lag_selection="optimize"`): пайплайн скорит матрицы по средней силе связи и выбирает лучший лаг.
- Структурированные ground-truth тесты ([tests/test_metric_ground_truth.py](tests/test_metric_ground_truth.py)) на VAR(1), независимые ряды и лаговую копию.
- Snapshot-регрессии для публичного pipeline ([tests/test_pipeline_snapshot.py](tests/test_pipeline_snapshot.py)).
- End-to-end subprocess-тест CLI ([tests/test_cli_integration.py](tests/test_cli_integration.py)).
- CI matrix `ubuntu-latest × windows-latest × Python 3.11/3.12` + `ruff check`, `ruff format --check`, `pytest --cov`.
- `neweds-group` console script для группового fMRI-CLI.

### Изменено
- Пакет переименован из неявного `src/` в стандартный `neweds/`. Импорты теперь `from neweds.* import ...`.
- `interfaces/` удалён; CLI-точки входа живут в `neweds/cli.py` и `neweds/cli_group.py`.
- `neweds/methods.py` — отдельный модуль с каталогом методов (`STABLE_METHODS`, `EXPERIMENTAL_METHODS`, `DIRECTED_METHODS`, `PVAL_METHODS`, `METHOD_INFO`); раньше всё это лежало в `config.py`.
- Описания методов (`METHOD_INFO`) переведены на английский.
- `AnalysisResult.tool_adapter_from_result` удалён из публичного API. Адаптер для существующих HTML/Excel writers перенесён в приватный `neweds/reporting/_adapter.py`.
- HTML/Excel writers теперь принимают `AnalysisResult` напрямую и не зависят от legacy-объекта `BigMasterTool`.
- `pyproject.toml` — `where=["."]`, `include=["neweds*"]`; `setuptools.packages.find` больше не использует `src` как пакет.
- Зависимости синхронизированы; добавлены extras `dev` и `advanced` (`dcor`, `hurst`, `nolds`).

### Исправлено
- `AnalysisConfig` имел дублирующие field-объявления (`master_seed`, `spatial_bin_size`, `spatial_grid_size`, `time_chunk` и др.) — Python молча брал последнее. Дубли удалены.
- `os.makedirs(SAVE_FOLDER, ...)` на верхнем уровне `config.py` создавал каталог `TimeSeriesAnalysis/` при любом импорте — удалено.
- `interfaces/cli.py` делал `sys.path.insert(...)` чтобы починить нестандартный layout — больше не нужно.

### Удалено
- Legacy-движок `BigMasterTool` (`src/core/engine.py`).
- Legacy-интерфейсы: `interfaces/gui.py`, `interfaces/web.py` (Streamlit), `interfaces/legacy_cli.py`.
- `START_TimeSeriesTool.bat`, `START_fMRI.bat`, `.streamlit/`.
- Пустой `setup.py` (PEP 517 не требует).
- Пустой каталог `src/export/` (`__init__.py` содержал только `# TODO`).
- `requirements.txt` — единый источник истины теперь `pyproject.toml`.
- `demo.csv` из корня репо (используем `examples/demo_timeseries.csv`).
- Тесты против removed legacy: `test_engine_metrics_api.py`, `test_exports_and_rank_metrics.py`.

### Метрики проекта

- Удалено ~6500 строк legacy-кода.
- 59 тестов проходят (было 49 преимущественно smoke-тестов).
- Архитектурные тесты гарантируют, что legacy-модули не возвращаются в импорт-граф.
