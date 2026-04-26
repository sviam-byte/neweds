# NewEDS

**NewEDS** — Python-инструмент для анализа связности (connectivity) многомерных временных рядов и групповых fMRI-данных.

Пакет показывает, как исследовательский прототип превращается в слоистый, тестируемый аналитический пайплайн со строгим публичным API.

---

## Что внутри

- **Central metric registry with decorator support** — 24 связностных метрики (Pearson, Spearman, Kendall, partial-correlation, distance correlation, mutual information, transfer entropy, Granger, coherence, ordinal MI, AH) живут в едином registry со структурированной метадатой; `@register_metric(...)` остаётся extension API для plugin/custom-метрик.
- **Структурированные контракты результатов** — `AnalysisResult` / `MetricResult` / `ComputationContract` (frozen `@dataclass(slots=True)`); каждый результат сопровождается воспроизводимым контекстом: параметры, конфиг-хэш, контролируемые переменные, лаг.
- **Чёткие границы форматов** — single-file `neweds` поддерживает CSV/Excel/Parquet/MAT/HDF5 через `load_or_generate`; directory batch поддерживает CSV/Excel/Parquet; `neweds-group` сейчас поддерживает subject-wise CSV/Excel/Parquet после spatial binning, не HDF5 group input.
- **Lag-optimization** — для направленных метрик пайплайн умеет искать лучший лаг в [1, max_lag].
- **HTML и Excel отчёты** — генерируются из `AnalysisResult`, без legacy-зависимостей.
- **Ground-truth тесты** — синтетические сценарии (VAR(1), независимые ряды, лаговая копия) с известным ответом, плюс snapshot-регрессии и subprocess-тест CLI.

---

## Установка

```bash
pip install -e ".[dev]"
pytest
```

Опционально:

```bash
pip install -e ".[dev,advanced]"   # + dcor, hurst, nolds для экспериментальных метрик
```

---

## Quick start

```bash
neweds examples/demo_timeseries.csv \
    --variants correlation_full,dcor_full,ordinal_full \
    --output-dir outputs/demo
```

После запуска в `outputs/demo/` появятся `report.html` и `report.xlsx`.

Программный API:

```python
from neweds import AnalysisConfig, run_analysis

result = run_analysis(
    "examples/demo_timeseries.csv",
    AnalysisConfig(
        variants=["correlation_full", "correlation_directed"],
        max_lag=3,
        lag_selection="optimize",
    ),
)

for name, metric in result.metrics.items():
    print(name, metric.matrix.shape, "lag=", metric.lag)
```

---

## Архитектура

```
neweds/
├── cli.py                    публичный CLI (time-series)
├── cli_group.py              CLI группового fMRI-сравнения
├── config.py                 AnalysisConfig, ComputationContract
├── methods.py                каталог методов (категории, описания, флаги)
├── core/
│   ├── pipeline.py           run_analysis (публичная точка входа)
│   ├── batch_pipeline.py     batch-режим + manifest + zip
│   ├── group_pipeline.py     fMRI: groupwise сравнение по canonical voxel space
│   ├── metric_runner.py      граница вычислений → registry
│   ├── results.py            AnalysisResult / MetricResult / WindowResult
│   ├── data_loader.py        ввод (CSV / Excel / Parquet / HDF5)
│   ├── preprocessing.py      нормализация, заполнение пропусков
│   ├── voxel_space.py        canonical voxel space для fMRI
│   └── window_scanner.py     сканирование по окнам (joblib)
├── metrics/
│   ├── registry.py           central metric registry (decorator support + dataclass)
│   └── connectivity.py       реализации метрик
├── reporting/
│   ├── html_generator.py     HTML-отчёт
│   ├── excel_writer.py       Excel-отчёт
│   └── _adapter.py           private shim: AnalysisResult → поверхность отчётов
├── io/                       загрузчики (HDF5, user_input)
├── analysis/                 dimred / graph / stats утилиты
├── validation/               синтетические сценарии для ground-truth
└── visualization/            heatmap / connectome / FFT plots
```

Поток данных:

```
CLI ─► run_analysis ─► load_or_generate ─► metric registry
                                               │
                            ComputationContract │
                                               ▼
                                       AnalysisResult ─► HTML / Excel
```

---

## Ключевые места для ревью кода

- [neweds/metrics/registry.py](neweds/metrics/registry.py) — central metric registry with decorator support, метаданные через `@dataclass(frozen=True, slots=True)`, read-only `Mapping`-view для обратной совместимости.
- [neweds/core/pipeline.py](neweds/core/pipeline.py) — публичная точка входа; реализует lag-selection через скоринг матриц связности, формирует `ComputationContract` для воспроизводимости.
- [neweds/core/results.py](neweds/core/results.py) — структурированные контракты результатов.
- [neweds/config.py](neweds/config.py) и [neweds/methods.py](neweds/methods.py) — разделение конфигурации и каталога методов.
- [tests/test_metric_ground_truth.py](tests/test_metric_ground_truth.py) — ground-truth сценарии (VAR(1), независимые ряды, лаговая копия) проверяют, что метрики действительно отвечают тому, что обещают.
- [tests/test_pipeline_snapshot.py](tests/test_pipeline_snapshot.py) — числовая регрессия публичного pipeline.
- [tests/test_cli_integration.py](tests/test_cli_integration.py) — subprocess-проверка, что `neweds <csv>` создаёт валидный HTML/Excel.

---

## Разработка

```bash
ruff check .
ruff format --check .
pytest --cov=neweds --cov-report=term-missing
```

CI (`.github/workflows/tests.yml`) гоняет матрицу `ubuntu-latest × windows-latest × Python 3.11/3.12`.

---

## Roadmap

- Расширение ground-truth сценариев (Lorenz, Rössler, NARMA).
- Группировка `*_partial` метрик в общий control-aware фреймворк.
- Экспорт результатов в Parquet и Arrow.

---

## Лицензия

MIT.
