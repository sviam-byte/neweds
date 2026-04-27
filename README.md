# NewEDS

**NewEDS** — Python-пакет для анализа связности многомерных временных рядов. Проект рассчитан на исследовательские данные, в том числе fMRI-подобные временные ряды: он загружает данные, выполняет настраиваемый preprocessing, считает набор connectivity-метрик, сохраняет воспроизводимые контракты вычислений и формирует HTML/Excel-отчёты.

Отдельно есть экспериментальный групповой пайплайн `neweds-group` для baseline-сравнения двух групп субъектов по edge-wise connectivity-признакам.

---

## Что делает проект

Проект решает задачу:

> взять многоканальные временные ряды и посчитать, как связаны между собой каналы разными математическими способами.

Типичный поток:

```text
таблица временных рядов
→ загрузка данных
→ preprocessing
→ расчёт connectivity-метрик
→ матрицы связности
→ контракты вычислений
→ HTML/Excel-отчёты
```

Если вход содержит `N` каналов, большинство метрик возвращает матрицу `N × N`, где элемент `[i, j]` описывает связь между каналами `i` и `j`. Для направленных метрик матрица может быть несимметричной: связь `i → j` не обязана совпадать со связью `j → i`.


- Загрузка CSV, Excel, Parquet, MAT и HDF5-файлов.
- Batch-режим для папок с CSV/Excel/Parquet.
- Настраиваемый preprocessing: заполнение пропусков, нормализация, обработка выбросов, AR-detrend.
- Набор connectivity-метрик: корреляционные, информационные, спектральные, ordinal и lag-based/направленные.
- Plugin registry для метрик: каждая метрика имеет имя, категорию, описание и metadata-флаги.
- Публичный Python API через `AnalysisConfig` и `run_analysis`.
- CLI-интерфейс `neweds` для одиночного файла и batch-запуска.
- HTML и Excel-отчёты.
- `ComputationContract` для воспроизводимости: каждый результат хранит сведения о входе, preprocessing, controls, лаге, типе метрики и конфигурации.
- Экспериментальный CLI `neweds-group` для сравнения двух групп субъектов.

---

## Установка

Требуется Python `>=3.11`.

```bash
pip install -e .
```

Для разработки и тестов:

```bash
pip install -e ".[dev]"
```

Дополнительные зависимости для некоторых продвинутых методов:

```bash
pip install -e ".[advanced]"
```

---

## Быстрый старт через CLI

Запуск на демо-файле:

```bash
neweds examples/demo_timeseries.csv \
    --variants correlation_full,dcor_full,ordinal_full \
    --output-dir outputs/demo
```

После запуска в `outputs/demo/` появятся отчёты:

```text
report.html
report.xlsx
```

Пример с лаговыми метриками:

```bash
neweds examples/demo_timeseries.csv \
    --variants correlation_directed,granger_full \
    --lags 3 \
    --lag-selection optimize \
    --output-dir outputs/lagged_demo
```

Отключение отдельных шагов preprocessing:

```bash
neweds examples/demo_timeseries.csv \
    --variants correlation_full,dcor_full \
    --no-normalize \
    --no-remove-outliers \
    --output-dir outputs/no_norm_demo
```

Batch-запуск по папке:

```bash
neweds data/ \
    --variants correlation_full,dcor_full,ordinal_full \
    --recursive \
    --batch-zip \
    --output-dir outputs/batch
```

Batch-режим создаёт отдельную папку результата для каждого файла и общий `batch_manifest.csv`.

--
## Конфигурация анализа

Настройки задаются через `AnalysisConfig`:

```python
from neweds import AnalysisConfig

config = AnalysisConfig(
    variants=["correlation_full", "correlation_partial", "granger_full"],
    max_lag=3,
    lag_selection="fixed",
    pvalue_correction="none",
    controls=["motion", "global_signal"],
    preprocess=True,
    fill_missing=True,
    normalize=True,
    remove_outliers=True,
    ar_order=0,
)
```

Ключевые параметры:

- `variants` — список метрик, которые нужно посчитать.
- `max_lag` — максимальный лаг для directed/lag-based метрик.
- `lag_selection` — `fixed` или `optimize`.
- `controls` — контрольные колонки для control-aware partial-метрик.
- `preprocess` — общий флаг preprocessing.
- `fill_missing` — заполнение пропусков.
- `normalize` — нормализация временных рядов.
- `remove_outliers` — обработка выбросов.
- `ar_order` — AR-detrend порядка `p`; `0` означает, что AR-detrend выключен.
- `pvalue_correction` — коррекция p-values, если метрика их возвращает.
- `window_sizes`, `window_stride`, `window_policy` — настройки sliding-window анализа.

---

## Метрики

Метрики регистрируются через plugin registry в `neweds.metrics.registry`. Каждая метрика хранит metadata:

- `name` — имя метрики;
- `category` — категория;
- `description` — описание;
- `directed` — направленная ли метрика;
- `pvalue_based` — возвращает ли p-value-based результат;
- `supports_control` — поддерживает ли control variables;
- `experimental` — экспериментальный статус;
- `stable` — стабильная метрика;
- `partial_mode` — семантика partial-режима.

Посмотреть доступные метрики можно так:

```python
from neweds.metrics import list_metrics

for metric in list_metrics():
    print(metric.name, metric.category, metric.directed, metric.experimental)
```

Основные группы метрик:

```text
correlation_full
correlation_spearman
correlation_kendall
correlation_partial
correlation_directed
h2_full
h2_partial
h2_directed

mutinf_full
mutinf_partial
dcor_full
dcor_partial
dcor_directed
ah_full
ah_partial
ah_directed

granger_full
granger_partial
te_full
te_partial

coherence_full
coherence_partial

ordinal_full
ordinal_directed
```
## Controls и partial-метрики

Контрольные колонки можно передать через Python API:

```python
config = AnalysisConfig(
    variants=["correlation_partial", "dcor_partial"],
    controls=["motion", "global_signal"],
)
```

или через CLI:

```bash
neweds data.csv \
    --variants correlation_partial,dcor_partial \
    --controls motion,global_signal \
    --output-dir outputs/partial
```

Важное различие:

- `precision_matrix` — partial-метрика через обратную корреляционную матрицу, то есть условно “контроль остальных каналов”.
- `explicit_controls_residualization` — метрика считается после регрессии пользовательских control-переменных.

Фактический режим сохраняется в `MetricResult.contract.partial_mode` и metadata результата. Это нужно проверять при интерпретации, потому что слово `partial` в разных метриках означает не одно и то же.

---

## Форматы входных данных

Одиночный запуск поддерживает:

```text
.csv
.xlsx / .xls
.parquet
.mat
.h5 / .hdf5 / .hdf
```

Batch-режим по папке поддерживает:

```text
.csv
.xlsx / .xls
.parquet
```

Для обычных табличных данных ожидается формат:

```text
time, channel_1, channel_2, channel_3, ...
0,    ...
1,    ...
2,    ...
```

Колонка времени может быть частью таблицы, но в анализ должны попадать именно числовые signal columns. Control columns, если они указаны, отделяются от signal-блока и используются только теми метриками, которые поддерживают controls.

HDF5/fMRI-подобные данные обрабатываются через spatial binning. Параметры пространственной агрегации задаются в `AnalysisConfig`:

```python
config = AnalysisConfig(
    spatial_grid_size=10,
    spatial_grid_method="mean",
    lazy_spatial_bin=False,
)
```

---

## Результаты

Один запуск возвращает `AnalysisResult` и может сохранять файлы отчёта.

В `AnalysisResult` есть:

- `input_info` — информация о входных данных;
- `config` — использованная конфигурация;
- `metrics` — словарь `имя_метрики → MetricResult`;
- `logs` — предупреждения и служебные сообщения;
- `windows` — результаты sliding-window анализа, если он включён;
- `artifacts` — пути к созданным артефактам.

В `MetricResult` есть:

- `matrix` — матрица связности;
- `pvalues` — p-values, если применимо;
- `labels` — имена каналов;
- `method` — имя метрики;
- `directed` — направленная ли матрица;
- `lag` — использованный лаг;
- `metadata` — дополнительные сведения;
- `contract` — `ComputationContract`.

`ComputationContract` нужен для воспроизводимости. Он фиксирует:

- какая метрика считалась;
- сколько было каналов и временных точек;
- долю пропусков во входе;
- какие preprocessing-шаги были применены;
- какие controls использовались;
- какой partial-mode был выбран;
- directed/lag-настройки;
- warnings;
- форму результата;
- seed и hash конфигурации.

---

## Групповое сравнение

Для группового анализа есть отдельный CLI:

```bash
neweds-group \
    --case-dir data/case \
    --control-dir data/control \
    --output-dir results/group \
    --spatial-grid-size 10 \
    --strategy intersection \
    --alpha 0.05
```

Текущий `neweds-group` делает:

```text
папка case
+ папка control
→ загрузка субъектов
→ построение canonical voxel space
→ выравнивание субъектов
→ расчёт connectivity-признаков
→ Mann–Whitney U по каждому edge
→ Benjamini–Hochberg FDR
→ CSV-выводы
```

Основные выходные файлы:

```text
group_comparison.csv
missing_bin_qc.csv
top_significant_pairs.csv
features_schiz.npy / features_healthy.npy, если сохранение признаков включено
subject_ids_schiz.csv / subject_ids_healthy.csv
```

Несмотря на названия `schiz/healthy` внутри некоторых файлов и старых API-аргументов, внешний CLI уже использует более универсальные названия `case/control`. Это legacy-след конкретной исходной задачи HC/SZ.

Групповой pipeline помечен как experimental. Его результаты подходят для разведочного анализа и инженерной демонстрации, но не должны использоваться как финальная публикационная статистика без расширения дизайна.

---

## Архитектура

Упрощённая структура проекта:

```text
neweds/
├── cli.py                    CLI для одиночного и batch-анализа
├── cli_group.py              CLI для группового сравнения
├── config.py                 AnalysisConfig и ComputationContract
├── methods.py                compatibility facade для старого API
├── core/
│   ├── pipeline.py           run_analysis — главный публичный pipeline
│   ├── batch_pipeline.py     batch-режим, manifest, zip
│   ├── group_pipeline.py     экспериментальный group pipeline
│   ├── metric_runner.py      граница между pipeline и registry метрик
│   ├── results.py            AnalysisResult, MetricResult, WindowResult
│   ├── data_loader.py        оркестрация загрузки данных
│   ├── preprocessing.py      preprocessing временных рядов
│   ├── voxel_space.py        canonical voxel space
│   └── window_scanner.py     sliding-window анализ
├── metrics/
│   ├── registry.py           plugin registry метрик
│   ├── correlation.py        корреляционные метрики
│   ├── information.py        MI, dCor, AH
│   ├── causal.py             Granger, transfer entropy
│   ├── spectral.py           coherence
│   └── ordinal.py            ordinal / permutation-pattern MI
├── io/                       загрузчики и обработка пользовательского ввода
├── reporting/                HTML, Excel и connectivity export
├── analysis/                 вспомогательные graph/stats/dimred функции
├── validation/               synthetic validation scenarios
├── visualization/            графики
└── tests/                    тесты
```


- Directed metrics чувствительны к preprocessing. AR-detrend может подавлять lag-структуру, которую затем пытаются найти Granger/TE/directed-метрики.
- Некоторые experimental-метрики вычислительно дорогие и должны проверяться на synthetic data перед содержательной интерпретацией.
- Внутри проекта остаются compatibility-слои и legacy-имена, особенно в group pipeline (`schiz/healthy` рядом с `case/control`).

