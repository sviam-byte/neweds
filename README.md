# NewEDS

NewEDS - Python-проект для анализа связности многоканальных табличных временных рядов.

Основной сценарий: взять CSV/Excel-таблицу, применить preprocessing, посчитать connectivity-метрики между каналами и сохранить HTML/Excel-отчет. Дополнительные сценарии для group comparison, HDF5/fMRI-like данных и voxel/bin-представлений остаются экспериментальными.

NewEDS is also an experimental pipeline for auditing how fMRI-like functional connectivity results depend on node definition, regional signal construction, temporal preprocessing, and connectivity metric choice. The goal is not to provide a clinical biomarker, but to make the preprocessing and metric-comparison steps explicit, auditable, and reproducible.

The project treats brain regions not as automatically valid units, but as objects that require quality control. Spatial bins or atlas parcels are candidate nodes: their mask validity and internal functional homogeneity should be checked before representative time series are used for connectivity analysis.

## Стабильное ядро

- загрузка табличных временных рядов из CSV и Excel;
- preprocessing: пропуски, нормализация, выбросы, AR/STL-опции;
- connectivity-метрики через единый registry;
- многомасштабная Haar-wavelet связность (`wavelet_full`, экспериментально);
- CLI `neweds` и Python API для одного файла;
- batch-запуск для набора табличных файлов;
- HTML/Excel-отчеты и экспорт connectivity-матриц;
- тесты для метрик, loader-ов, CLI и публичного pipeline.

Экспериментальные части вынесены ниже и не считаются основным контрактом проекта.

## Установка

```bash
pip install -e .
```

Для разработки и тестов:

```bash
pip install -e ".[dev,advanced,io]"
```

`advanced` добавляет часть опциональных метрик. `io` добавляет Parquet и старый XLS через `pyarrow` и `xlrd`.

## Быстрый старт

```bash
neweds examples/demo_timeseries.csv \
  --variants correlation_full,dcor_full,ordinal_full \
  --output-dir outputs/demo
```

После запуска появится:

```text
outputs/demo/
├── report.html
├── report.xlsx
└── connectivity_exports/
```

Минимальный Python API:

```python
from neweds import AnalysisConfig, run_analysis

result = run_analysis(
    "examples/demo_timeseries.csv",
    AnalysisConfig(
        variants=["correlation_full", "dcor_full", "ordinal_full"],
    ),
)

print(result.metrics.keys())
```

## Вход и выход

Обычный вход - таблица, где строки являются временными точками, а числовые столбцы являются сигналами:

```text
signal_a, signal_b_lagged, noise_control, seasonal_component
...
```

Основной выход:

- `report.html` - интерактивный отчет;
- `report.xlsx` - табличный отчет;
- экспортированные connectivity-матрицы;
- служебные сводки запуска.

## Экспериментальные сценарии

- signal QC before connectivity: ROI/bin homogeneity checks, regional-signal construction choices, and PCA sign-orientation metadata;
- `neweds-fmri-audit` для первичного аудита уже извлечённых ROI time series HC/SZ: inventory, ROI QC, temporal QC, baseline Pearson FC и FDR;
- `neweds-fmri-tissue-audit` для отдельного потокового аудита GM/WM/CSF HDF5 без смешивания с ROI/whole-brain результатами;
- node-definition QC for experimental voxel/bin workflows: mask coverage, volume-space adjacency constrained by mask, and explicit warnings that XYZ adjacency is not cortical-surface adjacency;
- `neweds-group` для case/control-сравнения;
- HDF5/fMRI-like эксперименты;
- выравнивание voxel/bin-пространства;
- validation-сценарии.

Эти сценарии полезны для раннего анализа и проверки pipeline на исследовательских данных, но их результаты нужно валидировать отдельно. Подробнее: [fMRI ROI audit MVP](docs/fmri_roi_audit.md), [Signal QC protocol](docs/fmri_signal_qc_protocol.md) и [Ограничения](docs/limitations.md).

## Документация

- [Архитектура](docs/architecture.md)
- [Метрики](docs/metrics.md)
- [Group pipeline](docs/group_pipeline.md)
- [fMRI ROI audit MVP](docs/fmri_roi_audit.md)
- [Independent GM/WM/CSF tissue audit](docs/fmri_tissue_audit.md)
- [fMRI audit separation contract](docs/fmri_audit_separation_contract.md)
- [fMRI-like signal QC protocol](docs/fmri_signal_qc_protocol.md)
- [Ограничения](docs/limitations.md)
- [Refactoring story](docs/refactoring_story.md)
- [Demo](examples/README.md)
