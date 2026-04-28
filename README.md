# NewEDS

NewEDS - Python-проект для анализа связности многоканальных табличных временных рядов.

Основной сценарий: взять CSV/Excel-таблицу, применить preprocessing, посчитать connectivity-метрики между каналами и сохранить HTML/Excel-отчет. Дополнительные сценарии для group comparison, HDF5/fMRI-like данных и voxel/bin-представлений остаются экспериментальными.

## Стабильное ядро

- загрузка табличных временных рядов из CSV и Excel;
- preprocessing: пропуски, нормализация, выбросы, AR/STL-опции;
- connectivity-метрики через единый registry;
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

- `neweds-group` для case/control-сравнения;
- HDF5/fMRI-like эксперименты;
- выравнивание voxel/bin-пространства;
- validation-сценарии.

Эти сценарии полезны для раннего анализа и проверки pipeline на исследовательских данных, но их результаты нужно валидировать отдельно. Подробнее: [Ограничения](docs/limitations.md).

## Документация

- [Архитектура](docs/architecture.md)
- [Метрики](docs/metrics.md)
- [Group pipeline](docs/group_pipeline.md)
- [Ограничения](docs/limitations.md)
- [Refactoring story](docs/refactoring_story.md)
- [Demo](examples/README.md)

## License

Open-source license сейчас не выдана.
