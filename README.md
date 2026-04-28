# NewEDS

NewEDS - исследовательский Python-проект для анализа связности многоканальных временных рядов.

Проект берет таблицу с временными рядами, применяет настраиваемый preprocessing, считает несколько семейств connectivity-метрик между каналами и собирает HTML/Excel-отчет. Это портфолио-проект и инженерный toolkit, а не медицинский диагностический инструмент.

## Что показывает проект

- загрузку CSV, Excel, MAT, HDF5 и опционально Parquet-файлов;
- preprocessing временных рядов: пропуски, нормализация, выбросы, AR/STL-опции;
- единый способ запуска метрик связности;
- CLI и Python API для одиночного файла и batch-запуска;
- HTML/Excel-отчеты с матрицами и сводками;
- тесты для численного кода, CLI и архитектурных границ;
- экспериментальный group pipeline для сравнения case/control-групп.

## Установка

```bash
pip install -e .
```

Для разработки и тестов:

```bash
pip install -e ".[dev,advanced,io]"
```

`io` добавляет поддержку Parquet и старого XLS через `pyarrow` и `xlrd`.

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

## Статус проекта

Стабильная часть:

- pipeline для одного файла;
- preprocessing табличных временных рядов;
- connectivity-метрики;
- CLI `neweds`;
- HTML/Excel-отчеты;
- публичный Python API.

Экспериментальная часть:

- `neweds-group` для case/control-сравнения;
- HDF5/fMRI-like эксперименты;
- выравнивание voxel/bin-пространства;
- validation-сценарии.

Экспериментальные части нужны для демонстрации работы с исследовательскими данными. Их результаты требуют отдельной статистической проверки.

## Документация

- [Архитектура](docs/architecture.md)
- [Метрики](docs/metrics.md)
- [Group pipeline](docs/group_pipeline.md)
- [Ограничения](docs/limitations.md)
- [Demo](examples/README.md)

## License

Репозиторий опубликован как портфолио/исследовательский проект. Open-source license сейчас не выдана.
