# Demo

Этот пример запускает основной pipeline на маленьком синтетическом многоканальном временном ряде.

```bash
neweds examples/demo_timeseries.csv \
  --variants correlation_full,dcor_full,ordinal_full \
  --output-dir outputs/demo
```

Ожидаемые файлы:

```text
outputs/demo/
├── report.html
├── report.xlsx
└── connectivity_exports/
```

Данные в `demo_timeseries.csv` специально небольшие: они нужны для быстрой проверки CLI и структуры отчета, а не для содержательных статистических выводов.
