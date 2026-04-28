# Group pipeline

`neweds-group` - экспериментальный сценарий для сравнения двух групп: `case` и `control`.

Пример:

```bash
neweds-group \
  --case-dir data/case \
  --control-dir data/control \
  --output-dir results/group \
  --canonical-reference all
```

Ожидаемый вход - папки с subject-wise CSV/Excel/Parquet-файлами. Pipeline строит общее bin/voxel-представление, считает connectivity-признаки по субъектам и сохраняет exploratory group comparison.

Основные выходы:

```text
group_comparison.csv
missing_bin_qc.csv
features_case.npy
features_control.npy
subject_ids_case.csv
subject_ids_control.csv
```

Этот сценарий не является финальной статистической моделью. Для реального исследования нужны дизайн эксперимента, проверка ковариат, permutation/bootstrap-подходы или другая независимая статистическая валидация.
