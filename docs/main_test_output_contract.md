# Main Test Output Contract

Этот документ описывает ожидаемый содержательный выход основного тестового
сценария NewEDS. Под main-тестом здесь понимается не только проверка, что CLI
создал `report.html` и `report.xlsx`, а полный исследовательский контракт:
какие метрики посчитаны, что они означают, как обработаны данные, где во
времени связь максимальна и можно ли по этим признакам различать группы.

## Цель

Main-тест должен отвечать на пять вопросов:

1. Какие связи обнаружены между каналами или регионами.
2. Каким семейством метрик эти связи найдены.
3. После какого препроцессинга получены результаты.
4. В каком временном окне, при каком старте и лаге связь максимальна.
5. Можно ли по рассчитанным признакам обучить модель, которая различает группы
   лучше случайного baseline.

## Минимальный Набор Метрик

Минимальный выход должен включать следующие метрики:

| Variant | Семейство | Выход | Смысл |
| --- | --- | --- | --- |
| `correlation_full` | linear correlation | симметричная матрица `N x N`, диапазон `[-1, 1]` | Базовая Pearson-связь между каналами. Показывает синхронную линейную связь. |
| `dcor_full` | nonlinear dependence | симметричная матрица `N x N`, диапазон `[0, 1]` | Distance correlation. Ловит зависимости шире линейной корреляции; `0` близко к независимости. |
| `ordinal_full` | ordinal patterns | симметричная матрица `N x N`, неотрицательные значения | Ordinal mutual information по Bandt-Pompe. Сравнивает локальные порядковые паттерны, а не абсолютные амплитуды. |
| `wavelet_full` | multiscale spectral | симметричная матрица `N x N`, диапазон `[0, 1]` | Новая Haar-wavelet связность. Показывает сходство каналов по многомасштабной структуре изменений. |
| `correlation_directed` | lagged directed | направленная матрица `N x N`, диагональ `0` | Связь вида `source(t) -> target(t + lag)` через лаговую корреляцию. |

Если main-тест запускается в расширенном режиме, к этому набору можно добавить:

| Variant | Семейство | Смысл |
| --- | --- | --- |
| `coherence_full` | spectral | Частотная согласованность каналов через magnitude-squared coherence. |
| `dcor_directed` | nonlinear directed | Направленная лаговая distance correlation. |
| `ordinal_directed` | ordinal directed | Направленная ordinal MI между прошлым источника и будущим цели. |
| `granger_full` | causal/statistical | p-value Granger-теста: помогает ли прошлое одного сигнала предсказывать другой. |
| `te_full` | information directed | Transfer entropy как направленная информационная зависимость. |

## Контракт Одной Метрики

Для каждой метрики ожидается единый результат:

```text
metrics[variant]:
  name
  matrix
  directed
  lag
  pvalue_based
  metadata
  contract
```

Обязательные поля по смыслу:

| Поле | Ожидаемый смысл |
| --- | --- |
| `matrix` | Матрица связности `N x N`. Для undirected-метрик симметричная, для directed-метрик направленная. |
| `directed` | `true`, если матрица имеет направление `source -> target`. |
| `lag` | Использованный лаг для directed-метрик; для undirected может быть `null`. |
| `pvalue_based` | `true`, если значения матрицы являются p-value, например у `granger_full`. |
| `metadata.category` | Семейство метрики: `correlation`, `information`, `spectral`, `ordinal`, `causal`. |
| `metadata.signal_columns` | Каналы, которые реально вошли в матрицу. |
| `metadata.control_columns` | Контрольные переменные, исключённые из основной матрицы. |
| `metadata.preprocess_steps` | Список шагов обработки данных до расчёта метрики. |
| `contract` | Воспроизводимый контракт вычисления: вход, лаг, preprocessing, controls, shape выхода, seed/config hash. |

## Препроцессинг-Анализ

Main-тест должен возвращать отдельный блок, объясняющий, на каких данных
считались метрики.

```text
preprocessing:
  raw_shape
  final_shape
  raw_columns
  signal_columns
  control_columns
  missing_fraction
  dropped_columns
  preprocess_steps
  outlier_summary
  normalization
  ar_diagnostics
```

Ожидаемый смысл полей:

| Поле | Смысл |
| --- | --- |
| `raw_shape` | Размер входной таблицы до обработки. |
| `final_shape` | Размер данных после препроцессинга и исключения control-колонок. |
| `signal_columns` | Каналы, по которым считались connectivity-матрицы. |
| `control_columns` | Колонки, использованные как ковариаты или исключённые controls. |
| `missing_fraction` | Доля пропусков после загрузки и обработки. |
| `dropped_columns` | Удалённые нечисловые, пустые или низковариативные колонки. |
| `preprocess_steps` | Фактически применённые шаги: low-variance filter, outlier handling, interpolation/fill, AR-cleaning, normalization, `auto_difference`. |
| `outlier_summary` | Сколько выбросов найдено и как они обработаны: mask, clip, winsorize или median. |
| `normalization` | Тип нормализации, например z-score или robust z-score. |
| `ar_diagnostics` | Если включён AR-препроцессинг: `phi`, автокорреляции до/после, Ljung-Box p-values. |

Препроцессинг-блок должен позволять отличить два разных вывода:

- "метрика изменилась из-за реального различия в сигналах";
- "метрика изменилась из-за очистки, нормализации, удаления колонок или AR-фильтрации".

## Окна, Лаги И Старт

Для динамического анализа main-тест должен возвращать результат сканирования
окна по времени.

```text
windows[variant]:
  policy
  stride
  sizes[window_size]:
    best_window:
      start
      end
      metric
    curve:
      x
      y
    ticks:
      - start
        end
        metric
    extremes:
      best
      median
      worst
```

Ожидаемый смысл:

| Поле | Смысл |
| --- | --- |
| `window_size` | Длина временного окна. |
| `stride` | Шаг сдвига окна. |
| `start` | Индекс начала окна. |
| `end` | Индекс конца окна. |
| `metric` | Score связности в этом окне. |
| `curve.x` | Все проверенные старты окон. |
| `curve.y` | Score для каждого старта. |
| `best_window` | Окно, где связь наиболее выражена. |
| `extremes.best` | Индекс лучшего окна в списке `ticks`. |
| `extremes.median` | Индекс типичного окна. |
| `extremes.worst` | Индекс слабейшего окна. |

Для directed-метрик дополнительно ожидается подбор лага:

```text
lag_search:
  strategy: fixed | optimize
  max_lag
  selected_lag
  score_by_lag
```

При `strategy = optimize` pipeline должен перебрать лаги от `1` до `max_lag`
и выбрать лаг, при котором directed-связь выражена сильнее всего. Для
p-value-метрик, например Granger, критерий выбора должен учитывать, что меньшее
p-value означает более сильное свидетельство направленной предсказательной
связи.

## Обучение И Проверка Различения

Main-тест должен включать supervised-блок, который проверяет, несут ли
connectivity-признаки различающую информацию о группах.

```text
classification:
  feature_sets
  labels
  validation
  models
  metrics
  predictions
  confusion_matrix
  permutation_baseline
  interpretation
```

### Feature Sets

Ожидаемые наборы признаков:

| Feature set | Источник признаков |
| --- | --- |
| `correlation_full` | Верхний треугольник матрицы Pearson-корреляции. |
| `dcor_full` | Верхний треугольник матрицы distance correlation. |
| `ordinal_full` | Верхний треугольник ordinal MI. |
| `wavelet_full` | Верхний треугольник Haar-wavelet матрицы. |
| `directed_lag` | Off-diagonal элементы directed-метрик с выбранным лагом. |
| `window_summary` | Сводки по окнам: best score, start/end лучшего окна, median/worst score, стабильность score по окнам. |
| `preprocessing_summary` | Не как основной классификационный сигнал, а как QC-ковариаты и контроль возможных артефактов. |
| `all_combined` | Объединённый набор признаков после стандартизации внутри train-fold. |

### Validation Contract

Валидация должна быть защищена от утечки данных:

```text
validation:
  strategy: train_test_split | cross_validation | leave_one_group_out
  n_splits
  random_seed
  leakage_guard: true
  scaling_fit_scope: train_fold_only
  feature_selection_fit_scope: train_fold_only
  preprocessing_fit_scope: train_fold_only_when_learned
```

Обязательное правило: всё, что обучается на данных, должно fit-иться только
на train-части. Это относится к scaler, feature selection, PCA, imputation,
model selection и любым learned-преобразованиям. Subject-local очистка сигнала
может выполняться на уровне отдельного объекта, но cross-subject операции должны
быть внутри train-fold.

### Метрики Качества

Для бинарного различения ожидаются:

| Метрика | Смысл |
| --- | --- |
| `accuracy` | Общая доля правильных ответов. |
| `balanced_accuracy` | Средняя точность по классам; предпочтительна при дисбалансе. |
| `roc_auc` | Способность ранжировать классы; `0.5` соответствует случайному уровню. |
| `sensitivity` | Насколько хорошо модель находит положительный класс. |
| `specificity` | Насколько хорошо модель не путает отрицательный класс с положительным. |
| `precision` | Насколько надёжны положительные предсказания. |
| `f1` | Баланс precision и sensitivity. |
| `train_test_gap` | Разрыв между train и test quality; большой gap указывает на переобучение. |

### Predictions

Таблица предсказаний должна иметь вид:

```text
predictions:
  sample_id
  group_id
  fold
  y_true
  y_pred
  y_score
  feature_set
  model_name
```

`group_id` нужен, если есть связанные наблюдения одного субъекта, сессии или
семейства. В таком случае split должен не допускать попадания связанных
наблюдений одновременно в train и test.

### Permutation Baseline

Для проверки, что качество не является случайным или следствием утечки,
ожидается permutation baseline:

```text
permutation_baseline:
  n_permutations
  metric
  observed_score
  null_mean
  null_std
  p_value
```

Смысл: метки классов перемешиваются, модель обучается тем же способом, и
наблюдаемое качество сравнивается с нулевым распределением. Main-тест должен
показывать не просто "модель обучилась", а "качество выше случайного baseline"
или явно фиксировать, что такого свидетельства нет.

### Interpretation

Интерпретационный блок должен показывать:

```text
interpretation:
  best_feature_set
  best_model
  top_features
  top_edges
  metric_family_ranking
  limitations
```

Ожидаемый смысл:

| Поле | Смысл |
| --- | --- |
| `best_feature_set` | Какое семейство признаков лучше всего различает группы. |
| `top_features` | Самые информативные признаки после стабильной оценки внутри CV. |
| `top_edges` | Пары каналов или регионов, которые дают основной вклад. |
| `metric_family_ranking` | Сравнение `correlation`, `dcor`, `ordinal`, `wavelet`, `directed`, `window`. |
| `limitations` | Ограничения интерпретации: малый размер выборки, дисбаланс, нестабильные признаки, отсутствие внешней валидации. |

## Рекомендуемый Полный Профиль

Для исследовательского main-прогона нужен режим "все метрики, но ограниченный
по окнам, лагам и стартам". Его цель - получить широкую картину связности без
ночного перебора всех возможных комбинаций.

Рекомендуемый профиль:

```text
profile: full_limited

metrics:
  correlation_full
  correlation_spearman
  correlation_kendall
  correlation_partial
  correlation_directed
  h2_full
  h2_partial
  h2_directed
  coherence_full
  coherence_partial
  wavelet_full
  wavelet_partial
  mutinf_full
  mutinf_partial
  dcor_full
  dcor_partial
  dcor_directed
  ordinal_full
  ordinal_directed
  granger_full
  granger_partial
  te_full
  te_partial

optional_opt_in_metrics:
  ah_full
  ah_partial
  ah_directed

windows:
  window_sizes: [60, 120]
  starts: [0, middle, late]
  full_sliding_scan: false

lags:
  candidate_lags: [1, 3]
  lag_selection: limited_optimize

classification:
  strict_validation: false
  strategy: simple_train_test_or_3_fold_cv
  model_family: linear_logistic_or_linear_svm
  permutation_p_test: false
```

Ожидаемый смысл профиля:

| Настройка | Смысл |
| --- | --- |
| `all metrics` | Считаются все основные семейства: linear/rank, nonlinear, spectral, wavelet, ordinal, directed, Granger, TE. |
| `window_sizes: [60, 120]` | Проверяются два масштаба локальной динамики, короткий и более длинный. |
| `starts: [0, middle, late]` | Проверяются ранний, средний и поздний участки ряда без полного sliding scan. |
| `candidate_lags: [1, 3]` | Проверяются короткий и более отложенный directed-эффект. |
| `strict_validation: false` | Обучение нужно как быстрая проверка различимости, а не как финальная публикационная модель. |
| `permutation_p_test: false` | Статистический permutation p-test отключён; это экономит время и не считается доказательством устойчивой классификации. |

Такой профиль должен давать широкую диагностическую картину, но оставаться
управляемым по времени. Если `N` велико, для тяжёлых pairwise-метрик нужно
использовать guardrails, ограничение пар или подвыборку пар.

## Оценка Времени Прогона

Main-прогон должен уметь выдавать грубую оценку времени выполнения. Это не
статистический тест качества модели и не permutation p-test; это инженерная
оценка runtime с небольшим разбросом.

Рекомендуемый режим оценки:

```text
runtime_estimate:
  enabled: true
  repeats: 3
  warmup_runs: 0 или 1
  same_config: true
  same_input: true
  collect_per_stage_timing: true
```

Ожидаемый выход:

```text
runtime_estimate:
  repeats: 3
  total_seconds:
    runs: [run_1, run_2, run_3]
    mean
    std
    min
    max
    spread_abs
    spread_pct
  per_stage_seconds:
    preprocessing
    metrics_total
    windows_total
    lag_search_total
    feature_building
    classification
    report_writing
  slowest_stages
  hardware_context
  config_context
```

Поля разброса:

| Поле | Формула | Смысл |
| --- | --- | --- |
| `mean` | среднее по 3 прогонам | Центральная оценка времени. |
| `std` | стандартное отклонение по 3 прогонам | Насколько нестабильно время. |
| `min` | минимум | Лучший наблюдавшийся прогон. |
| `max` | максимум | Худший наблюдавшийся прогон. |
| `spread_abs` | `max - min` | Абсолютный разброс. |
| `spread_pct` | `(max - min) / mean * 100` | Разброс в процентах от среднего. |

Пример интерпретации:

```text
runtime_estimate:
  repeats: 3
  total_seconds:
    runs: [3180.4, 3412.8, 3295.1]
    mean: 3296.1
    std: 116.2
    min: 3180.4
    max: 3412.8
    spread_abs: 232.4
    spread_pct: 7.1
```

Человекочитаемый вывод:

```text
Оценка времени: 54.9 мин в среднем по 3 прогонам.
Разброс: 53.0-56.9 мин, spread 7.1%.
Самые медленные этапы: dcor/mutinf/TE metrics, window scan, Granger.
Permutation p-test отключён.
```

Если полный трёхкратный прогон слишком дорогой, допускается probe-режим:

```text
runtime_probe:
  repeats: 3
  sample_channels: min(N, 40)
  sample_timepoints: min(T, 500)
  extrapolate_to_full: true
```

Probe-режим должен явно помечаться как приблизительная экстраполяция. Его
нельзя смешивать с фактическим временем полного прогона.

## Ожидаемое Дерево Выхода

Рекомендуемая структура артефактов:

```text
outputs/main_test/
  report.html
  report.xlsx
  run_summary.json
  preprocessing/
    preprocessing_summary.json
    preprocessing_steps.csv
  metrics/
    correlation_full.npy
    dcor_full.npy
    ordinal_full.npy
    wavelet_full.npy
    correlation_directed.npy
    metric_contracts.json
  windows/
    window_summary.json
    window_curves.csv
  lag_search/
    lag_search_summary.json
  classification/
    feature_sets.csv
    validation_summary.json
    predictions.csv
    confusion_matrix.csv
    permutation_baseline.csv
    top_features.csv
  runtime/
    timing_runs.csv
    runtime_summary.json
    stage_timing.csv
```

HTML/Excel отчёты являются человекочитаемым слоем. JSON/CSV/NPY файлы являются
машиночитаемым слоем, пригодным для повторной проверки и сравнения запусков.

## Критерии Прохождения Main-Теста

Main-тест считается успешно пройденным, если:

1. Все обязательные метрики рассчитаны и имеют матрицы ожидаемой формы `N x N`.
2. `correlation_full`, `dcor_full`, `ordinal_full` и `wavelet_full` являются
   undirected-результатами; directed-метрики имеют `directed = true` и
   зафиксированный `lag`.
3. Для каждой метрики есть `metadata` и `contract`, позволяющие понять входные
   каналы, препроцессинг, controls, shape выхода и конфигурацию.
4. Препроцессинг-блок явно показывает применённые шаги и итоговые каналы.
5. Оконный анализ возвращает `best_window`, `curve`, `ticks` и
   `best/median/worst` для заданных размеров окон.
6. Лаговый анализ для directed-метрик фиксирует `selected_lag` и критерий
   выбора.
7. Classification-блок возвращает качество различения на независимой проверке
   или cross-validation.
8. Есть baseline со случайно перемешанными метками или явно указано, что
   статистического свидетельства качества выше случайного уровня пока нет.
9. Runtime-блок, если включён, содержит не менее трёх повторов или явно помеченный
   probe-режим с экстраполяцией.
10. Отчёт не делает клинических или причинных утверждений сильнее, чем позволяют
   данные, дизайн валидации и размер выборки.

## Короткая Формулировка

Main-тест должен выдавать не просто набор файлов, а воспроизводимый протокол:
какие connectivity-метрики посчитаны, что они означают, как были обработаны
данные, где во времени и при каком лаге связь максимальна, и насколько хорошо
эти признаки позволяют различать группы на честной проверке.
