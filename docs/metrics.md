# Метрики

Проект сравнивает несколько семейств connectivity-метрик на одних и тех же данных.

Основные группы:

- линейные корреляционные метрики;
- ранговые корреляции;
- нелинейные и информационные метрики;
- спектральная coherence;
- лаговые и направленные метрики.

Примеры вариантов:

```text
correlation_full
correlation_spearman
correlation_kendall
correlation_partial
dcor_full
dcor_partial
mutinf_full
coherence_full
ordinal_full
granger_full
te_full
```

Посмотреть доступные метрики можно из Python:

```python
from neweds.metrics import list_metrics

for metric in list_metrics():
    print(metric.name, metric.category)
```

Directed-метрики чувствительны к preprocessing, длине ряда и выбору лага. Если метрика не может быть посчитана корректно, результат должен быть заметным: warning или `NaN`, а не тихая подмена на осмысленно выглядящий ноль.
