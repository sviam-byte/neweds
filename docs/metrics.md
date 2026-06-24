# Метрики

Проект сравнивает несколько семейств connectivity-метрик на одних и тех же данных.

Основные группы:

- линейные корреляционные метрики;
- ранговые корреляции;
- нелинейные и информационные метрики;
- спектральная coherence;
- многомасштабная Haar-wavelet связность;
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
wavelet_full
wavelet_partial
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

`wavelet_full` выполняет дискретное Haar-разложение каждого канала и усредняет
`r²` между detail-коэффициентами совпадающих масштабов с весом по числу
коэффициентов. Результат симметричен и лежит в `[0, 1]`. Это компактная
multiscale coupling-метрика, а не классическая continuous-wavelet coherence.
`wavelet_partial` сначала удаляет линейный вклад явно переданных control-переменных.
