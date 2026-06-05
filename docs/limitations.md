# Ограничения

## Не диагностический инструмент

NewEDS считает исследовательские connectivity-признаки. Его результаты подходят для проверки гипотез и инженерной отладки pipeline, но не заменяют валидированную модель, клиническое заключение или независимую статистическую проверку.

## Preprocessing меняет сигнал

Заполнение пропусков, нормализация, удаление выбросов, AR/STL-preprocessing, auto-differencing и spatial binning могут менять статистическую структуру временного ряда. Это особенно важно для lag-based и directed-метрик: после preprocessing сигнал может стать удобнее для вычислений, но часть исходной динамики может быть ослаблена или переинтерпретирована.

## Directed metrics are fragile

Lag-based и directed-метрики чувствительны к длине ряда, sampling rate, preprocessing и способу выбора лага. Направленная связь в отчёте не должна автоматически трактоваться как причинность: она показывает зависимость, найденную выбранным алгоритмом при заданных параметрах.

## Partial metrics are not equivalent

Метрики с названием `partial` не обязательно реализуют одинаковую математическую семантику. В отчётах нужно проверять, какой именно control/residualization использован: pairwise control, custom controls и residualization могут отвечать на разные статистические вопросы.

## Group pipeline is experimental

Group pipeline полезен для раннего case/control-сравнения, но требует независимой статистической проверки, контроля confounds, прозрачного protocol design и воспроизводимости на внешних данных.

## fMRI ROI audit is experimental

`neweds-fmri-audit` работает с уже извлечёнными ROI time series для групп HC/SZ.
Он может проверить форму файлов, ориентацию `ROI x time` или `time x ROI`,
нулевые и константные ROI, temporal QC и baseline Pearson FC с FDR.

Этот сценарий не проверяет voxel-wise функциональную однородность внутри ROI,
не доказывает корректность наложения атласа на BOLD, не реконструирует upstream
preprocessing и не является биомаркерным или диагностическим анализом. Ветка
`roi_level_gsr` является ROI-level approximation, а не строгим voxel-wise GSR.

## HDF5/fMRI-like поддержка экспериментальна

HDF5/fMRI-like сценарии, voxel/bin alignment и spatial grid остаются экспериментальными. Spatial binning может уменьшать размерность и стабилизировать расчёты, но одновременно сглаживает локальные различия и зависит от выбранной геометрии, размера бина и способа агрегации.

Spatial grid/binning is an engineering approximation for large voxel-like matrices. It is not a neuroanatomically valid cortical parcellation, grey-matter segmentation, atlas alignment procedure, or surface-based cortical model, and it should not be used as evidence of anatomical localization.

## Spatial and cortical limitations

The current voxel/bin mode operates in volume space and does not reconstruct cortical surface geometry. Spatial adjacency in XYZ coordinates should not be interpreted as anatomical adjacency along the cortical sheet. Due to cortical folding, nearby voxels in Euclidean space may belong to distinct gyri, opposite sulcal banks, mixed tissue boundaries, or functionally different regions.

HCP-MMP1 mask geometry QC in `neweds-fmri-audit` uses only voxel coordinates
and atlas region IDs. Connected components, 6/18/26-neighbour adjacency,
boundary voxel counts and `surface_to_volume_proxy` are volume-space engineering
diagnostics. They are not surface-based cortical adjacency, registration QC or
functional homogeneity QC.

For this reason, voxel/bin-level results are exploratory. Neuroanatomical interpretation requires grey-matter or cortical masking, atlas- or surface-based parcellation, registration QC, and ROI homogeneity QC.

## fMRI ROI audit sensitivity outputs

The `neweds-fmri-audit` conservative bad ROI baseline remains the primary
descriptive path. Threshold bad ROI sensitivity sets, Welch t-test edge tables,
permutation summaries and static PNG figures are exploratory robustness checks.
They are not classifier outputs, diagnostic scores, clinical biomarkers, or
evidence of voxel-wise functional homogeneity.

Threshold bad ROI sensitivity may recompute FC and subject-level summaries under
less strict ROI exclusion rules. Those tables are secondary robustness artifacts:
the conservative common-bad-ROI baseline remains the primary result to report.

Figures such as `fc_delta_matrix_HC_vs_SZ.png`,
`significant_edges_network.png`, temporal QC plots, and
`hcp_region_size_distribution.png` are report aids. HCP geometry figures remain
volume-space engineering diagnostics and do not establish surface/cortical
adjacency or functional validity.

## Качество данных и параметры важны

Результаты зависят от качества входных данных, количества наблюдений, пропусков, выбросов, выбранных метрик и параметров preprocessing. Для серьёзного анализа стоит сохранять конфигурацию запуска, фиксировать версии зависимостей, проверять устойчивость результатов на нескольких наборах параметров и сравнивать вывод с независимыми baseline-подходами.
