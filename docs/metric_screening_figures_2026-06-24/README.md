# Wide HC/SZ metric screening figures

Generated: `2026-06-24T20:31:52.157089+03:00`
Source result: `outputs\run_paths\new_results\2026-06-24_metric-classifier-wide-no-leak-screening`

## Figures

- `figures/metrics/01_metric_overall_ranking.png` - Exploratory per-metric ranking
- `figures/metrics/02_metric_scope_heatmap.png` - Metric by scope heatmap
- `figures/metrics/03_nested_vs_label_shuffle_null.png` - Nested performance vs label-shuffle null
- `figures/metrics/04_nested_selected_metrics.png` - Nested train-fold metric choices
- `figures/metrics/05_window_lag_grid.png` - Best score by window and lag
- `figures/preanalysis/01_subject_coverage.png` - Subject coverage by scope
- `figures/preanalysis/02_input_dimensions.png` - Input dimensions by scope
- `figures/preanalysis/03_qc_global_std_ar1.png` - QC global distributions
- `figures/preanalysis/04_valid_nodes.png` - Valid nodes by scope
- `figures/preanalysis/05_metric_compute_time.png` - Compute time by metric

## Interpretation guardrails

- The per-metric ranking figures are exploratory because the best task is selected after seeing the grid.
- The nested-vs-null figure is the leakage-audited performance view.
- All real HC/SZ figures here use 600 time points, not the separate 1000-point HDF5 voxel files.
- The 1000-point files require a separate ROI/voxel preprocessing adapter before they can be compared to these runs.
