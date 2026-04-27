# NewEDS

**NewEDS** — reproducible Python toolkit for connectivity analysis of multivariate time series and group fMRI comparisons.

The project demonstrates how a research prototype evolves into a layered, testable analytical pipeline with a strict public API: lazy plugin registry, structured result contracts, ground-truth tests, HTML/Excel reports, and an honest baseline group-comparison pipeline.

---

## What is neweds?

A reproducible Python package for analysis of multivariate time series and functional connectivity:

- **input** — CSV / Excel / Parquet / MAT / HDF5 (single-file mode); CSV / Excel / Parquet directories (batch mode)
- **preprocessing** — explicit, configurable: missing-value fill, normalization, outlier handling, AR-detrend
- **connectivity metrics** — 24 metrics across correlation / information / causal / spectral / ordinal categories, registered via a plugin registry
- **confound control** — explicit `controls` columns; metric metadata distinguishes precision-matrix partials from explicit-controls residualization
- **reports** — HTML + Excel with the full computation contract attached
- **group comparison** — baseline edge-wise pipeline (`neweds-group`), explicitly marked experimental

---

## Why it exists

Connectivity research code typically grows into a monolith where everything imports everything. NewEDS keeps:

- a layered import graph (`import neweds` does not pull `statsmodels` until a Granger metric is requested)
- structured, frozen result contracts (`AnalysisResult`, `MetricResult`, `ComputationContract`) so each number is paired with how it was produced
- ground-truth tests (VAR(1), independent series, lagged copies) that check whether each metric actually does what it advertises

---

## Quickstart

```bash
pip install -e ".[dev]"
pytest

# single-file analysis
neweds examples/demo_timeseries.csv \
    --variants correlation_full,dcor_full,ordinal_full \
    --output-dir outputs/demo

# baseline group comparison (experimental)
neweds-group \
    --case-dir examples/group/case \
    --control-dir examples/group/control \
    --output-dir outputs/group_demo \
    --spatial-grid-size 10
```

After the single-file run, `outputs/demo/` contains `report.html` and `report.xlsx`.

Programmatic API:

```python
from neweds import AnalysisConfig, run_analysis

result = run_analysis(
    "examples/demo_timeseries.csv",
    AnalysisConfig(
        variants=["correlation_full", "correlation_directed"],
        max_lag=3,
        lag_selection="optimize",
        # explicit preprocessing — surfaces in ComputationContract.preprocess_steps
        preprocess=True,
        normalize=True,
        fill_missing=True,
        remove_outliers=True,
        ar_order=0,
    ),
)

for name, metric in result.metrics.items():
    print(name, metric.matrix.shape, "lag=", metric.lag)
    print(metric.contract.summary_text())
```

---

## Architecture

```
neweds/
├── cli.py                    public CLI (time-series)
├── cli_group.py              CLI for group fMRI comparison (experimental)
├── config.py                 AnalysisConfig, ComputationContract
├── methods.py                lazy facade over the metric registry
├── core/
│   ├── pipeline.py           run_analysis (public entry point)
│   ├── batch_pipeline.py     batch mode + manifest + zip
│   ├── group_pipeline.py     fMRI group-wise comparison + GroupComparisonResult
│   ├── metric_runner.py      computation boundary → registry
│   ├── results.py            AnalysisResult / MetricResult / WindowResult
│   ├── data_loader.py        I/O orchestration (CSV / Excel / Parquet / MAT / HDF5)
│   ├── preprocessing.py      normalization, missing fill, outliers
│   ├── voxel_space.py        canonical voxel space for fMRI
│   └── window_scanner.py     sliding-window scanning (joblib)
├── metrics/
│   ├── registry.py           plugin registry (decorator + dataclass + lazy bootstrap)
│   ├── _shared.py            shared math utilities
│   ├── correlation.py        Pearson / Spearman / Kendall / partial / lagged-directed / H²
│   ├── information.py        KSG MI, distance correlation, Arnhold-H ratio
│   ├── causal.py             Granger F-test, Transfer Entropy
│   ├── spectral.py           coherence (full + partial)
│   ├── ordinal.py            Bandt-Pompe permutation MI
│   └── connectivity.py       backward-compat shim (re-exports)
├── reporting/                HTML / Excel generators
├── io/                       loaders (HDF5, user input)
├── analysis/                 dimred / graph / stats helpers
├── validation/               synthetic ground-truth scenarios
└── visualization/            heatmap / connectome / FFT plots
```

Data flow:

```
CLI ─► run_analysis ─► load_or_generate ─► metric registry (lazy)
                                               │
                            ComputationContract │
                                               ▼
                                       AnalysisResult ─► HTML / Excel
```

---

## Supported metrics

| Category    | Metrics                                                                         |
| ----------- | ------------------------------------------------------------------------------- |
| correlation | `correlation_full`, `correlation_spearman`, `correlation_kendall`, `correlation_partial`, `correlation_directed`, `h2_full`, `h2_partial`, `h2_directed` |
| information | `mutinf_full`, `mutinf_partial`, `dcor_full`, `dcor_partial`, `dcor_directed`, `ah_full`, `ah_partial`, `ah_directed`, `te_full`, `te_partial` |
| causal      | `granger_full`, `granger_partial`                                               |
| spectral    | `coherence_full`, `coherence_partial`                                           |
| ordinal     | `ordinal_full`, `ordinal_directed`                                              |

Use `from neweds.metrics import list_metrics; list_metrics()` to inspect categories, descriptions, and flags (`directed`, `pvalue_based`, `supports_control`, `experimental`, `stable`, `partial_mode`).

### Partial-mode semantics

`*_partial` metrics carry a `partial_mode` field that disambiguates *what* "partial" means:

- `precision_matrix` — `correlation_partial`, `h2_partial`. Conditioned on **all other variables** via the inverse correlation matrix (no explicit controls needed).
- `explicit_controls_residualization` — `mutinf_partial`, `dcor_partial`, `coherence_partial`, `granger_partial`, `te_partial`, `ah_partial`. Each pair is computed on residuals after regressing out the user-provided `controls`.

The chosen mode is stored in `MetricResult.contract.partial_mode` and in `MetricResult.metadata["partial_mode"]`, so downstream consumers can tell which interpretation applies.

---

## Data formats

- **single-file**: CSV, Excel, Parquet, MAT, HDF5 (`.h5`/`.hdf5`/`.hdf`); HDF5 is treated as 4D fMRI and reduced via spatial binning.
- **directory batch**: CSV / Excel / Parquet only.
- **group comparison**: subject-wise CSV / Excel / Parquet after spatial binning. HDF5 group input is **not** supported yet.

CSV encoding is auto-probed (utf-8, utf-8-sig, cp1251, cp1252, latin1).

---

## Controls / confounds

Pass control columns either via `AnalysisConfig.controls=[...]` or via the `--controls` CLI flag. They are removed from the signal block before metric computation and made available to `*_partial` metrics that support `explicit_controls_residualization`.

The active strategy and the resolved column list are recorded on every `ComputationContract`.

---

## Group comparison (experimental)

`neweds-group` performs an edge-wise Mann-Whitney U test on connectivity features extracted from a canonical voxel space, with Benjamini-Hochberg FDR correction.

```bash
neweds-group --case-dir data/case --control-dir data/control \
             --output-dir results/group --spatial-grid-size 10
```

The CLI prints an explicit `[experimental]` notice on start. The returned summary contains `design_metadata`, `warnings`, and `experimental: True`. A typed `GroupComparisonResult` dataclass is available via `from neweds.core.group_pipeline import GroupComparisonResult; GroupComparisonResult.from_summary(summary)`.

---

## Limitations

- The group pipeline currently supports **baseline edge-wise comparison only** (Mann-Whitney + BH-FDR). Covariate-aware GLM, permutation tests, and site-aware design are on the roadmap, not implemented.
- HDF5 group input is not supported by `neweds-group` (subject-wise CSV/Excel/Parquet only).
- Several metrics (`mutinf_*`, `te_*`, `ah_*`, `dcor_directed`, `ordinal_directed`) are computationally expensive and marked `experimental`. They should be validated on synthetic data before drawing conclusions.
- Directed metrics depend on careful preprocessing: enabling AR-detrend (`ar_order > 0`) can suppress lag structure that those metrics try to detect.
- `correlation_partial` (precision-matrix variant) requires `n_rows > n_cols + 2`; otherwise it returns NaN and emits a warning.
- Distance correlation falls back from O(N log N) (`dcor` package) to O(N²) with subsampling at N > 5000 — the result will note `_subsampled = True`.
- `data_loader.py` is intentionally large (3000+ lines) — splitting it is on the roadmap for the next refactor.

---

## Roadmap

- Split `data_loader.py` into `io/tabular.py`, `io/mat.py`, `io/voxel.py`, `io/hdf5.py`.
- Group pipeline v2: covariates, permutation GLM, site-aware design, effect sizes + confidence intervals.
- Extended ground-truth scenarios (Lorenz, Rössler, NARMA).
- Group `*_partial` metrics behind a unified control-aware framework.
- Result export to Parquet/Arrow.

---

## Key code locations for review

- [neweds/metrics/registry.py](neweds/metrics/registry.py) — plugin registry with decorator support, `partial_mode` metadata, lazy `ensure_builtins()`.
- [neweds/metrics/correlation.py](neweds/metrics/correlation.py), [information.py](neweds/metrics/information.py), [causal.py](neweds/metrics/causal.py), [spectral.py](neweds/metrics/spectral.py), [ordinal.py](neweds/metrics/ordinal.py) — per-category implementations.
- [neweds/core/pipeline.py](neweds/core/pipeline.py) — public entry point; lag-selection via matrix scoring, builds `ComputationContract`.
- [neweds/core/group_pipeline.py](neweds/core/group_pipeline.py) — group comparison + `GroupComparisonResult`.
- [neweds/core/results.py](neweds/core/results.py) — structured result contracts.
- [neweds/config.py](neweds/config.py) — `AnalysisConfig` (with explicit preprocessing flags), `ComputationContract`.
- [tests/test_metric_ground_truth.py](tests/test_metric_ground_truth.py) — VAR(1) / independent / lagged-copy scenarios.
- [tests/test_pipeline_snapshot.py](tests/test_pipeline_snapshot.py) — numerical regression of the public pipeline.
- [tests/test_cli_integration.py](tests/test_cli_integration.py) — subprocess-level CLI smoke test.

---

## Development

```bash
ruff check .
ruff format --check .
pytest --cov=neweds --cov-report=term-missing
```

CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs `ubuntu-latest × windows-latest × Python 3.11/3.12`.

---

## License

MIT.
