# NewEDS: Time-Series and fMRI Connectivity Toolkit

NewEDS is a Python toolkit for multivariate time-series connectivity analysis.

It demonstrates how an exploratory research prototype can be refactored into a layered, testable analysis tool with:

- a central metric registry;
- reproducible CLI runs;
- structured result contracts;
- HTML/Excel reporting;
- legacy isolation from the portfolio-facing pipeline.

## What this project does

For ordinary multivariate time-series data, NewEDS loads a CSV/Excel file, preprocesses the series, computes selected connectivity metrics, and writes human-readable reports.

The repository still contains a legacy GUI-compatible engine.
The portfolio-facing path is the modern CLI/core pipeline.

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

Run a simple time-series analysis:

```bash
neweds demo.csv \
  --variants correlation_full,dcor_full,ordinal_full \
  --output-dir outputs/demo
```

## Architecture

```text
interfaces/
  cli.py                  public command-line entry point
  legacy_cli.py           compatibility entry point for old workflow
  gui.py, web.py          legacy interactive interfaces

src/core/
  pipeline.py             public orchestration layer
  metric_runner.py        metric execution boundary
  results.py              structured result contracts
  data_loader.py          input loading and preprocessing

src/metrics/
  registry.py             canonical metric registry
  connectivity.py         metric implementations

src/reporting/
  html_generator.py       HTML report generation
  excel_writer.py         Excel report generation
```

The portfolio-facing path is `interfaces/cli.py -> src.core.pipeline.run_analysis -> src.metrics.registry -> AnalysisResult -> reporting`.

## Public API

The reviewed public API is the modern CLI/core path above.

Legacy compatibility is retained in:

- `interfaces/legacy_cli.py`
- `interfaces/gui.py`
- `interfaces/web.py`
- `src/core/engine.py`

Those modules are intentionally not the main portfolio surface.

## Implemented Metric Families

- Pearson correlation and partial correlation
- Distance correlation
- Mutual information and partial mutual information
- Coherence
- Granger-style lagged tests
- Transfer-entropy-style metrics
- Ordinal/permutation-pattern connectivity
- Directed lagged variants for selected metrics

## Portfolio Focus

This project is meant to show how research code can be turned into a clearer analytical toolkit:

- interfaces separated from computation;
- a public pipeline separated from legacy orchestration;
- explicit `AnalysisResult` / `MetricResult` result contracts;
- reproducible metadata attached to each run;
- reporting built on top of structured outputs.

## Implemented Now

- Modern CLI entry point for repeatable local runs
- Metric registry and explicit variant selection
- Structured `AnalysisResult`-based reporting pipeline
- HTML and Excel report generation
- Legacy engine retained but isolated from the main public flow

## Planned Next

- Synthetic validation scenarios with known ground truth
- fMRI-style spatial binning and canonical bin alignment
- Group-level connectivity comparison workflows

## Current Limitations

- The modern public pipeline currently executes metrics with a fixed user-specified lag.
- The legacy GUI/web path is retained for compatibility and is not the main public API.
- Some advanced metrics are exploratory and parameter-sensitive.
- `ah_*` metrics remain legacy-only and are not advertised in the modern public presets.
- Validation and generic fMRI group-comparison modules are not yet part of the portfolio-facing public workflow.
