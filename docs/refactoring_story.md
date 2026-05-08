# Refactoring story

NewEDS started as a compact research script and was later split into a small Python package for tabular time-series connectivity analysis.

## Initial state

- loading, preprocessing, metric execution and reporting lived in one flow;
- report generation depended on an older result shape;
- group comparison used domain-specific labels;
- fallback handling was broad in several exploratory branches.

## Current state

- the single-file tabular pipeline is the stable entry point;
- metric metadata and lookup live in the registry;
- case/control naming is neutral in the group layer;
- formula evaluation is isolated behind a restricted evaluator;
- tests cover the public API, CLI, loader behavior and metric regressions.

## Remaining technical debt

- reduce broad exception handling in analysis/reporting modules;
- split the HTML report generator;
- split preprocessing into smaller stages;
- make group pipeline thinner;
- tighten optional dependency handling.
