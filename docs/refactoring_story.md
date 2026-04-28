# Refactoring story

This project started as a compact exploratory analysis script and is being gradually refactored into a cleaner scientific Python package.

## Initial state

- monolithic analysis flow;
- mixed loading, preprocessing, metrics and reporting;
- domain-specific group labels;
- report generation coupled to an older result structure;
- broad fallback handling in exploratory branches.

## Current state

- stable single-file connectivity pipeline;
- metric registry as the source of truth;
- neutral case/control group layer;
- isolated restricted formula evaluator;
- split documentation;
- tests for public API, CLI and metric behavior.

## Remaining technical debt

- reduce broad exception handling in analysis/reporting modules;
- split the HTML report generator;
- split preprocessing into smaller stages;
- make group pipeline thinner;
- tighten optional dependency handling.

## Why this is kept visible

The repository is intended as a portfolio project: it shows both the final toolkit and the engineering process of turning exploratory scientific code into a more maintainable package.
