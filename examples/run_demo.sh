#!/usr/bin/env bash
set -euo pipefail

neweds examples/demo_timeseries.csv \
  --variants correlation_full,dcor_full,ordinal_full \
  --output-dir outputs/examples_demo
