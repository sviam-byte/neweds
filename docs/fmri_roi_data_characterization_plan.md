# fMRI ROI Data Characterization Plan

This artifact describes the already-extracted ROI time-series data and the
first analysis layer we want before running broad connectivity-metric scans.
It is a planning and audit document, not an execution log.

## Scope

We are working with fMRI ROI time series that have already been extracted by an
upstream atlas-based workflow. NewEDS starts from those regional time-series
CSV files. At this stage we do not reconstruct raw fMRI preprocessing, atlas
registration, voxel-to-ROI extraction, motion correction, or surface geometry.

The immediate goal is to characterize the data themselves:

- what files and subjects are present;
- what atlas each file belongs to;
- what orientation and shape each matrix has;
- which ROI are zero, constant, missing, unstable, or distributionally unusual;
- what temporal structure is visible before connectivity metrics are trusted;
- what frequency-domain structure is present;
- how HC and SZ differ at the signal-QC level before edge-wise FC testing.

Connectivity metrics and lag/window scans should be treated as the second
layer. The first layer is descriptive signal audit and dashboard construction.

## Known Data Shape

The expected directory layout is group-based:

```text
Group_HC/
  <subject_id>_AAL3_timeseries.csv
  <subject_id>_HCP_timeseries.csv
  ...

Group_SZ/
  <subject_id>_AAL3_timeseries.csv
  <subject_id>_HCP_timeseries.csv
  ...
```

The currently known AAL3 example shape is:

```text
167 x 600
```

Interpretation:

- 167 rows are ROI or atlas rows;
- 600 columns are time points;
- the raw file can therefore be `ROI x time`;
- internally NewEDS normalizes valid matrices to `time x ROI`.

The same logic must also support transposed files:

```text
600 x 167
```

Those are interpreted as `time x ROI`.

Any file where neither dimension matches the expected atlas ROI count is not
silently used. It must be marked as a shape/orientation problem.

## Atlas Families

Filename-based atlas detection currently recognizes:

- `AAL3`, `AAL_3`, `AAL3.2` as `AAL3`;
- `HCP`, `Glasser`, `MMP` as `HCP`;
- anything else as `unknown`.

Known AAL3 expectation:

```text
AAL3 expected ROI count: 167
```

Known HCP/Glasser expectations are less rigid because different exported
variants may include or omit background/non-cortical rows:

```text
HCP expected ROI counts: 360, 361, or 379
```

AAL3 and HCP/Glasser outputs must not be mixed at the ROI-index level. They are
separate node definitions and should produce separate inventories,
per-atlas QC, per-atlas figures, and per-atlas downstream connectivity runs.

## Important Indexing Caveat

For AAL3, a region-label file such as `aal3_regions.txt` may include a
background row. Therefore the pipeline must explicitly track:

- raw CSV row index;
- zero-based internal ROI index;
- one-based human ROI number;
- atlas label ID, if known;
- region name, if a mapping file is available;
- whether background is included or excluded.

We should not assume that row `i` in the CSV automatically equals line `i` in
the atlas label file without a mapping report.

## Known Problematic ROI Pattern

In the observed AAL3 example, these rows were fully zero:

```text
0-based indices: 34, 35, 80, 81
1-based ROI numbers: 35, 36, 81, 82
```

The first characterization run should verify whether this is:

- a single-subject issue;
- a repeated subject-level issue;
- group-specific;
- atlas-wide and systematic;
- caused by background/label mismatch;
- caused by extraction/mask coverage.

These ROI must not silently enter correlations or lagged metrics, because zero
variance creates undefined or misleading connectivity values.

## What We Can and Cannot Claim

We can claim:

- these are already-extracted regional time series;
- we audited file shapes, orientations, values, missingness, zero/constant ROI,
  temporal structure, frequency content, and group-level descriptive summaries;
- we identified ROI and subjects requiring exclusion or sensitivity analysis.

We cannot claim from ROI CSV files alone:

- voxel-wise ROI homogeneity;
- correct atlas overlay on anatomy;
- correct surface/cortical adjacency;
- clinical biomarker status;
- diagnostic validity;
- that ROI-level GSR equals strict voxel-wise GSR.

## Stage 1 Goal

Stage 1 should produce a data-characterization atlas, not a biomarker result.

The central question is:

```text
What kind of signals are these, and which parts are safe enough to enter
connectivity analysis?
```

The output should be readable as a dashboard:

- inventory of subjects/files;
- ROI health maps;
- subject health maps;
- group-level distribution summaries;
- temporal structure maps;
- frequency/spectral maps;
- warnings and exclusion recommendations;
- a concise narrative report.

## Stage 1 Inputs

Required inputs:

- HC directory;
- SZ directory;
- output directory;
- atlas filter or `all`;
- optional AAL3 region-name file;
- optional HCP/Glasser geometry metadata, if available.

Optional configuration:

- max ACF/PACF lag;
- trend model type;
- FFT/spectral band definitions;
- zero/constant variance threshold;
- missingness threshold;
- outlier threshold;
- whether to generate static figures;
- whether to generate an HTML dashboard.

## Stage 1 Output Structure

Recommended output tree:

```text
outputs/data_characterization/
  inventories/
    data_inventory.csv
    data_inventory.md
    atlas_region_mapping_report.csv

  qc/
    roi_qc_long.csv
    roi_qc_by_subject.csv
    roi_qc_by_region.csv
    subject_qc_summary.csv
    common_bad_rois_<atlas>.csv
    roi_exclusion_recommendations_<atlas>.csv

  distributions/
    signal_distribution_long.csv
    signal_distribution_by_roi.csv
    signal_distribution_by_subject.csv
    group_distribution_comparison.csv

  temporal/
    trend_qc_long.csv
    acf_qc_long.csv
    pacf_qc_long.csv
    ar_coefficients_long.csv
    temporal_qc_group_summary.csv

  spectral/
    fft_power_long.csv
    bandpower_by_roi.csv
    spectral_slope_by_roi.csv
    dominant_frequency_by_roi.csv
    spectral_group_summary.csv

  figures/
    <atlas>/
      inventory_overview.png
      missingness_heatmap.png
      zero_constant_roi_heatmap.png
      roi_std_heatmap.png
      roi_mean_distribution_by_group.png
      roi_variance_distribution_by_group.png
      trend_slope_heatmap.png
      acf_lag1_heatmap.png
      ar1_distribution_by_group.png
      fft_bandpower_heatmap.png
      dominant_frequency_distribution.png

  dashboards/
    data_characterization_dashboard.html

  reports/
    data_characterization_report.md
```

## Inventory Layer

The inventory answers:

- how many files were found per group;
- how many valid files per atlas;
- how many files were excluded by atlas filter;
- how many files failed to load;
- how many files had shape/orientation errors;
- how many subjects appear in both groups or collide by ID;
- whether HC and SZ have comparable file counts;
- whether time length is consistently 600;
- whether ROI count is consistent within each atlas.

Required columns:

```text
file_path
file_name
group
subject_id
atlas
shape_raw
n_rows
n_cols
orientation
n_regions
n_timepoints
n_nan
n_inf
n_zero_rows
n_zero_cols
n_constant_regions
status
error
```

Useful derived summaries:

- count by `group x atlas x status`;
- shape distribution by atlas;
- timepoint distribution by atlas;
- ROI-count distribution by atlas;
- number of usable subjects per group/atlas.

## ROI Health Layer

For each subject and ROI, compute:

- mean;
- median;
- min;
- max;
- standard deviation;
- variance;
- median absolute deviation;
- interquartile range;
- zero fraction;
- missing fraction;
- finite fraction;
- constant flag;
- all-zero flag;
- extreme-value flag;
- outlier fraction;
- linear trend slope;
- trend R2;
- absolute trend slope;
- basic signal validity label.

Suggested validity labels:

```text
ok
low_variance
zero
constant
missing
extreme
trend_heavy
review
```

The conservative bad-ROI rule for baseline FC should remain:

```text
If an ROI is zero or constant in any subject for a given atlas, exclude that
ROI from the conservative common FC set for that atlas.
```

But Stage 1 should also create softer review categories, because a useful
dashboard should separate "definitely unusable" from "statistically suspicious".

## Subject Health Layer

For each subject, summarize:

- number of ROI;
- number of valid ROI;
- number of zero ROI;
- number of constant ROI;
- number of missing-heavy ROI;
- median ROI standard deviation;
- median ROI absolute trend slope;
- median AR(1);
- global mean signal mean;
- global mean signal standard deviation;
- average absolute ROI correlation if computed;
- spectral low-frequency power fraction;
- subject-level warning count.

This gives a quick way to identify subjects that may dominate group differences
because of signal quality rather than functional organization.

## Signal Distribution Layer

For each ROI and subject:

- mean;
- standard deviation;
- skewness;
- kurtosis;
- selected quantiles, for example 1%, 5%, 25%, 50%, 75%, 95%, 99%;
- robust scale, such as IQR and MAD;
- min/max range;
- z-score outlier fraction;
- winsorized or clipped-value count if preprocessing has already clipped.

Group summaries should compare HC vs SZ at the descriptive level:

- median per-ROI mean by group;
- median per-ROI variance by group;
- difference in ROI variance between groups;
- difference in outlier fraction between groups;
- distribution shifts visible before connectivity.

These summaries are not clinical claims. They are checks for whether the two
groups differ in signal scale, noise, drift, or extraction artifacts.

## Trend Layer

For each subject and ROI:

- ordinary least-squares linear trend slope;
- trend intercept;
- trend R2;
- trend p-value if useful;
- robust trend estimate if added later;
- detrended variance ratio;
- sign of trend;
- absolute trend magnitude.

Group maps:

- median trend slope per ROI in HC;
- median trend slope per ROI in SZ;
- SZ minus HC trend difference;
- fraction of subjects with positive trend per ROI;
- fraction of subjects with high absolute trend per ROI.

Questions answered:

- are some ROI dominated by slow drift;
- are trend-heavy ROI concentrated in one atlas region group;
- are trends group-specific;
- would detrending be a harmless preprocessing branch or a major intervention.

## Autocorrelation Layer

For each subject and ROI:

- ACF at lags such as `1, 2, 3, 5, 10, 20, 30`;
- PACF at lags such as `1, 2, 3, 5, 10`;
- AR(1) coefficient;
- AR(2) coefficients;
- optional Ljung-Box summaries for selected lags;
- integrated autocorrelation proxy if needed.

Group maps:

- median ACF lag 1 per ROI;
- median ACF lag 5 per ROI;
- median AR(1) per ROI;
- HC/SZ AR(1) distribution comparison;
- ROI ranked by strongest autocorrelation;
- subjects ranked by strongest median autocorrelation.

Why this matters:

High autocorrelation can inflate apparent connectivity, affect lagged metrics,
and make window-level scans look more stable than they really are. This should
be visible before running causal or lag-sensitive metrics.

## Fourier and Spectral Layer

For each subject and ROI:

- FFT power spectrum;
- total spectral power;
- low-frequency power;
- mid-frequency power;
- high-frequency power;
- low/high power ratio;
- dominant frequency;
- spectral centroid;
- spectral entropy;
- spectral slope estimate;
- line-noise-like peaks if sampling metadata support this later.

Because TR/sampling frequency may not be known from the ROI CSV alone, the first
version can report frequencies in normalized units or cycles per sample. If TR
is provided later, the same tables can be interpreted in Hz.

Recommended first-pass bands in normalized cycles/sample:

```text
very_low: 0.000 - 0.010
low:      0.010 - 0.050
mid:      0.050 - 0.150
high:     0.150 - 0.500
```

These are engineering bands until TR is known. Reports must say this clearly.

Spectral outputs should answer:

- are signals dominated by very slow drift;
- do some subjects have abnormal high-frequency power;
- do HC and SZ differ in spectral distribution before connectivity;
- do zero/constant ROI appear as expected in spectral power maps;
- which ROI are unsafe for lag/window scans due to poor spectral behavior.

## Missing, Infinite, and Non-Finite Values

Every table should distinguish:

- true zeros;
- missing values;
- infinite values;
- non-numeric rows/columns removed during loading;
- header/index artifacts.

The data loader currently drops fully non-numeric rows/columns and can drop an
integer-index header row. The characterization report should state when this
happened.

## Dashboard Cards

The dashboard should contain compact cards for:

- total files discovered;
- usable files;
- HC subjects;
- SZ subjects;
- atlas families present;
- expected versus observed ROI counts;
- expected versus observed time length;
- zero ROI count;
- constant ROI count;
- common bad ROI count;
- subjects with warning flags;
- most problematic ROI;
- strongest trend ROI;
- strongest autocorrelation ROI;
- strongest low-frequency power ROI.

## Dashboard Figures

Recommended first dashboard figures:

- file status bar chart;
- shape/orientation table;
- subject-by-ROI zero/constant heatmap;
- subject-by-ROI standard deviation heatmap;
- ROI missingness heatmap;
- mean and variance distribution by group;
- trend slope heatmap;
- ACF lag-1 heatmap;
- AR(1) distribution by group;
- FFT bandpower heatmap;
- dominant-frequency distribution by group;
- ROI warning count map;
- subject warning count map.

For atlas datasets with known region names, figure labels should use region
names in hover text or companion CSV tables, not necessarily on dense static
axes.

## Map Definitions

In this context, "maps" means tabular and visual maps across:

- subject x ROI;
- group x ROI;
- atlas x ROI;
- ROI x QC metric;
- ROI x temporal metric;
- ROI x spectral metric.

These are not anatomical surface maps unless a proper anatomical/surface
mapping layer is provided later.

## Group Comparison at Stage 1

Stage 1 can compare HC and SZ descriptively on QC features, for example:

- ROI mean;
- ROI variance;
- zero/constant frequency;
- trend slope;
- ACF/AR coefficients;
- spectral bandpower;
- subject warning counts.

Recommended statistics:

- median difference;
- Mann-Whitney U for nonparametric comparison;
- rank-biserial effect size;
- Benjamini-Hochberg FDR across ROI/features;
- optional Welch t-test as sensitivity only.

Interpretation must remain conservative:

```text
These are signal-quality and temporal-structure differences, not disease
biomarkers.
```

## Relation to Stage 2 Metric Scans

Only after Stage 1 should we run the broad metric grid:

```text
metric x lag x window_size x window_start x atlas x preprocessing_branch
```

Stage 1 should decide:

- which ROI are excluded in the conservative set;
- which ROI are retained but flagged;
- which subjects are retained but flagged;
- which preprocessing branches are justified;
- which lags are plausible;
- which window sizes are too short for stable estimates;
- whether spectral/trend structure makes some metrics risky.

## Compute Safety Notes

There is an unrelated long-running Python/joblib process already on the
machine. We do not stop it and do not modify it.

For planning purposes, the characterization stage should be designed so it can
run with explicit worker limits:

```powershell
$env:TS_TOOL_N_JOBS="4"
$env:TS_TOOL_PARALLEL_BACKEND="threading"
```

The first characterization layer should be mostly linear in:

```text
subjects x ROI x timepoints
```

With around 100 ROI and 600 timepoints per subject, this is expected to be much
lighter than an exhaustive metric x lag x window grid. The heavier parts are
FFT summaries, ACF/PACF, and any optional pairwise ROI or within-region
homogeneity computations.

## Recommended Execution Order

1. Inventory all HC/SZ files.
2. Normalize valid matrices to `time x ROI`.
3. Build atlas mapping diagnostics.
4. Compute ROI-level value QC.
5. Compute subject-level QC.
6. Compute distribution summaries.
7. Compute trend summaries.
8. Compute ACF/PACF/AR summaries.
9. Compute FFT and bandpower summaries.
10. Build bad-ROI and review-ROI recommendations.
11. Build static figures.
12. Build dashboard.
13. Write narrative report.
14. Only then start broad connectivity metric scans.

## Open Decisions

Before implementation or full run, decide:

- exact HC/SZ input directories;
- whether AAL3, HCP, or both are in scope for the first run;
- whether TR is known;
- which FFT bands to use if TR is known;
- desired ACF/PACF max lag;
- whether to generate HTML dashboard in the first pass or only CSV/PNG/Markdown;
- whether to keep all ROI in descriptive tables even when excluded from FC;
- whether to run group-level QC statistics in Stage 1 or keep it descriptive.

## Minimal Stage 1 Success Criteria

Stage 1 is successful if it produces:

- a reliable file inventory;
- clear shape/orientation decisions;
- a list of unusable ROI;
- a list of suspicious ROI;
- per-subject QC summaries;
- trend and autocorrelation tables;
- spectral summaries;
- figures that make zero/constant ROI, drift, autocorrelation, and spectral
  dominance visible;
- a report that says what data are safe to pass into Stage 2 and what must be
  treated as sensitivity-only.

