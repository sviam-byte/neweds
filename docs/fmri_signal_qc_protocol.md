# fMRI-Like Signal QC Protocol

NewEDS treats fMRI-like voxel/bin workflows as experimental methodology work,
not as a validated neuroimaging or clinical pipeline. The purpose of this
protocol is to make node-definition and signal-construction decisions visible
before functional connectivity metrics are compared.

## Main Goal

The pipeline should evaluate how node definition, regional signal construction,
temporal preprocessing, and connectivity metric choice affect functional
connectivity results. The goal is not to produce a biomarker. The goal is to
make the preprocessing and metric-comparison steps explicit, auditable, and
reproducible.

In short: first check what the network nodes and regional signals mean, then
compare connectivity metrics.

For already-extracted HC/SZ ROI time series, the first implemented MVP entry
point is [`neweds-fmri-audit`](fmri_roi_audit.md). That command starts after ROI
extraction: it inventories subject files, checks ROI/time-series quality,
summarizes temporal structure, and runs baseline Pearson FC group comparison.
It should not be described as voxel-wise ROI homogeneity QC or atlas overlay
validation.

The intended audit chain is:

```text
cortical/grey-matter region definition
  -> ROI/bin homogeneity QC
  -> regional signal extraction
  -> temporal QC
  -> functional connectivity
  -> group/statistical analysis
```

## 1. Define Valid Spatial/Anatomical Units

The first question is not how to average an ROI. The first question is what
counts as a valid candidate node of the network.

Volume-space XYZ proximity is not the same as cortical adjacency. Because the
cortex is folded, two voxels can be close in Euclidean space while belonging to
different gyri, opposite sulcal banks, different functional tissue, or a
grey/white/boundary mixture. For that reason, ordinary cubic bins are only an
engineering baseline.

A spatial/anatomical validity layer should record:

- which mask defines valid tissue: brain, grey matter, cortical mask, or atlas
  validity mask;
- whether voxel adjacency is unconstrained or constrained by a valid mask;
- whether candidate bins/parcels cross mask boundaries;
- how much of each region is inside the valid mask;
- whether a region is too small, sparse, elongated, or hole-like to be trusted.

Current NewEDS helpers are volume-space only. `neweds.analysis.mask_qc` can
summarize mask coverage and per-region mask overlap. `neweds.analysis.spatial_adjacency`
can build 6/18/26-neighbor adjacency constrained by a mask. These helpers do not
reconstruct cortical surface geometry and should not be described as
surface-based cortical adjacency.

## 2. Choose a Parcellation / Node-Definition Mode

Different node-definition modes answer different questions:

- Grid/binning: fast, deterministic, useful as an engineering baseline, but not
  neuroanatomically valid cortical parcellation.
- Atlas parcels: more comparable between subjects and easier to relate to
  literature, but atlas membership does not prove subject-level functional
  homogeneity.
- Functional parcels: can follow subject-specific synchronous dynamics, but are
  harder to align across subjects and can leak information if learned outside a
  proper train/CV protocol.
- Surface-based parcels: better respect cortical folding, but require external
  surface reconstruction and registration pipelines such as FreeSurfer/HCP-style
  preprocessing.

NewEDS should treat atlas parcels and spatial bins as candidate nodes, not as
automatically valid analysis units.

## 3. Check Whether a Region Can Be Averaged

An atlas ROI or spatial bin does not guarantee that all voxels inside it carry
the same time-series structure. A naive mean can fail when the region contains
opposite-phase subgroups, weakly related signals, many constant/invalid series,
or local patterns that cancel out in the average.

Before a region-level signal is used, compute homogeneity diagnostics:

- number of input series and active non-constant series;
- median pairwise correlation inside the region;
- median absolute pairwise correlation;
- fraction of negative pairwise correlations;
- fraction of weak pairwise correlations, using `abs(r) < 0.2`;
- median correlation between the aggregate mean signal and each active series;
- `aggregation_risk` as `ok`, `weak`, or `bad`.

Regions marked `bad` should not silently enter downstream FC analysis. Regions
marked `weak` should be kept only when the report makes that uncertainty clear.

## 4. Compare Regional Signal Construction Methods

Connectivity results should be compared across region-signal construction
choices, not only across FC metrics. Useful variants include:

- mean signal;
- median or robust mean;
- PC1;
- sign-oriented PC1;
- future ICA or selected-voxel variants.

The scientific question is whether group differences and edge-level conclusions
change when the voxel-to-region compression changes. This is often more
important than a simple Pearson-versus-Granger comparison.

## 5. Handle PCA Sign Ambiguity

PCA components have arbitrary sign. The same component can be returned as `PC1`
or `-PC1` without changing the mathematical decomposition. For sign-sensitive
connectivity metrics, this is dangerous if the orientation rule is not recorded.

The conservative NewEDS rule is:

- by default, keep the historical PCA sign unchanged with `pca_orient_sign="none"`;
- when `pca_orient_sign="mean_corr"` is enabled, orient PC1 so that its
  correlation with the mean standardized regional signal is non-negative;
- if that correlation is undefined, optionally fall back to loading-sum
  orientation;
- store `pca_orient_sign`, `orientation_rule`, `sign_flip`,
  `corr_pc1_mean`, and `pc1_loading_sum` in PCA metadata.

Unoriented PCA/ICA components should not be compared directly with signed FC
metrics unless the report explicitly states that sign ambiguity remains.

## 6. Diagnose Temporal Structure Before AR Removal

Autocorrelation should be diagnosed before it is removed. Otherwise the pipeline
can hide the reason why a preprocessing choice was made.

A signal-QC run should report:

- trend diagnostics;
- stationarity checks when enabled;
- ACF over the requested lags;
- PACF over the requested lags;
- Ljung-Box summaries over multiple lags;
- before/after diagnostics only when AR removal is actually enabled.

This protects the analysis from interpreting shared slow drift or residual
autocorrelation as functional connectivity.

## 7. Treat GSR as a Sensitivity Branch

Global signal regression should be a paired sensitivity analysis, not a dogma.
For the same dataset and region-signal construction, run both branches:

- without GSR;
- with GSR or with the global signal included as an explicit control.

The report should show which edges and metrics are most sensitive to GSR, which
edges change sign, and which group/classification effects persist or disappear.

## 8. Compare FC Metrics Only After Signal QC

After region construction and temporal QC are visible, compare FC metrics in a
factorial matrix:

```text
node definition x region signal method x preprocessing x GSR mode x FC metric x group task
```

This makes it possible to separate meaningful FC differences from artifacts
introduced by mask choice, spatial binning, atlas parcellation, averaging,
temporal preprocessing, or global signal handling.

## Limitations

NewEDS does not make clinical claims, diagnostic claims, or anatomical
localization claims. Spatial grid/binning is an engineering approximation for
large voxel-like matrices. It is not a cortical parcellation, grey-matter
segmentation, atlas alignment procedure, or surface-based neuroanatomical model.
