# fMRI ROI Audit MVP

`neweds-fmri-audit` is an experimental pipeline for auditing already-extracted
ROI time series from two groups, `Group_HC` and `Group_SZ`. It is intended as a
first-pass data audit before broader connectivity experiments.

The command answers a narrow question: what is present in the ROI time-series
files, which ROI are unusable for baseline FC, what temporal structure is
visible, and which baseline Pearson FC edges differ between HC and SZ after FDR
correction.

It does not reconstruct upstream preprocessing, validate atlas overlay quality,
test voxel-wise functional homogeneity inside ROI, or produce a clinical
biomarker.

## Inputs

The MVP expects one CSV per subject in each group directory:

```text
Group_HC/
  subject_001_AAL3_timeseries.csv
  subject_002_AAL3_timeseries.csv

Group_SZ/
  subject_101_AAL3_timeseries.csv
  subject_102_AAL3_timeseries.csv
```

Supported atlas detection in filenames:

- `AAL3`, `AAL_3`, or `AAL3.2` -> `AAL3`;
- `HCP`, `Glasser`, or `MMP` -> `HCP`;
- anything else is marked `unknown` and excluded from atlas-level FC.

AAL3 orientation detection expects `167` ROI. Files shaped `167 x T` are read
as `ROI x time`; files shaped `T x 167` are read as `time x ROI`. Files where
neither dimension matches the expected ROI count are marked `shape_error`.

Optional metadata inputs:

- `aal3_regions.txt` for AAL3 row-to-region mapping diagnostics;
- `HCP-MMP1_atlas_voxel_map_from_xml.csv` for HCP volume-space geometry QC.

## CLI

Minimal run:

```bash
neweds-fmri-audit \
  --hc-dir data/raw/Group_HC \
  --sz-dir data/raw/Group_SZ \
  --output-dir outputs/fmri_roi_audit \
  --aal3-regions data/metadata/aal3_regions.txt \
  --atlas AAL3
```

Run all sensitivity branches:

```bash
neweds-fmri-audit \
  --hc-dir data/raw/Group_HC \
  --sz-dir data/raw/Group_SZ \
  --output-dir outputs/fmri_roi_audit \
  --atlas all \
  --include-sensitivity
```

Run optional statistical and reporting sensitivity outputs:

```bash
neweds-fmri-audit \
  --hc-dir data/raw/Group_HC \
  --sz-dir data/raw/Group_SZ \
  --output-dir outputs/fmri_roi_audit \
  --atlas AAL3 \
  --make-figures \
  --bad-roi-thresholds 0.05,0.10,0.20 \
  --include-ttest \
  --include-permutation \
  --n-permutations 1000 \
  --random-seed 0
```

Include HCP mask geometry QC:

```bash
neweds-fmri-audit \
  --hc-dir data/raw/Group_HC \
  --sz-dir data/raw/Group_SZ \
  --output-dir outputs/fmri_roi_audit \
  --hcp-voxel-map data/metadata/HCP-MMP1_atlas_voxel_map_from_xml.csv
```

## Python API

```python
from neweds import run_fmri_roi_audit

result = run_fmri_roi_audit(
    "data/raw/Group_HC",
    "data/raw/Group_SZ",
    "outputs/fmri_roi_audit",
    aal3_regions="data/metadata/aal3_regions.txt",
    atlas_filter="AAL3",
    branches=("raw_cleaned",),
)

print(result.as_dict())
```

The return value is `FmriRoiAuditResult`, a compact summary with subject counts,
atlases, preprocessing branches, bad ROI counts, tested edges, significant edges,
warnings, and output directory.

## Processing Contract

The pipeline runs the following steps:

1. Inventory subject CSV files and infer group, subject ID, atlas, shape,
   orientation, NaN/Inf counts, zero rows/columns, constant ROI, and file status.
2. Normalize valid ROI time-series matrices internally to `time x ROI`.
3. Build AAL3 region mapping diagnostics when AAL3 data are present.
4. Compute per-subject, per-ROI QC metrics: mean, standard deviation, variance,
   min, max, median, MAD, zero fraction, NaN fraction, linear trend slope/R2,
   and basic flags.
5. Build a conservative common bad ROI set: any ROI that is zero or constant in
   any subject is excluded from FC for all subjects in the same atlas.
6. Compute temporal QC: trend, ACF lags `1,2,3,5,10`, PACF lags `1,2,3`, AR(1),
   AR(2) coefficients in the long table, and compact HC/SZ summaries.
7. Run preprocessing branches. The MVP always supports `raw_cleaned`; optional
   sensitivity branches are `detrended`, `ar1_residualized`, and
   `roi_level_gsr`.
8. Compute Pearson FC matrices, apply Fisher z-transform, vectorize upper
   triangles, compare HC vs SZ with Mann-Whitney U, compute rank-biserial effect
   size, and apply Benjamini-Hochberg FDR.
9. Compute subject-level FC summaries from each Fisher-z matrix:
   `mean_fc`, `mean_abs_fc`, positive/negative edge fractions, FC variance,
   global strength, and fixed threshold densities for `abs(z) >= 0.2`, `0.4`,
   and `0.6`.
10. Compare HC vs SZ on the subject-level summary metrics with Mann-Whitney U,
    rank-biserial effect size, and FDR.
11. Optionally write sensitivity outputs: threshold-based bad ROI alternative FC
    summaries, Welch t-test edge comparisons, exploratory permutation summaries
    for subject-level metrics, and static PNG figures.
12. Write a final pilot report with explicit limitations.

`roi_level_gsr` is an ROI-level global signal approximation. It is not strict
voxel-wise GSR.

## Outputs

Inventory:

```text
outputs/inventories/data_inventory.csv
outputs/inventories/data_inventory.md
outputs/inventories/aal3_region_mapping_report.csv
```

ROI and temporal QC:

```text
outputs/qc/roi_timeseries_qc_long.csv
outputs/qc/roi_timeseries_qc_by_subject.csv
outputs/qc/roi_timeseries_qc_by_region.csv
outputs/qc/common_bad_rois_<atlas>.csv
outputs/temporal/temporal_qc_long.csv
outputs/temporal/temporal_qc_group_summary.csv
```

FC and group comparison:

```text
outputs/preprocessed/<atlas>/<branch>/<group>_<subject_id>.npy
outputs/fc_matrices/<atlas>/<branch>/<group>_<subject_id>_pearson_z.npy
outputs/fc_edges/<atlas>/<branch>/fc_edges_long.csv
outputs/group_comparison/<atlas>/<branch>/fc_group_comparison_edges.csv
outputs/group_comparison/<atlas>/<branch>/fc_group_comparison_edges_ttest.csv
outputs/group_comparison/<atlas>/<branch>/subject_level_fc_summary.csv
outputs/group_comparison/<atlas>/<branch>/subject_level_group_comparison.csv
outputs/group_comparison/<atlas>/<branch>/permutation_summary.csv
outputs/group_comparison/<atlas>/<branch>/summary.md
```

The Mann-Whitney edge table is the primary edge-wise comparison. The
`fc_group_comparison_edges_ttest.csv` file is opt-in sensitivity output, not a
replacement for the default result. `permutation_summary.csv` is an exploratory
subject-level sensitivity table with deterministic random seed support.

Sensitivity outputs:

```text
outputs/sensitivity/<atlas>/threshold_0_05/common_bad_rois_<atlas>.csv
outputs/sensitivity/<atlas>/threshold_0_10/common_bad_rois_<atlas>.csv
outputs/sensitivity/<atlas>/threshold_0_20/common_bad_rois_<atlas>.csv
outputs/sensitivity/<atlas>/threshold_0_05/<branch>/preprocessed/<group>_<subject_id>.npy
outputs/sensitivity/<atlas>/threshold_0_05/<branch>/fc_matrices/<group>_<subject_id>_pearson_z.npy
outputs/sensitivity/<atlas>/threshold_0_05/<branch>/fc_edges_long.csv
outputs/sensitivity/<atlas>/threshold_0_05/<branch>/fc_group_comparison_edges.csv
outputs/sensitivity/<atlas>/threshold_0_05/<branch>/subject_level_fc_summary.csv
outputs/sensitivity/<atlas>/threshold_0_05/<branch>/subject_level_group_comparison.csv
```

The conservative common-bad-ROI baseline remains the default FC path. Threshold
bad ROI outputs recompute an alternative FC/statistics path for review, but they
are exploratory sensitivity artifacts and do not replace the conservative
baseline. Subject output filenames include the group label to avoid collisions
when HC and SZ directories contain the same subject ID.

Figures:

```text
outputs/figures/<atlas>/<branch>/acf_profiles_by_group.png
outputs/figures/<atlas>/<branch>/ar1_distribution_HC_vs_SZ.png
outputs/figures/<atlas>/<branch>/trend_distribution_HC_vs_SZ.png
outputs/figures/<atlas>/<branch>/fc_delta_matrix_HC_vs_SZ.png
outputs/figures/<atlas>/<branch>/significant_edges_network.png
outputs/figures/HCP_geometry/hcp_region_size_distribution.png
```

Figures are static PNG summaries generated with matplotlib when available. They
are descriptive audit artifacts and should be read together with the CSV tables,
not as standalone evidence.

Final report:

```text
reports/final_pilot_report.md
```

Optional HCP mask geometry QC:

```text
outputs/qc/hcp_mmp1_mask_geometry_report.csv
outputs/qc/hcp_region_size_report.csv
outputs/qc/hcp_region_adjacency_report.csv
```

These HCP outputs describe volume-space atlas geometry only. They do not assess
functional homogeneity and they do not reconstruct surface-based cortical
adjacency.

`hcp_region_adjacency_report.csv` contains per-region diagnostics for 6-, 18-,
and 26-neighbour volume connectivity: connected component counts, neighbouring
region IDs, boundary voxel counts, and `surface_to_volume_proxy`. The proxy is
`boundary_voxel_count / n_voxels`; it is an engineering geometry flag, not a
neuroanatomical surface measurement.

`small_regions` in the geometry summary counts regions with fewer than 10 voxels.
That threshold is a simple engineering flag for review, not a validity criterion
for anatomical or functional interpretation.

## Interpretation Rules

Correct wording:

> We audited already-extracted ROI time series, identified problematic ROI,
> summarized temporal structure, and ran baseline Pearson FC group comparison
> with FDR correction.

Incorrect wording:

- "We found a schizophrenia biomarker."
- "We proved these atlas regions correctly match cortex."
- "We tested voxel-wise functional homogeneity inside ROI."
- "ROI-level GSR is equivalent to strict voxel-wise GSR."

The MVP is an engineering and methodology audit. Downstream scientific claims
need independent statistical design, covariates, confound checks, motion/site
controls, and external validation.

Subject-level FC summaries are descriptive group diagnostics. They are easier to
read than edge-wise tables, but they are not a classifier, diagnostic score, or
network-system interpretation. Lobe/system summaries are intentionally deferred
until a reliable ROI-to-system mapping is available.

Sensitivity analyses are secondary. Threshold bad ROI sets, Welch t-test tables,
permutation summaries, and figures help inspect robustness, but they do not
create biomarker evidence and they do not make the pipeline a classifier.
