# fMRI Audit Separation Contract

NewEDS treats the two available fMRI data products as separate audit scopes.
Their tables, reports, and scientific limitations must not be silently merged.

## Audit A: whole-brain / extracted ROI data

Canonical output directory:

```text
whole_brain_roi_audit/
```

This audit may contain:

- subject/file inventory;
- ROI shape and orientation checks;
- ROI value, trend, ACF/PACF, AR, and spectral QC;
- atlas-label mapping diagnostics;
- bad/review ROI decisions;
- descriptive HC/SZ comparisons of ROI-level QC features.

It must not claim:

- voxel-wise ROI homogeneity;
- correct tissue segmentation;
- valid cortical adjacency from volume XYZ;
- strict voxel-wise GSR.

## Audit B: GM/WM/CSF tissue HDF5 data

Canonical output directory:

```text
tissue_gm_wm_csf_audit/
```

This audit may contain:

- HDF5 schema and metadata inventory;
- GM/WM/CSF voxel counts and zero/constant/non-finite QC;
- active tissue-mean and combined global time series;
- ACF/PACF and AR1 before/after diagnostics;
- tissue-mean correlations;
- exploratory HC/SZ comparisons of tissue QC features with FDR.

It must not contain:

- ROI connectivity matrices or edge tables;
- atlas-region conclusions without voxel-to-region mapping;
- spatial neighbourhood or homogeneity claims when `*/xyz` is absent.

## Cross-dataset checks

Any explicit comparison between the two scopes must be written to a third,
separate directory:

```text
cross_dataset_checks/
```

Examples:

- subject-ID overlap;
- time-axis alignment;
- ROI-global versus GM-global correlation;
- availability of WM/CSF nuisance regressors for ROI sensitivity branches.

Cross-dataset checks do not change the evidence level of either source audit.

## Transcript-derived methodology requirements

The supplied meeting transcript adds the following requirements:

1. Do not divide folded cortex into unconstrained cubic volume bins.
2. Spatial growth must respect valid-mask boundaries and stop at background.
3. Atlas membership is only a candidate node definition, not proof of
   functional homogeneity.
4. Within-region pairwise voxel correlation distributions should be inspected
   before a regional signal is trusted.
5. Mean, sign-oriented PCA, ICA, and correlation-selected voxel subsets are
   competing regional-signal constructions.
6. PCA/ICA sign orientation must be explicit before signed connectivity is
   compared across subjects. Sign-invariant `r²` may be reported as sensitivity.
7. Trend/stationarity, ACF, PACF, and AR removal must be reported before and
   after preprocessing.
8. AR1 is a required sensitivity baseline; AR2-AR4 remain additional
   diagnostics rather than automatic defaults.
9. Global signal must be analysed in paired branches, with and without
   regression.
10. Group-level metric/edge scans require multiple-comparison correction.
11. Lag × window × metric grids are exploratory and require bounded compute.

When coordinates or voxel-to-region membership are absent, requirements 1-6
must be marked `blocked`; they must not be simulated with ordinary cubic bins.
