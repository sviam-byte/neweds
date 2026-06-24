# Independent GM/WM/CSF Tissue Audit

The tissue audit reads subject HDF5 files containing:

```text
GM/data
WM/data
CSF/data
```

Each dataset must have shape `voxels × time`. The audit is separate from the
whole-brain/ROI audit and never writes ROI edge or connectivity results.

## CLI

```powershell
neweds-fmri-tissue-audit `
  --input-dir "D:\data\Серое, белое и СМЖ" `
  --output-dir "D:\audits\tissue_gm_wm_csf_audit"
```

The reader is streaming: voxel matrices are processed in row blocks and are not
loaded into memory as one full multi-tissue array.

## Outputs

```text
tissue_gm_wm_csf_audit/
├── audit_manifest.json
├── inventories/
│   └── tissue_hdf5_inventory.csv
├── qc/
│   ├── tissue_dataset_qc.csv
│   ├── tissue_voxel_counts_wide.csv
│   └── tissue_audit_failures.csv
├── temporal/
│   ├── tissue_mean_timeseries.csv
│   ├── tissue_mean_temporal_qc.csv
│   ├── tissue_mean_acf_pacf.csv
│   └── tissue_mean_correlations.csv
├── group_comparison/
│   └── tissue_feature_group_comparison.csv
└── reports/
    ├── tissue_audit_report.md
    └── transcript_methodology_status.md
```

Fully zero and constant voxel rows are excluded from `active` tissue means.
Both all-voxel and active-voxel summaries are retained in QC tables.

## Spatial limitation

Voxel-wise spatial analysis is allowed only when all three tissue groups contain
matching coordinate arrays:

```text
GM/xyz
WM/xyz
CSF/xyz
```

Each must have shape `N_voxels × 3` and match the corresponding `data` row
count. Without these coordinates, NewEDS does not infer row positions from the
root volume shape and does not substitute unconstrained cubic bins.

## Interpretation

Tissue voxel counts and fractions are QC/confound candidates. They are not
automatically morphometric disease findings. Group comparisons are exploratory,
use Mann-Whitney U with rank-biserial effect size, and are corrected with
Benjamini-Hochberg FDR.
