# GM voxel recovery and regional signals

`neweds-fmri-gm-signals` restores GM voxel coordinates by exact matching of all
600 float32 samples. It never assigns a coordinate to an all-zero HDF5 row.

The recovery sidecar is ZSTD Parquet and contains one record per `GM/data` row:
`matched`, `unresolved_zero_signal`, or `unmatched_nonzero`. Any non-zero
miss, ambiguous digest, duplicate coordinate, subject/group mismatch, or
non-monotonic source order blocks that subject.

HCP-MMP1 uses a direct `(x,y,z)` join and node IDs
`1…180, 201…380`. AAL3v2 is fail-closed: its 91×109×91 image, 167-entry LUT,
local names, checksums, and reconstruction of ready ROI signals must all pass
before any AAL GM signal is emitted.

Four raw and z-standardized signals are stored:

- `active_mean`;
- `pca_pc1_oriented`;
- `ica_1_oriented`;
- `correlation_core`.

The stage does not compute connectivity, permutation tests, or HC/SZ
classification.
