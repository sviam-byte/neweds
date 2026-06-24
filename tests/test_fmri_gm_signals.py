from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from neweds.core.fmri_gm_signals import (
    AtlasDefinition,
    VoxelRecoveryConfig,
    _ica_one,
    _pca_pc1,
    assign_regions,
    connected_components_6,
    correlation_core_signal,
    load_aal3_atlas,
    load_hcp_atlas,
    read_whole_brain_roi_csv,
    recover_voxel_coordinates,
    timeseries_digest,
)


def _write_voxel_csv(
    path: Path,
    rows: np.ndarray,
    xyz: np.ndarray,
    *,
    subject: str = "1",
    group: str = "HC",
) -> None:
    frame = pd.DataFrame(rows, columns=[f"t{i}" for i in range(rows.shape[1])])
    frame.insert(0, "z", xyz[:, 2])
    frame.insert(0, "y", xyz[:, 1])
    frame.insert(0, "x", xyz[:, 0])
    frame["subject"] = subject
    frame["group"] = group
    frame.to_csv(path, index=False, encoding="cp1251")


def _write_h5(path: Path, rows: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("GM/data", data=rows.astype(np.float32))


def test_numeric_aal_header_is_not_a_data_row(tmp_path: Path) -> None:
    path = tmp_path / "aal.csv"
    array = np.arange(3 * 4, dtype=float).reshape(3, 4)
    pd.DataFrame(array).to_csv(path, index=False)
    loaded = read_whole_brain_roi_csv(path, expected_nodes=3, expected_timepoints=4)
    np.testing.assert_array_equal(loaded, array.astype(np.float32))


def test_text_hcp_header_is_not_a_data_row(tmp_path: Path) -> None:
    path = tmp_path / "hcp.csv"
    array = np.arange(3 * 4, dtype=float).reshape(3, 4)
    pd.DataFrame(array, columns=["TP_1", "TP_2", "TP_3", "TP_4"]).to_csv(path, index=False)
    loaded = read_whole_brain_roi_csv(path, expected_nodes=3, expected_timepoints=4)
    np.testing.assert_array_equal(loaded, array.astype(np.float32))


def test_full_float32_digest_is_stable() -> None:
    values = np.linspace(-1, 1, 600, dtype=np.float64)
    assert timeseries_digest(values) == timeseries_digest(values.astype(np.float32))
    changed = values.astype(np.float32)
    changed[-1] += np.float32(0.25)
    assert timeseries_digest(values) != timeseries_digest(changed)


def test_recovery_matches_exact_rows_and_keeps_zero_unresolved(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    rows = rng.normal(size=(3, 6)).astype(np.float32)
    h5_rows = np.vstack([rows[0], np.zeros(6, dtype=np.float32), rows[1], rows[2]])
    h5 = tmp_path / "1_HC.h5"
    csv = tmp_path / "1_voxels.csv"
    output = tmp_path / "mapping.parquet"
    _write_h5(h5, h5_rows)
    _write_voxel_csv(csv, rows, np.asarray([[1, 2, 3], [2, 2, 3], [3, 2, 3]]))
    result = recover_voxel_coordinates(
        VoxelRecoveryConfig(
            subject_id="1",
            group="HC",
            tissue_h5=str(h5),
            voxel_csv=str(csv),
            output_parquet=str(output),
            expected_timepoints=6,
            csv_block_size=1024,
        )
    )
    assert result.status == "ok"
    assert result.n_matched_rows == 3
    assert result.n_zero_rows == 1
    mapping = pd.read_parquet(output)
    assert mapping.loc[1, "status"] == "unresolved_zero_signal"
    assert pd.isna(mapping.loc[1, "x"])


def test_duplicate_nonzero_hash_blocks_recovery(tmp_path: Path) -> None:
    row = np.arange(6, dtype=np.float32)
    h5 = tmp_path / "1_HC.h5"
    csv = tmp_path / "1_voxels.csv"
    _write_h5(h5, np.vstack([row, row]))
    _write_voxel_csv(csv, row[None], np.asarray([[1, 2, 3]]))
    result = recover_voxel_coordinates(
        VoxelRecoveryConfig(
            subject_id="1",
            group="HC",
            tissue_h5=str(h5),
            voxel_csv=str(csv),
            output_parquet=str(tmp_path / "mapping.parquet"),
            expected_timepoints=6,
            csv_block_size=1024,
        )
    )
    assert result.status == "blocked_recovery"
    assert result.n_ambiguous_hashes == 1
    assert result.n_unmatched_rows == 2


def test_unordered_source_rows_are_detected(tmp_path: Path) -> None:
    rows = np.asarray(
        [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
        dtype=np.float32,
    )
    h5 = tmp_path / "1_HC.h5"
    csv = tmp_path / "1_voxels.csv"
    _write_h5(h5, rows)
    _write_voxel_csv(csv, rows[::-1], np.asarray([[1, 0, 0], [0, 0, 0]]))
    result = recover_voxel_coordinates(
        VoxelRecoveryConfig(
            subject_id="1",
            group="HC",
            tissue_h5=str(h5),
            voxel_csv=str(csv),
            output_parquet=str(tmp_path / "mapping.parquet"),
            expected_timepoints=6,
            csv_block_size=1024,
        )
    )
    assert result.status == "blocked_recovery"
    assert result.monotonic_source_order is False


def test_hcp_node_order_and_coordinate_join(tmp_path: Path) -> None:
    ids = [*range(1, 181), *range(201, 381)]
    frame = pd.DataFrame(
        {
            "x": np.arange(360),
            "y": 0,
            "z": 0,
            "region_id": ids,
            "region_name": [f"R{value}" for value in ids],
        }
    )
    path = tmp_path / "hcp.csv"
    frame.to_csv(path, index=False)
    atlas = load_hcp_atlas(path)
    assert atlas.n_nodes == 360
    assert atlas.node_table["region_id"].tolist() == ids
    mapping = pd.DataFrame(
        {
            "status": ["matched", "unresolved_zero_signal"],
            "x": pd.array([0, None], dtype="Int64"),
            "y": pd.array([0, None], dtype="Int64"),
            "z": pd.array([0, None], dtype="Int64"),
        }
    )
    assert assign_regions(mapping, atlas).tolist() == [1, -1]


def test_aal_loader_preserves_nonconsecutive_indices(tmp_path: Path) -> None:
    indices = [0, *range(1, 167)]
    indices[-1] = 170
    volume = np.zeros((91, 109, 91), dtype=np.int16)
    for offset, value in enumerate(indices[1:], start=1):
        x = offset % 91
        y = (offset // 91) % 109
        volume[x, y, 0] = value
    nifti = tmp_path / "aal.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), nifti)
    lut = tmp_path / "aal.xml"
    labels = "".join(
        f"<label><index>{value}</index><name>R{position}</name></label>"
        for position, value in enumerate(indices[1:], start=1)
    )
    lut.write_text(f"<atlas><data>{labels}</data></atlas>", encoding="utf-8")
    atlas = load_aal3_atlas(nifti, lut)
    assert atlas.n_nodes == 167
    assert atlas.node_table.iloc[-1]["region_id"] == 170


def test_pca_and_ica_are_oriented_to_nonnegative_mean_correlation() -> None:
    t = np.linspace(0, 4 * np.pi, 120)
    base = np.sin(t)
    values = np.vstack([base, 1.1 * base + 0.02 * np.cos(t), 0.9 * base])
    reference = values.mean(axis=0)
    pca = _pca_pc1(values, reference)
    ica = _ica_one(values, reference, seed=7)
    assert np.corrcoef(pca, reference)[0, 1] >= 0
    assert np.corrcoef(ica, reference)[0, 1] >= 0


def test_correlation_core_cannot_cross_a_mask_gap() -> None:
    t = np.linspace(0, 3 * np.pi, 100)
    base = np.sin(t)
    values = np.vstack([base] * 6)
    xyz = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]]
    )
    components = connected_components_6(xyz)
    assert [len(component) for component in components] == [3, 3]
    signal, details = correlation_core_signal(values, xyz)
    assert signal.shape == (100,)
    assert details["core_voxels"] == 3


def test_atlas_definition_reports_node_count() -> None:
    atlas = AtlasDefinition(
        atlas_id="tiny",
        display_name="Tiny",
        node_table=pd.DataFrame(
            {"node_index": [0, 1], "region_id": [0, 9], "region_name": ["B", "R"]}
        ),
        source_files=(),
        source_sha256={},
    )
    assert atlas.n_nodes == 2
