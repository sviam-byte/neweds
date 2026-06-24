"""Recover GM voxel geometry and build atlas-wise regional fMRI signals.

The module is intentionally fail-closed.  A subject is eligible for regional
signals only when every non-zero GM row is matched exactly to one voxel CSV
row by a BLAKE2b digest of all float32 time points.  Zero GM rows remain
explicit unresolved QC records and are never assigned invented coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.utils.extmath import randomized_svd

GM_SIGNAL_METHODS = (
    "active_mean",
    "pca_pc1_oriented",
    "ica_1_oriented",
    "correlation_core",
)
EXPECTED_TIMEPOINTS = 600
HASH_NAME = "blake2b-256-float32-c-order"


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def timeseries_digest(values: np.ndarray) -> bytes:
    """Return a full BLAKE2b-256 digest after canonical float32 conversion."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return hashlib.blake2b(array.tobytes(order="C"), digest_size=32).digest()


def is_numeric_time_header(values: Sequence[Any], n_timepoints: int = 600) -> bool:
    if len(values) != n_timepoints:
        return False
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(numeric, np.arange(n_timepoints, dtype=np.float64)))


def read_whole_brain_roi_csv(
    path: str | Path,
    *,
    expected_nodes: int,
    expected_timepoints: int = EXPECTED_TIMEPOINTS,
) -> np.ndarray:
    """Read ROI CSVs whose numeric ``0..599`` first line is a header."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        first = stream.readline().strip().split(",")
    numeric_header = is_numeric_time_header(first, expected_timepoints)
    has_text = False
    try:
        np.asarray(first, dtype=np.float64)
    except ValueError:
        has_text = True
    header = 0 if numeric_header or has_text else None
    frame = pd.read_csv(path, header=header)
    array = frame.to_numpy(dtype=np.float32)
    if array.shape != (expected_nodes, expected_timepoints):
        raise ValueError(
            f"{path}: expected {(expected_nodes, expected_timepoints)}, got {array.shape}"
        )
    return array


@dataclass(frozen=True, slots=True)
class VoxelRecoveryConfig:
    subject_id: str
    group: str
    tissue_h5: str
    voxel_csv: str
    output_parquet: str
    block_rows: int = 2048
    csv_block_size: int = 64 * 1024 * 1024
    expected_timepoints: int = EXPECTED_TIMEPOINTS
    resume: bool = True


@dataclass(frozen=True, slots=True)
class VoxelRecoveryResult:
    subject_id: str
    group: str
    status: str
    tissue_h5: str
    voxel_csv: str
    output_parquet: str
    h5_sha256: str
    voxel_csv_sha256: str
    hash_method: str
    n_gm_rows: int
    n_nonzero_rows: int
    n_zero_rows: int
    n_matched_rows: int
    n_unmatched_rows: int
    n_ambiguous_hashes: int
    n_duplicate_coordinates: int
    monotonic_source_order: bool
    coverage_nonzero: float
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AtlasDefinition:
    atlas_id: str
    display_name: str
    node_table: pd.DataFrame
    source_files: tuple[str, ...]
    source_sha256: dict[str, str]
    volume: np.ndarray | None = None
    coordinate_table: pd.DataFrame | None = None
    validation_status: str = "not_validated"
    validation_details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(len(self.node_table))


@dataclass(frozen=True, slots=True)
class RegionalSignalResult:
    subject_id: str
    group: str
    atlas_id: str
    status: str
    n_nodes: int
    n_timepoints: int
    signal_npz: str
    homogeneity_table: str
    method_status_table: str
    coverage: float
    unresolved_rows: int
    node_order: tuple[int, ...]
    method_metadata: dict[str, Any]
    input_sha256: dict[str, str]
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["node_order"] = list(self.node_order)
        return data


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _status_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".status.json")


def _load_resumed_recovery(config: VoxelRecoveryConfig) -> VoxelRecoveryResult | None:
    status_path = _status_path(Path(config.output_parquet))
    if not config.resume or not status_path.is_file() or not Path(config.output_parquet).is_file():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        result = VoxelRecoveryResult(**data)
        if result.status == "ok":
            return result
        metadata_only = (
            result.status == "blocked_recovery"
            and result.n_unmatched_rows == 0
            and result.n_ambiguous_hashes == 0
            and result.n_duplicate_coordinates == 0
            and result.monotonic_source_order
            and result.message.startswith("group metadata mismatch:")
        )
        if metadata_only:
            corrected = replace(result, status="ok", message="")
            _json_dump(status_path, corrected.as_dict())
            return corrected
        if result.status == "blocked_recovery":
            return result
        return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _h5_hash_index(
    h5_path: Path,
    *,
    expected_timepoints: int,
    block_rows: int,
) -> tuple[dict[bytes, int], list[int], set[bytes], int]:
    digest_to_row: dict[bytes, int] = {}
    zero_rows: list[int] = []
    ambiguous: set[bytes] = set()
    with h5py.File(h5_path, "r") as h5:
        if "GM/data" not in h5:
            raise KeyError(f"{h5_path}: missing GM/data")
        data = h5["GM/data"]
        if data.ndim != 2 or int(data.shape[1]) != expected_timepoints:
            raise ValueError(
                f"{h5_path}: GM/data expected (*,{expected_timepoints}), got {data.shape}"
            )
        n_rows = int(data.shape[0])
        for start in range(0, n_rows, max(1, block_rows)):
            block = np.asarray(data[start : start + block_rows], dtype=np.float32)
            for offset, row in enumerate(block):
                h5_row = start + offset
                if not np.isfinite(row).all():
                    raise ValueError(f"{h5_path}: non-finite GM row {h5_row}")
                if np.all(row == 0):
                    zero_rows.append(h5_row)
                    continue
                digest = timeseries_digest(row)
                if digest in digest_to_row:
                    ambiguous.add(digest)
                else:
                    digest_to_row[digest] = h5_row
    for digest in ambiguous:
        digest_to_row.pop(digest, None)
    return digest_to_row, zero_rows, ambiguous, n_rows


def _voxel_csv_batches(
    path: Path,
    *,
    expected_timepoints: int,
    block_size: int,
) -> Iterable[Any]:
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError as exc:  # pragma: no cover - exercised by optional dependency installs
        raise RuntimeError("pyarrow is required for streaming voxel CSV recovery") from exc

    time_columns = [f"t{i}" for i in range(expected_timepoints)]
    with path.open("r", encoding="cp1251", errors="strict") as stream:
        available = set(stream.readline().rstrip("\r\n").split(","))
    required = {"x", "y", "z", *time_columns}
    missing = required - available
    if missing:
        raise ValueError(f"{path}: missing required CSV columns: {sorted(missing)}")
    optional = [name for name in ("subject", "group") if name in available]
    include = ["x", "y", "z", *time_columns, *optional]
    column_types = {
        "x": pa.int32(),
        "y": pa.int32(),
        "z": pa.int32(),
        **{name: pa.float32() for name in time_columns},
        **{name: pa.string() for name in optional},
    }
    read = pacsv.ReadOptions(block_size=int(block_size), encoding="cp1251", use_threads=True)
    convert = pacsv.ConvertOptions(
        include_columns=include,
        column_types=column_types,
        strings_can_be_null=True,
    )
    reader = pacsv.open_csv(path, read_options=read, convert_options=convert)
    yield from reader


def _normalize_group(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"HC", "H", "N", "Н", "НОРМА", "HEALTHY", "CONTROL", "К"} or "НОРМ" in text:
        return "HC"
    if (
        text in {"SZ", "S", "Ш", "ШЗ", "ШК", "SCHIZOPHRENIA"}
        or "Ш" in text
        or "SCHIZ" in text
    ):
        return "SZ"
    return text


def recover_voxel_coordinates(config: VoxelRecoveryConfig) -> VoxelRecoveryResult:
    """Recover exact coordinates for one subject and write a ZSTD Parquet sidecar."""
    resumed = _load_resumed_recovery(config)
    if resumed is not None:
        return resumed

    h5_path = Path(config.tissue_h5)
    csv_path = Path(config.voxel_csv)
    output_path = Path(config.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest_to_row, zero_rows, ambiguous_hashes, n_rows = _h5_hash_index(
        h5_path,
        expected_timepoints=config.expected_timepoints,
        block_rows=config.block_rows,
    )
    matched: dict[int, tuple[int, int, int, int]] = {}
    coordinate_owner: dict[tuple[int, int, int], int] = {}
    duplicate_coordinates = 0
    source_row = 0
    subject_values: set[str] = set()
    group_values: set[str] = set()
    time_columns = [f"t{i}" for i in range(config.expected_timepoints)]

    for batch in _voxel_csv_batches(
        csv_path,
        expected_timepoints=config.expected_timepoints,
        block_size=config.csv_block_size,
    ):
        frame = batch.to_pandas()
        xyz = frame[["x", "y", "z"]].to_numpy(dtype=np.int32, copy=False)
        values = frame[time_columns].to_numpy(dtype=np.float32, copy=False)
        if "subject" in frame:
            subject_values.update(
                str(value).strip() for value in frame["subject"].dropna().unique()
            )
        if "group" in frame:
            group_values.update(_normalize_group(value) for value in frame["group"].dropna().unique())
        for local_row, row in enumerate(values):
            digest = timeseries_digest(row)
            h5_row = digest_to_row.get(digest)
            if h5_row is None:
                continue
            coordinate = tuple(int(value) for value in xyz[local_row])
            previous_owner = coordinate_owner.get(coordinate)
            if previous_owner is not None and previous_owner != h5_row:
                duplicate_coordinates += 1
                continue
            if h5_row in matched:
                # A repeated time series in the source is ambiguous even when its HDF5 hash was unique.
                ambiguous_hashes.add(digest)
                continue
            coordinate_owner[coordinate] = h5_row
            matched[h5_row] = (*coordinate, source_row + local_row)
        source_row += len(frame)

    expected_subject = str(config.subject_id)
    subject_ok = not subject_values or all(
        value == expected_subject or value.startswith(f"{expected_subject}_")
        for value in subject_values
    )
    expected_group = str(config.group).upper()
    group_ok = not group_values or group_values == {expected_group}
    matched_rows_sorted = sorted(matched)
    raw_order = [matched[row][3] for row in matched_rows_sorted]
    monotonic = all(left < right for left, right in zip(raw_order, raw_order[1:]))
    n_matched = len(matched)
    n_nonzero = n_rows - len(zero_rows)
    unmatched = n_nonzero - n_matched

    records: list[dict[str, Any]] = []
    zero_set = set(zero_rows)
    for h5_row in range(n_rows):
        if h5_row in zero_set:
            records.append(
                {
                    "subject_id": expected_subject,
                    "group": expected_group,
                    "h5_row": h5_row,
                    "x": None,
                    "y": None,
                    "z": None,
                    "source_row": None,
                    "status": "unresolved_zero_signal",
                    "hash_method": HASH_NAME,
                }
            )
        elif h5_row in matched:
            x, y, z, raw_row = matched[h5_row]
            records.append(
                {
                    "subject_id": expected_subject,
                    "group": expected_group,
                    "h5_row": h5_row,
                    "x": x,
                    "y": y,
                    "z": z,
                    "source_row": raw_row,
                    "status": "matched",
                    "hash_method": HASH_NAME,
                }
            )
        else:
            records.append(
                {
                    "subject_id": expected_subject,
                    "group": expected_group,
                    "h5_row": h5_row,
                    "x": None,
                    "y": None,
                    "z": None,
                    "source_row": None,
                    "status": "unmatched_nonzero",
                    "hash_method": HASH_NAME,
                }
            )
    mapping = pd.DataFrame.from_records(records)
    for column in ("h5_row", "source_row", "x", "y", "z"):
        mapping[column] = pd.array(mapping[column], dtype="Int64")
    mapping.to_parquet(output_path, compression="zstd", index=False)

    blockers: list[str] = []
    if unmatched:
        blockers.append(f"{unmatched} non-zero GM rows were not matched")
    if ambiguous_hashes:
        blockers.append(f"{len(ambiguous_hashes)} ambiguous time-series hashes")
    if duplicate_coordinates:
        blockers.append(f"{duplicate_coordinates} duplicate coordinates")
    if not monotonic:
        blockers.append("source order is not strictly monotonic")
    if not subject_ok:
        blockers.append(f"subject metadata mismatch: {sorted(subject_values)}")
    if not group_ok:
        blockers.append(f"group metadata mismatch: {sorted(group_values)}")
    status = "ok" if not blockers and n_matched == n_nonzero else "blocked_recovery"
    result = VoxelRecoveryResult(
        subject_id=expected_subject,
        group=expected_group,
        status=status,
        tissue_h5=str(h5_path),
        voxel_csv=str(csv_path),
        output_parquet=str(output_path),
        h5_sha256=sha256_file(h5_path),
        voxel_csv_sha256=sha256_file(csv_path),
        hash_method=HASH_NAME,
        n_gm_rows=n_rows,
        n_nonzero_rows=n_nonzero,
        n_zero_rows=len(zero_rows),
        n_matched_rows=n_matched,
        n_unmatched_rows=unmatched,
        n_ambiguous_hashes=len(ambiguous_hashes),
        n_duplicate_coordinates=duplicate_coordinates,
        monotonic_source_order=monotonic,
        coverage_nonzero=float(n_matched / n_nonzero) if n_nonzero else 1.0,
        message="; ".join(blockers),
    )
    _json_dump(_status_path(output_path), result.as_dict())
    return result


def load_hcp_atlas(path: str | Path) -> AtlasDefinition:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"x", "y", "z", "region_id", "region_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
    if frame.duplicated(["x", "y", "z"]).any():
        raise ValueError(f"{path}: duplicate voxel coordinates")
    nodes = (
        frame.loc[frame["region_id"] != 0, ["region_id", "region_name"]]
        .drop_duplicates()
        .sort_values("region_id")
        .reset_index(drop=True)
    )
    expected = [*range(1, 181), *range(201, 381)]
    if nodes["region_id"].astype(int).tolist() != expected:
        raise ValueError("HCP node order must be 1..180,201..380 (360 nodes)")
    nodes.insert(0, "node_index", np.arange(len(nodes), dtype=int))
    coord = frame[["x", "y", "z", "region_id"]].copy()
    return AtlasDefinition(
        atlas_id="HCP-MMP1-360",
        display_name="HCP-MMP1 (360 regions)",
        node_table=nodes,
        coordinate_table=coord,
        source_files=(str(path),),
        source_sha256={str(path): sha256_file(path)},
        validation_status="validated_local_map",
    )


def load_aal3_atlas(
    nifti_path: str | Path,
    lut_path: str | Path,
    *,
    local_regions_path: str | Path | None = None,
) -> AtlasDefinition:
    """Load Nilearn-compatible AAL3v2 NIfTI + XML/LUT and validate 167 nodes."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for AAL3 mapping") from exc

    nifti_path = Path(nifti_path)
    lut_path = Path(lut_path)
    image = nib.load(str(nifti_path))
    if tuple(image.shape) != (91, 109, 91):
        raise ValueError(f"AAL atlas shape must be 91x109x91, got {image.shape}")
    volume = np.asarray(image.dataobj, dtype=np.int16)
    labels: list[tuple[int, str]] = [(0, "Background")]
    if lut_path.suffix.lower() == ".xml":
        root = ET.parse(lut_path).getroot()
        for label in root.iter("label"):
            labels.append(
                (int(label.findtext("index", default="-1")), label.findtext("name", default=""))
            )
    else:
        lut = pd.read_csv(lut_path, sep=None, engine="python")
        index_column = next(
            (name for name in ("index", "region_id", "id", "value") if name in lut.columns),
            None,
        )
        name_column = next(
            (name for name in ("name", "region_name", "label") if name in lut.columns),
            None,
        )
        if index_column is None or name_column is None:
            raise ValueError(f"{lut_path}: LUT needs index/id and name/label columns")
        labels.extend(
            (int(index), str(name))
            for index, name in zip(lut[index_column], lut[name_column])
            if int(index) != 0
        )
    if len(labels) != 167:
        raise ValueError(f"AAL3v2 LUT must contain 167 nodes including background, got {len(labels)}")
    unique_values = set(int(value) for value in np.unique(volume))
    indices = [value for value, _ in labels]
    if set(indices) != unique_values:
        raise ValueError(
            "AAL3v2 NIfTI values and LUT indices differ: "
            f"missing={sorted(set(indices)-unique_values)}, extra={sorted(unique_values-set(indices))}"
        )
    nodes = pd.DataFrame(labels, columns=["region_id", "region_name"])
    nodes.insert(0, "node_index", np.arange(len(nodes), dtype=int))
    sources = [nifti_path, lut_path]
    if local_regions_path is not None:
        local = Path(local_regions_path)
        names = []
        for line in local.read_text(encoding="utf-8-sig").splitlines():
            parts = line.strip().split(maxsplit=1)
            if parts:
                names.append(parts[-1].strip())
        if names != nodes["region_name"].tolist():
            raise ValueError("AAL3 local region-name order does not match the supplied LUT")
        sources.append(local)
    return AtlasDefinition(
        atlas_id="AAL3v2-167",
        display_name="AAL3v2 (167 regions including background)",
        node_table=nodes,
        volume=volume,
        source_files=tuple(str(path) for path in sources),
        source_sha256={str(path): sha256_file(path) for path in sources},
    )


def assign_regions(mapping: pd.DataFrame, atlas: AtlasDefinition) -> np.ndarray:
    """Assign atlas region IDs to recovered matched coordinates."""
    result = np.full(len(mapping), -1, dtype=np.int32)
    matched_mask = mapping["status"].eq("matched").to_numpy()
    matched = mapping.loc[matched_mask, ["x", "y", "z"]].astype(int)
    if atlas.coordinate_table is not None:
        left = matched.reset_index().rename(columns={"index": "_row"})
        joined = left.merge(
            atlas.coordinate_table,
            on=["x", "y", "z"],
            how="left",
            validate="many_to_one",
        )
        if joined["region_id"].isna().any():
            raise ValueError(f"{atlas.atlas_id}: coordinates outside local voxel map")
        result[joined["_row"].to_numpy(dtype=int)] = joined["region_id"].to_numpy(dtype=np.int32)
    elif atlas.volume is not None:
        xyz = matched.to_numpy(dtype=np.intp)
        shape = np.asarray(atlas.volume.shape)
        if np.any(xyz < 0) or np.any(xyz >= shape):
            raise ValueError(f"{atlas.atlas_id}: coordinates outside atlas volume")
        values = atlas.volume[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        result[np.flatnonzero(matched_mask)] = values.astype(np.int32)
    else:
        raise ValueError(f"{atlas.atlas_id}: atlas has neither volume nor coordinate table")
    return result


_NEIGHBOURS = np.asarray(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int16,
)


def connected_components_6(xyz: np.ndarray) -> list[np.ndarray]:
    """Return 6-neighbour components; gaps in an atlas mask cannot be crossed."""
    xyz = np.asarray(xyz, dtype=np.int32)
    if len(xyz) == 0:
        return []
    lookup = {tuple(coord): index for index, coord in enumerate(xyz)}
    unseen = set(range(len(xyz)))
    components: list[np.ndarray] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            for delta in _NEIGHBOURS:
                neighbour = lookup.get(tuple(xyz[current] + delta))
                if neighbour is not None and neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
                    component.append(neighbour)
        components.append(np.asarray(sorted(component), dtype=int))
    components.sort(key=lambda values: (-len(values), int(values[0])))
    return components


def _zscore_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    valid = np.isfinite(std[:, 0]) & (std[:, 0] > 1e-12)
    output = np.full_like(values, np.nan)
    output[valid] = (values[valid] - mean[valid]) / std[valid]
    return output, valid


def _orient_to_reference(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.std(signal) <= 1e-12 or np.std(reference) <= 1e-12:
        return signal
    return -signal if np.corrcoef(signal, reference)[0, 1] < 0 else signal


def _pca_pc1(values: np.ndarray, reference: np.ndarray, seed: int = 1729) -> np.ndarray:
    standardized, valid = _zscore_rows(values)
    matrix = standardized[valid]
    if len(matrix) < 2:
        raise ValueError("requires_at_least_two_nonconstant_voxels")
    _, _, vh = randomized_svd(matrix, n_components=1, random_state=int(seed))
    return _orient_to_reference(vh[0], reference).astype(np.float32)


def _ica_one(values: np.ndarray, reference: np.ndarray, seed: int) -> np.ndarray:
    standardized, valid = _zscore_rows(values)
    matrix = standardized[valid]
    if len(matrix) < 2:
        raise ValueError("requires_at_least_two_nonconstant_voxels")
    n_components = min(8, len(matrix), matrix.shape[1] - 1)
    if n_components < 2:
        raise ValueError("insufficient_rank_for_ica")
    _, singular_values, vh = randomized_svd(
        matrix,
        n_components=n_components,
        random_state=int(seed),
    )
    reduced = vh.T * singular_values
    model = FastICA(
        n_components=1,
        whiten="unit-variance",
        random_state=int(seed),
        max_iter=1000,
        tol=1e-4,
    )
    signal = model.fit_transform(reduced)[:, 0]
    return _orient_to_reference(signal, reference).astype(np.float32)


def correlation_core_signal(values: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    standardized, valid = _zscore_rows(values)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 3:
        raise ValueError("requires_at_least_three_nonconstant_voxels")
    standardized = standardized[valid]
    reference = np.median(standardized, axis=0)
    reference_std = float(np.std(reference))
    if reference_std <= 1e-12:
        raise ValueError("constant_median_standardized_reference")
    reference_z = (reference - reference.mean()) / reference_std
    correlations = standardized @ reference_z / standardized.shape[1]
    center = float(np.median(correlations))
    mad = float(np.median(np.abs(correlations - center)))
    radius = 1.4826 * mad
    selected_local = np.flatnonzero(
        (correlations > 0) & (correlations >= center - radius) & (correlations <= center + radius)
    )
    if len(selected_local) < 3:
        raise ValueError("fewer_than_three_correlation_core_candidates")
    selected_original = valid_indices[selected_local]
    components = connected_components_6(np.asarray(xyz)[selected_original])
    if not components or len(components[0]) < 3:
        raise ValueError("largest_correlation_core_component_has_fewer_than_three_voxels")
    core_rows = selected_original[components[0]]
    return (
        np.asarray(values[core_rows].mean(axis=0), dtype=np.float32),
        {
            "candidate_voxels": int(len(selected_original)),
            "core_voxels": int(len(core_rows)),
            "correlation_median": center,
            "correlation_mad": mad,
        },
    )


def _pairwise_correlation_sample(
    standardized: np.ndarray,
    valid: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> np.ndarray:
    matrix = standardized[valid]
    n = len(matrix)
    pair_count = n * (n - 1) // 2
    if pair_count == 0:
        return np.empty(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if pair_count <= max_pairs:
        left, right = np.triu_indices(n, 1)
    else:
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < max_pairs:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n - 1))
            if b >= a:
                b += 1
            pairs.add((min(a, b), max(a, b)))
        ordered = sorted(pairs)
        left = np.fromiter((pair[0] for pair in ordered), dtype=int)
        right = np.fromiter((pair[1] for pair in ordered), dtype=int)
    return np.sum(matrix[left] * matrix[right], axis=1).astype(np.float32) / matrix.shape[1]


def _standardize_signal_rows(signals: np.ndarray) -> np.ndarray:
    output = np.full_like(signals, np.nan, dtype=np.float32)
    for index, signal in enumerate(signals):
        if np.isfinite(signal).all() and np.std(signal) > 1e-12:
            output[index] = ((signal - signal.mean()) / signal.std()).astype(np.float32)
    return output


def build_regional_signals(
    *,
    subject_id: str,
    group: str,
    tissue_h5: str | Path,
    mapping_parquet: str | Path,
    atlas: AtlasDefinition,
    output_dir: str | Path,
    random_seed: int = 1729,
    max_pairwise_correlations: int = 10_000,
) -> RegionalSignalResult:
    """Create four GM-only signal variants and spatial/homogeneity QC."""
    if atlas.atlas_id.startswith("AAL") and atlas.validation_status != "validated_ready_roi":
        return RegionalSignalResult(
            subject_id=str(subject_id),
            group=str(group),
            atlas_id=atlas.atlas_id,
            status="blocked_atlas_validation",
            n_nodes=atlas.n_nodes,
            n_timepoints=EXPECTED_TIMEPOINTS,
            signal_npz="",
            homogeneity_table="",
            method_status_table="",
            coverage=0.0,
            unresolved_rows=0,
            node_order=tuple(atlas.node_table["region_id"].astype(int)),
            method_metadata={},
            input_sha256=atlas.source_sha256,
            message="AAL atlas was not validated by reconstruction of ready ROI files",
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_parquet(mapping_parquet).sort_values("h5_row").reset_index(drop=True)
    if not mapping["h5_row"].astype(int).equals(pd.Series(np.arange(len(mapping)))):
        raise ValueError("mapping h5_row must be complete and ordered")
    unresolved = int((mapping["status"] != "matched").sum())
    recovery_blockers = int((mapping["status"] == "unmatched_nonzero").sum())
    if recovery_blockers:
        return RegionalSignalResult(
            subject_id=str(subject_id),
            group=str(group),
            atlas_id=atlas.atlas_id,
            status="blocked_recovery",
            n_nodes=atlas.n_nodes,
            n_timepoints=EXPECTED_TIMEPOINTS,
            signal_npz="",
            homogeneity_table="",
            method_status_table="",
            coverage=0.0,
            unresolved_rows=unresolved,
            node_order=tuple(atlas.node_table["region_id"].astype(int)),
            method_metadata={},
            input_sha256={
                str(tissue_h5): sha256_file(tissue_h5),
                str(mapping_parquet): sha256_file(mapping_parquet),
                **atlas.source_sha256,
            },
            message="non-zero GM rows remain unresolved",
        )
    region_for_row = assign_regions(mapping, atlas)
    n_nodes = atlas.n_nodes
    signals = {
        method: np.full((n_nodes, EXPECTED_TIMEPOINTS), np.nan, dtype=np.float32)
        for method in GM_SIGNAL_METHODS
    }
    homogeneity_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    node_ids = atlas.node_table["region_id"].astype(int).to_numpy()
    matched = mapping["status"].eq("matched").to_numpy()
    xyz_all = mapping[["x", "y", "z"]].to_numpy(dtype=float)

    with h5py.File(tissue_h5, "r") as h5:
        data = np.asarray(h5["GM/data"], dtype=np.float32)
        for node_index, region_id in enumerate(node_ids):
            row_indices = np.flatnonzero(matched & (region_for_row == region_id))
            values = data[row_indices]
            xyz = xyz_all[row_indices].astype(np.int32)
            standardized, valid = _zscore_rows(values) if len(values) else (
                np.empty_like(values, dtype=float),
                np.empty(0, dtype=bool),
            )
            components = connected_components_6(xyz)
            largest_fraction = float(len(components[0]) / len(xyz)) if components else math.nan
            reference = values.mean(axis=0) if len(values) else np.full(EXPECTED_TIMEPOINTS, np.nan)
            voxel_reference_corr = (
                standardized[valid]
                @ ((reference - reference.mean()) / reference.std())
                / EXPECTED_TIMEPOINTS
                if len(values) and np.std(reference) > 1e-12 and valid.any()
                else np.empty(0)
            )
            pairwise = _pairwise_correlation_sample(
                standardized,
                valid,
                max_pairs=max_pairwise_correlations,
                seed=random_seed + node_index,
            )
            homogeneity_rows.append(
                {
                    "subject_id": str(subject_id),
                    "group": str(group),
                    "atlas_id": atlas.atlas_id,
                    "node_index": node_index,
                    "region_id": int(region_id),
                    "region_name": str(atlas.node_table.iloc[node_index]["region_name"]),
                    "active_gm_voxels": int(len(values)),
                    "nonconstant_gm_voxels": int(valid.sum()),
                    "components_6n": int(len(components)),
                    "largest_component_fraction": largest_fraction,
                    "voxel_reference_corr_median": float(np.median(voxel_reference_corr))
                    if len(voxel_reference_corr)
                    else math.nan,
                    "voxel_reference_corr_mad": float(
                        np.median(np.abs(voxel_reference_corr - np.median(voxel_reference_corr)))
                    )
                    if len(voxel_reference_corr)
                    else math.nan,
                    "pairwise_corr_n": int(len(pairwise)),
                    "pairwise_corr_median": float(np.median(pairwise))
                    if len(pairwise)
                    else math.nan,
                    "pairwise_corr_q05": float(np.quantile(pairwise, 0.05))
                    if len(pairwise)
                    else math.nan,
                    "pairwise_corr_q95": float(np.quantile(pairwise, 0.95))
                    if len(pairwise)
                    else math.nan,
                }
            )
            for method in GM_SIGNAL_METHODS:
                status = "ok"
                details: dict[str, Any] = {}
                try:
                    if method == "active_mean":
                        if len(values) < 1:
                            raise ValueError("requires_at_least_one_voxel")
                        signal = values.mean(axis=0)
                    elif method == "pca_pc1_oriented":
                        signal = _pca_pc1(values, reference, random_seed + node_index)
                    elif method == "ica_1_oriented":
                        signal = _ica_one(values, reference, random_seed + node_index)
                    else:
                        signal, details = correlation_core_signal(values, xyz)
                    signals[method][node_index] = signal
                except Exception as exc:
                    status = f"failed:{type(exc).__name__}:{exc}"
                method_rows.append(
                    {
                        "subject_id": str(subject_id),
                        "group": str(group),
                        "atlas_id": atlas.atlas_id,
                        "node_index": node_index,
                        "region_id": int(region_id),
                        "method": method,
                        "status": status,
                        **details,
                    }
                )

    npz_path = output_dir / f"{subject_id}_{atlas.atlas_id}_gm_signals.npz"
    arrays: dict[str, np.ndarray] = {}
    for method, raw in signals.items():
        arrays[f"{method}_raw"] = raw
        arrays[f"{method}_z"] = _standardize_signal_rows(raw)
    np.savez_compressed(npz_path, **arrays)
    homogeneity_path = output_dir / f"{subject_id}_{atlas.atlas_id}_homogeneity.parquet"
    method_path = output_dir / f"{subject_id}_{atlas.atlas_id}_method_status.parquet"
    pd.DataFrame(homogeneity_rows).to_parquet(homogeneity_path, compression="zstd", index=False)
    pd.DataFrame(method_rows).to_parquet(method_path, compression="zstd", index=False)
    covered = int(sum(row["active_gm_voxels"] > 0 for row in homogeneity_rows))
    return RegionalSignalResult(
        subject_id=str(subject_id),
        group=str(group),
        atlas_id=atlas.atlas_id,
        status="ok",
        n_nodes=n_nodes,
        n_timepoints=EXPECTED_TIMEPOINTS,
        signal_npz=str(npz_path),
        homogeneity_table=str(homogeneity_path),
        method_status_table=str(method_path),
        coverage=float(covered / n_nodes),
        unresolved_rows=unresolved,
        node_order=tuple(int(value) for value in node_ids),
        method_metadata={
            "methods": list(GM_SIGNAL_METHODS),
            "random_seed": int(random_seed),
            "max_pairwise_correlations": int(max_pairwise_correlations),
            "correlation_core_threshold": "positive and median +/- 1.4826*MAD",
            "connectivity_computed": False,
            "classification_computed": False,
        },
        input_sha256={
            str(tissue_h5): sha256_file(tissue_h5),
            str(mapping_parquet): sha256_file(mapping_parquet),
            **atlas.source_sha256,
        },
    )


def reconstruct_ready_roi(
    *,
    voxel_csv: str | Path,
    atlas: AtlasDefinition,
    expected_timepoints: int = EXPECTED_TIMEPOINTS,
    csv_block_size: int = 64 * 1024 * 1024,
) -> np.ndarray:
    """Stream all voxels and reconstruct whole-brain atlas means for validation."""
    node_ids = atlas.node_table["region_id"].astype(int).to_numpy()
    id_to_node = {int(value): index for index, value in enumerate(node_ids)}
    sums = np.zeros((atlas.n_nodes, expected_timepoints), dtype=np.float64)
    counts = np.zeros(atlas.n_nodes, dtype=np.int64)
    time_columns = [f"t{i}" for i in range(expected_timepoints)]
    for batch in _voxel_csv_batches(
        Path(voxel_csv),
        expected_timepoints=expected_timepoints,
        block_size=csv_block_size,
    ):
        frame = batch.to_pandas()
        xyz = frame[["x", "y", "z"]].to_numpy(dtype=np.intp, copy=False)
        if atlas.volume is not None:
            region_values = atlas.volume[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        elif atlas.coordinate_table is not None:
            joined = frame[["x", "y", "z"]].merge(
                atlas.coordinate_table, on=["x", "y", "z"], how="left", validate="many_to_one"
            )
            if joined["region_id"].isna().any():
                raise ValueError("voxel coordinate not present in atlas coordinate table")
            region_values = joined["region_id"].to_numpy(dtype=int)
        else:
            raise ValueError("atlas mapping is unavailable")
        values = frame[time_columns].to_numpy(dtype=np.float32, copy=False)
        for region_id in np.unique(region_values):
            node = id_to_node.get(int(region_id))
            if node is None:
                raise ValueError(f"atlas value {region_id} is absent from node table")
            selected = region_values == region_id
            sums[node] += values[selected].sum(axis=0, dtype=np.float64)
            counts[node] += int(selected.sum())
    return np.divide(
        sums,
        counts[:, None],
        out=np.full_like(sums, np.nan),
        where=counts[:, None] > 0,
    ).astype(np.float32)


def validate_atlas_against_ready_roi(
    atlas: AtlasDefinition,
    validation_cases: Sequence[tuple[str | Path, str | Path]],
    *,
    atol: float = 1e-4,
    min_correlation: float = 0.999999,
) -> AtlasDefinition:
    """Validate exact node order by reconstruction; mutate only validation metadata."""
    details: list[dict[str, Any]] = []
    all_ok = True
    for voxel_csv, ready_csv in validation_cases:
        reconstructed = reconstruct_ready_roi(voxel_csv=voxel_csv, atlas=atlas)
        ready = read_whole_brain_roi_csv(ready_csv, expected_nodes=atlas.n_nodes)
        finite = np.isfinite(reconstructed) & np.isfinite(ready)
        max_abs = float(np.max(np.abs(reconstructed[finite] - ready[finite]))) if finite.any() else math.inf
        row_correlations: list[float] = []
        for left, right in zip(reconstructed, ready):
            valid = np.isfinite(left) & np.isfinite(right)
            if valid.sum() < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
                continue
            row_correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
        minimum_correlation = min(row_correlations, default=-math.inf)
        case_ok = bool(max_abs <= atol or minimum_correlation >= min_correlation)
        all_ok &= case_ok
        details.append(
            {
                "voxel_csv": str(voxel_csv),
                "ready_roi_csv": str(ready_csv),
                "max_abs_difference": max_abs,
                "minimum_nonconstant_row_correlation": minimum_correlation,
                "status": "ok" if case_ok else "mismatch",
            }
        )
    atlas.validation_status = "validated_ready_roi" if all_ok else "blocked_atlas_validation"
    atlas.validation_details = {"cases": details, "all_ok": all_ok}
    return atlas


def align_whole_brain_input(
    *,
    subject_id: str,
    group: str,
    atlas: AtlasDefinition,
    ready_roi_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Store a compact node-aligned whole-brain input beside GM-only signals."""
    values = read_whole_brain_roi_csv(ready_roi_csv, expected_nodes=atlas.n_nodes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{subject_id}_{atlas.atlas_id}_whole_brain.npz"
    np.savez_compressed(output, raw=values, z=_standardize_signal_rows(values))
    return {
        "subject_id": str(subject_id),
        "group": str(group),
        "atlas_id": atlas.atlas_id,
        "status": "ok",
        "input_csv": str(ready_roi_csv),
        "input_sha256": sha256_file(ready_roi_csv),
        "output_npz": str(output),
        "output_sha256": sha256_file(output),
        "shape": list(values.shape),
        "node_order": atlas.node_table["region_id"].astype(int).tolist(),
    }
