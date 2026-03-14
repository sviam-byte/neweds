"""Тесты ранговых корреляций и экспортных артефактов запуска."""

import json

import pandas as pd

from src.config import AnalysisConfig
from src.core.engine import BigMasterTool
from src.metrics.registry import get_metric_func


def test_rank_correlation_metrics_are_registered() -> None:
    """Проверяем регистрацию и базовую корректность Spearman/Kendall в реестре."""
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
        "z": [5, 4, 3, 2, 1],
    })
    spearman = get_metric_func("correlation_spearman")(df)
    kendall = get_metric_func("correlation_kendall")(df)
    assert spearman.shape == (3, 3)
    assert kendall.shape == (3, 3)
    assert spearman[0, 1] > 0.99
    assert kendall[0, 2] < -0.99


def test_export_binned_timeseries_and_manifest(tmp_path) -> None:
    """Проверяем экспорт binned-рядов и manifest для воспроизводимости запуска."""
    df = pd.DataFrame({
        "bin_0_0_0": [0.1, 0.2, 0.3],
        "bin_0_0_1": [1.0, 1.1, 1.2],
    })
    df.attrs["format"] = "spatial_bins"
    df.attrs["coords"] = pd.DataFrame(
        {
            "voxel_id": ["bin_0_0_0", "bin_0_0_1"],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 1.0],
            "bin_key": ["0_0_0", "0_0_1"],
            "n_voxels": [4, 5],
            "n_active": [4, 5],
        }
    )

    tool = BigMasterTool(config=AnalysisConfig())
    tool.data = df.copy()
    tool.data.attrs = df.attrs.copy()
    tool.coords_df = df.attrs["coords"].copy()
    tool.results_meta = {"__run__": {"lag": 2, "window_size": 128, "window_sizes": [64, 128]}}

    exported = tool.export_binned_timeseries(str(tmp_path / "binned.csv"))
    assert (tmp_path / "binned.csv").exists()
    assert exported["binned_coords_csv"].endswith("binned_coords.csv")
    assert (tmp_path / "binned_meta.json").exists()

    manifest_path = tmp_path / "run_manifest.json"
    tool.export_run_manifest(str(manifest_path), extra={"source": "unit-test"})
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["results_meta"]["__run__"]["lag"] == 2
    assert payload["extra"]["source"] == "unit-test"
