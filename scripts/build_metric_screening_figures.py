"""Build figures for the wide HC/SZ metric screening result."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.write_result_provenance import write_result_provenance


SCOPE_LABELS = {
    "AAL3_whole_roi": "AAL3 whole ROI",
    "HCP360_GM_active_mean": "HCP360 GM",
    "HCP360_whole_brain": "HCP360 whole",
    "tissue_mean_GM_WM_CSF": "Tissue GM/WM/CSF",
}


def _read_csv(result_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(result_dir / name)


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _barh_metric_ranking(rank: pd.DataFrame, out: Path) -> None:
    df = rank.sort_values("mean_best_balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(df)))
    ax.barh(df["metric"], df["mean_best_balanced_accuracy"], color=colors)
    ax.axvline(0.5, color="#666", lw=1, ls="--", label="chance BA=0.5")
    ax.set_xlabel("Mean best balanced accuracy across scopes")
    ax.set_ylabel("Metric")
    ax.set_title("Exploratory per-metric ranking")
    ax.set_xlim(0.45, max(0.9, float(df["mean_best_balanced_accuracy"].max()) + 0.03))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, out)


def _heatmap_metric_scope(detail: pd.DataFrame, out: Path) -> pd.DataFrame:
    best = (
        detail.sort_values(["metric", "scope", "balanced_accuracy_mean"], ascending=[True, True, False])
        .groupby(["metric", "scope"], as_index=False)
        .first()
    )
    pivot = best.pivot(index="metric", columns="scope", values="balanced_accuracy_mean")
    metric_order = pivot.mean(axis=1).sort_values(ascending=False).index
    scope_order = [scope for scope in SCOPE_LABELS if scope in pivot.columns]
    pivot = pivot.loc[metric_order, scope_order]
    fig, ax = plt.subplots(figsize=(8.8, 10))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0.5, vmax=0.9)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([SCOPE_LABELS.get(c, c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Best balanced accuracy per metric and data scope")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.68 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Balanced accuracy")
    _save(fig, out)
    return pivot.reset_index()


def _nested_vs_null(nested: pd.DataFrame, null: pd.DataFrame, out: Path) -> None:
    combined = nested[nested["model"].eq("nested_all_tasks_combined")].copy()
    single = nested[nested["model"].eq("nested_single_task_selection")].copy()
    scopes = [scope for scope in SCOPE_LABELS if scope in set(nested["scope"])]
    x = np.arange(len(scopes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    comb_vals = [float(combined.loc[combined["scope"].eq(s), "balanced_accuracy"].iloc[0]) for s in scopes]
    single_vals = [float(single.loc[single["scope"].eq(s), "balanced_accuracy"].iloc[0]) for s in scopes]
    null_vals = [float(null.loc[null["scope"].eq(s), "null_balanced_accuracy_mean"].iloc[0]) for s in scopes]
    null_q95 = [float(null.loc[null["scope"].eq(s), "null_balanced_accuracy_q95"].iloc[0]) for s in scopes]
    ax.bar(x - width, single_vals, width, label="Nested single metric", color="#6aaed6")
    ax.bar(x, comb_vals, width, label="Nested all metrics", color="#2a9d8f")
    ax.bar(x + width, null_vals, width, label="Label-shuffle mean", color="#b8b8b8")
    ax.scatter(x + width, null_q95, label="Label-shuffle q95", color="#444", marker="_", s=180, zorder=3)
    ax.axhline(0.5, color="#666", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SCOPE_LABELS.get(s, s) for s in scopes], rotation=15, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.35, 0.9)
    ax.set_title("Leakage-safe nested performance vs label-shuffle null")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    _save(fig, out)


def _nested_choices(choices: pd.DataFrame, out: Path) -> pd.DataFrame:
    counts = choices.groupby(["scope", "metric"], as_index=False).size()
    counts["scope_label"] = counts["scope"].map(SCOPE_LABELS).fillna(counts["scope"])
    pivot = counts.pivot(index="metric", columns="scope_label", values="size").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.38 * len(pivot))))
    left = np.zeros(len(pivot))
    for col in pivot.columns:
        vals = pivot[col].to_numpy()
        ax.barh(pivot.index, vals, left=left, label=col)
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("Outer-fold selections")
    ax.set_title("Which metric was selected inside nested training folds")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out)
    return counts


def _window_lag_grid(detail: pd.DataFrame, out: Path) -> pd.DataFrame:
    grid = (
        detail.groupby(["scope", "window_label", "lag"], as_index=False)["balanced_accuracy_mean"]
        .max()
        .sort_values(["scope", "window_label", "lag"])
    )
    scopes = [scope for scope in SCOPE_LABELS if scope in set(grid["scope"])]
    fig, axes = plt.subplots(1, len(scopes), figsize=(4.2 * len(scopes), 4), sharey=True)
    if len(scopes) == 1:
        axes = [axes]
    for ax, scope in zip(axes, scopes):
        sub = grid[grid["scope"].eq(scope)]
        pivot = sub.pivot(index="lag", columns="window_label", values="balanced_accuracy_mean").sort_index()
        im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0.55, vmax=0.9, aspect="auto")
        ax.set_title(SCOPE_LABELS.get(scope, scope))
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Window")
        ax.set_ylabel("Lag")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iat[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Best BA")
    fig.suptitle("Best metric score by window start and lag", y=1.03)
    _save(fig, out)
    return grid


def _subject_coverage(manifest: pd.DataFrame, out: Path) -> None:
    counts = manifest.drop_duplicates(["scope", "subject_id"]).groupby(["scope", "group"]).size().unstack(fill_value=0)
    counts = counts.reindex([s for s in SCOPE_LABELS if s in counts.index])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bottom = np.zeros(len(counts))
    for group, color in [("HC", "#4c78a8"), ("SZ", "#f58518")]:
        vals = counts[group].to_numpy() if group in counts else np.zeros(len(counts))
        ax.bar([SCOPE_LABELS.get(s, s) for s in counts.index], vals, bottom=bottom, label=group, color=color)
        bottom += vals
    ax.set_ylabel("Subjects")
    ax.set_title("Subject coverage by data scope")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def _dimension_table_plot(manifest: pd.DataFrame, out: Path) -> pd.DataFrame:
    dim = (
        manifest.groupby("scope", as_index=False)
        .agg(n_subjects=("subject_id", "nunique"), n_nodes=("n_nodes", "median"), n_timepoints=("n_timepoints", "median"))
    )
    dim["scope_label"] = dim["scope"].map(SCOPE_LABELS).fillna(dim["scope"])
    x = np.arange(len(dim))
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.bar(x - 0.18, dim["n_nodes"], width=0.36, color="#59a14f", label="nodes/series")
    ax1.set_ylabel("Nodes / series")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, dim["n_timepoints"], color="#e15759", marker="o", lw=2, label="time points")
    ax2.set_ylabel("Time points")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dim["scope_label"], rotation=20, ha="right")
    ax1.set_title("Input dimensions used by the screen")
    ax1.grid(axis="y", alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    _save(fig, out)
    return dim


def _qc_distributions(qc: pd.DataFrame, out: Path) -> None:
    scopes = [s for s in SCOPE_LABELS if s in set(qc["scope"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, col, title in [
        (axes[0], "global_std", "Global signal scale by scope/window/group"),
        (axes[1], "global_ar1", "Global AR(1) by scope/window/group"),
    ]:
        positions = []
        data = []
        labels = []
        pos = 0
        for scope in scopes:
            for group in ["HC", "SZ"]:
                vals = qc[(qc["scope"].eq(scope)) & (qc["group"].eq(group))][col].dropna().to_numpy()
                if len(vals):
                    data.append(vals)
                    positions.append(pos)
                    labels.append(f"{SCOPE_LABELS.get(scope, scope)}\n{group}")
                    pos += 1
            pos += 0.7
        ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    _save(fig, out)


def _valid_nodes(qc: pd.DataFrame, out: Path) -> None:
    valid = (
        qc.groupby(["scope", "group"], as_index=False)
        .agg(n_nodes_total=("n_nodes_total", "median"), n_nodes_valid=("n_nodes_valid", "median"))
    )
    valid["invalid"] = valid["n_nodes_total"] - valid["n_nodes_valid"]
    scopes = [s for s in SCOPE_LABELS if s in set(valid["scope"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = []
    valid_vals = []
    invalid_vals = []
    for scope in scopes:
        sub = valid[valid["scope"].eq(scope)]
        labels.append(SCOPE_LABELS.get(scope, scope))
        valid_vals.append(float(sub["n_nodes_valid"].median()))
        invalid_vals.append(float(sub["invalid"].median()))
    x = np.arange(len(labels))
    ax.bar(x, valid_vals, label="valid", color="#2a9d8f")
    ax.bar(x, invalid_vals, bottom=valid_vals, label="invalid/constant", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Median nodes")
    ax.set_title("Valid nodes after preprocessing checks")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out)


def _compute_time(status: pd.DataFrame, out: Path) -> pd.DataFrame:
    timing = (
        status.groupby(["metric"], as_index=False)
        .agg(total_seconds=("seconds", "sum"), median_seconds=("seconds", "median"), rows=("seconds", "size"))
        .sort_values("total_seconds", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(timing["metric"], timing["total_seconds"] / 60.0, color="#9c755f")
    ax.set_xlabel("Total compute time, minutes")
    ax.set_title("Metric compute cost in the wide screen")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out)
    return timing


def _write_report(out_dir: Path, source_dir: Path, generated: dict[str, str], tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Wide HC/SZ metric screening figures",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat()}`",
        f"Source result: `{source_dir}`",
        "",
        "## Figures",
        "",
    ]
    for key, rel in generated.items():
        lines.append(f"- `{rel}` - {key}")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The per-metric ranking figures are exploratory because the best task is selected after seeing the grid.",
            "- The nested-vs-null figure is the leakage-audited performance view.",
            "- All real HC/SZ figures here use 600 time points, not the separate 1000-point HDF5 voxel files.",
            "- The 1000-point files require a separate ROI/voxel preprocessing adapter before they can be compared to these runs.",
        ]
    )
    (out_dir / "FIGURE_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    cards = "\n".join(
        f'<section class="card"><h2>{title}</h2><a href="{rel}"><img src="{rel}" alt="{title}"></a></section>'
        for title, rel in generated.items()
    )
    gallery = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wide HC/SZ Metric Screening Figures</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f5f7fb; color: #15202b; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; }}
    .muted {{ color: #5b6878; }}
    .card {{ background: white; border: 1px solid #d9e0ea; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .card h2 {{ margin: 0 0 14px; font-size: 20px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #e5eaf1; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Wide HC/SZ Metric Screening Figures</h1>
  <p class="muted">Source: <code>{source_dir}</code>. Real HC/SZ figures use 600 time points; 1000-point HDF5 voxel files are not included in this result.</p>
  {cards}
</main>
</body>
</html>
"""
    (out_dir / "FIGURE_GALLERY.html").write_text(gallery, encoding="utf-8")


def _scope_data_label(scope: str, n_subjects: int, n_hc: int, n_sz: int) -> str:
    if scope == "AAL3_whole_roi":
        return f"AAL3 whole ROI; {n_subjects} subj: HC={n_hc}/SZ={n_sz}; 167 ROI; 600 точек"
    if scope == "HCP360_GM_active_mean":
        return f"HCP360 GM active-mean; {n_subjects} subj: HC={n_hc}/SZ={n_sz}; 360 ROI; 600 точек"
    if scope == "HCP360_whole_brain":
        return f"HCP360 whole-brain; {n_subjects} subj: HC={n_hc}/SZ={n_sz}; 360 ROI; 600 точек"
    if scope == "tissue_mean_GM_WM_CSF":
        return f"tissue mean GM/WM/CSF; {n_subjects} subj: HC={n_hc}/SZ={n_sz}; 3 tissue rows; 600 точек"
    return f"{scope}; {n_subjects} subj: HC={n_hc}/SZ={n_sz}; 600 точек"


def _nested_readable(nested: pd.DataFrame) -> pd.DataFrame:
    model_names = {
        "nested_all_tasks_combined": "all 26 combined",
        "nested_single_task_selection": "nested-selected single metric",
    }
    rows = []
    scope_order = [scope for scope in SCOPE_LABELS if scope in set(nested["scope"])]
    for scope in scope_order:
        sub = nested[nested["scope"].eq(scope)]
        for model in ["nested_all_tasks_combined", "nested_single_task_selection"]:
            if model not in set(sub["model"]):
                continue
            row = sub[sub["model"].eq(model)].iloc[0]
            if model == "nested_all_tasks_combined":
                params = "nested outer=5/inner=4; all 156 tasks"
            else:
                params = "train-only выбор одной task"
            rows.append(
                {
                    "Метрика / модель": model_names[model],
                    "Результат": f"BA {row['balanced_accuracy']:.3f}, AUC {row['roc_auc']:.3f}",
                    "По каким данным": _scope_data_label(
                        str(row["scope"]), int(row["n_subjects"]), int(row["n_hc"]), int(row["n_sz"])
                    ),
                    "Параметры": params,
                }
            )
    return pd.DataFrame(rows)


def _exploratory_readable(rank: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    best = (
        detail.sort_values(["metric", "balanced_accuracy_mean", "roc_auc_mean"], ascending=[True, False, False])
        .groupby("metric", as_index=False)
        .first()
    )
    merged = rank.merge(
        best[
            [
                "metric",
                "scope",
                "window_label",
                "lag",
                "n_subjects",
                "n_hc",
                "n_sz",
                "balanced_accuracy_mean",
                "roc_auc_mean",
            ]
        ],
        on="metric",
        how="left",
    ).sort_values("overall_rank")
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Метрика": row["metric"],
                "Результат": (
                    f"mean best BA {row['mean_best_balanced_accuracy']:.3f}, "
                    f"AUC {row['mean_best_roc_auc']:.3f}; "
                    f"best task BA {row['balanced_accuracy_mean']:.3f}, AUC {row['roc_auc_mean']:.3f}"
                ),
                "По каким данным": _scope_data_label(
                    str(row["scope"]), int(row["n_subjects"]), int(row["n_hc"]), int(row["n_sz"])
                ),
                "Параметры": f"{row['window_label']}, lag={int(row['lag'])}",
            }
        )
    return pd.DataFrame(rows)


def _write_excel(
    out_dir: Path,
    generated: dict[str, str],
    nested_readable: pd.DataFrame,
    exploratory_readable: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> Path:
    excel_dir = out_dir / "excel"
    excel_dir.mkdir(parents=True, exist_ok=True)
    path = excel_dir / "metric_screening_summary_tables.xlsx"
    manifest = pd.DataFrame(
        [{"Figure": title, "Path": rel} for title, rel in generated.items()]
    )
    with pd.ExcelWriter(path) as writer:
        nested_readable.to_excel(writer, sheet_name="nested_summary", index=False)
        exploratory_readable.to_excel(writer, sheet_name="exploratory_metrics", index=False)
        manifest.to_excel(writer, sheet_name="figure_manifest", index=False)
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            for col_cells in worksheet.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                worksheet.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 12), 70)
    return path


def _export_for_git(out_dir: Path, export_dir: Path) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "FIGURE_README.md", export_dir / "README.md")
    shutil.copy2(out_dir / "FIGURE_GALLERY.html", export_dir / "FIGURE_GALLERY.html")
    shutil.copy2(out_dir / "figure_manifest.json", export_dir / "figure_manifest.json")
    shutil.copy2(out_dir / "excel" / "metric_screening_summary_tables.xlsx", export_dir / "metric_screening_summary_tables.xlsx")
    shutil.copytree(out_dir / "figures", export_dir / "figures")
    shutil.copytree(out_dir / "tables", export_dir / "tables")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--new-results", default="outputs/run_paths/new_results")
    parser.add_argument("--result-date", default="2026-06-24")
    parser.add_argument("--slug", default="metric-screening-figures")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--export-dir", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_dir = Path(args.source_result)
    out_dir = Path(args.new_results) / f"{args.result_date}_{args.slug}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    figures = out_dir / "figures"
    metrics_dir = figures / "metrics"
    pre_dir = figures / "preanalysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rank = _read_csv(source_dir, "overall_metric_ranking.csv")
    detail = _read_csv(source_dir, "metric_ranking_by_scope_window_lag.csv")
    nested = _read_csv(source_dir, "nested_selection_summary.csv")
    choices = _read_csv(source_dir, "nested_selection_choices.csv")
    null = _read_csv(source_dir, "label_shuffle_negative_control.csv")
    manifest = _read_csv(source_dir, "subject_input_manifest.csv")
    qc = _read_csv(source_dir, "subject_preprocessing_qc.csv")
    status = _read_csv(source_dir, "metric_compute_status.csv")

    generated: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    jobs = [
        ("Exploratory per-metric ranking", metrics_dir / "01_metric_overall_ranking"),
        ("Metric by scope heatmap", metrics_dir / "02_metric_scope_heatmap"),
        ("Nested performance vs label-shuffle null", metrics_dir / "03_nested_vs_label_shuffle_null"),
        ("Nested train-fold metric choices", metrics_dir / "04_nested_selected_metrics"),
        ("Best score by window and lag", metrics_dir / "05_window_lag_grid"),
        ("Subject coverage by scope", pre_dir / "01_subject_coverage"),
        ("Input dimensions by scope", pre_dir / "02_input_dimensions"),
        ("QC global distributions", pre_dir / "03_qc_global_std_ar1"),
        ("Valid nodes by scope", pre_dir / "04_valid_nodes"),
        ("Compute time by metric", pre_dir / "05_metric_compute_time"),
    ]

    _barh_metric_ranking(rank, jobs[0][1])
    generated[jobs[0][0]] = jobs[0][1].with_suffix(".png").relative_to(out_dir).as_posix()
    tables["metric_scope_best_balanced_accuracy"] = _heatmap_metric_scope(detail, jobs[1][1])
    generated[jobs[1][0]] = jobs[1][1].with_suffix(".png").relative_to(out_dir).as_posix()
    _nested_vs_null(nested, null, jobs[2][1])
    generated[jobs[2][0]] = jobs[2][1].with_suffix(".png").relative_to(out_dir).as_posix()
    tables["nested_selected_metric_counts"] = _nested_choices(choices, jobs[3][1])
    generated[jobs[3][0]] = jobs[3][1].with_suffix(".png").relative_to(out_dir).as_posix()
    tables["window_lag_best_balanced_accuracy"] = _window_lag_grid(detail, jobs[4][1])
    generated[jobs[4][0]] = jobs[4][1].with_suffix(".png").relative_to(out_dir).as_posix()
    _subject_coverage(manifest, jobs[5][1])
    generated[jobs[5][0]] = jobs[5][1].with_suffix(".png").relative_to(out_dir).as_posix()
    tables["input_dimensions_by_scope"] = _dimension_table_plot(manifest, jobs[6][1])
    generated[jobs[6][0]] = jobs[6][1].with_suffix(".png").relative_to(out_dir).as_posix()
    _qc_distributions(qc, jobs[7][1])
    generated[jobs[7][0]] = jobs[7][1].with_suffix(".png").relative_to(out_dir).as_posix()
    _valid_nodes(qc, jobs[8][1])
    generated[jobs[8][0]] = jobs[8][1].with_suffix(".png").relative_to(out_dir).as_posix()
    tables["metric_compute_time"] = _compute_time(status, jobs[9][1])
    generated[jobs[9][0]] = jobs[9][1].with_suffix(".png").relative_to(out_dir).as_posix()

    _write_report(out_dir, source_dir, generated, tables)
    nested_readable = _nested_readable(nested)
    exploratory_readable = _exploratory_readable(rank, detail)
    tables["nested_summary_readable"] = nested_readable
    tables["exploratory_per_metric_readable"] = exploratory_readable
    nested_readable.to_csv(out_dir / "tables" / "nested_summary_readable.csv", index=False, encoding="utf-8-sig")
    exploratory_readable.to_csv(out_dir / "tables" / "exploratory_per_metric_readable.csv", index=False, encoding="utf-8-sig")
    excel_path = _write_excel(out_dir, generated, nested_readable, exploratory_readable, tables)
    (out_dir / "figure_manifest.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_dir:
        _export_for_git(out_dir, Path(args.export_dir))

    command = " ".join(
        [
            "python scripts/build_metric_screening_figures.py",
            f"--source-result {args.source_result}",
            f"--new-results {args.new_results}",
            f"--result-date {args.result_date}",
            f"--slug {args.slug}",
            f"--export-dir {args.export_dir}" if args.export_dir else "",
        ]
    )
    write_result_provenance(
        result_dir=out_dir,
        result_id=f"{args.slug}-{args.result_date}",
        title="Wide HC/SZ metric screening figures",
        result_type="derived_visualization_package",
        status="completed",
        execution_mode="derived_run",
        summary="Generated metric and preanalysis figures from the wide HC/SZ metric-screening result.",
        meaning="The package separates exploratory per-metric views from leakage-audited nested performance and documents the data coverage/QC used by the screen.",
        command=command,
        inputs=[str(source_dir)],
        code_files=[
            Path("scripts/build_metric_screening_figures.py"),
            Path("scripts/write_result_provenance.py"),
        ],
        repository=Path(args.repository),
        findings=[
            "Generated metric ranking, metric-by-scope, nested-vs-null, nested-choice, window-lag, subject coverage, input-dimension, QC, valid-node, and compute-time figures.",
            f"Wrote Excel workbook with readable nested and exploratory tables: {excel_path.relative_to(out_dir).as_posix()}.",
            "All figures are derived from the 600-timepoint wide HC/SZ screen; the separate 1000-timepoint HDF5 voxel files were not part of this source result.",
        ],
        limitations=[
            "Per-metric ranking plots are exploratory and selection-optimistic.",
            "Nested performance plots are still small-n exploratory method-comparison results, not diagnostic validation.",
            "No new model was trained by this visualization package.",
        ],
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
