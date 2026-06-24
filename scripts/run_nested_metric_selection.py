"""Leakage-safe nested metric/window/lag selection for HC/SZ screening.

The fast screen (``run_fast_metric_classifier_screen.py``) computes, per subject,
a compact summary feature vector for every (metric, window, lag) task and then
reports a per-task cross-validated ranking.  That ranking is optimistic: picking
the best task *after* seeing all the data is a form of selection leakage, and a
label-shuffle control at small n produces near-perfect "winners".

This script consumes ``subject_metric_features.csv`` from a screening result and
produces the defensible estimate instead:

* Outer subject-level StratifiedKFold (several passes).
* Inside every training fold only, an inner CV ranks all candidate tasks and the
  single best (metric, window, lag) is chosen.  The chosen task is refit on the
  full training fold and applied to the held-out subjects.
* A combined "all tasks" model concatenates every task's summary features into
  one subject vector and is evaluated with the same outer folds; all imputation,
  scaling and feature selection are fit inside the training fold.
* A label-shuffle negative control repeats the whole single-task procedure on
  permuted labels.  If the pipeline is leakage-free this collapses to ~0.5
  balanced accuracy.

Nothing here is a diagnostic claim; it is an exploratory, leakage-audited
method comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Compact per-task summary feature columns emitted by the fast screen.
FEATURE_COLUMNS = (
    "value_count",
    "finite_fraction",
    "mean",
    "std",
    "median",
    "q05",
    "q25",
    "q75",
    "q95",
    "min",
    "max",
    "mean_abs",
    "std_abs",
    "q95_abs",
    "positive_fraction",
    "negative_fraction",
    "zero_fraction",
    "node_strength_mean",
    "node_strength_std",
    "node_strength_q95",
    "node_strength_max",
    "p_mean",
    "p_median",
    "p_min",
    "p_lt_005_fraction",
    "neglog10p_mean",
    "neglog10p_q95",
)

TASK_KEYS = ("window_label", "lag", "metric")


def _make_estimator(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _scores_from_estimator(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)


def _n_splits_for(labels: np.ndarray, requested: int) -> int:
    counts = Counter(int(v) for v in labels)
    if len(counts) < 2:
        return 0
    return int(max(2, min(requested, min(counts.values()))))


def _build_task_matrices(
    scope_df: pd.DataFrame,
    subjects: list[str],
) -> dict[tuple[Any, ...], np.ndarray]:
    """Return ``task -> (n_subjects, n_features)`` with NaN for missing subjects."""
    subject_index = {subject: i for i, subject in enumerate(subjects)}
    feature_cols = [c for c in FEATURE_COLUMNS if c in scope_df.columns]
    matrices: dict[tuple[Any, ...], np.ndarray] = {}
    for task, task_df in scope_df.groupby(list(TASK_KEYS), sort=True):
        matrix = np.full((len(subjects), len(feature_cols)), np.nan, dtype=np.float64)
        for row in task_df.itertuples(index=False):
            idx = subject_index.get(str(getattr(row, "subject_id")))
            if idx is None:
                continue
            matrix[idx] = [float(getattr(row, col)) for col in feature_cols]
        matrices[task] = matrix
    return matrices


def _inner_rank_tasks(
    task_matrices: dict[tuple[Any, ...], np.ndarray],
    train_rows: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_splits: int,
    seed: int,
) -> list[tuple[tuple[Any, ...], float]]:
    """Rank tasks by inner-CV balanced accuracy on the training fold only."""
    n_inner = _n_splits_for(y_train, inner_splits)
    rankings: list[tuple[tuple[Any, ...], float]] = []
    if n_inner < 2:
        return rankings
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for task, matrix in task_matrices.items():
        X = matrix[train_rows]
        # Require enough finite information to be a usable task.
        if np.isfinite(X).sum() == 0:
            continue
        fold_scores: list[float] = []
        for inner_train, inner_test in inner_cv.split(X, y_train):
            if len(np.unique(y_train[inner_train])) < 2:
                continue
            estimator = _make_estimator(seed + 17)
            try:
                estimator.fit(X[inner_train], y_train[inner_train])
                pred = estimator.predict(X[inner_test]).astype(int)
                fold_scores.append(float(balanced_accuracy_score(y_train[inner_test], pred)))
            except Exception:
                continue
        if fold_scores:
            rankings.append((task, float(np.mean(fold_scores))))
    rankings.sort(key=lambda item: item[1], reverse=True)
    return rankings


def _oof_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(np.mean(pred == y)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, score))
    except Exception:
        out["roc_auc"] = math.nan
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else math.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else math.nan
    return out


def nested_single_task(
    task_matrices: dict[tuple[Any, ...], np.ndarray],
    y: np.ndarray,
    subjects: list[str],
    *,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    record_choices: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Outer CV with task selection nested inside each training fold."""
    n_outer = _n_splits_for(y, outer_splits)
    if n_outer < 2 or not task_matrices:
        return {}, [], []
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    n = len(subjects)
    oof_pred = np.full(n, -1, dtype=int)
    oof_score = np.full(n, np.nan, dtype=float)
    choices: list[dict[str, Any]] = []
    rows = np.arange(n)
    for fold_id, (train_rows, test_rows) in enumerate(outer_cv.split(rows, y), start=1):
        y_train = y[train_rows]
        ranking = _inner_rank_tasks(
            task_matrices, train_rows, y_train, inner_splits=inner_splits, seed=seed + fold_id
        )
        if not ranking:
            continue
        best_task, inner_ba = ranking[0]
        matrix = task_matrices[best_task]
        estimator = _make_estimator(seed + fold_id)
        estimator.fit(matrix[train_rows], y_train)
        oof_pred[test_rows] = estimator.predict(matrix[test_rows]).astype(int)
        oof_score[test_rows] = _scores_from_estimator(estimator, matrix[test_rows])
        if record_choices:
            choices.append(
                {
                    "outer_fold": fold_id,
                    "window_label": best_task[0],
                    "lag": int(best_task[1]),
                    "metric": best_task[2],
                    "inner_cv_balanced_accuracy": inner_ba,
                    "n_train": int(len(train_rows)),
                    "n_test": int(len(test_rows)),
                }
            )
    evaluated = oof_pred >= 0
    if evaluated.sum() < 4 or len(np.unique(y[evaluated])) < 2:
        return {}, choices, []
    metrics = _oof_metrics(y[evaluated], oof_pred[evaluated], oof_score[evaluated])
    predictions = [
        {
            "subject_id": subjects[i],
            "true_group": "SZ" if int(y[i]) == 1 else "HC",
            "oof_predicted_group": "SZ" if int(oof_pred[i]) == 1 else "HC",
            "oof_score_SZ": float(oof_score[i]),
        }
        for i in np.flatnonzero(evaluated)
    ]
    return metrics, choices, predictions


def nested_all_combined(
    task_matrices: dict[tuple[Any, ...], np.ndarray],
    y: np.ndarray,
    *,
    outer_splits: int,
    seed: int,
) -> dict[str, float]:
    """Concatenate every task's features per subject; select features inside folds."""
    n_outer = _n_splits_for(y, outer_splits)
    if n_outer < 2 or not task_matrices:
        return {}
    combined = np.hstack([task_matrices[task] for task in sorted(task_matrices)])
    n = combined.shape[0]
    oof_pred = np.full(n, -1, dtype=int)
    oof_score = np.full(n, np.nan, dtype=float)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    rows = np.arange(n)
    for fold_id, (train_rows, test_rows) in enumerate(outer_cv.split(rows, y), start=1):
        k = int(min(50, combined.shape[1]))
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                        random_state=int(seed + fold_id),
                    ),
                ),
            ]
        )
        try:
            pipe.fit(combined[train_rows], y[train_rows])
            oof_pred[test_rows] = pipe.predict(combined[test_rows]).astype(int)
            oof_score[test_rows] = _scores_from_estimator(pipe, combined[test_rows])
        except Exception:
            continue
    evaluated = oof_pred >= 0
    if evaluated.sum() < 4 or len(np.unique(y[evaluated])) < 2:
        return {}
    return _oof_metrics(y[evaluated], oof_pred[evaluated], oof_score[evaluated])


def label_shuffle_control(
    task_matrices: dict[tuple[Any, ...], np.ndarray],
    y: np.ndarray,
    subjects: list[str],
    *,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    n_shuffles: int,
) -> dict[str, float]:
    """Run the single-task nested procedure on permuted labels."""
    rng = np.random.default_rng(seed)
    null_ba: list[float] = []
    for i in range(int(n_shuffles)):
        y_perm = rng.permutation(y)
        if len(np.unique(y_perm)) < 2:
            continue
        metrics, _, _ = nested_single_task(
            task_matrices,
            y_perm,
            subjects,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            seed=seed + 1000 + i,
            record_choices=False,
        )
        if metrics:
            null_ba.append(metrics["balanced_accuracy"])
    if not null_ba:
        return {}
    arr = np.asarray(null_ba, dtype=float)
    return {
        "n_shuffles": int(len(arr)),
        "null_balanced_accuracy_mean": float(np.mean(arr)),
        "null_balanced_accuracy_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "null_balanced_accuracy_q95": float(np.quantile(arr, 0.95)),
        "null_balanced_accuracy_max": float(np.max(arr)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, help="Fast-screen result directory.")
    parser.add_argument("--features-csv", default="", help="Override path to subject_metric_features.csv.")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--label-shuffles", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=1729)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    features_path = Path(args.features_csv) if args.features_csv else result_dir / "subject_metric_features.csv"
    if not features_path.is_file():
        print(f"ERROR: features file not found: {features_path}")
        return 1
    features = pd.read_csv(features_path)
    features["subject_id"] = features["subject_id"].astype(str)

    summary_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for scope, scope_df in features.groupby("scope", sort=True):
        subjects = sorted(scope_df["subject_id"].unique())
        label_by_subject = (
            scope_df.drop_duplicates("subject_id").set_index("subject_id")["group"].map({"HC": 0, "SZ": 1})
        )
        y = label_by_subject.reindex(subjects).to_numpy(dtype=float)
        if np.isnan(y).any() or len(np.unique(y)) < 2:
            summary_rows.append({"scope": scope, "model": "all", "status": "skipped_labels"})
            continue
        y = y.astype(int)
        counts = Counter(int(v) for v in y)
        task_matrices = _build_task_matrices(scope_df, subjects)

        single_metrics, choices, predictions = nested_single_task(
            task_matrices,
            y,
            subjects,
            outer_splits=int(args.outer_splits),
            inner_splits=int(args.inner_splits),
            seed=int(args.random_seed),
            record_choices=True,
        )
        combined_metrics = nested_all_combined(
            task_matrices, y, outer_splits=int(args.outer_splits), seed=int(args.random_seed)
        )
        control = label_shuffle_control(
            task_matrices,
            y,
            subjects,
            outer_splits=int(args.outer_splits),
            inner_splits=int(args.inner_splits),
            seed=int(args.random_seed),
            n_shuffles=int(args.label_shuffles),
        )

        base = {
            "scope": scope,
            "n_subjects": len(subjects),
            "n_hc": counts.get(0, 0),
            "n_sz": counts.get(1, 0),
            "n_tasks": len(task_matrices),
        }
        if single_metrics:
            summary_rows.append({**base, "model": "nested_single_task_selection", "status": "ok", **single_metrics})
        else:
            summary_rows.append({**base, "model": "nested_single_task_selection", "status": "insufficient"})
        if combined_metrics:
            summary_rows.append({**base, "model": "nested_all_tasks_combined", "status": "ok", **combined_metrics})
        else:
            summary_rows.append({**base, "model": "nested_all_tasks_combined", "status": "insufficient"})
        for row in choices:
            choice_rows.append({"scope": scope, **row})
        for row in predictions:
            prediction_rows.append({"scope": scope, "model": "nested_single_task_selection", **row})
        if control:
            control_rows.append({**base, **control})
        print(
            f"SCOPE {scope}: single_task_BA="
            f"{single_metrics.get('balanced_accuracy', float('nan')):.3f} "
            f"combined_BA={combined_metrics.get('balanced_accuracy', float('nan')):.3f} "
            f"null_BA_mean={control.get('null_balanced_accuracy_mean', float('nan')):.3f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    choices_df = pd.DataFrame(choice_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    control_df = pd.DataFrame(control_rows)
    summary.to_csv(result_dir / "nested_selection_summary.csv", index=False, encoding="utf-8-sig")
    choices_df.to_csv(result_dir / "nested_selection_choices.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(result_dir / "nested_selection_predictions.csv", index=False, encoding="utf-8-sig")
    control_df.to_csv(result_dir / "label_shuffle_negative_control.csv", index=False, encoding="utf-8-sig")

    _write_report(result_dir / "reports" / "nested_selection_audit.md", summary, control_df, choices_df, args)
    print(f"DONE nested selection -> {result_dir}", flush=True)
    return 0


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    keep = [c for c in cols if c in df.columns]
    if df.empty or not keep:
        return "_No rows._"
    view = df[keep].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = ["| " + " | ".join(keep) + " |", "| " + " | ".join("---" for _ in keep) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in keep) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    control: pd.DataFrame,
    choices: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Leakage-safe nested metric/window/lag selection",
        "",
        "Defensible HC/SZ estimate. The (metric, window, lag) choice is made inside",
        "training folds only; the held-out subjects never influence task selection.",
        "",
        f"- Outer folds: `{args.outer_splits}` | inner folds: `{args.inner_splits}`",
        f"- Label shuffles (negative control): `{args.label_shuffles}`",
        f"- Generated: `{datetime.now().astimezone().isoformat()}`",
        "",
        "## Out-of-fold performance",
        "",
        _md_table(
            summary,
            [
                "scope",
                "model",
                "status",
                "n_subjects",
                "n_hc",
                "n_sz",
                "n_tasks",
                "balanced_accuracy",
                "roc_auc",
                "sensitivity",
                "specificity",
            ],
        ),
        "",
        "## Label-shuffle negative control",
        "",
        "If the pipeline is leakage-free, null balanced accuracy concentrates near 0.5.",
        "",
        _md_table(
            control,
            [
                "scope",
                "n_shuffles",
                "null_balanced_accuracy_mean",
                "null_balanced_accuracy_q95",
                "null_balanced_accuracy_max",
            ],
        ),
        "",
        "## Selected task per outer fold",
        "",
        _md_table(choices, ["scope", "outer_fold", "metric", "window_label", "lag", "inner_cv_balanced_accuracy"]),
        "",
        "## Interpretation",
        "",
        "- The nested single-task estimate is the honest 'best single metric' number.",
        "- Compare every scope against its own label-shuffle null, not against 0.5 by assumption.",
        "- These are exploratory method-comparison results, not diagnostic validation.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
