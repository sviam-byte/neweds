"""Build Stage 1.5 pipeline decisions from fMRI ROI characterization tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONSERVATIVE_EXCLUDE_ROIS_1BASED = {35, 36, 81, 82}
V2_CONSERVATIVE_EXCLUDE_ROIS_1BASED = {1, 35, 36, 81, 82}
REVIEW_ROIS_1BASED = {106, 111, 133, 134, 160, 167}
V2_SENSITIVITY_ONLY_ROIS_1BASED = {133, 134, 167}
V2_QC_FLAG_KEEP_ROIS_1BASED = {106, 111, 160}
EXTREME_DIAGNOSTIC_ROIS_1BASED = {133, 134, 160, 167}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _read_regions(path: Path | None) -> pd.DataFrame:
    columns = ["roi_index_1based", "region_name"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        match = re.match(r"^\s*(\d+)\s+(.+?)\s*$", text)
        if match:
            rows.append(
                {
                    "roi_index_1based": int(match.group(1)),
                    "region_name": match.group(2).strip(),
                }
            )
        else:
            rows.append({"roi_index_1based": len(rows) + 1, "region_name": text})
    return pd.DataFrame(rows, columns=columns)


def _frequency(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0).astype(bool).mean())


def _join_reasons(reasons: list[str]) -> str:
    return "; ".join(reason for reason in reasons if reason)


def build_roi_decisions(features: pd.DataFrame, regions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    bool_cols = ["zero_flag", "constant_flag", "extreme_amplitude_flag", "nan_flag"]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    numeric_cols = [
        "roi_index_1based",
        "acf_lag_1",
        "linear_trend_slope",
        "linear_trend_r2",
        "mean_shift_second_minus_first",
        "std_ratio_second_to_first",
        "low_high_power_ratio",
        "spectral_entropy",
        "spectral_slope",
        "amplitude",
        "std",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Split the old broad "trend_or_shift" idea into explicit flags.
    df["linear_trend_flag"] = (
        df["linear_trend_r2"].fillna(0).gt(0.05)
        & df["linear_trend_slope"].abs().gt(df["linear_trend_slope"].abs().median(skipna=True))
    )
    shift_threshold = max(1.0, float(df["std"].median(skipna=True) or 0.0) * 0.5)
    df["mean_shift_flag"] = df["mean_shift_second_minus_first"].abs().gt(shift_threshold)
    df["variance_shift_flag"] = (
        df["std_ratio_second_to_first"].lt(0.5) | df["std_ratio_second_to_first"].gt(2.0)
    )
    df["high_acf_flag"] = df["acf_lag_1"].gt(0.7)
    df["spectral_warning_flag"] = (
        df["low_high_power_ratio"].gt(5.0)
        | df["spectral_entropy"].lt(0.65)
        | df["spectral_slope"].lt(-1.2)
    )
    df["stationarity_review_flag"] = (
        df["linear_trend_flag"] | df["mean_shift_flag"] | df["variance_shift_flag"]
    )

    grouped = df.groupby(["atlas", "roi_index_0based", "roi_index_1based"], as_index=False)
    roi = grouped.agg(
        n_subjects=("subject_id", "nunique"),
        zero_subject_count=("zero_flag", "sum"),
        constant_subject_count=("constant_flag", "sum"),
        nan_subject_count=("nan_flag", "sum"),
        extreme_amplitude_count=("extreme_amplitude_flag", "sum"),
        high_acf_count=("high_acf_flag", "sum"),
        spectral_warning_count=("spectral_warning_flag", "sum"),
        linear_trend_count=("linear_trend_flag", "sum"),
        mean_shift_count=("mean_shift_flag", "sum"),
        variance_shift_count=("variance_shift_flag", "sum"),
        stationarity_review_count=("stationarity_review_flag", "sum"),
        median_acf1=("acf_lag_1", "median"),
        median_low_high_power_ratio=("low_high_power_ratio", "median"),
        median_spectral_entropy=("spectral_entropy", "median"),
        median_spectral_slope=("spectral_slope", "median"),
        median_amplitude=("amplitude", "median"),
        max_amplitude=("amplitude", "max"),
    )
    for count_col in [
        "zero_subject_count",
        "constant_subject_count",
        "nan_subject_count",
        "extreme_amplitude_count",
        "high_acf_count",
        "spectral_warning_count",
        "linear_trend_count",
        "mean_shift_count",
        "variance_shift_count",
        "stationarity_review_count",
    ]:
        roi[count_col] = pd.to_numeric(roi[count_col], errors="coerce").fillna(0).astype(int)
        roi[count_col.replace("_count", "_frequency")] = roi[count_col] / roi["n_subjects"].clip(lower=1)

    zero_all = roi["zero_subject_count"].eq(roi["n_subjects"])
    constant_all = roi["constant_subject_count"].eq(roi["n_subjects"])
    constant_any = roi["constant_subject_count"].gt(0)
    roi["zero_all_subjects"] = zero_all
    roi["constant_any_subject"] = constant_any

    decisions: list[str] = []
    reasons_all: list[str] = []
    for _, row in roi.iterrows():
        roi_num = int(row["roi_index_1based"])
        reasons: list[str] = []
        if roi_num in CONSERVATIVE_EXCLUDE_ROIS_1BASED or bool(row["zero_all_subjects"]) or (
            bool(row["constant_any_subject"]) and float(row["constant_subject_frequency"]) >= 1.0
        ):
            decision = "exclude_conservative"
            if roi_num in CONSERVATIVE_EXCLUDE_ROIS_1BASED:
                reasons.append("systemic zero/constant ROI predefined for conservative FC exclusion")
            if bool(row["zero_all_subjects"]):
                reasons.append("zero in all subjects")
            if float(row["constant_subject_frequency"]) >= 1.0:
                reasons.append("constant in all subjects")
        elif roi_num in REVIEW_ROIS_1BASED:
            decision = "review"
            reasons.append("preselected Stage 1.5 review ROI")
        elif float(row["constant_subject_frequency"]) > 0.0:
            decision = "review"
            reasons.append("zero/constant in at least one subject")
        elif float(row["extreme_amplitude_frequency"]) >= 0.10:
            decision = "review"
            reasons.append("extreme amplitude in >=10% subjects")
        elif float(row["high_acf_frequency"]) >= 0.25:
            decision = "review"
            reasons.append("high ACF1 in >=25% subjects")
        elif float(row["spectral_warning_frequency"]) >= 0.25:
            decision = "review"
            reasons.append("spectral warning in >=25% subjects")
        elif float(row["stationarity_review_frequency"]) >= 0.25:
            decision = "review"
            reasons.append("stationarity review flag in >=25% subjects")
        else:
            decision = "keep"
            reasons.append("no Stage 1.5 exclusion/review rule triggered")
        decisions.append(decision)
        reasons_all.append(_join_reasons(reasons))
    roi["decision"] = decisions
    roi["reason"] = reasons_all

    if not regions.empty:
        roi = roi.merge(regions, on="roi_index_1based", how="left")
    else:
        roi["region_name"] = ""

    # Subject-level special case requested by Stage 1.5: SZ/1177 ROI 106/111.
    diagnostic = df[df["roi_index_1based"].isin(EXTREME_DIAGNOSTIC_ROIS_1BASED)].copy()
    if not regions.empty:
        diagnostic = diagnostic.merge(regions, on="roi_index_1based", how="left")
    diagnostic_cols = [
        "atlas",
        "group",
        "subject_id",
        "roi_index_0based",
        "roi_index_1based",
        "region_name",
        "amplitude",
        "extreme_amplitude_flag",
        "acf_lag_1",
        "high_acf_flag",
        "linear_trend_slope",
        "linear_trend_r2",
        "mean_shift_second_minus_first",
        "std_ratio_second_to_first",
        "low_high_power_ratio",
        "spectral_entropy",
        "spectral_slope",
        "spectral_warning_flag",
        "stationarity_review_flag",
    ]
    diagnostic = diagnostic[[c for c in diagnostic_cols if c in diagnostic.columns]]
    return roi, diagnostic


def build_subject_decisions(features: pd.DataFrame, subject_summary: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    for col in ["roi_index_1based", "zero_flag", "constant_flag"]:
        if col in df.columns and col != "zero_flag" and col != "constant_flag":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["zero_flag"] = df["zero_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
    df["constant_flag"] = df["constant_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
    for col in ["low_high_power_ratio", "spectral_entropy", "spectral_slope", "acf_lag_1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["high_acf_flag"] = df["acf_lag_1"].gt(0.7)
    df["spectral_warning_flag"] = (
        df["low_high_power_ratio"].gt(5.0)
        | df["spectral_entropy"].lt(0.65)
        | df["spectral_slope"].lt(-1.2)
    )
    df["low_freq_dominance_flag"] = df["low_high_power_ratio"].gt(5.0)
    subject_flags = (
        df.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            roi_106_zero_or_constant=(
                "constant_flag",
                lambda s: bool(
                    (
                        df.loc[s.index, "roi_index_1based"].eq(106)
                        & (df.loc[s.index, "zero_flag"] | df.loc[s.index, "constant_flag"])
                    ).any()
                ),
            ),
            roi_111_zero_or_constant=(
                "constant_flag",
                lambda s: bool(
                    (
                        df.loc[s.index, "roi_index_1based"].eq(111)
                        & (df.loc[s.index, "zero_flag"] | df.loc[s.index, "constant_flag"])
                    ).any()
                ),
            ),
            high_acf_frequency=("high_acf_flag", _frequency),
            spectral_warning_frequency=("spectral_warning_flag", _frequency),
            low_freq_dominance_frequency=("low_freq_dominance_flag", _frequency),
        )
        .reset_index(drop=True)
    )
    out = subject_summary.merge(subject_flags, on=["atlas", "group", "subject_id"], how="left")
    for col in [
        "n_zero_roi",
        "n_constant_roi",
        "n_extreme_amplitude_roi",
        "warning_count",
        "median_ar1",
        "median_low_high_power_ratio",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    decisions: list[str] = []
    reasons_all: list[str] = []
    for _, row in out.iterrows():
        reasons: list[str] = []
        decision = "keep"
        if str(row["group"]) == "SZ" and str(row["subject_id"]) == "1177_Ivanov_S_A":
            decision = "review"
            reasons.append("explicit subject-level warning: extra zero/constant ROI 106/111")
        if _safe_float(row.get("n_zero_roi")) > 4:
            decision = "review"
            reasons.append("more zero ROI than systemic baseline")
        if _safe_float(row.get("n_extreme_amplitude_roi")) >= 5:
            decision = "review"
            reasons.append("many extreme-amplitude ROI")
        if _safe_float(row.get("median_ar1")) > 0.7:
            decision = "review"
            reasons.append("high median AR1")
        if _safe_float(row.get("low_freq_dominance_frequency")) >= 0.25:
            decision = "review"
            reasons.append("low-frequency dominance in >=25% ROI")
        if _safe_float(row.get("warning_count")) >= 15:
            decision = "review"
            reasons.append("high warning count")
        if not reasons:
            reasons.append("no subject-level Stage 1.5 rule triggered")
        decisions.append(decision)
        reasons_all.append(_join_reasons(reasons))
    out["decision"] = decisions
    out["reason"] = reasons_all
    return out


def _prepare_feature_flags(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    for col in ["zero_flag", "constant_flag", "extreme_amplitude_flag", "nan_flag"]:
        df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    for col in [
        "roi_index_0based",
        "roi_index_1based",
        "acf_lag_1",
        "linear_trend_slope",
        "linear_trend_r2",
        "mean_shift_second_minus_first",
        "std_ratio_second_to_first",
        "low_high_power_ratio",
        "spectral_entropy",
        "spectral_slope",
        "amplitude",
        "std",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["linear_trend_flag"] = (
        df["linear_trend_r2"].fillna(0).gt(0.05)
        & df["linear_trend_slope"].abs().gt(df["linear_trend_slope"].abs().median(skipna=True))
    )
    shift_threshold = max(1.0, float(df["std"].median(skipna=True) or 0.0) * 0.5)
    df["mean_shift_flag"] = df["mean_shift_second_minus_first"].abs().gt(shift_threshold)
    df["variance_shift_flag"] = (
        df["std_ratio_second_to_first"].lt(0.5) | df["std_ratio_second_to_first"].gt(2.0)
    )
    df["high_acf_flag"] = df["acf_lag_1"].gt(0.7)
    df["spectral_warning_flag"] = (
        df["low_high_power_ratio"].gt(5.0)
        | df["spectral_entropy"].lt(0.65)
        | df["spectral_slope"].lt(-1.2)
    )
    df["stationarity_review_flag"] = (
        df["linear_trend_flag"] | df["mean_shift_flag"] | df["variance_shift_flag"]
    )
    df["low_freq_dominance_flag"] = df["low_high_power_ratio"].gt(5.0)
    return df


def build_roi_decisions_v2(
    features: pd.DataFrame,
    regions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build normalized Stage 1.5 v2 ROI decisions.

    v2 separates temporal/spectral QC flags from conservative exclusions.
    """
    df = _prepare_feature_flags(features)
    grouped = df.groupby(["atlas", "roi_index_0based", "roi_index_1based"], as_index=False)
    roi = grouped.agg(
        n_subjects=("subject_id", "nunique"),
        zero_subject_count=("zero_flag", "sum"),
        constant_subject_count=("constant_flag", "sum"),
        nan_subject_count=("nan_flag", "sum"),
        extreme_amplitude_count=("extreme_amplitude_flag", "sum"),
        high_acf_count=("high_acf_flag", "sum"),
        spectral_warning_count=("spectral_warning_flag", "sum"),
        linear_trend_count=("linear_trend_flag", "sum"),
        mean_shift_count=("mean_shift_flag", "sum"),
        variance_shift_count=("variance_shift_flag", "sum"),
        stationarity_review_count=("stationarity_review_flag", "sum"),
        median_acf1=("acf_lag_1", "median"),
        median_low_high_power_ratio=("low_high_power_ratio", "median"),
        median_spectral_entropy=("spectral_entropy", "median"),
        median_spectral_slope=("spectral_slope", "median"),
        median_amplitude=("amplitude", "median"),
        max_amplitude=("amplitude", "max"),
    )
    for count_col in [
        "zero_subject_count",
        "constant_subject_count",
        "nan_subject_count",
        "extreme_amplitude_count",
        "high_acf_count",
        "spectral_warning_count",
        "linear_trend_count",
        "mean_shift_count",
        "variance_shift_count",
        "stationarity_review_count",
    ]:
        roi[count_col] = pd.to_numeric(roi[count_col], errors="coerce").fillna(0).astype(int)
        roi[count_col.replace("_count", "_frequency")] = roi[count_col] / roi["n_subjects"].clip(lower=1)

    roi["zero_all_subjects"] = roi["zero_subject_count"].eq(roi["n_subjects"])
    roi["constant_all_subjects"] = roi["constant_subject_count"].eq(roi["n_subjects"])
    roi["constant_any_subject"] = roi["constant_subject_count"].gt(0)

    if not regions.empty:
        roi = roi.merge(regions, on="roi_index_1based", how="left")
    else:
        roi["region_name"] = ""

    decisions: list[str] = []
    reason_categories_all: list[str] = []
    reasons_all: list[str] = []
    primary_include: list[bool] = []
    include_review_roi_include: list[bool] = []
    for _, row in roi.iterrows():
        roi_num = int(row["roi_index_1based"])
        region_name = str(row.get("region_name", ""))
        cats: list[str] = []
        reasons: list[str] = []
        is_background = roi_num == 1 or region_name.strip().lower() == "background"
        hard_bad = (
            roi_num in V2_CONSERVATIVE_EXCLUDE_ROIS_1BASED
            or is_background
            or bool(row["zero_all_subjects"])
            or bool(row["constant_all_subjects"])
        )
        if hard_bad:
            decision = "exclude_conservative"
            if is_background:
                cats.append("background_node")
                reasons.append("Background is a non-anatomical node")
            if roi_num in V2_CONSERVATIVE_EXCLUDE_ROIS_1BASED and not is_background:
                cats.append("systemic_zero_constant")
                reasons.append("predefined conservative hard exclusion")
            if bool(row["zero_all_subjects"]):
                cats.append("systemic_zero_constant")
                reasons.append("zero in all subjects")
            if bool(row["constant_all_subjects"]):
                cats.append("systemic_zero_constant")
                reasons.append("constant in all subjects")
        elif roi_num in V2_SENSITIVITY_ONLY_ROIS_1BASED:
            decision = "sensitivity_only"
            cats.append("extreme_amplitude")
            cats.append("anatomical_review")
            reasons.append("stable extreme-amplitude review ROI; exclude from primary")
        else:
            flagged = False
            if roi_num in V2_QC_FLAG_KEEP_ROIS_1BASED:
                flagged = True
                cats.append("subject_specific_zero_constant" if roi_num in {106, 111} else "extreme_amplitude")
                reasons.append("preselected QC-flag ROI retained in primary")
            if float(row["constant_subject_frequency"]) > 0.0:
                flagged = True
                cats.append("subject_specific_zero_constant")
                reasons.append("zero/constant in at least one subject")
            if float(row["extreme_amplitude_frequency"]) >= 0.10:
                flagged = True
                cats.append("extreme_amplitude")
                reasons.append("extreme amplitude in >=10% subjects")
            if float(row["high_acf_frequency"]) >= 0.25:
                flagged = True
                cats.append("temporal_qc_flag")
                reasons.append("high ACF1 in >=25% subjects")
            if float(row["spectral_warning_frequency"]) >= 0.25:
                flagged = True
                cats.append("spectral_qc_flag")
                reasons.append("spectral warning in >=25% subjects")
            if float(row["stationarity_review_frequency"]) >= 0.25:
                flagged = True
                cats.append("stationarity_review")
                reasons.append("stationarity review in >=25% subjects")
            if flagged:
                decision = "qc_flag_keep"
            else:
                decision = "keep"
                reasons.append("no Stage 1.5 v2 rule triggered")
        decisions.append(decision)
        reason_categories_all.append(";".join(sorted(set(cats))))
        reasons_all.append(_join_reasons(reasons))
        primary_include.append(decision in {"keep", "qc_flag_keep"})
        include_review_roi_include.append(decision in {"keep", "qc_flag_keep", "sensitivity_only"})

    roi["decision"] = decisions
    roi["reason_category"] = reason_categories_all
    roi["reason"] = reasons_all
    roi["primary_stage2_include"] = primary_include
    roi["include_review_roi_include"] = include_review_roi_include

    diagnostic = df[df["roi_index_1based"].isin(EXTREME_DIAGNOSTIC_ROIS_1BASED)].copy()
    if not regions.empty:
        diagnostic = diagnostic.merge(regions, on="roi_index_1based", how="left")
    diagnostic_cols = [
        "atlas",
        "group",
        "subject_id",
        "roi_index_0based",
        "roi_index_1based",
        "region_name",
        "amplitude",
        "extreme_amplitude_flag",
        "acf_lag_1",
        "high_acf_flag",
        "linear_trend_slope",
        "linear_trend_r2",
        "mean_shift_second_minus_first",
        "std_ratio_second_to_first",
        "low_high_power_ratio",
        "spectral_entropy",
        "spectral_slope",
        "spectral_warning_flag",
        "stationarity_review_flag",
    ]
    diagnostic = diagnostic[[c for c in diagnostic_cols if c in diagnostic.columns]]
    return roi, diagnostic


def build_subject_decisions_v2(features: pd.DataFrame, subject_summary: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_feature_flags(features)
    subject_flags = (
        df.groupby(["atlas", "group", "subject_id"], as_index=False)
        .agg(
            roi_106_zero_or_constant=(
                "constant_flag",
                lambda s: bool(
                    (
                        df.loc[s.index, "roi_index_1based"].eq(106)
                        & (df.loc[s.index, "zero_flag"] | df.loc[s.index, "constant_flag"])
                    ).any()
                ),
            ),
            roi_111_zero_or_constant=(
                "constant_flag",
                lambda s: bool(
                    (
                        df.loc[s.index, "roi_index_1based"].eq(111)
                        & (df.loc[s.index, "zero_flag"] | df.loc[s.index, "constant_flag"])
                    ).any()
                ),
            ),
            high_acf_frequency=("high_acf_flag", _frequency),
            spectral_warning_frequency=("spectral_warning_flag", _frequency),
            low_freq_dominance_frequency=("low_freq_dominance_flag", _frequency),
        )
        .reset_index(drop=True)
    )
    out = subject_summary.merge(subject_flags, on=["atlas", "group", "subject_id"], how="left")
    for col in [
        "n_zero_roi",
        "n_constant_roi",
        "n_extreme_amplitude_roi",
        "warning_count",
        "median_ar1",
        "low_freq_dominance_frequency",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    decisions: list[str] = []
    categories_all: list[str] = []
    reasons_all: list[str] = []
    for _, row in out.iterrows():
        cats: list[str] = []
        reasons: list[str] = []
        decision = "keep"
        is_1177 = str(row["group"]) == "SZ" and str(row["subject_id"]) == "1177_Ivanov_S_A"
        if is_1177:
            decision = "sensitivity_review"
            cats.append("subject_specific_zero_constant")
            reasons.append("explicit warning: extra zero/constant ROI 106/111")
        if _safe_float(row.get("n_zero_roi")) > 4:
            decision = "sensitivity_review"
            cats.append("subject_specific_zero_constant")
            reasons.append("more zero ROI than systemic baseline")
        if _safe_float(row.get("n_extreme_amplitude_roi")) >= 5:
            decision = "sensitivity_review" if decision == "sensitivity_review" else "qc_flag_keep"
            cats.append("extreme_amplitude")
            reasons.append("many extreme-amplitude ROI")
        if _safe_float(row.get("median_ar1")) > 0.7:
            decision = "sensitivity_review" if decision == "sensitivity_review" else "qc_flag_keep"
            cats.append("temporal_qc_flag")
            reasons.append("high median AR1")
        if _safe_float(row.get("low_freq_dominance_frequency")) >= 0.25:
            decision = "sensitivity_review" if decision == "sensitivity_review" else "qc_flag_keep"
            cats.append("spectral_qc_flag")
            reasons.append("low-frequency dominance in >=25% ROI")
        if _safe_float(row.get("warning_count")) >= 15:
            decision = "sensitivity_review"
            cats.append("stationarity_review")
            reasons.append("high warning count")
        if not reasons:
            reasons.append("no subject-level Stage 1.5 v2 rule triggered")
        decisions.append(decision)
        categories_all.append(";".join(sorted(set(cats))))
        reasons_all.append(_join_reasons(reasons))
    out["decision"] = decisions
    out["reason_category"] = categories_all
    out["reason"] = reasons_all
    out["primary_stage2_include"] = True
    out["exclude_review_subjects_include"] = out["decision"].eq("keep")
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in view.columns) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    roi_decisions: pd.DataFrame,
    subject_decisions: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision_counts = roi_decisions.groupby(["atlas", "decision"]).size().reset_index(name="n")
    subject_counts = subject_decisions.groupby(["atlas", "group", "decision"]).size().reset_index(name="n")
    key_roi = roi_decisions[
        roi_decisions["roi_index_1based"].isin(
            sorted(CONSERVATIVE_EXCLUDE_ROIS_1BASED | REVIEW_ROIS_1BASED)
        )
    ][
        [
            "atlas",
            "roi_index_1based",
            "region_name",
            "zero_subject_frequency",
            "constant_subject_frequency",
            "extreme_amplitude_frequency",
            "high_acf_frequency",
            "spectral_warning_frequency",
            "stationarity_review_frequency",
            "decision",
            "reason",
        ]
    ].sort_values("roi_index_1based")
    top_subjects = subject_decisions.sort_values("warning_count", ascending=False).head(12)
    diag_summary = (
        diagnostic.groupby(["atlas", "roi_index_1based", "region_name", "group"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            extreme_frequency=("extreme_amplitude_flag", _frequency),
            high_acf_frequency=("high_acf_flag", _frequency),
            spectral_warning_frequency=("spectral_warning_flag", _frequency),
            stationarity_review_frequency=("stationarity_review_flag", _frequency),
            median_acf1=("acf_lag_1", "median"),
            median_low_high_power_ratio=("low_high_power_ratio", "median"),
            median_amplitude=("amplitude", "median"),
        )
        if not diagnostic.empty
        else pd.DataFrame()
    )
    text = [
        "# Stage 1.5 fMRI ROI Decision Layer",
        "",
        "This report converts Stage 1 QC features into pipeline decisions for Stage 2.",
        "",
        "## Decision Vocabulary",
        "",
        "- `keep`: allowed into conservative Stage 2 inputs.",
        "- `review`: can be used, but must be called out and tested in sensitivity.",
        "- `exclude_conservative`: excluded from conservative FC/metric runs; may appear only in sensitivity or diagnostics.",
        "- `exclude_all`: reserved for future rules when a subject/ROI is unusable in every branch.",
        "",
        "## Hard Conservative ROI Rule",
        "",
        "ROI 35, 36, 81, and 82 are excluded from conservative FC because they are zero/constant in every subject.",
        "",
        "ROI 106 and 111 are not global exclusions, but they are subject-level warnings for SZ/1177_Ivanov_S_A.",
        "",
        "## ROI Decision Counts",
        "",
        _markdown_table(decision_counts),
        "",
        "## Key ROI Decisions",
        "",
        _markdown_table(key_roi),
        "",
        "## Subject Decision Counts",
        "",
        _markdown_table(subject_counts),
        "",
        "## Highest-Warning Subjects",
        "",
        _markdown_table(
            top_subjects[
                [
                    "atlas",
                    "group",
                    "subject_id",
                    "n_zero_roi",
                    "n_extreme_amplitude_roi",
                    "median_ar1",
                    "low_freq_dominance_frequency",
                    "warning_count",
                    "decision",
                    "reason",
                ]
            ]
        ),
        "",
        "## Extreme ROI Mini-Diagnostic: 133, 134, 160, 167",
        "",
        _markdown_table(diag_summary),
        "",
        "## Preprocessing Branch Recommendation",
        "",
        "Minimum Stage 2 branches:",
        "",
        "- `baseline`: conservative ROI set, no extra temporal correction.",
        "- `detrended`: conservative ROI set after linear detrending.",
        "- `AR1_residualized`: conservative ROI set after AR(1) residualization.",
        "- `AR1_plus_detrended`: conservative ROI set after both operations.",
        "",
        "Sensitivity branches:",
        "",
        "- `AR2_residualized`: sensitivity only.",
        "- `with_GSR`: sensitivity only unless a valid global signal definition is explicitly available.",
        "- `include_review_roi`: sensitivity only, includes review ROI such as 133, 134, 160, 167.",
        "- `include_subject_1177_extra_roi`: sensitivity only for ROI 106/111 behavior in SZ/1177.",
        "",
        "## Stage 2 Gate",
        "",
        "Do not start the full metric x lag x window scan until conservative ROI and subject review decisions are acknowledged in the report.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def write_branch_recommendations(path: Path) -> None:
    rows = [
        {
            "branch": "baseline",
            "tier": "minimum_stage2",
            "roi_policy": "exclude_conservative ROI only",
            "subject_policy": "keep all subjects, report review subjects",
            "description": "Conservative ROI set with no extra temporal correction.",
        },
        {
            "branch": "detrended",
            "tier": "minimum_stage2",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "Linear detrending sensitivity for drift/shift robustness.",
        },
        {
            "branch": "AR1_residualized",
            "tier": "minimum_stage2",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "AR(1) residualization branch for autocorrelation robustness.",
        },
        {
            "branch": "AR1_plus_detrended",
            "tier": "minimum_stage2",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "Combined detrending and AR(1) residualization branch.",
        },
        {
            "branch": "AR2_residualized",
            "tier": "sensitivity",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "AR(2) residualization is sensitivity-only until stability is checked.",
        },
        {
            "branch": "with_GSR",
            "tier": "sensitivity",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "Use only if a valid global signal definition is explicit; otherwise report as ROI-level approximation.",
        },
        {
            "branch": "include_review_roi",
            "tier": "sensitivity",
            "roi_policy": "include review ROI, still exclude conservative systemic-zero ROI",
            "subject_policy": "same as baseline",
            "description": "Tests whether review ROI such as 133, 134, 160, and 167 change results.",
        },
        {
            "branch": "exclude_review_subjects",
            "tier": "sensitivity",
            "roi_policy": "same as baseline",
            "subject_policy": "exclude subjects marked review",
            "description": "Tests whether subject-level warnings drive the result.",
        },
    ]
    _write_csv(pd.DataFrame(rows), path)


def write_branch_recommendations_v2(path: Path) -> None:
    rows = [
        {
            "branch": "baseline",
            "tier": "minimum_stage2",
            "roi_policy": "include keep + qc_flag_keep; exclude exclude_conservative + sensitivity_only + exclude_all",
            "subject_policy": "include all subjects; report subject decisions",
            "description": "Primary conservative ROI set with no extra temporal correction.",
        },
        {
            "branch": "detrended",
            "tier": "minimum_stage2",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "Linear detrending branch for drift/shift robustness.",
        },
        {
            "branch": "AR1_residualized",
            "tier": "minimum_stage2",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "AR(1) residualization branch for autocorrelation robustness.",
        },
        {
            "branch": "AR1_plus_detrended",
            "tier": "minimum_stage2",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "Combined detrending and AR(1) residualization branch.",
        },
        {
            "branch": "AR2_residualized",
            "tier": "sensitivity",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "AR(2) residualization is sensitivity-only until stability is checked.",
        },
        {
            "branch": "with_GSR",
            "tier": "sensitivity",
            "roi_policy": "same as baseline",
            "subject_policy": "same as baseline",
            "description": "Use only if a valid global signal definition is explicit; otherwise report as ROI-level approximation.",
        },
        {
            "branch": "include_review_roi",
            "tier": "sensitivity",
            "roi_policy": "include keep + qc_flag_keep + sensitivity_only; still exclude exclude_conservative + exclude_all",
            "subject_policy": "same as baseline",
            "description": "Tests whether sensitivity-only ROI change results without reintroducing background/systemic-zero ROI.",
        },
        {
            "branch": "exclude_review_subjects",
            "tier": "sensitivity",
            "roi_policy": "same as baseline",
            "subject_policy": "exclude subjects not marked keep; report changed HC/SZ balance",
            "description": "Sensitivity-only because subject review is group-imbalanced and cannot be interpreted as primary.",
        },
    ]
    _write_csv(pd.DataFrame(rows), path)


def write_report_v2(
    path: Path,
    roi_decisions: pd.DataFrame,
    subject_decisions: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision_counts = roi_decisions.groupby(["atlas", "decision"]).size().reset_index(name="n")
    subject_counts = subject_decisions.groupby(["atlas", "group", "decision"]).size().reset_index(name="n")
    include_counts = (
        roi_decisions.groupby(["atlas", "primary_stage2_include", "include_review_roi_include"])
        .size()
        .reset_index(name="n")
    )
    key_roi = roi_decisions[
        roi_decisions["roi_index_1based"].isin(
            sorted(V2_CONSERVATIVE_EXCLUDE_ROIS_1BASED | REVIEW_ROIS_1BASED)
        )
    ][
        [
            "atlas",
            "roi_index_1based",
            "region_name",
            "zero_subject_frequency",
            "constant_subject_frequency",
            "extreme_amplitude_frequency",
            "high_acf_frequency",
            "spectral_warning_frequency",
            "stationarity_review_frequency",
            "decision",
            "reason_category",
            "reason",
        ]
    ].sort_values("roi_index_1based")
    top_subjects = subject_decisions.sort_values("warning_count", ascending=False).head(12)
    diag_summary = (
        diagnostic.groupby(["atlas", "roi_index_1based", "region_name", "group"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            extreme_frequency=("extreme_amplitude_flag", _frequency),
            high_acf_frequency=("high_acf_flag", _frequency),
            spectral_warning_frequency=("spectral_warning_flag", _frequency),
            stationarity_review_frequency=("stationarity_review_flag", _frequency),
            median_acf1=("acf_lag_1", "median"),
            median_low_high_power_ratio=("low_high_power_ratio", "median"),
            median_amplitude=("amplitude", "median"),
        )
        if not diagnostic.empty
        else pd.DataFrame()
    )
    keep_balance = (
        subject_decisions[subject_decisions["exclude_review_subjects_include"].astype(bool)]
        .groupby(["atlas", "group"])
        .size()
        .reset_index(name="n_keep_if_exclude_review_subjects")
    )
    text = [
        "# Stage 1.5 v2 fMRI ROI Decision Layer",
        "",
        "This report normalizes Stage 1.5 decision semantics before any full Stage 2 grid scan.",
        "",
        "## Decision Vocabulary",
        "",
        "ROI decisions: `keep`, `qc_flag_keep`, `sensitivity_only`, `exclude_conservative`, `exclude_all`.",
        "",
        "Subject decisions: `keep`, `qc_flag_keep`, `sensitivity_review`, `exclude_all`.",
        "",
        "High ACF and spectral warnings are QC flags, not removal reasons by themselves.",
        "",
        "## Primary Stage 2 Policy",
        "",
        "- ROI included: `keep + qc_flag_keep`.",
        "- ROI excluded: `exclude_conservative + sensitivity_only + exclude_all`.",
        "- Subjects included: all subjects, with warnings reported.",
        "- Primary branches: `baseline`, `detrended`, `AR1_residualized`, `AR1_plus_detrended`.",
        "",
        "## Sensitivity Policy",
        "",
        "- `include_review_roi` includes `sensitivity_only` ROI but still excludes Background/systemic-zero ROI.",
        "- `exclude_review_subjects` is sensitivity-only and changes HC/SZ balance.",
        "- `with_GSR` is sensitivity-only unless a valid global signal definition is explicit.",
        "",
        "## Hard Conservative ROI Rule",
        "",
        "ROI 1, 35, 36, 81, and 82 are excluded from conservative FC. ROI 1 is Background.",
        "",
        "## ROI Decision Counts",
        "",
        _markdown_table(decision_counts),
        "",
        "## ROI Inclusion Counts",
        "",
        _markdown_table(include_counts),
        "",
        "## Key ROI Decisions",
        "",
        _markdown_table(key_roi),
        "",
        "## Subject Decision Counts",
        "",
        _markdown_table(subject_counts),
        "",
        "## Exclude-Review-Subjects Balance Warning",
        "",
        _markdown_table(keep_balance),
        "",
        "## Highest-Warning Subjects",
        "",
        _markdown_table(
            top_subjects[
                [
                    "atlas",
                    "group",
                    "subject_id",
                    "n_zero_roi",
                    "n_extreme_amplitude_roi",
                    "median_ar1",
                    "low_freq_dominance_frequency",
                    "warning_count",
                    "decision",
                    "reason_category",
                    "reason",
                ]
            ]
        ),
        "",
        "## Extreme ROI Mini-Diagnostic: 133, 134, 160, 167",
        "",
        _markdown_table(diag_summary),
        "",
        "## Stage 2 Gate",
        "",
        "Run only the sanity scan next. Do not start the full metric x lag x window grid until the sanity report is reviewed.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characterization-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--aal3-regions", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.characterization_dir)
    output = Path(args.output_dir) if args.output_dir else root / "decisions"
    regions = _read_regions(Path(args.aal3_regions) if args.aal3_regions else None)
    features = pd.read_csv(root / "roi_signal_characterization_all_features.csv")
    subject_summary = pd.read_csv(root / "qc" / "subject_qc_summary.csv")
    roi_decisions, diagnostic = build_roi_decisions(features, regions)
    subject_decisions = build_subject_decisions(features, subject_summary)
    _write_csv(roi_decisions, output / "roi_decision_layer.csv")
    _write_csv(subject_decisions, output / "subject_decision_layer.csv")
    _write_csv(diagnostic, output / "extreme_roi_133_134_160_167_diagnostic.csv")
    write_branch_recommendations(output / "preprocessing_branch_recommendations.csv")
    write_report(output / "stage15_decision_report.md", roi_decisions, subject_decisions, diagnostic)
    roi_decisions_v2, diagnostic_v2 = build_roi_decisions_v2(features, regions)
    subject_decisions_v2 = build_subject_decisions_v2(features, subject_summary)
    _write_csv(roi_decisions_v2, output / "roi_decision_layer_v2.csv")
    _write_csv(subject_decisions_v2, output / "subject_decision_layer_v2.csv")
    _write_csv(diagnostic_v2, output / "extreme_roi_133_134_160_167_diagnostic_v2.csv")
    write_branch_recommendations_v2(output / "preprocessing_branch_recommendations_v2.csv")
    write_report_v2(
        output / "stage15_decision_report_v2.md",
        roi_decisions_v2,
        subject_decisions_v2,
        diagnostic_v2,
    )


if __name__ == "__main__":
    main()
