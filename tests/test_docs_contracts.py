from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_architecture_contract_defines_two_axis_evidence_model() -> None:
    text = _read("docs/architecture.md")
    assert "EvidenceStrength" in text
    assert "MaterialScope" in text
    assert "idea_only" in text
    assert "paper_plus_code_plus_outputs" in text


def test_architecture_contract_defines_search_trace_and_paper_records() -> None:
    text = _read("docs/architecture.md")
    assert "PaperRecord" in text
    assert "EnemyPaperRecord" in text
    assert "SearchTrace" in text
    assert "trace_basis" in text
    assert "user_reported_freshness_safe" in text


def test_architecture_contract_separates_claim_atoms_and_lego_components() -> None:
    text = _read("docs/architecture.md")
    assert "claim_atoms" in text
    assert "lego_components" in text
    assert "component_1" in text


def test_architecture_contract_contains_score_caps_and_neuroimaging_nuance() -> None:
    text = _read("docs/architecture.md")
    assert "novelty_mechanism" in text
    assert "confound_safety <= 1" in text
    assert "Subject-local preprocessing" in text
    assert "Must be fit inside train folds" in text


def test_runtime_contract_contains_required_flow_constraints() -> None:
    text = _read("docs/runtime_contract.md")
    assert "Create one `AuditRequest`" in text
    assert "Keep at most five active records at a time" in text
    assert "Never produce a novelty verdict without `SearchTrace`" in text
    assert "forbidden-verdict check" in text


def test_fmri_roi_audit_docs_include_subject_level_outputs_and_limits() -> None:
    text = _read("docs/fmri_roi_audit.md")
    assert "subject_level_fc_summary.csv" in text
    assert "subject_level_group_comparison.csv" in text
    assert "outputs/figures" in text
    assert "outputs/preprocessed" in text
    assert "<group>_<subject_id>_pearson_z.npy" in text
    assert "AR(2) coefficients" in text
    assert "fc_group_comparison_edges_ttest.csv" in text
    assert "permutation_summary.csv" in text
    assert "conservative common-bad-ROI baseline remains the default FC path" in text
    assert "alternative FC/statistics path" in text
    assert "threshold" in text
    assert "hcp_region_adjacency_report.csv" in text
    assert "volume-space atlas geometry only" in text
    assert "functional homogeneity" in text
    assert "mean_abs_fc" in text
    assert "not a classifier" in text
    assert "Lobe/system summaries are intentionally deferred" in text
