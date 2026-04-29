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
