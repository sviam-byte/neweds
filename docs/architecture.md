# Architecture Contract

This document defines the contract for a literature-aware research-audit skill.
It is intentionally stricter than a narrative architecture note: the goal is to
remove ambiguity between evidence levels, material scope, novelty search
requirements, direction scoring, and runtime transitions.

## Design goals

- Separate "how strong is the evidence" from "what material was inspected".
- Prevent fake novelty confidence when no verifiable trace exists.
- Keep claim ontology stable across opportunity mining and direction selection.
- Encode neuroimaging leakage nuance instead of overgeneralizing tabular ML rules.
- Make runtime behavior compact, explicit, and testable.

## Two-axis evidence model

Do not encode evidence strength and inspected material in the same numeric level.
They represent different questions and must remain separate.

### EvidenceStrength

`EvidenceStrength` measures how strong the audit evidence is.

| Value | Meaning |
| --- | --- |
| `0` | `idea_only` |
| `1` | `static_text` |
| `2` | `static_code` |
| `3` | `evidence_bound` |
| `4` | `enemy_aware` |
| `5` | `publication_ready` |

### MaterialScope

`MaterialScope` measures which materials were actually inspected.

| Value | Meaning |
| --- | --- |
| `literature_only` | Search or bibliography-level literature scan |
| `paper_only` | Paper or manuscript text only |
| `code_only` | Code repository or implementation only |
| `paper_plus_code` | Paper/manuscript plus code |
| `paper_plus_code_plus_outputs` | Paper, code, and execution outputs |

### Normalization rule

- Never write `Level 3`, `Level 4`, or similar shorthand without naming the axis.
- Every verdict that mentions evidence must carry both `evidence_strength` and
  `material_scope`.

## Formal objects

The following objects are the minimum schema required for novelty and enemy
search reasoning.

### AuditRequest

```json
{
  "request_id": "REQ-001",
  "user_goal": "Assess whether a claim is new and reproducible",
  "claim_text": "Dynamic connectivity marker X predicts outcome Y",
  "claim_atoms": {
    "domain": "resting-state fMRI",
    "object": "dynamic connectivity marker X",
    "operation": "predicts",
    "endpoint": "outcome Y",
    "control": "age and motion",
    "consequence": "improved discrimination"
  },
  "user_materials": [],
  "requested_mode": "audit"
}
```

### PaperRecord

`PaperRecord` is mandatory whenever a search trace names, counts, compares, or
screens papers.

```json
{
  "paper_id": "P001",
  "title": "Example title",
  "authors": ["Author A", "Author B"],
  "year": 2025,
  "venue_or_server": "bioRxiv",
  "identifier_type": "DOI",
  "identifier": "10.1101/2025.01.01.123456",
  "raw_url": "https://doi.org/10.1101/2025.01.01.123456",
  "verification_status": "verified",
  "relevance_summary": "Closest prior work on the same endpoint",
  "overlap_with_claim": [
    "same population",
    "same endpoint",
    "different operation"
  ]
}
```

Allowed `verification_status` values:

- `verified`
- `unverified`
- `user_provided_unverified`

### EnemyPaperRecord

`EnemyPaperRecord` must reference a `paper_id` instead of duplicating bibliographic
fields as a second loose object.

```json
{
  "paper_id": "P001",
  "enemy_rank": 1,
  "threat_type": "direct_overlap",
  "threat_summary": "Same endpoint and similar feature family",
  "difference_from_current_claim": [
    "different cohort",
    "stronger baseline",
    "weaker controls in current claim"
  ]
}
```

### SearchTrace

`SearchTrace` is required before any novelty, freshness, or enemy-density claim.

```json
{
  "trace_id": "TRACE-001",
  "trace_basis": "tool_executed",
  "material_scope": "literature_only",
  "queries": [
    "resting-state fMRI dynamic connectivity outcome Y biomarker",
    "dynamic connectivity outcome Y classification site harmonization"
  ],
  "sources_used": [
    "PubMed",
    "Google Scholar",
    "arXiv"
  ],
  "paper_records": ["P001", "P002", "P003"],
  "enemy_papers": ["P001"],
  "screened_count": 12,
  "direct_enemy_count": 1,
  "nearest_hits": ["P001", "P002"],
  "search_date": "2026-04-29",
  "freshness_status": "freshness_uncertain",
  "notes": "One direct overlap found; two partial overlaps"
}
```

Allowed `trace_basis` values:

- `tool_executed`
- `user_provided_bibliography`
- `manual_user_report`
- `model_estimate_forbidden`

Allowed `freshness_status` values:

- `freshness_unchecked`
- `freshness_uncertain`
- `freshness_safe`
- `user_reported_freshness_safe`

### Search-trace safety rules

- `screened_count`, `direct_enemy_count`, `nearest_hits`, and `freshness_safe`
  are forbidden unless backed by `SearchTrace` or explicit user-provided
  bibliography.
- If `trace_basis = manual_user_report`, the maximum freshness verdict is
  `freshness_uncertain` unless the output explicitly uses
  `user_reported_freshness_safe`.
- If no `SearchTrace` exists, freshness must remain `freshness_unchecked`.
- `model_estimate_forbidden` exists only to mark invalid attempted outputs and
  must never appear in a final accepted record.

## Claim ontology

Use separate names for claim atoms and lego combinations.

### `claim_atoms`

These are semantic parts of the user's claim:

- `domain`
- `object`
- `operation`
- `endpoint`
- `control`
- `consequence`

### `lego_components`

These are recombination slots used for direction mining, not claim parsing:

- `component_1`
- `component_2`
- `component_3`

### Naming rule

- Do not reuse `A/B/C` labels across both systems.
- Do not describe a lego combination as if it were a parsed claim atom list.

## Direction scoring

Direction scores are useful only if they are explicitly capped by missing
evidence and safety failures.

### Score dimensions

Use a bounded total score from `0` to `85`, but apply the following caps before
selecting a "best next direction".

### Hard caps

- If no `SearchTrace` exists, `novelty_mechanism` is capped at `3/5`.
- If no execution outputs exist, `evidence` is capped at `2/5`.
- If no code execution occurred, `reproducibility` is capped at `3/5`.
- If `confound_safety <= 1`, total score is capped at `50`.
- If the direction is `lego_incremental`, it cannot be the best next direction
  unless reframed as a benchmark, audit, or negative-control study.

### Selection rule

Narrative enthusiasm must never override caps. If a cap applies, the capped
value wins even when the prose sounds persuasive.

## Leakage policy with neuroimaging nuance

Default leakage language must distinguish subject-local preprocessing from
cohort-level fitted transforms.

### Allowed before cross-validation

Subject-local preprocessing may occur before CV when all of the following are
true:

- It is fitted only within one subject's own time series.
- It is label-independent.
- It does not borrow statistics from other subjects or folds.

Typical examples include nuisance regression inside a subject run and
subject-local temporal detrending.

### Must be fit inside train folds

The following remain train-fold-only when they use cross-subject information:

- imputation across subjects
- scaling across subjects
- PCA across subjects
- harmonization or site correction
- feature selection
- any learned transformation using pooled cohort statistics

### Circularity flags

Treat the following as leakage or selection circularity:

- atlas selection based on test AUC
- branch or model selection based on test performance
- choosing preprocessing settings after seeing held-out endpoint performance

## Runtime architecture

The runtime model is intentionally small. During a run, the system should hold a
limited number of active records and avoid long, free-floating scratchpads.

See [runtime_contract.md](runtime_contract.md) for state-by-state rules.

## Test-case contract

Golden cases should also carry machine-checkable assertions.

Recommended fields:

```yaml
case_id: case_12
must_include:
  - user_reported_freshness_safe
  - lego novelty defense
  - required searches are listed
must_not_include:
  - "This is novel"
  - "therefore novel"
pass_condition:
  - verdict is downgraded
  - no strong novelty verdict without SearchTrace
```

### Test authoring rules

- Every case must define `must_include`.
- Every case must define `must_not_include`.
- Every case must define at least one `pass_condition`.
- Test fixtures must not rely on implicit schema values that are absent from the
  formal objects above.
