# Runtime Contract

This file makes the state machine executable enough for implementation and
review. It does not attempt to encode every internal heuristic. It defines the
minimum required inputs, outputs, transitions, downgrade behavior, and terminal
verdict constraints.

## Global run rules

For each run:

1. Create one `AuditRequest`.
2. Select one mode before producing any verdict.
3. Keep at most five active records at a time.
4. Never produce a novelty verdict without `SearchTrace` or explicitly
   user-scoped freshness language.
5. Always end with an allowed verdict and an explicit forbidden-verdict check.

## States

### `request_created`

Input:

- raw user request

Output:

- `AuditRequest`

Allowed transitions:

- `mode_selected`
- `failed`

Failure conditions:

- no extractable claim text
- contradictory user scope that cannot be normalized

Downgrade behavior:

- convert to a scope-clarification request

Compact output contract:

- one normalized `AuditRequest`

### `mode_selected`

Input:

- `AuditRequest`

Output:

- mode record with `audit`, `literature_scan`, `code_audit`, or `direction_scan`

Allowed transitions:

- `materials_loaded`
- `failed`

Failure conditions:

- impossible mode request, such as publication-grade verdict with no materials

Downgrade behavior:

- lower to the strongest admissible mode

Compact output contract:

- one mode record

### `materials_loaded`

Input:

- `AuditRequest`
- user materials, if any

Output:

- material inventory
- `material_scope`

Allowed transitions:

- `search_traced`
- `code_audited`
- `direction_scored`
- `failed`

Failure conditions:

- inaccessible required material

Downgrade behavior:

- reduce `material_scope`

Compact output contract:

- one material inventory

### `search_traced`

Input:

- material inventory
- search results or user bibliography

Output:

- `SearchTrace`
- zero or more `PaperRecord`
- zero or more `EnemyPaperRecord`

Allowed transitions:

- `novelty_evaluated`
- `direction_scored`
- `failed`

Failure conditions:

- claims about screening counts with no trace basis
- papers named without identifiers or raw URL

Downgrade behavior:

- set `freshness_status` to `freshness_unchecked` or `freshness_uncertain`
- delete unsupported counts

Compact output contract:

- one `SearchTrace`
- up to five total paper-related records active at once

### `code_audited`

Input:

- code materials

Output:

- code audit summary
- reproducibility notes
- leakage findings with neuroimaging nuance applied

Allowed transitions:

- `novelty_evaluated`
- `direction_scored`
- `failed`

Failure conditions:

- leakage conclusions stated without distinguishing subject-local and
  cross-subject fitting

Downgrade behavior:

- convert categorical leakage claims into conditional findings

Compact output contract:

- one code audit summary

### `novelty_evaluated`

Input:

- `SearchTrace`
- optional code audit summary

Output:

- novelty verdict
- freshness verdict
- enemy-density summary

Allowed transitions:

- `direction_scored`
- `finalized`
- `failed`

Failure conditions:

- strong novelty verdict with no `SearchTrace`
- `freshness_safe` with `manual_user_report`

Downgrade behavior:

- downgrade to `freshness_unchecked`, `freshness_uncertain`, or
  `user_reported_freshness_safe`

Compact output contract:

- one novelty verdict block

### `direction_scored`

Input:

- novelty block
- code audit summary
- output evidence status

Output:

- scored direction list
- cap explanations
- best-next-direction decision or refusal

Allowed transitions:

- `finalized`
- `failed`

Failure conditions:

- score caps ignored
- lego-incremental direction selected despite exclusion rule

Downgrade behavior:

- cap scores
- reframe direction as benchmark or audit

Compact output contract:

- ranked directions with cap annotations

### `finalized`

Input:

- any admissible prior state output

Output:

- final compact report

Allowed transitions:

- terminal only

Failure conditions:

- forbidden verdict present

Downgrade behavior:

- remove forbidden verdict
- re-emit downgraded final report

Compact output contract:

- allowed verdict
- forbidden verdict check
- explicit uncertainty statements where required

### `failed`

Input:

- any state

Output:

- failure record
- fallback recommendation

Allowed transitions:

- terminal only

Compact output contract:

- short explanation
- strongest safe fallback
