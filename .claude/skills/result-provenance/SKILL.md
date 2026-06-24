---
name: result-provenance
description: >-
  Use when producing, packaging, or auditing any research/analysis result in this repo
  (model runs, screenings, QC tables, figures, reports). Enforces the canonical
  new_results provenance protocol: one immutable dated folder per run, with
  ABOUT_THIS_RESULT.md, run_metadata.json, output_inventory.csv and a code_snapshot.
  Trigger on phrases like "по протоколу", "оформи результат", "provenance", "package
  this result", "сделай результат воспроизводимым", or before reporting a finished run.
---

# Research Result Provenance Protocol

Goal: a month from now, a result folder must be understandable **without the chat,
without the operator's memory, and without guessing**. Everything known is recorded
exactly; everything unknown is marked `unknown` — never reconstructed after the fact.

## 1. One canonical root

All results live under a single root: **`new_results/`**.

The root must always contain:

- `README.md`
- `results_catalog.json`

## 2. One immutable dated folder per run

```
new_results/<YYYY-MM-DD>_<short-result-slug>/
```

Example: `new_results/2026-06-24_metric-classifier-allsubj/`

- **Never overwrite an old result to "update the output".**
- A new run **or a new interpretation** = a **new dated folder**.
- Separate different datasets, scopes, evidence levels, and analysis types into
  different result folders.

## 3. Mandatory contents of every result folder

```
<result-folder>/
  ABOUT_THIS_RESULT.md
  run_metadata.json
  output_inventory.csv
  code_snapshot/
    environment.json
    python_packages.txt
    git_diff.patch
    git_status.txt
    git_log_last_10.txt
    files/                # copies of the exact code that produced the result
  <the scientific artifacts>   # tables, figures, json, csv, reports, models, logs
```

If the Git worktree is dirty, the dirty patch **must** be saved to
`code_snapshot/git_diff.patch` (the generator does this automatically).

## 4. ABOUT_THIS_RESULT.md must honestly answer

- what was done;
- why, and what it means;
- which input data were used;
- which command or workflow produced it;
- when it was produced or packaged;
- which code version was used;
- whether the Git tree was dirty;
- which conclusions are **descriptive** vs **inferential/statistical**;
- what limits stronger claims;
- whether it is `fresh_run`, `derived_run`, or `legacy_snapshot_packaged`.

## 5. execution_mode and honesty rules

- `fresh_run` — computed now from inputs.
- `derived_run` — a new interpretation/audit derived from an existing result.
- `legacy_snapshot_packaged` — an old result packaged after the fact without
  proper provenance.

For legacy results, do **not** invent the command, SHA, timestamp, or source data
revision. Write `unknown`. Honest `unknown` beats a plausible reconstruction.

## 6. Generate provenance with the protocol script

Do not hand-write these files. Use:

```powershell
python scripts/write_result_provenance.py `
  --result-dir "<result-dir>" `
  --result-id "<stable-id>" `
  --title "<human title>" `
  --result-type "<type>" `
  --status "<status>" `
  --execution-mode "<fresh_run|derived_run|legacy_snapshot_packaged>" `
  --summary "<what was done>" `
  --meaning "<why it matters>" `
  --command "<exact command, or an honest unknown notice>" `
  --repository "<repo>" `
  --input "<input path>" `        # repeatable
  --code-file "<code path>" `     # repeatable; copied into code_snapshot/files/
  --finding "<short finding>" `   # repeatable
  --limitation "<claim guardrail>" `  # repeatable
  --note "<optional note>"        # repeatable
```

The script writes `run_metadata.json` (schema `research-result-provenance/v1`),
`ABOUT_THIS_RESULT.md`, `output_inventory.csv` (SHA-256 of every artifact), copies
code into `code_snapshot/files/`, and captures the git + environment snapshot. It
also runs an integrity check, so **do not write into the result folder while the
generator runs**.

## 7. After generating, update the catalog

Add or update the run's entry in `new_results/results_catalog.json`
(`result_id`, `title`, `path`, `status`, `execution_mode`, `summary`) and bump
`result_count` + `updated_at`.

## Checklist before calling a result "done"

1. Folder is `new_results/<date>_<slug>/`, not an overwrite of an old one.
2. `ABOUT_THIS_RESULT.md`, `run_metadata.json`, `output_inventory.csv`, `code_snapshot/` all present.
3. `code_snapshot/` has environment.json, python_packages.txt, git_diff.patch, git_status.txt, git_log_last_10.txt, files/.
4. Descriptive vs inferential claims are separated; limitations are explicit.
5. Unknown provenance is written as `unknown`, not invented.
6. Dirty worktree → diff saved.
7. `results_catalog.json` updated.
