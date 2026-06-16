---
step_id: 00-literature-rv-master
phase: E
status: completed
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/37
branches:
  - step/00-literature-rv-master
depends_on: []
blocks: [08-external-rv-crosscheck]
master_todo_id: literature-rv-master
related_legacy_plans: []
repo_docs_to_update: []
---

# Step 00: Literature RV master table

## Goal / science outcome

Single committed reference of published RV epochs from compact-object follow-up papers for external validation and orbit-fit QA.

## Scope (in) / non-goals (out)

**In:** Merge El-Badry 2024 population + Gaia NS1 + Gaia BH1 + Simon 2026 subset into one CSV with uniform columns and reference metadata.

**Out:** Pipeline comparison CLI (step 08); automatic Gaia TAP fetch of these RVs.

## Prerequisites

Local arXiv source trees under `~/Downloads/arXiv-*`.

## Implementation tasks

- [x] `validation/build_literature_rv_master.py` parser and merger
- [x] `calibration/literature_rv_master.csv` (417 rows, 23 systems, 4 references)
- [x] `tests/validation/test_literature_rv_master.py`
- [x] Land on branch `step/00-literature-rv-master` via PR

## Key files

- `/Users/rfoley/darkhunter/rvs/dark-hunter_rv/calibration/literature_rv_master.csv`
- `/Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation/build_literature_rv_master.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.build_literature_rv_master
PYTHONPATH=. python -m pytest tests/validation/test_literature_rv_master.py -q
```

## Acceptance criteria

- 417 observations across 4 references; J1432-1021 present in both population and NS1 papers with distinct `reference_key`
- Every row has reference metadata; missing columns empty where source table lacks them
- Rebuild script is idempotent from default Download paths

## Tests / validation

`tests/validation/test_literature_rv_master.py` (5 tests).

## Propagation checklist (on merge)

- [x] Master plan todo `literature-rv-master` → completed
- [x] PR https://github.com/astrofoley/dark-hunter_rv/pull/46 (merged 2026-06-07)
- [x] INDEX.md merge date (2026-06-07)
- [x] GitHub issue #37 closed (auto via Closes on merge)

## Role in per-method precision program

Step 00 is the **external validation anchor** for all measurement lanes (mask, template, strong-lines), not only the adopted cascade:

- **Phase A absolute gate** (APF ↔ literature, \|ΔRV\| < 1 km/s, `--no-bias`) uses `literature_rv_master.csv` via `validation/rv_phase_a_baseline.py`.
- **Step 08** will join pipeline summaries to this table for orbit-fit QA and per-reference bias.
- **Template lane (step 10)** does not replace literature truth; mask−template residuals are internal; literature pairs test zeropoint after each lane is deployed.

Rebuild when arXiv source trees update; row count and `reference_key` disambiguation remain acceptance criteria.

## Open decisions

None.
