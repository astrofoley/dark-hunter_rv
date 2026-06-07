---
step_id: 08-external-rv-crosscheck
phase: E
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/45
branches:
  - step/08-external-rv-crosscheck
depends_on: [00-literature-rv-master, 07-sb2-search]
blocks: []
master_todo_id: external-rv-crosscheck
related_legacy_plans:
  - pipeline_legacy_diagnostics_e98cdb98.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 08: External RV cross-check (literature + catalogs)

## Goal / science outcome

Systematic comparison of pipeline adopted RVs and orbit fits to published literature (primary: El-Badry 2024) and LAMOST/RAVE.

## Scope (in) / non-goals (out)

**In:** `validation/compare_literature_rvs.py`; join on `gaia_dr3_id` + nearest BJD; orbit-fit overlay from master CSV; LAMOST/RAVE extension.

**Out:** Rebuilding literature master (step 00).

## Prerequisites

- `calibration/literature_rv_master.csv`
- Pipeline summaries and diagnostics for overlapping Gaia IDs

## Implementation tasks

- [ ] CLI: load master CSV + pipeline `*_summary.txt` / diagnostics
- [ ] Per-epoch ΔRV vs published err; per-star bias/RMS tables
- [ ] Optional: wire literature points in `fit_apf_rv_keplerian.py` plots from master CSV
- [ ] Extend to `external_rvs` from star summaries (LAMOST/RAVE)
- [ ] Playbook recipes and example output paths

## Key files

- `calibration/literature_rv_master.csv`
- `validation/compare_literature_rvs.py` (new)
- `validation/diagnose_legacy_campaign.py` (reference joins)
- `fit_apf_rv_keplerian.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.compare_literature_rvs \
  --master calibration/literature_rv_master.csv \
  --summary-dir output \
  --report-dir validation_output/literature_crosscheck
```

## Acceptance criteria

- Report covers ≥10 El-Badry 2024 stars with both APF and literature epochs
- Published M_star, M2, P_orb joined for orbit-fit QA table
- LAMOST/RAVE rows compared where present in summaries

## Tests / validation

- Unit test: nearest BJD join logic
- Compare Gaia NS1 literature vs pipeline epoch table

## Propagation checklist (on merge)

- [ ] Master todo `external-rv-crosscheck` → completed
- [ ] Update master plan literature section with CLI path

## Open decisions

- Match adopted RV vs mask-only vs all methods in comparison?
