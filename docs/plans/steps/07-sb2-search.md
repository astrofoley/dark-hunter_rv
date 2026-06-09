---
step_id: 07-sb2-search
phase: D
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/44
branches:
  - step/07a-sb2-detection
  - step/07b-sb2-reporting
  - step/07c-sb2-orbit-optional
depends_on: [06-strong-line-line-list]
blocks: []
master_todo_id: sb2-search
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 07: SB2 search and reporting

## Goal / science outcome

Detect and report double-lined systems where appropriate; optional two-lined orbit extension.

## Scope (in) / non-goals (out)

**In:** CCF asymmetry / dual-Gaussian / profile flags; `sb2_candidate`, primary/secondary RV columns when resolved; Gaia NSS cross-link.

**Out (07c optional):** Full SB2 Keplerian MCMC in `fit_apf_rv_keplerian.py`.

## Prerequisites

- Mask CCF diagnostics (`gauss_ok`, peak shape)
- Multi-epoch data for known binaries

## Implementation tasks

### 07a (`step/07a-sb2-detection`)

- [ ] Prototype dual-Gaussian CCF fit or bisector metric in `rv_core.py`
- [ ] Per-chunk SB2 scores in diagnostics

### 07b (`step/07b-sb2-reporting`)

- [ ] Exposure-level `sb2_candidate` flag + columns in CSV/summary
- [ ] Validation report: fraction flagged vs Gaia NSS SB2

### 07c (`step/07c-sb2-orbit-optional`, defer if needed)

- [ ] Two-lined Keplerian likelihood sketch or separate module

## Key files

- `darkhunter_rv/rv_core.py`
- `darkhunter_rv/pipeline.py`
- `fit_apf_rv_keplerian.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m darkhunter_rv.pipeline ... --run-all-methods
```

## Acceptance criteria

- SB2 flag does not fire on high-S/N cool calibration stars (low false positive)
- Known asymmetric CCF test case flagged
- Documented limitations (single-lined orbit fitter unchanged unless 07c done)

## Tests / validation

- Synthetic double-lined injection test
- Manual check on suspect epochs from overlap discordance

## Propagation checklist (on merge)

- [ ] Master todo `sb2-search` → completed
- [ ] Update `rv_pipeline_roadmap` phase 3 binary section

## Open decisions

- Spectral decomposition per epoch vs time-series only?
- 07c in scope for this step or separate future step?
