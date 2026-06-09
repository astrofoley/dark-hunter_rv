---
step_id: 05-short-pair-epoch-ccf
phase: E
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/42
branches:
  - step/05a-short-pair-calibration
  - step/05b-epoch-ccf-consistency
depends_on: [04-adopted-rv-match-plots]
blocks: [06-strong-line-line-list]
master_todo_id: short-pair-epoch-ccf
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
  - pr_repo_organization_326c519c.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
  - docs/operations.md
---

# Step 05: Short-pair calibration and epoch–epoch CCF QC

## Goal / science outcome

Use closely spaced epoch pairs (~0 RV variation) for pipeline calibration checks; use spectrum–spectrum CCF as zeropoint-free consistency monitor.

## Scope (in) / non-goals (out)

**In:** `validation/find_short_pairs.py`; `validation/epoch_ccf_consistency.py`; optional hook in calibration setup.

**Out:** Primary RV measurement via epoch CCF.

## Prerequisites

- Multi-epoch stars in `output/*_summary.txt`
- Normalized spectra or pipeline re-read path

## Implementation tasks

### 05a (`step/05a-short-pair-calibration`)

- [ ] Find pairs with Δt &lt; configurable threshold (default: same night)
- [ ] Report ΔRV per method; flag pairs violating ~0 km/s assumption
- [ ] Integrate into `run_calibration_setup` docs

### 05b (`step/05b-epoch-ccf-consistency`)

- [ ] CCF normalized epoch A vs B on log-λ grid
- [ ] Compare ΔRV(CCF) vs adopted RV difference
- [ ] Per-star CSV + flag epochs above threshold

## Key files

- `validation/find_short_pairs.py` (new)
- `validation/epoch_ccf_consistency.py` (new)
- `validation/run_calibration_setup.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.find_short_pairs --summary-dir output --max-delta-days 1
python -m validation.epoch_ccf_consistency --gaia-id 6328149636482597888 ...
```

## Acceptance criteria

- Short-pair report runs on full survey output
- Epoch CCF flags disagreeing epochs on test star (e.g. Gaia NS1)
- Playbook documents Δt and threshold parameters

## Tests / validation

- Unit tests with synthetic pair timestamps
- Manual run on J1432-1021 epochs

## Propagation checklist (on merge)

- [ ] Master todo `short-pair-epoch-ccf` → completed

## Open decisions

- Same night only vs &lt;24 h?
- Phase-gate known binaries?
