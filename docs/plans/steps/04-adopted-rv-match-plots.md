---
step_id: 04-adopted-rv-match-plots
phase: D
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/41
branches:
  - step/04-adopted-rv-match-plots
depends_on: [03-method-fusion-coverage]
blocks: [05-short-pair-epoch-ccf]
master_todo_id: adopted-rv-match-plots
related_legacy_plans:
  - legacy_plot_and_ccf_qc_2f3b70cd.plan.md
  - gaia_cache_and_ccf_diagnostics_2e731512.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 04: Adopted-RV match diagnostic plots

## Goal / science outcome

Routine per-exposure figure: continuum-normalized spectrum with stellar mask and strong-line markers at **debiased adopted RV** (not only outlier PDFs).

## Scope (in) / non-goals (out)

**In:** New plot function in `plotting.py`; hook from `pipeline.py` when `--plots` / `--plots-focus`; uses cascade adopted RV + debias.

**Out:** Replacing `plot_legacy_outlier_orders` (keep for disagreements).

## Prerequisites

- Adopted RV cascade stable
- Existing mask schematic helpers in `plotting.py`

## Implementation tasks

- [ ] `plot_adopted_rv_match(spec, adopted_rv, debiased, mask, strong_lines)` in `plotting.py`
- [ ] Write `{stem}_adopted_rv_match.png` from `process_spectrum`
- [ ] Include in `--plots-focus` default bundle
- [ ] One fixture smoke test or validation script check

## Key files

- `darkhunter_rv/plotting.py`
- `darkhunter_rv/pipeline.py`
- `validation/plot_legacy_outlier_orders.py` (reference layout)

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m darkhunter_rv.pipeline /path/to/Gaia_DR3_*.txt --instrument APF --plots --plots-focus
```

## Acceptance criteria

- Every plotted exposure shows mask overlay at adopted RV after debias
- Strong-line rest positions marked when `strong_lines` row exists
- Plot generated without requiring outlier threshold

## Tests / validation

- Visual check on 3 calibration stars + 1 hot star
- Optional: PNG exists assertion in smoke test

## Propagation checklist (on merge)

- [ ] Master todo `adopted-rv-match-plots` → completed
- [ ] Note in `legacy_plot_and_ccf_qc` plan

## Open decisions

- All orders vs single representative order panel?
