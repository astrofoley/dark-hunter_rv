---
step_id: 03-method-fusion-coverage
phase: C
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/40
branches:
  - step/03-method-fusion-coverage
depends_on: [02-chunk-weights-subchunks]
blocks: [04-adopted-rv-match-plots]
master_todo_id: method-fusion-coverage
related_legacy_plans:
  - rv_method_fusion_plan_b4fd46f0.plan.md
  - rv_methods_evaluation_plan_fcd09d94.plan.md
  - three_rv_methods_e1b72701.plan.md
repo_docs_to_update:
  - docs/rv_methods_evaluation.md
---

# Step 03: Method fusion and coverage reporting

## Goal / science outcome

Honest adopted RVs with calibrated uncertainties, explicit rejections, and coverage tables (`frac_finite` per method vs Teff/S/N).

## Scope (in) / non-goals (out)

**In:** `darkhunter_rv/method_fusion.py`; `rv_accepted` + `reject_reason`; coverage CSVs in diagnostics report; optional `rv_calibrated_kms` columns.

**Out:** Learned ML scorer (v2); changing raw per-method RVs without labeled calibrated columns.

## Prerequisites

- Overlap report CSVs from calibration campaign
- `method_evaluation.py`, `method_regions.py`

## Implementation tasks

- [ ] Add `method_fusion.py` (bias surfaces, σ inflation, discordance gates)
- [ ] Extend `rv_method_diagnostics_report.py` with `binned_method_coverage_vs_teff.csv`
- [ ] Wire optional fusion columns to diagnostics or post-process CSV
- [ ] Tests for reject rules and coverage denominators
- [ ] Document adoption v2 in `docs/rv_methods_evaluation.md`

## Key files

- `darkhunter_rv/method_fusion.py` (new)
- `darkhunter_rv/method_evaluation.py`
- `validation/rv_method_diagnostics_report.py`
- `validation/rv_method_overlap_report.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.rv_method_diagnostics_report --diagnostics-glob 'output/*_diagnostics.csv' ...
PYTHONPATH=. python -m pytest tests/test_method_evaluation.py -q
```

## Acceptance criteria

- Coverage report shows N_total vs N_finite per method (not only frac_bad among finite)
- Fusion rejects exposures with large method discordance despite small formal σ
- Unit tests cover tiered policy edge cases

## Tests / validation

- `tests/test_method_fusion.py` (new)
- Compare high-σ fraction plots before/after on held-out star

## Propagation checklist (on merge)

- [ ] Master todo `method-fusion-coverage` → completed
- [ ] Mark `rv_method_fusion_plan` todos addressed

## Open decisions

- Apply fusion in pipeline by default or opt-in flag first?
