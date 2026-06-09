---
step_id: 06-strong-line-line-list
phase: C
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/43
branches:
  - step/06-strong-line-line-list
depends_on: [05-short-pair-epoch-ccf]
blocks: [07-sb2-search]
master_todo_id: strong-line-line-list
related_legacy_plans:
  - template_grid_and_hβ_rv_dff5dfce.plan.md
  - three_rv_methods_e1b72701.plan.md
repo_docs_to_update:
  - docs/broad_line_method.md
---

# Step 06: Strong-line rest wavelengths and multi-line Voigt+Lorentz

## Goal / science outcome

Empirically chosen strong lines per Teff/S/N; extend `measure_h_beta_rv` API beyond Hβ-only for the `strong_lines` method.

## Scope (in) / non-goals (out)

**In:** Line list study; generalized Voigt+Lorentz per line; pipeline still exposes one `strong_lines` row.

**Out:** Reviving Gaussian multi-line centroids as product RV.

## Prerequisites

- Hβ path in `rv_core.py`
- Overlap report Teff strata

## Implementation tasks

- [ ] Survey candidates from `STRONG_LINES` / literature (`docs/broad_line_method.md`)
- [ ] Validation sweep: recovery vs mask/template per Teff bin
- [ ] Refactor `measure_strong_line_voigt_lorentz(rest=...)` from Hβ code
- [ ] Wire best line(s) into pipeline `strong_lines` row
- [ ] Update tests in `tests/test_h_beta_rv.py` or new module

## Key files

- `darkhunter_rv/rv_core.py`
- `darkhunter_rv/continuum.py` (`STRONG_LINES`)
- `validation/h_beta_profile_method_report.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.h_beta_profile_method_report ...
PYTHONPATH=. python -m pytest tests/test_h_beta_rv.py -q
```

## Acceptance criteria

- Documented line list with Teff applicability
- At least one additional line beyond Hβ tested on real spectra OR explicit decision to stay Hβ-only with rationale
- No regression on hot-star Hβ performance

## Tests / validation

- Synthetic line recovery tests per rest wavelength
- Overlap residuals for strong_lines vs mask on cool stars

## Propagation checklist (on merge)

- [ ] Master todo `strong-line-line-list` → completed
- [ ] Update `three_rv_methods` plan

## Open decisions

- Single best line per exposure vs multi-line joint fit?
