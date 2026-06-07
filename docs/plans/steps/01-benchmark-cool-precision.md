---
step_id: 01-benchmark-cool-precision
phase: C
status: in_progress
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/38
branches:
  - step/01-benchmark-cool-precision
depends_on: []
blocks: [02-chunk-weights-subchunks]
master_todo_id: benchmark-cool-precision
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
  - rv_methods_evaluation_plan_fcd09d94.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 01: Benchmark cool high-S/N mask precision

## Goal / science outcome

Quantify current APF mask-CCF precision on cool, high-S/N calibration spectra and document gap to **<0.1 km/s** per-epoch target.

## Scope (in) / non-goals (out)

**In:** Repeatability / chunk-scatter metrics on bias-training or overlap cool-star set; report panels vs S/N; baseline before chunk-weight changes.

**Out:** Implementing trust weights (step 02); method fusion (step 03).

## Prerequisites

- Calibration list (`calibration/bias_train.txt` or equivalent)
- Pipeline run with bias on, `--run-all-methods`
- Existing overlap/diagnostics reports

## Implementation tasks

- [x] Define metric: per-exposure chunk scatter (initial); night-pair / jackknife deferred
- [x] `validation/benchmark_cool_precision.py` for cool-star high-S/N subset
- [x] Phase A: `validation/rv_phase_a_baseline.py` — overlap inventory, absolute (APF–lit) and relative (APF–APF) gates, diagnostic plots
- [x] Frozen baseline: `calibration/phase_a_baseline/` (goals.yaml, reference_manifest.json)
- [ ] Produce `validation_output/benchmark_cool_precision/` tables + plots (RMS vs log10 mask CCF S/N)
- [ ] Document 0.1 km/s goal interpretation (single epoch vs night-mean) in report README or playbook
- [ ] Record baseline numbers in step md for step 02 comparison
- [ ] `--no-bias` pipeline rerun on overlap stars for canonical absolute gate

## Key files

- `validation/rv_method_diagnostics_report.py`
- `validation/rv_method_overlap_report.py`
- `darkhunter_rv/pipeline.py` (`chunk_scatter_kms`, mask stack)
- `validation/build_bias_set.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.run_calibration_setup --bias-list calibration/bias_train.txt ...
python -m validation.rv_method_overlap_report --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' ...
```

## Acceptance criteria

- Report identifies median/p90 chunk scatter for cool stars with `log10(median_mask_ccf_peak_snr) > 1.0`
- Explicit statement: met / not met / how far from 0.1 km/s goal
- Phase A overlap list + gate reports under `validation_output/rv_phase_a_baseline/`
- Reproducible one-command recipe in `docs/validation_playbook.md`

### Phase A baseline (2026-06-07, bias applied)

| Metric | Value |
|--------|-------|
| Overlap stars (APF ∩ literature) | 8 |
| APF–literature pairs (7 d window) | 0 (min separation 189–971 d) |
| APF–APF pairs (7 d) | 14 across 4 stars |
| Relative gate median \|ΔRV\| | 0.30 km/s |
| Relative gate p90 \|ΔRV\| | 16.3 km/s (outliers: Gaia BH1, J1449+6919) |
| Stars with good relative precision | J2102+3703 (~0.009 km/s), J0824+5254 (~0.30 km/s) |

## Tests / validation

- No new unit tests required unless adding pure metric helpers
- Manual: rerun on fixed diagnostics glob, compare CSV hash or key statistics

## Propagation checklist (on merge)

- [ ] Master todo `benchmark-cool-precision` → completed
- [ ] INDEX.md status + issue closed
- [ ] Legacy plans: note benchmark baseline in `rv_methods_evaluation_plan`

## Open decisions

- 0.1 km/s: single exposure vs night-averaged?
- Which stars qualify as “cool high-S/N” (Teff cut + S/N cut)?
