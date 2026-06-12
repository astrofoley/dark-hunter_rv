---
step_id: 09-ccf-rv-estimator
phase: D
status: complete_gauss_offset_adopted
depends_on: [01-benchmark-cool-precision]
blocks: []
---

# Step 09: CCF RV estimator study

## Goal

Compare mask-CCF RV estimators (Gaussian, parabolic, smoothed peak, bi-Gaussian), deploy an S/N-adaptive router, and rebuild bias with stellar-parameter trend correction. Phase gates ensure precision does not regress.

## Decision (2026-06-09)

**Production estimator: `gauss_offset` (symmetric offset + Gaussian).** Keep `MASK_CCF_ESTIMATOR = "gauss_offset"` in `darkhunter_rv/config.py`. Do not deploy `auto` or `smooth_peak` despite marginal post-debias σ_RV differences — Gaussian is simpler, essentially tied on the primary metric, and was the pre-debias composite winner.

Stellar-regression debias for `gauss_offset` uses a **curve** model (`validation_output/ccf_estimator_study/bias_gauss_offset/`). Rebuild production `bias_statistics.txt` from a full mask-only pipeline on `calibration/bias_train.txt` before catalog refit (campaign regression used raw lag-frame RVs).

## Key modules

- `darkhunter_rv/ccf_rv_estimators.py` — estimator library + `select_ccf_estimator`
- `darkhunter_rv/config.py` — `MASK_CCF_ESTIMATOR = "gauss_offset"`
- `validation/ccf_rv_estimator_benchmark.py` — multi-estimator campaign
- `validation/ccf_rv_post_debias.py` — post-debias σ_RV ranking
- `validation/ccf_rv_precision_gates.py` — phase gate checks
- `validation/ccf_estimator_bias.py` — Teff/logg/[M/H] bias regression
- `calibration/ccf_estimator_baseline/` — frozen reference metrics

## Phase C campaign (114 spectra, 19 029 chunks)

Raw pre-debias metrics (not used for final decision — see post-debias below):

| Estimator | median σ_RV (formal, pre-debias) | median chunk scatter (pre-debias) | composite |
|-----------|-----------------------------------|-----------------------------------|-----------|
| gauss_offset | 0.0087 km/s | 18.9 km/s | **9.73** |
| auto | 0.0079 | 23.4 | 10.76 |
| smooth_peak | 0.0077 | 22.0 | 10.12 |
| bi_gauss | 0.0106 | 21.3 | 10.93 |

Pre-debias σ_RV ≈ 0.008 km/s is an artifact of the 0.1 km/s per-chunk error floor, not science precision. Pre-debias scatter (~19 km/s) is order-to-order structure before debias.

Artifacts: `validation_output/ccf_estimator_study/per_chunk_rv.csv`, `estimator_comparison.csv`, `phase_C_metrics.json`.

## Phase C post-debias results (primary metric)

Method: per estimator, fit nested bias regression (chunk curve + Teff/logg/[M/H] + per-object offset), subtract `adjusted_bias_kms`, IVW stack per exposure (`stack_calibrated_exposure`). All chunks with finite RV in the campaign; no extra high-S/N cut.

| Estimator | median σ_RV (debiased) | p90 σ_RV | debiased chunk scatter | chosen bias model |
|-----------|------------------------|----------|------------------------|-------------------|
| auto | **0.0444 km/s** | 0.055 | 23.2 km/s | intercept |
| **gauss_offset** | **0.0458** | 0.057 | 23.1 | curve |
| parabolic_ls | 0.0469 | 0.058 | 22.7 | intercept |
| smooth_peak | 0.0469 | 0.058 | **22.0** | intercept |
| grid | 0.0484 | 0.060 | 23.3 | intercept |
| bi_gauss | 0.0697 | 0.118 | 24.5 | curve |

`auto` is marginally best on median σ_RV (~0.001 km/s over Gaussian). `smooth_peak` has lowest debiased chunk scatter but worse σ_RV. **Adopted: gauss_offset.**

Artifacts: `validation_output/ccf_estimator_study/estimator_comparison_post_debias.csv`, `phase_C_post_debias_winner.json`, `epochs_post_debias__*.csv`.

```bash
PYTHONPATH=. python -m validation.ccf_rv_post_debias \
  --wide-diagnostics validation_output/ccf_estimator_study/per_chunk_rv.csv \
  --summary-dir output \
  --out-dir validation_output/ccf_estimator_study
```

### Caveats

- Debiased chunk scatter remains ~22 km/s because the benchmark used raw lag-frame mask CCF RVs, not full pipeline debias with per-order `bias_statistics.txt`.
- Follow-up: mask-only pipeline on `calibration/bias_train.txt` with `gauss_offset`, then rebuild `bias_statistics.txt` for catalog refit.

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. pytest tests/test_ccf_rv_estimators.py \
  tests/validation/test_ccf_rv_precision_gates.py \
  tests/validation/test_ccf_estimator_bias.py \
  tests/validation/test_ccf_rv_post_debias.py -q

PYTHONPATH=. python -m validation.ccf_rv_estimator_benchmark \
  --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \
  --chunk-layout calibration/chunk_layouts/subchunks_4.yaml \
  --phase C --check-gate \
  --baseline calibration/ccf_estimator_baseline/phase_c_reference.json \
  --out-dir validation_output/ccf_estimator_study

PYTHONPATH=. python -m validation.ccf_estimator_bias \
  --wide-diagnostics validation_output/ccf_estimator_study/per_chunk_rv.csv \
  --summary-dir output \
  --estimator gauss_offset \
  --out-dir validation_output/ccf_estimator_study/bias_gauss_offset
```

## Phase gates

| Phase | Gate | Outcome |
|-------|------|---------|
| A | Synthetic estimator tests pass; `gauss_offset` backward compatible | Pass |
| B | Best param sweep ≤ Phase A `median_sigma_rv_kms` | Not run on full bias_train |
| C | Post-debias σ_RV comparison on chunk-campaign list | Done — see table above |
| D | Stellar regression debias + `bias_statistics.txt` for gauss_offset | Campaign export done; pipeline rebuild pending |

## Remaining

- [ ] Mask-only pipeline on `calibration/bias_train.txt` with `gauss_offset` + `subchunks_4`
- [ ] Replace `bias_statistics.txt` from pipeline orders (not campaign lag-frame regression)
- [ ] Optional: re-run post-debias on pipeline diagnostics to confirm σ_RV after proper debias
