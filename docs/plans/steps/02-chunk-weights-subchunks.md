---
step_id: 02-chunk-weights-subchunks
phase: C
status: in_progress
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/39
branches:
  - step/02a-subchunk-study
  - step/02b-trust-weights-stack
depends_on: [01-benchmark-cool-precision]
blocks: [10-template-fft-precision]
master_todo_id: chunk-weights-subchunks
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
  - rv_mismatch_diagnosis_24a11615.plan.md
  - pr_repo_organization_326c519c.plan.md
  - rv_precision_framework_8bc25b55.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
  - validation/chunk_optimization_advice.md
---

# Step 02: Wavelength chunks and trust-weighted stack

## Goal / science outcome

Replace arbitrary whole-order chunks with validated sub-chunking and post-debias weights that improve exposure RV precision and robustness. **Shared chunk layout** applies to mask_ccf and template_fft; trust-weight tuning (02b) is mask-stack focused but should not break template stacks.

## Scope (in) / non-goals (out)

**In:** Chunk layout campaign on APF; telluric/mask-line/consistency weights; persist in diagnostics; update `order_chunk_qc.yaml`.

**Out:** Template measurement knobs (step 10); full method fusion (step 03); exhaustive mixed per-order tilings (ruled out — uniform layout wins).

## Prerequisites

- Step 01 baseline metrics
- `validation/chunk_campaign.py`, `validation/chunk_calibration.py`

## Implementation tasks

### 02a (`step/02a-subchunk-study`) — **largely complete**

- [x] Run APF campaign with subchunks 2,3,4 (+ merge layouts) on 114-exposure list
- [x] Phase 1b: `subchunks_8` campaign (114/114 diagnostics, cache ingested)
- [x] `per_order_chunk_baseline`, `chunk_adaptive_stack`, `spectrum_tiling_search` tooling
- [x] **Decision:** uniform **`subchunks_8`** beats `subchunks_4` (median σ_RV 0.0189 vs 0.0223 km/s); adaptive mix adds no gain over pure s8
- [x] Ruled out: N=5,6,7,>8; per-order n=2/3/4 greedy mix (worse than uniform s8 under production stack)
- [x] **Production defaults:** `subchunks_8.yaml` in config + refit scripts
- [x] `calibration/bias_train.txt`, `scripts/rebuild_mask_bias.sh`, `calibration/mask_lane_deploy.md`
- [ ] Rebuild + commit `bias_statistics.txt` for subchunks_8
- [ ] Refit catalog on ziggy

### 02b (`step/02b-trust-weights-stack`) — **pending**

- [ ] Implement trust weights (residual vs robust mean, telluric fraction, CCF quality) scaling IVW in `pipeline.py`
- [ ] Add weight columns to diagnostics CSV
- [ ] Version `order_chunk_qc.yaml` thresholds
- [ ] Validate on relative gate + median σ_RV vs current IVW-only stack

**Defer 02b** until template lane baseline (step 10) is captured — avoids retuning weights twice.

## Key files

- `calibration/chunk_layouts/subchunks_8.yaml`
- `validation/chunk_campaign.py`, `validation/chunk_adaptive_stack.py`
- `validation/per_order_chunk_baseline.py`, `validation/chunk_optimization_advice.md`
- `darkhunter_rv/pipeline.py`, `validation/build_bias_set.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
CAMPAIGN=validation_output/chunk_campaign

# Offline re-eval after campaign
PYTHONPATH=. python3 -m validation.chunk_adaptive_stack --campaign-dir "$CAMPAIGN"
PYTHONPATH=. python3 -m validation.per_order_chunk_baseline \
  --campaign-dir "$CAMPAIGN" --split-ns 1,2,3,4,8

# Deploy (production)
CHUNK_LAYOUT=calibration/chunk_layouts/subchunks_8.yaml bash scripts/refit_star_rvs.sh
```

## Acceptance criteria

- [x] Subchunk study shows improved median σ_RV vs subchunks_4 on common cohort
- [ ] Production layout + bias committed and refit on ziggy
- [ ] Trust-weighted stack (02b) — future; not blocking step 10

## Tests / validation

- `tests/validation/test_chunk_adaptive_stack.py`
- `tests/validation/test_per_order_chunk_baseline.py`
- `tests/validation/test_spectrum_tiling_search.py`

## Propagation checklist (on merge)

- [ ] Close 02a when deploy lands; keep issue #39 open for 02b
- [ ] Update `chunk_optimization_advice.md` deploy section (subchunks_8 winner)

## Open decisions

- **Resolved:** global uniform `subchunks_8`, not per-order mix.
- **Open:** trust weights before or after template lane? → **after** template baseline.
