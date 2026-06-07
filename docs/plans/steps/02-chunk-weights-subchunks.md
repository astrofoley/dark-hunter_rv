---
step_id: 02-chunk-weights-subchunks
phase: C
status: pending
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/39
branches:
  - step/02a-subchunk-study
  - step/02b-trust-weights-stack
depends_on: [01-benchmark-cool-precision]
blocks: [03-method-fusion-coverage]
master_todo_id: chunk-weights-subchunks
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
  - rv_mismatch_diagnosis_24a11615.plan.md
  - pr_repo_organization_326c519c.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 02: Wavelength chunks and trust-weighted stack

## Goal / science outcome

Replace arbitrary whole-order chunks with validated sub-chunking and post-debias weights that improve exposure RV precision and robustness.

## Scope (in) / non-goals (out)

**In:** `--subchunks` campaign on APF; telluric/mask-line/consistency weights; persist in diagnostics; update `order_chunk_qc.yaml`.

**Out:** Full method fusion (step 03).

## Prerequisites

- Step 01 baseline metrics
- `darkhunter_rv/chunking.py`, `validation/build_bias_set.py`

## Implementation tasks

### 02a (`step/02a-subchunk-study`)

- [ ] Run APF campaign with `--subchunks 2,4` on calibration set
- [ ] Compare chunk scatter vs step 01 baseline
- [ ] Document recommended default N per instrument

### 02b (`step/02b-trust-weights-stack`)

- [ ] Implement trust weights (residual vs robust mean, telluric fraction, CCF quality) scaling IVW in `pipeline.py`
- [ ] Add weight columns to diagnostics CSV
- [ ] Version `order_chunk_qc.yaml` thresholds
- [ ] Re-run bias build if chunk keys change

## Key files

- `darkhunter_rv/pipeline.py`
- `darkhunter_rv/chunking.py`
- `order_chunk_qc.yaml`
- `validation/build_bias_set.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m darkhunter_rv.pipeline ... --subchunks 4 --instrument APF
python -m validation.build_bias_set --input-dir output ...
```

## Acceptance criteria

- Subchunk study shows improved or equal scatter vs baseline on cool calibration set
- Trust-weighted stack reduces impact of outlier orders in test cases
- Diagnostics record per-chunk weights; debias keys remain consistent

## Tests / validation

- `tests/test_chunking.py` extended if schema changes
- Campaign comparison CSV archived under `validation_output/`

## Propagation checklist (on merge)

- [ ] Master todo `chunk-weights-subchunks` → completed
- [ ] Update `rv_mismatch_diagnosis` plan (trust weights item)

## Open decisions

- Enable `subchunks>1` globally or per-instrument config?
