# Chunk optimization — post-campaign workflow

## North star

**Primary metric:** `median_sigma_rv_kms` — median per-exposure calibrated RV uncertainty after per-chunk debias and intrinsic-scatter IVW stacking (`validation/chunk_calibration.summarize_sigma_rv_metrics`).

**Secondary:** `p90_sigma_rv_kms`, APF–APF relative gate (`relative_median_abs_delta_kms`).

Do **not** use `composite_score` as the primary objective (it blends σ_RV with relative gate).

---

## Phase 1 — Per-order N + merge baseline (run first)

From existing campaign cache (no pipeline), pick the best equal split **N ∈ {1,2,3,4}** or **merge_w{2,3,4}** per echelle order, then validate with greedy exposure-level stacks.

**Normalization to n=1 (approximate):** for **sub-chunks**, σ_norm = σ_stack / √n_sub; for **merges**, σ_norm = σ_stack × √(orders merged). Combined on a stack: multiply σ_stack by `(∏√merge_width) / √(n_subchunk_meas)`. Ignores per-chunk IVW weight differences.

**N=1 (one chunk per order):** candidate `split_1` uses pipelined `whole_order` measurements when present in cache; otherwise an **IVW proxy** coarsened from `subchunks_2/3/4` (see `split_1_uses_ivw_proxy` in `per_order_chunk_summary.csv`). Pipeline `whole_order` for true n=1:

```bash
PYTHONPATH=. python3 -m validation.chunk_campaign --run-pipeline --no-stage-b \
  --out-dir "$CAMPAIGN" --only-layouts whole_order
```

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
CAMPAIGN=validation_output/chunk_campaign

PYTHONPATH=. python3 -m validation.per_order_chunk_baseline --campaign-dir "$CAMPAIGN"
```

**Outputs** (`$CAMPAIGN/`):

| File | Content |
|------|---------|
| `per_order_candidate_scores.csv` | Marginal median \|residual\| per (order, candidate) |
| `per_order_chunk_map.csv` | Modal greedy winner per order across exposures |
| `per_order_greedy_epochs.csv` | Per-exposure stacks using per-order choices |
| `per_order_chunk_summary.csv` | Campaign median σ_RV vs reference layouts |
| `plots/per_order_candidate_heatmap.png` | Order × candidate score heatmap |
| `plots/per_order_choice_counts.png` | How many orders chose each candidate |

**Latest run (local):** 114 exposures, `median_sigma_rv_kms ≈ 0.029` km/s (`needs_subchunks_8=True` because orders 37, 38, 49 prefer `split_4`).

### Phase 1b — Gated `subchunks_8`

Run **only if** `per_order_chunk_summary.csv` has `needs_subchunks_8=True`:

```bash
PYTHONPATH=. python3 -m validation.chunk_campaign --run-pipeline --no-stage-b \
  --out-dir "$CAMPAIGN" --only-layouts subchunks_8

PYTHONPATH=. python3 -m validation.per_order_chunk_baseline \
  --campaign-dir "$CAMPAIGN" --split-ns 1,2,3,4,8
```

**After n=8:** compare N=4 vs N=8 on orders that preferred `split_4`. Pipeline N=5,6,7 only if n=8 results are ambiguous (see decision table in plan).

---

## Step 0 — Refresh global layout rankings

```bash
PYTHONPATH=. python3 -m validation.chunk_campaign --out-dir "$CAMPAIGN"

python3 - <<'PY'
import pandas as pd
g = pd.read_csv("validation_output/chunk_campaign/campaign_grid_summary.csv")
cols = ["layout","n_exposures","median_sigma_rv_kms","p90_sigma_rv_kms","relative_median_abs_delta_kms"]
print(g[cols].sort_values("median_sigma_rv_kms").to_string(index=False))
PY
```

**Redundant layout:** `n3_equal` ≡ `subchunks_3` (equal edges) — ignore `n3_equal`.

---

## Adaptive per-order mix

```bash
PYTHONPATH=. python3 -m validation.chunk_adaptive_stack --campaign-dir "$CAMPAIGN"
```

Compare `adaptive_mix` vs best fixed layout on **`adaptive_stack_common_cohort.csv`** (identical exposure set).

---

## Chunk weight lookup (incremental)

Persist IVW weights keyed by `(layout_name, chunk_key)`. Re-running for a layout updates only that layout’s rows.

```bash
PYTHONPATH=. python3 -m validation.chunk_weight_lookup \
  --campaign-dir "$CAMPAIGN" \
  --layouts subchunks_4,merge_w4,n3_red_heavy \
  --lookup-csv "$CAMPAIGN/chunk_weight_lookup.csv"
```

Weight model per chunk: `σ = hypot(stat_err, intrinsic)`; `w = 1/σ²`; exposure `σ_RV = 1/sqrt(Σw)`.

---

## Decision rules

1. Minimize `median_sigma_rv_kms` among layouts with `n_exposures` within ~5% of cohort max.
2. Tie-break: lower `p90_sigma_rv_kms`, then lower `relative_median_abs_delta_kms`.
3. Use per-order baseline map for **context** (which orders want merge vs fine split); production layout may still be a single global YAML if adaptive gain is small.
4. Adopt `adaptive_mix` only if it beats the best fixed layout on median σ_RV on the common cohort.

---

## Deploy (#57) — mask lane

**Production defaults:** `subchunks_8` (`config.DEFAULT_CHUNK_LAYOUT`, refit scripts).

1. Rebuild debias: `bash scripts/rebuild_mask_bias.sh` (see `calibration/mask_lane_deploy.md`).
2. Refit RVs: `scripts/refit_star_rvs.sh` / `scripts/refit_all_per_object_parallel.sh`.
3. Snapshot mask−template overlap before step 10.

---

## Future phases

- **Phase 2:** Flexible YAML — per-order unequal edges + cross-order pixel spans (`validation/chunk_layout.py` v2).
- **Phase 3:** Curated search (`validation/flexible_chunk_search.py`); discrete per-order GA on cached layouts if greedy adaptive plateaus.
- **Edge refinement:** coordinate descent / presets — `validation_output/chunk_grid_search/EDGE_SEARCH_ADVICE.md`.
- **ML ranking (issue #53):** BDT on chunk features to prioritize edge candidates before pipeline reruns.
