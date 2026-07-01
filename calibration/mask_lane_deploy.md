# Mask lane deploy record

**Status:** production defaults updated to `subchunks_8` (2026-06).

## Production config

| Item | Value |
|------|--------|
| Chunk layout | `calibration/chunk_layouts/subchunks_8.yaml` |
| Default | `darkhunter_rv.config.DEFAULT_CHUNK_LAYOUT` |
| Continuum (mask CCF) | `sinc_blaze_only` via `--continuum-mode split` + `calibration/blaze_orders_apf.json` |
| Continuum (template / strong lines) | `sinc_blaze` (blaze then spline) on same split mode |
| Legacy spline-only | `--no-blaze-continuum` or `--continuum-mode spline` |
| CCF estimator | `gauss_offset` (`config.MASK_CCF_ESTIMATOR`) |
| Debias table | `bias_statistics.txt` (per **chunk_key**, e.g. `10_2`; rebuild after layout change) |
| Bias training set | `calibration/bias_train.txt` (114 campaign spectra) |

## Campaign decision (114 exposures)

- Uniform **subchunks_8** median σ_RV **0.0189 km/s** vs **0.0223** for subchunks_4
- Adaptive per-order mix ≈ pure s8 (no heterogeneous layout gain)
- Per-order greedy σ_norm mix **not** deployed (regresses vs uniform s8)

## Mask debias rebuild (`subchunks_8`)

Rebuild per-chunk `bias_statistics.txt` from the committed bias training set (mask-only, no debias applied during training):

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
bash scripts/rebuild_mask_bias.sh
```

The script runs `run_calibration_setup --bias-only --clean-after-bias`, which:

1. Runs mask-only `--no-bias` pipeline on `calibration/bias_train.txt` (unless `SKIP_PIPELINE=1`).
2. Aggregates **only** those spectra's `*_orders.txt` (ignores stale files in `output/`).
3. Sigma-clips bad mask chunks per exposure, then pools residuals **per chunk_key** (`10_0`, …).
4. Installs `bias_statistics.txt` at the repo root and removes bias-training intermediates.

Quick re-aggregate when `*_orders.txt` for the bias list already exist:

```bash
SKIP_PIPELINE=1 bash scripts/rebuild_mask_bias.sh
```

Ziggy:

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull
PY=/home/marley/anaconda2/envs/gaia-env/bin/python \
  OUT=/data2/darkhunter/dark-hunter_rv/output \
  bash scripts/rebuild_mask_bias.sh
```

## Refit catalog after bias rebuild

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
bash scripts/refit_all_per_object_parallel.sh
```

Or single star:

```bash
STAR_ID=1702370142434513152 bash scripts/refit_star_rvs.sh
```

## Handoff

Next active lane: **step 10 template FFT** — `docs/plans/steps/10-template-fft-precision.md`.
