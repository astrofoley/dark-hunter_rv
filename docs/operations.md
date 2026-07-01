# Operations: calibration, processing, and cron

**See also:** [rv_methods_evaluation.md](rv_methods_evaluation.md) (cascade adoption), [contributing.md](contributing.md) (PR checklist), [README.md](../README.md) (quick start).

## Overview

1. **Calibration** (occasional): order/chunk biases (`bias_statistics.txt`) and optional global method offsets (`method_rv_offsets.txt`, mask as truth).
2. **Processing**: run the pipeline on spectra; by default all three exposure-level methods are computed and the **adopted** RV uses a **cascade** (mask → template → strong lines) with applicability regions and a σ threshold (`config.ADOPTED_CASCADE_MAX_SIGMA_KMS`).
3. **Incremental runs**: `--update` skips inputs whose `*_diagnostics.csv` is newer than the spectrum file (for cron).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DARKHUNTER_OUTPUT_DIR` | Output directory (default: `output/` under repo) |
| `DARKHUNTER_PLOT_DIR` | Plot directory |
| `DARKHUNTER_MASK_DIR` | Stellar mask library |
| `DARKHUNTER_PHOENIX_DIR` | PHOENIX HiRes grid root |
| `DARKHUNTER_METHOD_OFFSETS_FILE` | Path to `method_rv_offsets.txt` (overrides default auto-detect) |
| `DARKHUNTER_ADOPTED_MAX_SIGMA_KMS` | Max σ (km/s) for “good” tier in cascade (default: same as comparison reports, typically 2.5) |
| `DARKHUNTER_CHUNK_LAYOUT` | YAML chunk layout (default: `calibration/chunk_layouts/subchunks_8.yaml`) |
| `DARKHUNTER_BIAS_FILE` | Per-order debias table (default: repo-root `bias_statistics.txt`) |

## One-command calibration (recommended)

Prepare two text files (one path per line, `#` comments allowed):

- **`bias_train.txt`** — cool / mask-friendly spectra for **per-order bias** (stellar mask only).
- **`offset_train.txt`** — spectra in the **method-overlap** region for **template vs strong vs mask** offsets.

Then:

```bash
python -m validation.run_calibration_setup \
  --bias-list calibration/bias_train.txt \
  --offset-list calibration/offset_train.txt \
  --instrument APF \
  --clean-after-bias
```

This will:

1. Run `python -m darkhunter_rv.pipeline … --mask-only --no-bias` on the bias list (mask CCF chunk RVs only).
2. Run `validation.build_bias_set` → install `bias_statistics.txt` at the **repo root**.
   Only `*_orders.txt` stems listed in `--bias-list` are used (stale files in `output/` are ignored).
3. Optionally **delete** `*_orders.txt` / `*_diagnostics.csv` for those bias spectra (`--clean-after-bias`; add `--clean-plots` to remove PNGs).
4. Run the **full** default pipeline (multi-method, **with** bias) on the offset list.
5. Run `validation.compute_method_rv_offsets` → install `method_rv_offsets.txt` at the repo root.
6. Write **`calibration/manifest.json`** (paths used for bias vs offset phases).

Optional: **`--rerun-offset-with-corrections`** — re-run the offset list so adopted RVs use the new `method_rv_offsets.txt`.

**Note:** Root `rv_bias.py` expects an older `*_orders.txt` format (epoch/suborder columns). The automated path uses `build_bias_set`, which matches **current** chunk keys (`0_a`, …).

Quick reference only:

```bash
python -m validation.setup_calibration
```

## Production batch (skip offset-training spectra)

After calibration, process the full survey list but **skip** spectra that were already run in the offset phase (if their `*_diagnostics.csv` exists), limiting repeat Gaia work:

```bash
python -m validation.run_production_remaining \
  --manifest calibration/manifest.json \
  --spectrum-list all_spectra.txt \
  --instrument APF \
  --update
```

Use **`--include-offset-calibration`** to force reprocessing of offset-training targets too.

## Manual / piecemeal calibration

1. Mask-only bias training: `python -m darkhunter_rv.pipeline <files…> --mask-only --no-bias …`
2. Build biases: `python -m validation.build_bias_set --input-dir output --out-dir …`
3. Install `bias_statistics.txt` next to `InstrumentProfile.bias_file` (APF: repo root).
4. Offsets: `python -m validation.compute_method_rv_offsets --diagnostics-list … --output method_rv_offsets.txt`

## Production RV refit (full catalog, parallel)

Re-measure every star with `subchunks_8` + debias, Keplerian fit, and **per-star website update** (plots, summary, `data.csv` row). Low CPU priority; parallel workers use `flock` on `data.csv`.

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull

screen -dmS darkhunter_parallel_refit bash -lc '
  REPO=/data2/darkhunter/dark-hunter_rv
  cd "$REPO"
  JOBS=4 NICE_LEVEL=10 PIPELINE_FORCE=1 FIT_FORCE=1 FIT_JITTER=1 \
    bash scripts/refit_all_per_object_parallel.sh
'
```

Attach with `screen -r darkhunter_parallel_refit`. Logs: `logs/refit_all_per_object_parallel.log` and `logs/refit_parallel/<gaia_id>.log`. Tune `JOBS` (default half of CPU count, max 8) and `NICE_LEVEL` (default 10).

## Production RV refit (one star: re-measure + Keplerian fit)

Re-run the pipeline with the **production chunk layout** (`subchunks_8`), **committed debias table**, and mask-CCF summary RVs, then fit:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
STAR_ID=1702370142434513152 \
SPEC_ROOT=/Users/rfoley/darkhunter/rvs/data \
bash scripts/refit_star_rvs.sh
```

**Server (ziggy):**

```bash
cd /data2/darkhunter/dark-hunter_rv
git fetch origin && git checkout step/01-benchmark-cool-precision && git pull
STAR_ID=1702370142434513152 bash scripts/refit_star_rvs.sh
```

Use `python3` (or set `PY=...`) on the server — bare `python` may be Python 2.7. Set `DARKHUNTER_PHOENIX_DIR` if HiRes PHOENIX is not under `~/phoenix/HiResFITS` or `phoenix_models/`.

**Debias:** `bias_statistics.txt` at repo root (one row per **chunk_key**, e.g. `9_2` for subchunks_8). Rebuild after layout changes **or after enabling blaze split continuum** (mask debias must match mask CCF continuum):

```bash
bash scripts/rebuild_mask_bias.sh
```

Default pipeline uses `--continuum-mode split` with `calibration/blaze_orders_apf.json` (mask: `sinc_blaze_only`, template/strong: `sinc_blaze`). Legacy spline-only: `--no-blaze-continuum` or `--continuum-mode spline`. See `calibration/mask_lane_deploy.md`. Legacy per-echelle-order debias rows (integer key `9`) still work as a fallback when a chunk_key is missing.

## Daily / cron processing

```bash
python -m darkhunter_rv.pipeline /path/to/*.txt --instrument APF --update --log-level INFO
```

Use `--force` to ignore mtime skip. Logs should record skipped vs processed files.

## Legacy single-method mode

```bash
python -m darkhunter_rv.pipeline … --no-run-all-methods
```

Adopted RV reverts to cool/hot **chunk-stack** rules (see `config.RV_METHOD_SELECTION_NOTES`).

## Future: cheaper S/N proxy

Warm-star mask applicability and strong-line regions currently use `log10(median mask CCF peak S/N)`, which requires mask CCF chunks to have run. A **cheaper proxy** (e.g. continuum S/N) could allow skipping expensive paths when a star is clearly out of region; this is not implemented in v1 (selection uses post-hoc regions while methods still run).
