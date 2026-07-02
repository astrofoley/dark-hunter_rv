# Dark Hunter RV website (`/var/www/html/darkhunter/rv`)

The public explorer lives in a **subdirectory** of the Apache base path so `/var/www/html/darkhunter/` stays free of `index.html` and mixed deploy scripts.

## Directory layout

```
/var/www/html/darkhunter/          # base (no site index required)
  README.html                      # optional pointer to rv/
  rv/                              # WEB_ROOT — document root for the app
    index.html
    script.js
    style.css
    tables/
      data.csv
      keck_targets.csv
      simbad_gaia_ids.csv   # optional
    stars/
      Gaia_DR3_<id>/
        Gaia/
          <id>_summary.txt
          Plots/*.png
          RV_Fit/<id>_keplerian_fit.png
    output/                 # mirror of pipeline output (rsync)
    rv_fit_reports/         # fit JSON/PNG archive (rsync)
```

Canonical URL shape (after Apache maps `/darkhunter/rv/`):

`https://ziggy.ucolick.org/darkhunter/rv/?rows=all&page=1`

## One-time setup on ziggy

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull

# 1) Static HTML/JS/CSS into /var/www/html/darkhunter/rv
bash scripts/setup_website.sh

# 2) Copy catalog tables from legacy site (once)
LEGACY_WEB_ROOT=/var/www/html/ktaggart/rv_website_v1 \
  bash scripts/bootstrap_website_tables.sh

# 3) Optional: tiny pointer at base (not a full site)
echo '<!DOCTYPE html><html><body><p><a href="rv/">Dark Hunter RV explorer</a></p></body></html>' \
  | sudo tee /var/www/html/darkhunter/README.html
```

Ensure Apache serves `/var/www/html/darkhunter/rv/` (existing `Alias` or symlink under `html/`).

## Populate / refresh star assets

`populate_website.sh` runs Keplerian fits (pipeline + literature epochs, excluding legacy sentinels such as −9999 km/s). The **RV Curve** plot (`*_rv_plot.png`) is always built from the summary via `scripts/plot_rv_from_summaries.py`, including when the Keplerian fit is skipped or fails. Writes:

| File | Description |
|------|-------------|
| `Gaia_DR3_<id>_rv_plot.png` | Our data only (APF/KPF/…); Today + APF window |
| `Gaia_DR3_<id>_keplerian_fit.png` | Four fits + all epochs (lit + ours); no legend/text |
| `Gaia_DR3_<id>_keplerian_residuals.png` | Same top panel + residuals (±5 km/s) + fit P, e, M₂ sin i |
| `Gaia_DR3_<id>_28_hbeta.png` | All epochs on one axes (viridis by MJD) |
| Table **RV Fit** thumb | `RV_Fit/<id>_keplerian_fit.png` (fits only) |
| Table **RV Fit** click | `Plots/<id>_keplerian_residuals.png` (fits + residuals) |

**Table columns** (from `*_keplerian_fit.json` and summaries after a fit pass):

| Column | Meaning |
|--------|---------|
| **DATA PRODUCTS** | APF / KPF / Swift product links (instrument trees) |
| **GAIA DATA** | Link to `stars/Gaia_DR3_<id>/Gaia/` (star file directory) |
| **SOURCE IMAGE** | UV image from legacy CSV (restored; not hidden) |
| **N_obs** | Count of APF pipeline epochs in the star summary |
| **DAYS SINCE LAST APF** | Days since latest APF pipeline epoch (sortable; drives &lt;7d / &lt;30d filters) |
| **NEXT RV EVENT (DATE)** | Sooner of next max/min RV from **P & e fixed** fit, as `YYYY/MM/DD` (legacy `NEXT RV EVENT (MJD)` column is merged away on repair) |

**Mass columns:**

| Column | Meaning |
|--------|---------|
| **ECCENTRICITY** | Catalog eccentricity |
| **INCLINATION (deg)** | Astrometric i (Gaia NSS or Thiele-Innes); `N/A` if unknown or edge-on |
| **M1 (Msun)** | Luminous primary mass (catalog); **used first** for RV-derived M2 sin i / M2 at i |
| **M2 (Msun)** | Gaia NSS astrometric secondary mass (`used_m2_msun`), not from the RV fit |
| **M2sin i (Msun)** | RV-only Keplerian fit f(M) with table M1 |
| **M2sin i error (Msun)** | Uncertainty on M2 sin i from the RV-only fit (P/K/e Jacobian propagation) |
| **(M2sin i)/(sin i) (Msun)** | Same f(M) and M1 with Gaia astrometric inclination; `N/A` without valid i |
| **M2 at i P,e fixed (Msun)** | M2 at i from **P & e fixed** RV fit and astrometric i; `N/A` without valid i |

**RV fit plots:** blue shaded regions mark the current APF visibility season at Lick (nautical twilight −12°; airmass ≤ 1.7 for ≥30 min/night). **RV Curve** plots (`plot_rv_from_summaries.py`) include the same shading from summary RA/Dec even when there are zero APF epochs or no Keplerian fit. Circumpolar = observable on **every** scanned nautical night under those rules (requires dec &gt; 52.66° at Lick). Stale year-long cache dates are repaired at plot time via `window_mjd_bounds`.

Plot scripts (`plot_rv_from_summaries.py`, `build_hbeta_website_plots.py`, `replot_rv_figures_from_fits.py`) **auto-copy** contract PNGs into `WEB_ROOT/stars/Gaia_DR3_<id>/Gaia/Plots/` when `WEB_ROOT` is set (or pass `--web-root`). Use `--no-sync-website` to skip. Pipeline output under `output/Gaia_DR3_<id>/` remains the canonical build tree; the website copy is updated immediately after each plot is written.

**Sample filters:** checkboxes ATF22 / E24 NS / E24 Full (union when multiple checked); membership in `tables/sample_tags.json` and embedded `sample_tags_data.js`.

## Four-phase workflow (ziggy)

Gaia TAP queries for ATF22/E24 sample stars are **slow** (~70 sources). They are separated from fast table/UI updates. None of these phases run `rebuild_mask_bias.sh`; phase 4 uses the existing repo-root `bias_statistics.txt`.

Paths: `REPO=/data2/darkhunter/dark-hunter_rv`, `PY=/home/marley/anaconda2/envs/gaia-env/bin/python`, `WEB_ROOT=/var/www/html/darkhunter/rv`, `OUT=$REPO/output`, `REPORTS_DIR=$REPO/rv_fit_reports`, `SPEC_ROOT=/data2/gaia_stars/apf_reductions`.

```bash
export REPO=/data2/darkhunter/dark-hunter_rv
export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
export PYTHONPATH="$REPO"
export WEB_ROOT=/var/www/html/darkhunter/rv
export OUT="$REPO/output"
export REPORTS_DIR="$REPO/rv_fit_reports"
export SPEC_ROOT=/data2/gaia_stars/apf_reductions
cd "$REPO" && git pull origin website-updates
```

**Recommended order:** phase 1 → 2 → (3 and/or 4). After a full refit (phase 4), rerun phase 3 if you want replotted figures from stored JSON without relying on per-star staging alone.

### Phase 1 — Gaia queries (slow; run once)

`bash scripts/query_website_gaia.sh` — writes `output/Gaia_DR3_<id>_summary.txt` (metadata + external RVs) and patches G/BP/RP photometry.

```bash
screen -dmS dh_gaia_query bash -lc '
  export REPO=/data2/darkhunter/dark-hunter_rv
  export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
  export PYTHONPATH="$REPO" OUT="$REPO/output"
  cd "$REPO" && bash scripts/query_website_gaia.sh
'
```

Log: `logs/query_website_gaia.log` · Attach: `screen -r dh_gaia_query`

### Phase 2 — Website table + UI (fast)

`bash scripts/update_website_table_only.sh` — deploys static site, adds sample rows to `data.csv`, fills `N_obs`, `G (mag)`, masses, schedule columns. No Gaia queries.

```bash
screen -dmS dh_table_update bash -lc '
  export REPO=/data2/darkhunter/dark-hunter_rv
  export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
  export PYTHONPATH="$REPO"
  export WEB_ROOT=/var/www/html/darkhunter/rv
  export OUT="$REPO/output" REPORTS_DIR="$REPO/rv_fit_reports"
  cd "$REPO" && bash scripts/update_website_table_only.sh
'
```

Optional slow Gaia NSS for inclination/masses: `PREFETCH_GAIA_NSS=1 bash scripts/update_website_table_only.sh`

Log: `logs/update_website_table_only.log` · Hard-refresh browser after completion.

`bash scripts/repair_website_table.sh` is a thin wrapper around phase 2. Set `RUN_GAIA_QUERIES=1` to include phase 1 in the same run.

### Phase 3 — Replot RV + fit figures (no refit)

`bash scripts/replot_rv_website.sh` — observability cache, `replot_rv_figures_from_fits.py` (Keplerian fit + residuals + RV data), literature-only summaries via `--also-summaries-without-fits`, table column refresh.

```bash
screen -dmS dh_replot_rv bash -lc '
  export REPO=/data2/darkhunter/dark-hunter_rv
  export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
  export PYTHONPATH="$REPO"
  export WEB_ROOT=/var/www/html/darkhunter/rv
  export OUT="$REPO/output" REPORTS_DIR="$REPO/rv_fit_reports"
  cd "$REPO" && bash scripts/replot_rv_website.sh
'
```

Log: `logs/replot_rv_website.log`

### Phase 4 — Full refit + plots + website (long)

Per-star: pipeline (debias with current `bias_statistics.txt`) → Keplerian fit → RV plots → Hβ → stage to `WEB_ROOT`. Discovers stars from `SPEC_ROOT` only (literature-only sample stars without APF spectra are covered in phases 1–3).

**Parallel (recommended):**

```bash
screen -dmS dh_full_refit bash -lc '
  export REPO=/data2/darkhunter/dark-hunter_rv
  export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
  export PYTHONPATH="$REPO"
  export WEB_ROOT=/var/www/html/darkhunter/rv
  export OUT="$REPO/output" REPORTS_DIR="$REPO/rv_fit_reports"
  export SPEC_ROOT=/data2/gaia_stars/apf_reductions
  cd "$REPO"
  JOBS=4 NICE_LEVEL=10 PIPELINE_FORCE=1 FIT_FORCE=1 FIT_JITTER=1 QUERY_GAIA_ONLINE=0 \
    bash scripts/refit_all_per_object_parallel.sh
'
```

**Sequential:** replace the last line with `bash scripts/refit_all_per_object.sh`.

Canary: `STAR_ID=77413727493690112 PIPELINE_FORCE=1 FIT_FORCE=1 bash scripts/refit_all_per_object.sh`

Logs: `logs/refit_all_per_object_parallel.log`, `logs/refit_parallel/<gaia_id>.log`

## Three commands (ziggy)

Paths assume `REPO=/data2/darkhunter/dark-hunter_rv`, `WEB_ROOT=/var/www/html/darkhunter/rv`, `SPEC_ROOT=/data2/gaia_stars/apf_reductions`.

### 1 — Hβ plots + website staging (fast; no pipeline, no refit)

Rebuilds `Gaia_DR3_<id>_28_hbeta.png` (−300…+300 km/s, flux 1–99%), copies star trees to `WEB_ROOT`, rsyncs reports. Summaries must already be complete.

```bash
cd /data2/darkhunter/dark-hunter_rv && git pull
bash scripts/update_hbeta_website.sh
```

Deploys `script.js` (table H-beta links always point at `Gaia_DR3_<id>_28_hbeta.png`, not legacy CSV URLs) and prunes old `*_h_beta_rv.png` files from `Plots/`.

### 2 — Full refit in detached screen (per-star pipeline → fit → Hβ → live website row)

Repairs `data.csv` columns, deploys static assets, then **`refit_all_per_object.sh`**: for each star, analyze all `Gaia_DR3_*_epoch_*.txt` spectra, run Keplerian fit, rebuild plots/Hβ, **stage that star and update its table row immediately**.

```bash
screen -dmS darkhunter_full_refresh bash -lc '
  REPO=/data2/darkhunter/dark-hunter_rv
  cd "$REPO" && git pull && bash scripts/full_website_refresh.sh
'
```

Log: `logs/full_website_refresh.log`, `logs/refit_all_per_object.log`

### 3 — Per-object refit only (same star-by-star website updates; no CSV layout repair)

Use when static site / `data.csv` columns are already correct. Cron now includes `Gaia_DR3_*_epoch_*.txt` in pipeline discovery (not only `*_ap1.*`).

```bash
screen -dmS darkhunter_per_object bash -lc '
  REPO=/data2/darkhunter/dark-hunter_rv
  cd "$REPO" && git pull && bash scripts/refit_all_per_object.sh
'
```

Single-star test: `STAR_ID=1551542027851147904 bash scripts/refit_all_per_object.sh`

**Bulk alternative** (website updates only at the end): `bash scripts/full_website_refresh_bulk.sh`

### Populate from existing summaries only (no pipeline)

```bash
cd /data2/darkhunter/dark-hunter_rv
WEB_ROOT=/var/www/html/darkhunter/rv MIN_POINTS=5 FIT_FORCE=1 bash scripts/populate_website.sh
```

Attach: `screen -r darkhunter_full_refresh` — Detach: `Ctrl-a d`  
Logs: `/data2/darkhunter/dark-hunter_rv/logs/full_website_refresh.log` and `logs/batch_fits_plots_sync.log`

Same without screen:

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull
bash scripts/full_website_refresh.sh
```

Plots/staging only (no refit):

```bash
WEB_ROOT=/var/www/html/darkhunter/rv RUN_FITS=0 RUN_RV_PLOTS=1 MIN_POINTS=5 \
  bash scripts/populate_website.sh
```

Audit missing summaries / RV data plots / fits:

```bash
cd /data2/darkhunter/dark-hunter_rv
PYTHONPATH=. python3 scripts/audit_pipeline_coverage.py --only-issues
```

**Symptoms without a full deploy:** RV thumbnails under **M2 sin i** (stray `<img>` in `data.csv`), correct RV plots only in **RV Curve** (new `script.js`), no **H-beta** (PNGs missing or not built).

### Step 1 — Repair table + frontend only (fast, no refit)

Run phase 1 once if ATF22/E24 summaries are missing, then phase 2:

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull
bash scripts/query_website_gaia.sh                    # once, slow
bash scripts/update_website_table_only.sh             # fast (same as repair_website_table.sh)
```

Or combined legacy: `RUN_GAIA_QUERIES=1 bash scripts/repair_website_table.sh`

Optional Gaia NSS: `PREFETCH_GAIA_NSS=1 bash scripts/update_website_table_only.sh`

```bash
PYTHONPATH=. python3 scripts/build_apf_observability_cache.py \
  --data-csv /var/www/html/darkhunter/rv/tables/data.csv \
  --output-dir /data2/darkhunter/dark-hunter_rv/output
```

Hard-refresh the browser. After a code update that changes Keplerian fit logic or observability shading, rerun per-object refit (command 3) or full refresh (command 2).

**Inclination from Thiele-Innes** (Gaia DR3 astrometric binaries often have empty `inclination` in `nss_two_body_orbit`; derive from A,B,F,G in summaries):

```bash
cd /data2/darkhunter/dark-hunter_rv
PYTHONPATH=. python3 scripts/patch_summary_inclination_from_thiele_innes.py \
  --output-dir /data2/darkhunter/dark-hunter_rv/output
PREFETCH_GAIA_NSS=1 bash scripts/repair_website_table.sh
```

New Gaia queries and summary reads apply the same derivation automatically when TI elements are present.

### Step 2 — Full refresh when ready

Same as **command 2** above (`full_website_refresh.sh` → per-star `refit_all_per_object.sh`). Use when summaries are missing epochs (e.g. only four of nine `epoch_*.txt` in `[PIPELINE RESULTS]`).

CSV-only normalize (no column value fill):

```bash
PYTHONPATH=. python3 scripts/fix_data_csv_column_order.py \
  --data-csv /var/www/html/darkhunter/rv/tables/data.csv
```

Hβ overlays read spectra under `SPEC_ROOT` (not per-epoch pipeline PNGs). Build alone:

```bash
PYTHONPATH=. python3 scripts/build_hbeta_website_plots.py \
  --summary-dir output \
  --plots-root /var/www/html/darkhunter/rv/stars \
  --spec-root /data2/gaia_stars/apf_reductions
```

## Cron (daily: new spectra → RVs → fits → website)

`scripts/cron_update_rv_website.sh` runs:

1. **Pipeline** `--update` on `SPEC_ROOT` for full-epoch `Gaia_DR3_*_epoch_<N>.txt` (not `*_order_*` Hβ extracts), `*_ap1.{flm,txt}`, and `*.fits` (skips spectra whose `*_diagnostics.csv` is newer than the input).
2. **Populate**: Keplerian fits (≥`MIN_POINTS`, literature included, bad RVs filtered), RV/Hβ plots, `data.csv` mass columns, staging to `WEB_ROOT`. Skips refit when the JSON is newer than the summary (`FIT_FORCE=0`).

**Missing epochs in summaries:** Cron used to match only `*_ap1.*` / `*.fits`, not `Gaia_DR3_*_epoch_*.txt`. Stars with only epoch `.txt` reductions (e.g. nine epochs on disk but four in `[PIPELINE RESULTS]`) need a one-time **`bash scripts/full_website_refresh.sh`** (`RUN_PIPELINE=1`, default), which runs the pipeline on all epoch files then refits. Incremental cron picks up new epoch `.txt` files after that.

**Spectra on disk but empty GAIA DATA folder:** The catalog table (`data.csv`) was copied from legacy `rv_website_v1`; star assets are built separately from `$REPO/output/Gaia_DR3_*_summary.txt`. Cron runs `pipeline --update`, which **skips** spectra whose diagnostics CSV is newer than the input and does **not** create a summary when every epoch is skipped. After the daily pipeline pass, cron runs `scripts/ensure_pipeline_summaries.py` to backfill missing/incomplete summaries for any `Gaia_DR3_*` tree under `SPEC_ROOT` (all subdirectories). Audit gaps (no matplotlib required):

```bash
cd /data2/darkhunter/dark-hunter_rv
PY=/home/marley/anaconda2/envs/gaia-env/bin/python PYTHONPATH=. \
  python3 scripts/audit_pipeline_coverage.py --only-issues
```

Use the gaia-env Python on ziggy (base conda lacks matplotlib). One-star backfill: `PY=... python3 scripts/ensure_pipeline_summaries.py --gaia-id <id>`.

Install crontab: run **`crontab -e`** alone (do not put the schedule on the same shell line), then add:

```cron
# Optional defaults for all jobs on this machine:
REPO=/data2/darkhunter/dark-hunter_rv
WEB_ROOT=/var/www/html/darkhunter/rv
SPEC_ROOT=/data2/gaia_stars/apf_reductions
MIN_POINTS=5

# 10:00 daily — use absolute path to the script (no >> $REPO/... on the job line)
0 10 * * * /bin/bash /data2/darkhunter/dark-hunter_rv/scripts/cron_update_rv_website.sh
```

Do **not** use `>> $REPO/logs/...` on the same line as `REPO=/path/...`; cron expands `$REPO` in redirects before that assignment, which yields `/logs/cron_rv_website.log`. The script logs to `$REPO/logs/cron_rv_website.log` internally.

Manual test:

```bash
cd /data2/darkhunter/dark-hunter_rv
REPO=/data2/darkhunter/dark-hunter_rv \
WEB_ROOT=/var/www/html/darkhunter/rv \
SPEC_ROOT=/data2/gaia_stars/apf_reductions \
MIN_POINTS=5 \
bash scripts/cron_update_rv_website.sh
```

Log: `/data2/darkhunter/dark-hunter_rv/logs/cron_rv_website.log`

## Repo source of truth

Static assets: `website/rv/` in this repository.

Batch integration: `scripts/batch_fits_plots_sync.sh` (default `WEB_ROOT=/var/www/html/darkhunter/rv`).
