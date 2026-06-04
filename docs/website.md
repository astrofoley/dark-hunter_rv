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

`populate_website.sh` runs Keplerian fits (pipeline + literature epochs, excluding legacy sentinels such as −9999 km/s), writes:

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
| **DAYS SINCE LAST APF** | Days since latest APF pipeline epoch (sortable; drives &lt;7d / &lt;30d filters) |
| **NEXT RV EVENT (DATE)** | Sooner of next max/min RV from **P & e fixed** fit, as `YYYY/MM/DD` |

**Mass columns:**

| Column | Meaning |
|--------|---------|
| **M2 (Msun)** | Gaia NSS astrometric secondary mass (`used_m2_msun`), not from the RV fit |
| **M2sin i (Msun)** | RV-only Keplerian fit, with assumed M1 |
| **(M2sin i)/(sin i) (Msun)** | Same RV-only f(M) and M1, with Gaia astrometric inclination |

```bash
cd /data2/darkhunter/dark-hunter_rv

# Regenerate everything (≥5 epochs per star, literature included in fits)
WEB_ROOT=/var/www/html/darkhunter/rv MIN_POINTS=5 FIT_FORCE=1 bash scripts/populate_website.sh
```

### Full overnight refresh (detached screen)

Refits every star with **≥5** valid epochs (pipeline + literature; drops −9999 / NaN / |RV|≥5000 km/s), **force**-rebuilds all Keplerian and Hβ plots, repairs `data.csv`, and stages to the website. Run after merging website fixes (#19+):

```bash
screen -dmS darkhunter_full_refresh bash -lc '
  REPO=/data2/darkhunter/dark-hunter_rv
  cd "$REPO" && git pull && bash scripts/full_website_refresh.sh
'
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

**Symptoms without a full deploy:** RV thumbnails under **M2 sin i** (stray `<img>` in `data.csv`), correct RV plots only in **RV Curve** (new `script.js`), no **H-beta** (PNGs missing or not built).

### Step 1 — Repair table + frontend only (fast, no refit)

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull
bash scripts/repair_website_table.sh
```

Hard-refresh the browser (Cmd+Shift+R). This deploys `script.js`, fixes column alignment, clears stray plot HTML, and fills **DAYS SINCE LAST APF** / mass / **NEXT RV EVENT** from existing summaries and `rv_fit_reports` JSON if they are already on disk.

### Step 2 — Full refresh when ready (long: refit + all plots + Hβ + staging)

```bash
bash scripts/full_website_refresh.sh
```

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

1. **Pipeline** `--update` on `SPEC_ROOT` (new/changed `.flm` / `.txt` / `.fits` only).
2. **Populate**: Keplerian fits (≥`MIN_POINTS`, literature included, bad RVs filtered), RV/Hβ plots, `data.csv` mass columns, staging to `WEB_ROOT`. Skips refit when the JSON is newer than the summary (`FIT_FORCE=0`).

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
