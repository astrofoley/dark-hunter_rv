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
| `Gaia_DR3_<id>_keplerian_fit.png` | Four fits (free P/e, fixed P, fixed e, fixed P+e) |
| `Gaia_DR3_<id>_keplerian_residuals.png` | Residuals vs RV-only fit (±5 km/s cap) |
| `Gaia_DR3_<id>_28_hbeta.png` | All epochs on one axes (viridis by MJD) |
| Table **RV Fit** thumb | `RV_Fit/<id>_keplerian_fit.png` (fits only) |
| Table **RV Fit** click | `Plots/<id>_keplerian_residuals.png` (fits + residuals) |

```bash
cd /data2/darkhunter/dark-hunter_rv

# Regenerate everything (≥5 epochs per star, literature included in fits)
WEB_ROOT=/var/www/html/darkhunter/rv MIN_POINTS=5 FIT_FORCE=1 bash scripts/populate_website.sh
```

Detached (full fits):

```bash
screen -dmS darkhunter_fits bash -lc '
cd /data2/darkhunter/dark-hunter_rv
WEB_ROOT=/var/www/html/darkhunter/rv MIN_POINTS=5 FIT_FORCE=1 RUN_FITS=1 \
  bash scripts/populate_website.sh >> batch_fits_plots_sync.log 2>&1
'
```

Plots/staging only (no refit):

```bash
WEB_ROOT=/var/www/html/darkhunter/rv RUN_FITS=0 RUN_RV_PLOTS=1 MIN_POINTS=5 \
  bash scripts/populate_website.sh
```

**Repair shifted columns / stale embedded `<img>` in `data.csv`** (after a bad populate, or once after upgrading):

```bash
cd /data2/darkhunter/dark-hunter_rv
git pull
bash scripts/setup_website.sh
PYTHONPATH=. python3 scripts/fix_data_csv_column_order.py \
  --data-csv /var/www/html/darkhunter/rv/tables/data.csv
WEB_ROOT=/var/www/html/darkhunter/rv RUN_FITS=0 RUN_HBETA_PLOTS=1 \
  SPEC_ROOT=/data2/gaia_stars/apf_reductions \
  bash scripts/populate_website.sh
```

Hard-refresh the browser once (`script.js` builds plot URLs from Gaia ID; thumbnails use a per-load cache-bust query).

Hβ overlays read spectra under `SPEC_ROOT` (not per-epoch pipeline PNGs). Build alone:

```bash
PYTHONPATH=. python3 scripts/build_hbeta_website_plots.py \
  --summary-dir output \
  --plots-root /var/www/html/darkhunter/rv/stars \
  --spec-root /data2/gaia_stars/apf_reductions
```

## Cron (new spectra → RVs → website)

```bash
REPO=/data2/darkhunter/dark-hunter_rv \
WEB_ROOT=/var/www/html/darkhunter/rv \
SPEC_ROOT=/data2/gaia_stars/apf_reductions \
MIN_POINTS=5 \
bash scripts/cron_update_rv_website.sh
```

Example crontab line:

```cron
0 6 * * * REPO=/data2/darkhunter/dark-hunter_rv WEB_ROOT=/var/www/html/darkhunter/rv SPEC_ROOT=/data2/gaia_stars/apf_reductions bash $REPO/scripts/cron_update_rv_website.sh >> $REPO/logs/cron_rv_website.log 2>&1
```

## Repo source of truth

Static assets: `website/rv/` in this repository.

Batch integration: `scripts/batch_fits_plots_sync.sh` (default `WEB_ROOT=/var/www/html/darkhunter/rv`).
