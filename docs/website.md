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

Plots + staging write directly under `WEB_ROOT` (no separate Kirsty mirror).

```bash
cd /data2/darkhunter/dark-hunter_rv

# Full refresh (fits + plots + table M2 columns + rsync archives)
WEB_ROOT=/var/www/html/darkhunter/rv MIN_POINTS=5 bash scripts/populate_website.sh

# Skip refitting; reuse existing rv_fit_reports/*.json
WEB_ROOT=/var/www/html/darkhunter/rv RUN_FITS=0 FIT_FORCE=0 MIN_POINTS=5 \
  bash scripts/populate_website.sh
```

Detached:

```bash
screen -dmS darkhunter_web bash -lc '
cd /data2/darkhunter/dark-hunter_rv
WEB_ROOT=/var/www/html/darkhunter/rv RUN_FITS=0 FIT_FORCE=0 MIN_POINTS=5 \
  bash scripts/populate_website.sh >> batch_fits_plots_sync.log 2>&1
'
```

## Repo source of truth

Static assets: `website/rv/` in this repository.

Batch integration: `scripts/batch_fits_plots_sync.sh` (default `WEB_ROOT=/var/www/html/darkhunter/rv`).
