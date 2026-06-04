#!/usr/bin/env bash
# Fast path: rebuild Hβ table thumbnails + stage stars to WEB_ROOT (no pipeline, no Keplerian refit).
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/update_hbeta_website.sh
#
#   STAR_ID=1551542027851147904 bash scripts/update_hbeta_website.sh
#   DEPLOY_STATIC=1 bash scripts/update_hbeta_website.sh   # also copy script.js / style.css

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
export DEPLOY_STATIC="${DEPLOY_STATIC:-0}"

cd "$REPO"
if [[ "$DEPLOY_STATIC" == "1" ]]; then
  bash scripts/setup_website.sh
fi

export RUN_FITS=0 RUN_RV_PLOTS=0 RUN_HBETA_PLOTS=1 FIT_FORCE=0
exec bash scripts/batch_fits_plots_sync.sh
