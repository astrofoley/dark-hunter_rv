#!/usr/bin/env bash
# Populate website star tree + update tables/data.csv from pipeline output.
#
# Usage:
#   bash scripts/populate_website.sh
#   RUN_FITS=0 MIN_POINTS=5 bash scripts/populate_website.sh
#   STAR_ID=1702370142434513152 bash scripts/populate_website.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing. Run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

cd "$REPO"
exec bash scripts/batch_fits_plots_sync.sh
