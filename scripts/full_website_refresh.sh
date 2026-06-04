#!/usr/bin/env bash
# One-shot: deploy static site assets, repair data.csv, refit all stars (≥MIN_POINTS),
# rebuild RV / Keplerian / Hβ plots, update mass columns, stage to WEB_ROOT.
#
# Fits use pipeline + literature epochs; invalid RVs (|RV|≥5000, NaN, -9999, etc.) are dropped
# via darkhunter_rv.rv_point_filters.rv_value_is_valid.
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv
#   git pull   # merge website fixes (#19+) before first run
#   bash scripts/full_website_refresh.sh
#
# Detached:
#   screen -dmS darkhunter_full_refresh bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO" && git pull && bash scripts/full_website_refresh.sh
#   '

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
MIN_POINTS="${MIN_POINTS:-5}"
LOG="${LOG:-$REPO/logs/full_website_refresh.log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$OUT" "$REPO/logs"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT OUT SPEC_ROOT MIN_POINTS
export RUN_FITS=1 RUN_RV_PLOTS=1 RUN_HBETA_PLOTS=1 FIT_FORCE=1 MIN_POINTS="${MIN_POINTS:-5}" QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
export LOG="$REPO/logs/batch_fits_plots_sync.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) full_website_refresh start (pid $$) ==="
echo "repo=$REPO out=$OUT web_root=$WEB_ROOT min_points=$MIN_POINTS"

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing — run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

echo "=== Deploy script.js / style.css / index.html ==="
bash scripts/setup_website.sh

if [[ -f "$REPO/scripts/fix_data_csv_column_order.py" ]]; then
  echo "=== Repair tables/data.csv column alignment + stray plot HTML ==="
  "$PY" scripts/fix_data_csv_column_order.py --data-csv "$WEB_ROOT/tables/data.csv"
else
  echo "[WARN] scripts/fix_data_csv_column_order.py not found (git pull #19+?); skipping CSV repair"
fi

echo "=== Keplerian fits (all stars, force) + plots + Hβ + website staging ==="
bash scripts/populate_website.sh

echo "=== $(date -Is) full_website_refresh done ==="
echo "Log: $LOG"
echo "Batch detail: $REPO/logs/batch_fits_plots_sync.log"
