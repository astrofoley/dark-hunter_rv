#!/usr/bin/env bash
# Phase 2: Deploy static site + refresh data.csv (fast; no Gaia TAP queries).
#
# - setup_website.sh (index.html, script.js, sample_tags_data.js)
# - ensure_sample_stars_website.py --table-only (add ATF22/E24 rows from summaries)
# - update_website_table_columns.py (N_obs, G mag, masses, schedule columns)
#
# Prerequisite: bash scripts/query_website_gaia.sh (or existing summaries on disk).
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/update_website_table_only.sh
#
# Optional Gaia NSS prefetch (slow, separate from phase 1):
#   PREFETCH_GAIA_NSS=1 bash scripts/update_website_table_only.sh
#
# Detached screen:
#   screen -dmS dh_table_update bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
#     export WEB_ROOT=/var/www/html/darkhunter/rv
#     cd "$REPO" && bash scripts/update_website_table_only.sh
#   '

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
LOG="${LOG:-$REPO/logs/update_website_table_only.log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$REPO/logs"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing — run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) update_website_table_only start (pid $$) ==="
echo "repo=$REPO web_root=$WEB_ROOT out=$OUT reports=$REPORTS_DIR"

echo "=== Deploy script.js / style.css / index.html ==="
bash scripts/setup_website.sh

echo "=== Add ATF22/E24 sample rows from existing summaries (no Gaia queries) ==="
"$PY" scripts/ensure_sample_stars_website.py \
  --data-csv "$WEB_ROOT/tables/data.csv" \
  --output-dir "$OUT" \
  --table-only

if [[ "${PREFETCH_GAIA_NSS:-0}" == "1" ]]; then
  echo "=== Prefetch Gaia NSS (inclination, binary masses) for table stars ==="
  "$PY" scripts/prefetch_gaia_nss_for_table.py \
    --data-csv "$WEB_ROOT/tables/data.csv" \
    --reports-dir "$REPORTS_DIR"
fi

echo "=== Normalize data.csv + fill columns from existing summaries/reports ==="
"$PY" scripts/update_website_table_columns.py \
  --data-csv "$WEB_ROOT/tables/data.csv" \
  --output-dir "$OUT" \
  --reports-dir "$REPORTS_DIR"

echo "=== $(date -Is) update_website_table_only done ==="
echo "Hard-refresh the browser. Next (optional): bash scripts/replot_rv_website.sh"
echo "Log: $LOG"
