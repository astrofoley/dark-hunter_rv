#!/usr/bin/env bash
# Full refit with incremental website updates (per star: pipeline → fit → Hβ → stage).
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/full_website_refresh.sh
#
# Detached screen (command 2 — full refit, site updates star-by-star):
#   screen -dmS darkhunter_full_refresh bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO" && git pull && bash scripts/full_website_refresh.sh
#   '

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
LOG="${LOG:-$REPO/logs/full_website_refresh.log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$REPO/logs"
export PYTHONPATH="$REPO"
export WEB_ROOT

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) full_website_refresh start (pid $$) ==="

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing — run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

echo "=== Deploy script.js / style.css / index.html ==="
bash scripts/setup_website.sh

if [[ -f "$REPO/scripts/fix_data_csv_column_order.py" ]]; then
  echo "=== Repair tables/data.csv column alignment ==="
  "$PY" scripts/fix_data_csv_column_order.py --data-csv "$WEB_ROOT/tables/data.csv"
fi

echo "=== APF observability windows cache ==="
"$PY" scripts/build_apf_observability_cache.py \
  --data-csv "$WEB_ROOT/tables/data.csv" \
  --output-dir "$REPO/output" \
  --cache "$REPO/rv_fit_reports/observability_windows_cache.json"

echo "=== Per-object pipeline + fit + website (see logs/refit_all_per_object.log) ==="
bash scripts/refit_all_per_object.sh

echo "=== $(date -Is) full_website_refresh done ==="
echo "Logs: $LOG  $REPO/logs/refit_all_per_object.log"
