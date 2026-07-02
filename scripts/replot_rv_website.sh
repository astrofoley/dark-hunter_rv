#!/usr/bin/env bash
# Phase 3: Replot RV data + Keplerian fit figures and stage to WEB_ROOT (no refit).
#
# - Refreshes APF observability cache
# - replot_rv_figures_from_fits.py (fit PNGs + RV data from fit JSON)
# - --also-summaries-without-fits (literature-only sample stars)
# - Refreshes table columns that depend on fit reports
#
# Does not rerun pipeline, refit Keplerian models, or touch bias_statistics.txt.
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/replot_rv_website.sh
#
# Detached screen:
#   screen -dmS dh_replot_rv bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
#     export WEB_ROOT=/var/www/html/darkhunter/rv
#     cd "$REPO" && bash scripts/replot_rv_website.sh
#   '

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
LOG="${LOG:-$REPO/logs/replot_rv_website.log}"
OBS_CACHE="${OBS_CACHE:-$REPORTS_DIR/observability_windows_cache.json}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$REPO/logs" "$REPORTS_DIR"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing" >&2
  exit 2
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) replot_rv_website start (pid $$) ==="
echo "repo=$REPO web_root=$WEB_ROOT out=$OUT reports=$REPORTS_DIR"

echo "=== APF observability windows cache ==="
"$PY" scripts/build_apf_observability_cache.py \
  --data-csv "$WEB_ROOT/tables/data.csv" \
  --output-dir "$OUT" \
  --cache "$OBS_CACHE"

echo "=== Replot RV fit + RV data figures from stored fits ==="
"$PY" scripts/replot_rv_figures_from_fits.py \
  --output-dir "$OUT" \
  --reports-dir "$REPORTS_DIR" \
  --observability-cache "$OBS_CACHE" \
  --also-summaries-without-fits \
  --web-root "$WEB_ROOT"

echo "=== Refresh table columns from reports ==="
"$PY" scripts/update_website_table_columns.py \
  --data-csv "$WEB_ROOT/tables/data.csv" \
  --output-dir "$OUT" \
  --reports-dir "$REPORTS_DIR"

echo "=== $(date -Is) replot_rv_website done ==="
echo "Log: $LOG"
