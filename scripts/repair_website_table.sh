#!/usr/bin/env bash
# Quick website table + frontend repair (no Keplerian refit, no plot rebuild).
#
# - Deploys website/rv → WEB_ROOT (script.js builds plot cells by header name)
# - Normalizes tables/data.csv column order and clears stray <img> in mass columns
# - Fills DAYS SINCE LAST APF from output summaries (if present)
# - Fills M2 / M2 sin i / M2 at i / NEXT RV EVENT from existing rv_fit_reports JSON (if present)
#
# Run this first, then full_website_refresh.sh when you are ready for a long refit pass.
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv
#   git pull   # needs PR #22+ (website_table_csv.py, script.js)
#   bash scripts/repair_website_table.sh
#
# Three operational commands (see docs/website.md):
#   bash scripts/update_hbeta_website.sh              # Hβ + stage only
#   screen ... bash scripts/full_website_refresh.sh   # per-star pipeline+fit+site
#   screen ... bash scripts/refit_all_per_object.sh   # same (no CSV repair preamble)

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
LOG="${LOG:-$REPO/logs/repair_website_table.log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$REPO/logs"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) repair_website_table start (pid $$) ==="
echo "repo=$REPO web_root=$WEB_ROOT out=$OUT reports=$REPORTS_DIR"

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing — run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

echo "=== Deploy script.js / style.css / index.html ==="
export WEB_ROOT
bash scripts/setup_website.sh

if [[ ! -f "$REPO/scripts/update_website_table_columns.py" ]]; then
  echo "[ERROR] scripts/update_website_table_columns.py missing — git pull (merge PR #22+) first." >&2
  exit 2
fi

echo "=== Normalize data.csv + fill columns from existing summaries/reports ==="
"$PY" scripts/update_website_table_columns.py \
  --data-csv "$WEB_ROOT/tables/data.csv" \
  --output-dir "$OUT" \
  --reports-dir "$REPORTS_DIR"

echo "=== $(date -Is) repair_website_table done ==="
echo "Next: hard-refresh the browser, then when ready:"
echo "  bash scripts/full_website_refresh.sh"
echo "Log: $LOG"
