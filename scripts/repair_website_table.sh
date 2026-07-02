#!/usr/bin/env bash
# Quick website table + frontend repair (no Keplerian refit, no plot rebuild).
#
# By default runs phase 2 only (fast): deploy static site + refresh data.csv.
# Slow Gaia TAP queries are separated — run once first:
#   bash scripts/query_website_gaia.sh
#
# To include Gaia queries in this run (legacy combined behavior):
#   RUN_GAIA_QUERIES=1 bash scripts/repair_website_table.sh
#
# Phased workflow (see docs/website.md):
#   1. bash scripts/query_website_gaia.sh          # Gaia summaries (slow)
#   2. bash scripts/update_website_table_only.sh   # table + UI (fast)
#   3. bash scripts/replot_rv_website.sh           # RV + fit plots (no refit)
#   4. bash scripts/refit_all_per_object.sh        # full pipeline refit
#
# Does not touch repo-root bias_statistics.txt (mask debias table is manual-only).
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv
#   git pull
#   bash scripts/query_website_gaia.sh              # once, if summaries missing
#   bash scripts/repair_website_table.sh            # or: update_website_table_only.sh

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
LOG="${LOG:-$REPO/logs/repair_website_table.log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$REPO/logs"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) repair_website_table start (pid $$) ==="

if [[ "${RUN_GAIA_QUERIES:-0}" == "1" ]]; then
  echo "=== RUN_GAIA_QUERIES=1: phase 1 Gaia queries ==="
  bash scripts/query_website_gaia.sh
fi

bash scripts/update_website_table_only.sh

echo "=== $(date -Is) repair_website_table done ==="
echo "Optional next steps:"
echo "  bash scripts/replot_rv_website.sh"
echo "  bash scripts/full_website_refresh.sh"
echo "Log: $LOG"
