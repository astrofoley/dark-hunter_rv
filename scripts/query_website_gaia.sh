#!/usr/bin/env bash
# Phase 1: Gaia TAP queries for ATF22/E24 sample stars (slow; run once).
#
# Writes output/Gaia_DR3_<id>_summary.txt (metadata + external RVs) and backfills
# G/BP/RP photometry. Does not touch data.csv, plots, or bias_statistics.txt.
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/query_website_gaia.sh
#
# Detached screen:
#   screen -dmS dh_gaia_query bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     export PY=/home/marley/anaconda2/envs/gaia-env/bin/python
#     cd "$REPO" && bash scripts/query_website_gaia.sh
#   '

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
LOG="${LOG:-$REPO/logs/query_website_gaia.log}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")" "$OUT" "$REPO/logs"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) query_website_gaia start (pid $$) ==="
echo "repo=$REPO out=$OUT"

echo "=== Sample star Gaia queries (summaries only) ==="
"$PY" scripts/ensure_sample_stars_website.py \
  --output-dir "$OUT" \
  --summaries-only

echo "=== Patch Gaia G/BP/RP photometry in sample summaries ==="
"$PY" scripts/patch_summary_gaia_photometry.py --output-dir "$OUT"

echo "=== $(date -Is) query_website_gaia done ==="
echo "Next: bash scripts/update_website_table_only.sh"
echo "Log: $LOG"
