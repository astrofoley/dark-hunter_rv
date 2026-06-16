#!/usr/bin/env bash
# Per-object backfill: for each Gaia star, run pipeline on all its spectra, Keplerian fit,
# Hβ overlay, then stage that star to the website and update its data.csv row.
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/refit_all_per_object.sh
#
#   STAR_ID=1551542027851147904 bash scripts/refit_all_per_object.sh
#   MIN_POINTS=5 FIT_FORCE=1 bash scripts/refit_all_per_object.sh
#
# Detached sequential (one star at a time):
#   screen -dmS darkhunter_per_object bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO" && PIPELINE_FORCE=1 bash scripts/refit_all_per_object.sh
#   '
#
# Detached parallel (recommended for full catalog refit):
#   screen -dmS darkhunter_parallel_refit bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO"
#     JOBS=4 NICE_LEVEL=10 PIPELINE_FORCE=1 FIT_FORCE=1 FIT_JITTER=1 \
#       bash scripts/refit_all_per_object_parallel.sh
#   '

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
if [[ -n "${PY:-}" ]] && command -v "$PY" >/dev/null 2>&1; then
  :
elif [[ -x /home/marley/anaconda2/envs/gaia-env/bin/python ]]; then
  PY=/home/marley/anaconda2/envs/gaia-env/bin/python
else
  PY=python3
fi
CHUNK_LAYOUT="${CHUNK_LAYOUT:-$REPO/calibration/chunk_layouts/subchunks_8.yaml}"
PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
MASK_PRIMARY="${MASK_PRIMARY:-1}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
MIN_POINTS="${MIN_POINTS:-5}"
FIT_FORCE="${FIT_FORCE:-1}"
FIT_JITTER="${FIT_JITTER:-1}"
PIPELINE_UPDATE="${PIPELINE_UPDATE:-0}"
RUN_HBETA="${RUN_HBETA:-1}"
QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
STAR_ID="${STAR_ID:-}"
LOG="${LOG:-$REPO/logs/refit_all_per_object.log}"

cd "$REPO"
# shellcheck source=scripts/lib/spec_find_patterns.sh
source "$REPO/scripts/lib/spec_find_patterns.sh"
# shellcheck source=scripts/lib/discover_gaia_star_ids.sh
source "$REPO/scripts/lib/discover_gaia_star_ids.sh"
mkdir -p "$(dirname "$LOG")" "$OUT" "$REPO/logs" "$REPORTS_DIR"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT OUT SPEC_ROOT REPORTS_DIR PY MIN_POINTS
export WEBSITE_STARS_DIR="$WEB_ROOT/stars"
export DATA_CSV="$WEB_ROOT/tables/data.csv"
export CHUNK_LAYOUT PIPELINE_FORCE MASK_PRIMARY FIT_FORCE FIT_JITTER
export PIPELINE_UPDATE RUN_HBETA QUERY_GAIA_ONLINE

if [[ ! -f "$DATA_CSV" ]]; then
  echo "[ERROR] $DATA_CSV missing — run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) refit_all_per_object start (pid $$) ==="
echo "repo=$REPO spec_root=$SPEC_ROOT out=$OUT web_root=$WEB_ROOT min_points=$MIN_POINTS"

mapfile -t STAR_IDS < <(
  if [[ -n "$STAR_ID" ]]; then
    echo "$STAR_ID"
  else
    discover_gaia_star_ids "$SPEC_ROOT"
  fi
)

if [[ "${#STAR_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No Gaia_DR3_* stars under $SPEC_ROOT" >&2
  exit 2
fi
echo "stars_to_process=${#STAR_IDS[@]}"

export REFIT_PARALLEL_LOG_DIR="$REPO/logs/refit_per_object"
if [[ ! -f "$REPO/bias_statistics.txt" ]]; then
  echo "[WARN] $REPO/bias_statistics.txt missing — mask RVs will not be debiased" >&2
fi
mkdir -p "$REFIT_PARALLEL_LOG_DIR"
worker="$REPO/scripts/lib/refit_one_object.sh"
chmod +x "$worker"

n_ok=0
n_skip=0
for gid in "${STAR_IDS[@]}"; do
  echo ""
  echo "=== Gaia_DR3_${gid} ($(date '+%Y-%m-%dT%H:%M:%S%z')) ==="
  if bash "$worker" "$gid"; then
    n_ok=$((n_ok + 1))
  else
    n_skip=$((n_skip + 1))
  fi
done

echo ""
echo "=== $(date -Is) refit_all_per_object done: website_ok=$n_ok skipped=$n_skip ==="
echo "Log: $LOG"
