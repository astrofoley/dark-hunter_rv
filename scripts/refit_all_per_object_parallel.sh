#!/usr/bin/env bash
# Parallel per-object refit: pipeline (subchunks_4 + debias) → Keplerian fit → website per star.
#
# Each worker runs scripts/lib/refit_one_object.sh (pipeline, fit, Hβ, stage plots + data.csv row).
# Uses flock on data.csv updates so parallel workers do not corrupt the website table.
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   PIPELINE_FORCE=1 JOBS=4 bash scripts/refit_all_per_object_parallel.sh
#
# Detached screen (recommended — low CPU priority, 4 parallel stars):
#   screen -dmS darkhunter_parallel_refit bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO"
#     JOBS=4 NICE_LEVEL=10 PIPELINE_FORCE=1 FIT_FORCE=1 FIT_JITTER=1 \
#       bash scripts/refit_all_per_object_parallel.sh
#   '
#
# Attach: screen -r darkhunter_parallel_refit
# Per-star logs: logs/refit_parallel/<gaia_id>.log
#
# Env:
#   JOBS          parallel workers (default: half of nproc, max 8)
#   NICE_LEVEL    nice priority 0–19 (default 10)
#   PIPELINE_FORCE=1  re-measure all epochs (default 1 here)
#   STAR_ID       optional single-star mode

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
STAR_ID="${STAR_ID:-}"
PIPELINE_FORCE="${PIPELINE_FORCE:-1}"
JOBS="${JOBS:-}"
NICE_LEVEL="${NICE_LEVEL:-10}"
LOG="${LOG:-$REPO/logs/refit_all_per_object_parallel.log}"

cd "$REPO"
# shellcheck source=scripts/lib/discover_gaia_star_ids.sh
source "$REPO/scripts/lib/discover_gaia_star_ids.sh"

if [[ -z "$JOBS" ]]; then
  ncpu=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  JOBS=$(( ncpu / 2 ))
  [[ "$JOBS" -lt 1 ]] && JOBS=1
  [[ "$JOBS" -gt 8 ]] && JOBS=8
fi

mkdir -p "$(dirname "$LOG")" "$REPO/logs/refit_parallel"
export REPO SPEC_ROOT PIPELINE_FORCE REFIT_PARALLEL_LOG_DIR="$REPO/logs/refit_parallel"
export OUT WEB_ROOT PY CHUNK_LAYOUT MASK_PRIMARY REPORTS_DIR MIN_POINTS
export FIT_FORCE FIT_JITTER PIPELINE_UPDATE RUN_HBETA QUERY_GAIA_ONLINE
export WEBSITE_STARS_DIR DATA_CSV

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%Y-%m-%dT%H:%M:%S%z') refit_all_per_object_parallel start pid=$$ ==="
echo "repo=$REPO jobs=$JOBS nice=$NICE_LEVEL pipeline_force=$PIPELINE_FORCE spec_root=$SPEC_ROOT"

STAR_IDS=()
if [[ -n "$STAR_ID" ]]; then
  STAR_IDS=("$STAR_ID")
else
  while IFS= read -r id; do
    [[ -n "$id" ]] && STAR_IDS+=("$id")
  done < <(discover_gaia_star_ids "$SPEC_ROOT")
fi

if [[ "${#STAR_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No Gaia_DR3_* stars under $SPEC_ROOT" >&2
  exit 2
fi
echo "stars_to_process=${#STAR_IDS[@]}"

worker="$REPO/scripts/lib/refit_one_object.sh"
chmod +x "$worker"

if ! command -v flock >/dev/null 2>&1 && [[ "$JOBS" -gt 1 ]]; then
  echo "[ERROR] flock not found — set JOBS=1 or install util-linux flock" >&2
  exit 2
fi

n_ok=0
n_fail=0
idx=0
total=${#STAR_IDS[@]}

for gid in "${STAR_IDS[@]}"; do
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$JOBS" ]]; do
    sleep 2
  done
  idx=$((idx + 1))
  echo "[launch $idx/$total] Gaia_DR3_${gid}"
  (
    if nice -n "$NICE_LEVEL" bash "$worker" "$gid"; then
      echo "[done OK] Gaia_DR3_${gid}"
      exit 0
    fi
    echo "[done FAIL] Gaia_DR3_${gid}"
    exit 1
  ) &
done

while [[ "$(jobs -rp | wc -l | tr -d ' ')" -gt 0 ]]; do
  if wait -n; then
    n_ok=$((n_ok + 1))
  else
    n_fail=$((n_fail + 1))
  fi
done

echo ""
echo "=== $(date '+%Y-%m-%dT%H:%M:%S%z') refit_all_per_object_parallel done ok=$n_ok failed=$n_fail ==="
echo "Master log: $LOG"
echo "Per-star logs: $REPO/logs/refit_parallel/<gaia_id>.log"
