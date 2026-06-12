#!/usr/bin/env bash
# Wait for observability_windows_cache.json (or an apf_obs_cache screen) to finish,
# then rerun Keplerian fits and sync plots + table to the website.
#
# Usage:
#   bash scripts/wait_obs_cache_then_batch_sync.sh
#
# Detached on ziggy:
#   screen -dmS darkhunter_after_obs_cache bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO" && bash scripts/wait_obs_cache_then_batch_sync.sh
#   '

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
CACHE="${CACHE:-$REPO/rv_fit_reports/observability_windows_cache.json}"
LOG="${LOG:-$REPO/logs/wait_obs_cache_then_batch_sync.log}"
WAIT_SCREEN="${WAIT_SCREEN:-apf_obs_cache}"
POLL_SEC="${POLL_SEC:-60}"
STABLE_SEC="${STABLE_SEC:-120}"
MIN_CACHE_BYTES="${MIN_CACHE_BYTES:-20}"

cd "$REPO"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

_now() { date '+%Y-%m-%dT%H:%M:%S%z' 2>/dev/null || date; }

echo "=== $(_now) wait_obs_cache_then_batch_sync start (pid $$) ==="
echo "cache=$CACHE wait_screen=$WAIT_SCREEN poll=${POLL_SEC}s stable=${STABLE_SEC}s"

wait_for_screen() {
  if ! command -v screen >/dev/null 2>&1; then
    return 0
  fi
  while screen -ls 2>/dev/null | grep -q "[[:space:]]*[0-9]*\.${WAIT_SCREEN}[[:space:]]"; do
    echo "$(_now) screen session ${WAIT_SCREEN} still running; sleeping ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
  echo "$(_now) screen session ${WAIT_SCREEN} not running (or never started)"
}

wait_for_stable_cache() {
  local stable_for=0
  local last_size=-1
  while true; do
    if [[ -f "$CACHE" ]]; then
      size=$(stat -c%s "$CACHE" 2>/dev/null || echo 0)
      if [[ "$size" -ge "$MIN_CACHE_BYTES" && "$size" == "$last_size" ]]; then
        stable_for=$((stable_for + POLL_SEC))
        if [[ "$stable_for" -ge "$STABLE_SEC" ]]; then
          echo "$(_now) cache stable at ${size} bytes for >= ${STABLE_SEC}s: $CACHE"
          return 0
        fi
        echo "$(_now) cache ${size} bytes unchanged ${stable_for}s / ${STABLE_SEC}s"
      else
        stable_for=0
        echo "$(_now) cache size ${size} bytes (waiting for ${STABLE_SEC}s stability)"
      fi
      last_size="$size"
    else
      stable_for=0
      last_size=-1
      echo "$(_now) waiting for $CACHE to appear"
    fi
    sleep "$POLL_SEC"
  done
}

wait_for_screen
wait_for_stable_cache

echo "=== $(_now) starting batch_fits_plots_sync (FIT_FORCE=${FIT_FORCE:-1}) ==="
FIT_FORCE="${FIT_FORCE:-1}" MIN_POINTS="${MIN_POINTS:-5}" bash scripts/batch_fits_plots_sync.sh

echo "=== $(_now) wait_obs_cache_then_batch_sync done ==="
echo "Log: $LOG"
