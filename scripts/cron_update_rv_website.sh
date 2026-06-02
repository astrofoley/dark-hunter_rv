#!/usr/bin/env bash
# Cron-friendly: measure RVs on new spectra, then refresh website assets for updated stars.
#
# Example crontab (daily 06:00):
#   0 6 * * * REPO=/data2/darkhunter/dark-hunter_rv WEB_ROOT=/var/www/html/darkhunter/rv SPEC_ROOT=/data2/gaia_stars/apf_reductions bash $REPO/scripts/cron_update_rv_website.sh >> $REPO/logs/cron_rv_website.log 2>&1
#
# Env:
#   REPO, OUT, WEB_ROOT, SPEC_ROOT, PY, MIN_POINTS, LOG

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
MIN_POINTS="${MIN_POINTS:-5}"
LOG="${LOG:-$REPO/logs/cron_rv_website.log}"

mkdir -p "$(dirname "$LOG")" "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"

exec >>"$LOG" 2>&1
echo "=== $(date -Is) cron_update_rv_website start ==="

# 1) Pipeline: new/changed spectra under SPEC_ROOT (--update skips unchanged diagnostics).
if [[ -d "$SPEC_ROOT" ]]; then
  echo "=== Pipeline --update on $SPEC_ROOT ==="
  find "$SPEC_ROOT" -type f \( -name '*_ap1.flm' -o -name '*_ap1.txt' -o -name '*.fits' \) -print0 2>/dev/null \
    | xargs -0 -r "$PY" -m darkhunter_rv.pipeline --instrument APF --update --plots --plots-focus 2>/dev/null \
    || echo "[WARN] pipeline pass had errors (continuing)"
else
  echo "[WARN] SPEC_ROOT missing: $SPEC_ROOT"
fi

# 2) Fits (incl. literature), plots, Hβ stacks, website staging.
echo "=== Website populate (fits + assets) ==="
export WEB_ROOT MIN_POINTS
RUN_FITS=1 FIT_FORCE=0 bash scripts/populate_website.sh

echo "=== $(date -Is) cron_update_rv_website done ==="
