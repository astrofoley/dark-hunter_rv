#!/usr/bin/env bash
# Cron-friendly: measure RVs on new/changed spectra, then Keplerian fits + website assets.
#
# Daily crontab (06:00 local) — append log; do not wrap in screen:
#   0 6 * * * REPO=/data2/darkhunter/dark-hunter_rv WEB_ROOT=/var/www/html/darkhunter/rv SPEC_ROOT=/data2/gaia_stars/apf_reductions MIN_POINTS=5 /bin/bash $REPO/scripts/cron_update_rv_website.sh >> $REPO/logs/cron_rv_website.log 2>&1
#
# Env: REPO, OUT, WEB_ROOT, SPEC_ROOT, PY, MIN_POINTS, LOG, RUN_PIPELINE (default 1)

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
MIN_POINTS="${MIN_POINTS:-5}"
LOG="${LOG:-$REPO/logs/cron_rv_website.log}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"

mkdir -p "$(dirname "$LOG")" "$OUT" "$REPO/logs"
cd "$REPO"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT OUT SPEC_ROOT MIN_POINTS
export RUN_FITS=1 RUN_RV_PLOTS=1 RUN_HBETA_PLOTS=1 FIT_FORCE=0 QUERY_GAIA_ONLINE=0
export LOG="$REPO/logs/batch_fits_plots_sync.log"

exec >>"$REPO/logs/cron_rv_website.log" 2>&1
echo "=== $(date -Is) cron_update_rv_website start (pid $$) ==="

# 1) Pipeline: new/changed spectra (--update skips unchanged). Updates summaries in output/.
if [[ "$RUN_PIPELINE" == "1" && -d "$SPEC_ROOT" ]]; then
  echo "=== Pipeline --update on $SPEC_ROOT ==="
  find "$SPEC_ROOT" -type f \( -name '*_ap1.flm' -o -name '*_ap1.txt' -o -name '*.fits' \) -print0 2>/dev/null \
    | xargs -0 -r "$PY" -m darkhunter_rv.pipeline --instrument APF --update --plots --plots-focus 2>/dev/null \
    || echo "[WARN] pipeline pass had errors (continuing)"
elif [[ "$RUN_PIPELINE" == "1" ]]; then
  echo "[WARN] SPEC_ROOT missing: $SPEC_ROOT"
fi

# 2) Keplerian fits (pipeline + literature; bad RVs filtered), plots, Hβ, data.csv masses, staging.
# FIT_FORCE=0: refit only when summary is newer than existing *_keplerian_fit.json.
echo "=== Website populate (incremental fits + assets) ==="
bash scripts/populate_website.sh

echo "=== $(date -Is) cron_update_rv_website done ==="
