#!/usr/bin/env bash
# Bulk refit: pipeline all spectra, then Keplerian fits + plots + Hβ + website in one batch.
# Website updates only at the end (not per star). Prefer refit_all_per_object.sh for incremental site updates.
#
# Detached:
#   screen -dmS darkhunter_bulk_refresh bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO" && git pull && bash scripts/full_website_refresh_bulk.sh
#   '

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
MIN_POINTS="${MIN_POINTS:-5}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"
LOG="${LOG:-$REPO/logs/full_website_refresh_bulk.log}"

cd "$REPO"
# shellcheck source=scripts/lib/spec_find_patterns.sh
source "$REPO/scripts/lib/spec_find_patterns.sh"
mkdir -p "$(dirname "$LOG")" "$OUT" "$REPO/logs"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT OUT SPEC_ROOT MIN_POINTS
export RUN_FITS=1 RUN_RV_PLOTS=1 RUN_HBETA_PLOTS=1 FIT_FORCE=1 QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
export LOG="$REPO/logs/batch_fits_plots_sync.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) full_website_refresh_bulk start (pid $$) ==="

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  echo "[ERROR] $WEB_ROOT/tables/data.csv missing — run scripts/bootstrap_website_tables.sh first." >&2
  exit 2
fi

if [[ "$RUN_PIPELINE" == "1" && -d "$SPEC_ROOT" ]]; then
  echo "=== Pipeline on all spectra under $SPEC_ROOT ==="
  find_apf_spectra_print0 "$SPEC_ROOT" \
    | xargs -0 -r "$PY" -m darkhunter_rv.pipeline --instrument APF --plots --plots-focus 2>/dev/null \
    || echo "[WARN] pipeline pass had errors (continuing)"
fi

bash scripts/setup_website.sh
if [[ -f "$REPO/scripts/fix_data_csv_column_order.py" ]]; then
  "$PY" scripts/fix_data_csv_column_order.py --data-csv "$WEB_ROOT/tables/data.csv"
fi

bash scripts/populate_website.sh
echo "=== $(date -Is) full_website_refresh_bulk done ==="
