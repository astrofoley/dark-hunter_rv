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
# Detached (recommended for full catalog):
#   screen -dmS darkhunter_per_object bash -lc '
#     REPO=/data2/darkhunter/dark-hunter_rv
#     cd "$REPO" && git pull && bash scripts/refit_all_per_object.sh
#   '

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
MIN_POINTS="${MIN_POINTS:-5}"
FIT_FORCE="${FIT_FORCE:-1}"
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
# shellcheck source=scripts/lib/website_sync_one_star.sh
source "$REPO/scripts/lib/website_sync_one_star.sh"

mkdir -p "$(dirname "$LOG")" "$OUT" "$REPO/logs" "$REPORTS_DIR"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT OUT SPEC_ROOT REPORTS_DIR PY MIN_POINTS
export WEBSITE_STARS_DIR="$WEB_ROOT/stars"
export DATA_CSV="$WEB_ROOT/tables/data.csv"

run_cmd() { "$@"; }

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

pipeline_args=(--instrument APF --plots --plots-focus)
if [[ "$PIPELINE_UPDATE" == "1" ]]; then
  pipeline_args+=(--update)
fi

n_ok=0
n_skip=0
for gid in "${STAR_IDS[@]}"; do
  echo ""
  echo "=== Gaia_DR3_${gid} ($(date -Is)) ==="

  mapfile -d '' -t SPEC_FILES < <(
    find "$SPEC_ROOT" -type f \( \
      -name "Gaia_DR3_${gid}_epoch_*.txt" -o \
      -name "Gaia_DR3_${gid}_*_ap1.flm" -o \
      -name "Gaia_DR3_${gid}_*_ap1.txt" \
    \) -print0 2>/dev/null
  )
  if [[ "${#SPEC_FILES[@]}" -eq 0 ]]; then
    echo "[WARN] no spectra files for Gaia_DR3_${gid}; skip"
    n_skip=$((n_skip + 1))
    continue
  fi
  echo "pipeline: ${#SPEC_FILES[@]} spectrum file(s)"
  printf '%s\0' "${SPEC_FILES[@]}" | xargs -0 -r "$PY" -m darkhunter_rv.pipeline "${pipeline_args[@]}" \
    || echo "[WARN] pipeline errors for Gaia_DR3_${gid} (continuing)"

  summ="$OUT/Gaia_DR3_${gid}_summary.txt"
  if [[ ! -f "$summ" ]]; then
    echo "[WARN] no summary after pipeline; skip fit/website"
    n_skip=$((n_skip + 1))
    continue
  fi

  fit_args=(
    fit_apf_rv_keplerian.py
    --summary "$summ"
    --output-dir "$OUT"
    --reports-dir "$REPORTS_DIR"
    --use-gaia-nss
    --min-points "$MIN_POINTS"
  )
  if [[ "$FIT_FORCE" == "1" ]]; then
    fit_args+=(--force)
  fi
  if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
    fit_args+=(--query-gaia-online)
  fi
  "$PY" "${fit_args[@]}" || echo "[WARN] fit failed for Gaia_DR3_${gid} (continuing)"

  if [[ "$RUN_HBETA" == "1" ]]; then
    "$PY" scripts/build_hbeta_website_plots.py \
      --summary-dir "$OUT" \
      --plots-root "$OUT" \
      --spec-root "$SPEC_ROOT" \
      --star-id "$gid" \
      || echo "[WARN] Hβ plot failed for Gaia_DR3_${gid} (continuing)"
  fi

  if website_sync_one_star "$gid"; then
    n_ok=$((n_ok + 1))
    echo "[OK] website updated for Gaia_DR3_${gid}"
  else
    n_skip=$((n_skip + 1))
  fi
done

echo ""
echo "=== $(date -Is) refit_all_per_object done: website_ok=$n_ok skipped=$n_skip ==="
echo "Log: $LOG"
