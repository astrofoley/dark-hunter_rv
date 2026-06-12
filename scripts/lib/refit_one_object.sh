#!/usr/bin/env bash
# Pipeline → Keplerian fit → Hβ plots → website sync for one Gaia_DR3_<id>.
# Called by refit_all_per_object.sh and refit_all_per_object_parallel.sh.

set -euo pipefail

gid="${1:?Gaia source id required}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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
CHUNK_LAYOUT="${CHUNK_LAYOUT:-$REPO/calibration/chunk_layouts/subchunks_4.yaml}"
PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
MASK_PRIMARY="${MASK_PRIMARY:-1}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
MIN_POINTS="${MIN_POINTS:-5}"
FIT_FORCE="${FIT_FORCE:-1}"
FIT_JITTER="${FIT_JITTER:-1}"
PIPELINE_UPDATE="${PIPELINE_UPDATE:-0}"
RUN_HBETA="${RUN_HBETA:-1}"
RUN_RV_PLOTS="${RUN_RV_PLOTS:-1}"
QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
REFIT_PARALLEL_LOG_DIR="${REFIT_PARALLEL_LOG_DIR:-$REPO/logs/refit_parallel}"

cd "$REPO"
# shellcheck source=scripts/lib/website_sync_one_star.sh
source "$REPO/scripts/lib/website_sync_one_star.sh"

export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export WEB_ROOT OUT SPEC_ROOT REPORTS_DIR PY MIN_POINTS
export WEBSITE_STARS_DIR="${WEBSITE_STARS_DIR:-$WEB_ROOT/stars}"
export DATA_CSV="${DATA_CSV:-$WEB_ROOT/tables/data.csv}"

run_cmd() { "$@"; }

mkdir -p "$OUT" "$REPORTS_DIR" "$REFIT_PARALLEL_LOG_DIR"
LOG="$REFIT_PARALLEL_LOG_DIR/${gid}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Gaia_DR3_${gid} worker pid=$$ $(date '+%Y-%m-%dT%H:%M:%S%z') ==="

if [[ ! -f "$DATA_CSV" ]]; then
  echo "[ERROR] $DATA_CSV missing" >&2
  exit 2
fi
if [[ ! -f "$REPO/bias_statistics.txt" ]]; then
  echo "[WARN] $REPO/bias_statistics.txt missing — mask RVs will not be debiased" >&2
fi

SPEC_FILES=()
while IFS= read -r f; do
  [[ -n "$f" ]] && SPEC_FILES+=("$f")
done < <(
  find "$SPEC_ROOT" -type f \( \
    \( -name "Gaia_DR3_${gid}_epoch_*.txt" ! -name '*_order_*' \) -o \
    -name "Gaia_DR3_${gid}_*_ap1.flm" -o \
    -name "Gaia_DR3_${gid}_*_ap1.txt" \
  \) 2>/dev/null | sort
)
if [[ "${#SPEC_FILES[@]}" -eq 0 ]]; then
  echo "[WARN] no spectra for Gaia_DR3_${gid}; skip"
  exit 1
fi

pipeline_args=(--instrument APF --plots --plots-focus)
if [[ -f "$CHUNK_LAYOUT" ]]; then
  pipeline_args+=(--chunk-layout "$CHUNK_LAYOUT")
fi
if [[ "$PIPELINE_FORCE" == "1" ]]; then
  pipeline_args+=(--force)
elif [[ "$PIPELINE_UPDATE" == "1" ]]; then
  pipeline_args+=(--update)
fi
if [[ "$MASK_PRIMARY" == "1" ]]; then
  pipeline_args+=(--no-run-all-methods)
fi

echo "pipeline: ${#SPEC_FILES[@]} spectrum file(s)"
if ! "$PY" -m darkhunter_rv.pipeline "${pipeline_args[@]}" "${SPEC_FILES[@]}"; then
  echo "[WARN] pipeline errors for Gaia_DR3_${gid} (continuing to fit if summary exists)"
fi

summ="$OUT/Gaia_DR3_${gid}_summary.txt"
if [[ ! -f "$summ" ]]; then
  echo "[WARN] no summary after pipeline; skip fit/website"
  exit 1
fi

fit_args=(
  fit_apf_rv_keplerian.py
  --summary "$summ"
  --output-dir "$OUT"
  --reports-dir "$REPORTS_DIR"
  --use-gaia-nss
  --min-points "$MIN_POINTS"
  --data-csv "$DATA_CSV"
)
if [[ "$FIT_FORCE" == "1" ]]; then
  fit_args+=(--force)
fi
if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
  fit_args+=(--query-gaia-online)
fi
if [[ "$FIT_JITTER" == "1" ]]; then
  fit_args+=(--fit-jitter)
fi
if ! "$PY" "${fit_args[@]}"; then
  echo "[WARN] fit failed for Gaia_DR3_${gid} (continuing to website if assets exist)"
fi

if [[ "$RUN_RV_PLOTS" == "1" ]]; then
  "$PY" scripts/plot_rv_from_summaries.py \
    --summary-dir "$OUT" \
    --plots-root "$OUT" \
    --star-id "$gid" \
    || echo "[WARN] RV data plot failed for Gaia_DR3_${gid} (continuing)"
fi

if [[ "$RUN_HBETA" == "1" ]]; then
  "$PY" scripts/build_hbeta_website_plots.py \
    --summary-dir "$OUT" \
    --plots-root "$OUT" \
    --spec-root "$SPEC_ROOT" \
    --star-id "$gid" \
    || echo "[WARN] Hβ plot failed for Gaia_DR3_${gid} (continuing)"
fi

if website_sync_one_star "$gid"; then
  echo "[OK] Gaia_DR3_${gid} pipeline + fit + website complete"
  exit 0
fi
echo "[WARN] website sync failed for Gaia_DR3_${gid}"
exit 1
