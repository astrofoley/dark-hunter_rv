#!/usr/bin/env bash
# Re-measure per-epoch mask-CCF RVs (production chunk layout + debias) and Keplerian fit one star.
#
# Production defaults:
#   - chunk layout: calibration/chunk_layouts/subchunks_4.yaml (4 equal pixel splits / order)
#   - debias: repo-root bias_statistics.txt (per-order b0/b1/b2)
#   - summary RV: mask-CCF stack (--no-run-all-methods)
#   - pipeline: --force (ignore stale diagnostics from older layouts)
#
# Local example:
#   cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
#   STAR_ID=1702370142434513152 \
#   SPEC_ROOT=/Users/rfoley/darkhunter/rvs/data \
#   bash scripts/refit_star_rvs.sh
#
# Server (ziggy) example:
#   cd /data2/darkhunter/dark-hunter_rv
#   git fetch origin && git checkout step/01-benchmark-cool-precision && git pull
#   STAR_ID=1702370142434513152 bash scripts/refit_star_rvs.sh
#
# Optional env:
#   CHUNK_LAYOUT  — override layout YAML
#   PIPELINE_FORCE=0 — skip --force (not recommended for layout changes)
#   MASK_PRIMARY=0 — allow multi-method adopted RV in summary (default 1 = mask stack)
#   FIT_JITTER=1, FIT_FORCE=1, MIN_POINTS=5, QUERY_GAIA_ONLINE=0
#   PY — Python 3 interpreter (server: python3 or gaia-env python)
#   DARKHUNTER_PHOENIX_DIR — HiRes PHOENIX grid if not auto-detected

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-$REPO/output}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
CHUNK_LAYOUT="${CHUNK_LAYOUT:-$REPO/calibration/chunk_layouts/subchunks_4.yaml}"
PIPELINE_FORCE="${PIPELINE_FORCE:-1}"
MASK_PRIMARY="${MASK_PRIMARY:-1}"
MIN_POINTS="${MIN_POINTS:-5}"
FIT_FORCE="${FIT_FORCE:-1}"
FIT_JITTER="${FIT_JITTER:-1}"
QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
STAR_ID="${STAR_ID:-${1:-}}"

if [[ -z "$STAR_ID" ]]; then
  echo "Usage: STAR_ID=<gaia_id> bash scripts/refit_star_rvs.sh" >&2
  echo "   or: bash scripts/refit_star_rvs.sh <gaia_id>" >&2
  exit 2
fi

if [[ -n "${PY:-}" ]] && command -v "$PY" >/dev/null 2>&1; then
  :
elif [[ -x /home/marley/anaconda2/envs/gaia-env/bin/python ]]; then
  PY=/home/marley/anaconda2/envs/gaia-env/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "[ERROR] No Python 3 found (set PY=python3 or gaia-env python)." >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$OUT" "$REPORTS_DIR" "$(dirname "$OUT")"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"

if [[ ! -f "$REPO/bias_statistics.txt" ]]; then
  echo "[ERROR] Missing $REPO/bias_statistics.txt (required for debiased RVs)." >&2
  exit 2
fi
if [[ ! -f "$CHUNK_LAYOUT" ]]; then
  echo "[ERROR] Chunk layout not found: $CHUNK_LAYOUT" >&2
  exit 2
fi

SPEC_FILES=()
while IFS= read -r f; do
  [[ -n "$f" ]] && SPEC_FILES+=("$f")
done < <(
  find "$SPEC_ROOT" -type f \( \
    \( -name "Gaia_DR3_${STAR_ID}_epoch_*.txt" ! -name '*_order_*' \) -o \
    -name "Gaia_DR3_${STAR_ID}_*_ap1.flm" -o \
    -name "Gaia_DR3_${STAR_ID}_*_ap1.txt" \
  \) 2>/dev/null | sort
)
if [[ "${#SPEC_FILES[@]}" -eq 0 ]]; then
  echo "[ERROR] No spectra for Gaia_DR3_${STAR_ID} under $SPEC_ROOT" >&2
  exit 2
fi

echo "=== refit_star_rvs Gaia_DR3_${STAR_ID} $(date '+%Y-%m-%dT%H:%M:%S%z') ==="
echo "repo=$REPO out=$OUT spec_root=$SPEC_ROOT"
echo "py=$PY chunk_layout=$CHUNK_LAYOUT bias=$REPO/bias_statistics.txt"
echo "spectra=${#SPEC_FILES[@]} pipeline_force=$PIPELINE_FORCE mask_primary=$MASK_PRIMARY"

pipeline_args=(
  --instrument APF
  --plots-focus
  --chunk-layout "$CHUNK_LAYOUT"
  --log-level INFO
)
if [[ "$PIPELINE_FORCE" == "1" ]]; then
  pipeline_args+=(--force)
fi
if [[ "$MASK_PRIMARY" == "1" ]]; then
  pipeline_args+=(--no-run-all-methods)
fi

if ! "$PY" -m darkhunter_rv.pipeline "${pipeline_args[@]}" "${SPEC_FILES[@]}"; then
  echo "[ERROR] Pipeline failed for Gaia_DR3_${STAR_ID}" >&2
  exit 1
fi

summ="$OUT/Gaia_DR3_${STAR_ID}_summary.txt"
if [[ ! -f "$summ" ]]; then
  echo "[ERROR] Pipeline did not write $summ" >&2
  exit 2
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
if [[ "$FIT_JITTER" == "1" ]]; then
  fit_args+=(--fit-jitter)
fi
if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
  fit_args+=(--query-gaia-online)
fi

"$PY" "${fit_args[@]}"

echo ""
echo "=== Done ==="
echo "Summary:  $summ"
echo "Fit JSON: $REPORTS_DIR/${STAR_ID}_keplerian_fit.json"
echo "Fit PNG:  $REPORTS_DIR/${STAR_ID}_keplerian_fit.png"
