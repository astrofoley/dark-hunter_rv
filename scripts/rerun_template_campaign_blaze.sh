#!/usr/bin/env bash
# Re-run 114-exposure template campaign with split blaze continuum (mask: blaze-only;
# template/strong: blaze+spline). Writes diagnostics to a separate tree for comparison.
#
#   cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
#   bash scripts/rerun_template_campaign_blaze.sh
#
# Optional env:
#   OUT_ROOT  — pipeline output root (default: validation_output/template_fft_baseline/pipeline_blaze_split)
#   BATCH     — spectra per pipeline invocation (default: 20)
#   PY        — python3 interpreter

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
LIST="${LIST:-$REPO/validation_output/chunk_campaign/spectrum_list.txt}"
OUT_ROOT="${OUT_ROOT:-$REPO/validation_output/template_fft_baseline/pipeline_blaze_split}"
CHUNK_LAYOUT="${CHUNK_LAYOUT:-$REPO/calibration/chunk_layouts/subchunks_8.yaml}"
BLAZE_CAL="${BLAZE_CAL:-$REPO/calibration/blaze_orders_apf.json}"
LOG="${LOG:-$REPO/logs/template_campaign_blaze_split.log}"
BATCH="${BATCH:-5}"

if [[ ! -f "$LIST" ]]; then
  echo "[ERROR] Missing spectrum list: $LIST" >&2
  exit 2
fi
if [[ ! -f "$BLAZE_CAL" ]]; then
  echo "[ERROR] Missing blaze calibration: $BLAZE_CAL" >&2
  echo "Build with: python3 -m validation.build_blaze_calibration --spectrum-list $LIST ..." >&2
  exit 2
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT_ROOT"
export DARKHUNTER_CHUNK_LAYOUT="$CHUNK_LAYOUT"
mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"

SPECS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -n "$line" ]] && SPECS+=("$line")
done < "$LIST"
echo "=== template campaign blaze-split $(date '+%Y-%m-%dT%H:%M:%S%z') ===" | tee "$LOG"
echo "spectra=${#SPECS[@]} out=$OUT_ROOT blaze=$BLAZE_CAL" | tee -a "$LOG"

for ((i=0; i<${#SPECS[@]}; i+=BATCH)); do
  chunk=("${SPECS[@]:i:BATCH}")
  echo "=== batch $i..$((i+${#chunk[@]}-1)) ===" | tee -a "$LOG"
  "$PY" -m darkhunter_rv.pipeline \
    "${chunk[@]}" \
    --instrument APF \
    --chunk-layout "$CHUNK_LAYOUT" \
    --continuum-mode split \
    --blaze-calibration "$BLAZE_CAL" \
    --run-all-methods \
    --force \
    --log-level INFO \
    2>&1 | tee -a "$LOG"
done

echo "=== overlap report ===" | tee -a "$LOG"
OVERLAP_OUT="$REPO/validation_output/template_fft_baseline/overlap_blaze_split"
"$PY" -m validation.rv_method_overlap_report \
  --diagnostics-glob "$OUT_ROOT/Gaia_DR3_*_diagnostics.csv" \
  --gaia-summary-dir "$OUT_ROOT" \
  --out-dir "$OVERLAP_OUT" \
  2>&1 | tee -a "$LOG"

echo "=== teff residual report ===" | tee -a "$LOG"
TEFF_OUT="$REPO/validation_output/template_fft_baseline/teff_residuals_blaze_split"
"$PY" -m validation.rv_method_diagnostics_report \
  --diagnostics-glob "$OUT_ROOT/Gaia_DR3_*_diagnostics.csv" \
  --legacy-summary-dir "$OUT_ROOT" \
  --out-dir "$TEFF_OUT" \
  2>&1 | tee -a "$LOG"

echo "Done. Diagnostics: $OUT_ROOT" | tee -a "$LOG"
echo "Overlap: $OVERLAP_OUT" | tee -a "$LOG"
echo "Teff residuals: $TEFF_OUT" | tee -a "$LOG"
