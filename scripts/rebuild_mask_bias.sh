#!/usr/bin/env bash
# Rebuild repo-root bias_statistics.txt for the production mask chunk layout.
#
# Runs mask-only pipeline (--no-bias) on calibration/bias_train.txt, then
# validation.build_bias_set → bias_statistics.txt (per-chunk b0/b1/b2).
#
# Usage:
#   cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
#   bash scripts/rebuild_mask_bias.sh
#
#   SKIP_PIPELINE=1 bash scripts/rebuild_mask_bias.sh   # aggregate existing *_orders.txt only
#   CHUNK_LAYOUT=calibration/chunk_layouts/subchunks_8.yaml bash scripts/rebuild_mask_bias.sh
#   DRY_RUN=1 bash scripts/rebuild_mask_bias.sh   # print commands only
#
# Server (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv
#   PY=/home/marley/anaconda2/envs/gaia-env/bin/python bash scripts/rebuild_mask_bias.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-$REPO/output}"
BIAS_LIST="${BIAS_LIST:-$REPO/calibration/bias_train.txt}"
CHUNK_LAYOUT="${CHUNK_LAYOUT:-$REPO/calibration/chunk_layouts/subchunks_8.yaml}"
PY="${PY:-python3}"
DRY_RUN="${DRY_RUN:-0}"
BOOTSTRAP="${BOOTSTRAP:-200}"
BACKUP="${BACKUP:-1}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"

if [[ ! -f "$BIAS_LIST" ]]; then
  echo "[ERROR] Missing bias list: $BIAS_LIST" >&2
  exit 2
fi
if [[ ! -f "$CHUNK_LAYOUT" ]]; then
  echo "[ERROR] Missing chunk layout: $CHUNK_LAYOUT" >&2
  exit 2
fi

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

cd "$REPO"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"
export DARKHUNTER_CHUNK_LAYOUT="$CHUNK_LAYOUT"

mkdir -p "$OUT" "$REPO/calibration/bias_build"

echo "=== rebuild_mask_bias: layout=$CHUNK_LAYOUT out=$OUT ==="
echo "=== bias list: $BIAS_LIST ($(grep -cve '^[[:space:]]*$' -e '^#' "$BIAS_LIST" || true) spectra) ==="

if [[ "$BACKUP" == "1" && -f "$REPO/bias_statistics.txt" ]]; then
  ts=$(date +%Y%m%d)
  bak="$REPO/bias_statistics.bak_${ts}"
  if [[ ! -f "$bak" ]]; then
    /bin/cp "$REPO/bias_statistics.txt" "$bak"
  fi
fi

SETUP_ARGS=(
  --bias-list "$BIAS_LIST"
  --bias-only
  --chunk-layout "$CHUNK_LAYOUT"
  --bootstrap "$BOOTSTRAP"
  --log-level INFO
  --clean-after-bias
)
if [[ "$SKIP_PIPELINE" == "1" ]]; then
  SETUP_ARGS+=(--skip-bias-pipeline)
fi

run "$PY" -m validation.run_calibration_setup "${SETUP_ARGS[@]}"

echo "=== Done. Installed: $REPO/bias_statistics.txt ==="
run head -5 "$REPO/bias_statistics.txt" 2>/dev/null || true
