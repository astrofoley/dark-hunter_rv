#!/usr/bin/env bash
# Remove legacy pipeline Hβ PNGs from deployed website Plots/ (keeps _28_hbeta, rv, residuals).
#
# Usage (ziggy):
#   cd /data2/darkhunter/dark-hunter_rv && git pull
#   bash scripts/prune_legacy_website_plots.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib/website_plot_sync.sh
source "$SCRIPT_DIR/lib/website_plot_sync.sh"

WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
STARS="$WEB_ROOT/stars"

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

if [[ ! -d "$STARS" ]]; then
  echo "[ERROR] missing $STARS" >&2
  exit 2
fi

n=0
for plot_dir in "$STARS"/Gaia_DR3_*/Gaia/Plots; do
  [[ -d "$plot_dir" ]] || continue
  before=$(find "$plot_dir" -maxdepth 1 -type f \( \
    -name '*_h_beta_rv.png' -o \
    -name '*_h_beta_order*.png' -o \
    -name '*_h_beta_three*.png' \
  \) 2>/dev/null | wc -l | tr -d ' ')
  website_prune_legacy_gaia_plots "$plot_dir"
  n=$((n + before))
done

echo "Pruned legacy Hβ plot files under $STARS (removed $n files)."
