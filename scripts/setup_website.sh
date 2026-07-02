#!/usr/bin/env bash
# Install static website files into /var/www/html/darkhunter/rv (no star data).
#
# Usage:
#   bash scripts/setup_website.sh
#   WEB_BASE=/var/www/html/darkhunter WEB_ROOT=/var/www/html/darkhunter/rv bash scripts/setup_website.sh
#
# After setup, seed tables/ (once):
#   bash scripts/bootstrap_website_tables.sh
#
# Populate stars/ + refresh M2 columns:
#   bash scripts/populate_website.sh

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
WEB_BASE="${WEB_BASE:-/var/www/html/darkhunter}"
WEB_ROOT="${WEB_ROOT:-$WEB_BASE/rv}"
SRC="$REPO/website/rv"

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] missing template dir: $SRC" >&2
  exit 2
fi

echo "web_base=$WEB_BASE"
echo "web_root=$WEB_ROOT (site document root)"
echo "source=$SRC"

gen_sample_tags_js() {
  local src_json="$SRC/tables/sample_tags.json"
  local out_js="$WEB_ROOT/sample_tags_data.js"
  if [[ ! -f "$src_json" ]]; then
    echo "[WARN] missing $src_json — sample filter counts may stay at 0" >&2
    return 0
  fi
  run_cmd "$PY" - "$src_json" "$out_js" <<'PY'
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(src.read_text(encoding="utf-8"))
dst.write_text("window.__SAMPLE_TAG_DATA__ = " + json.dumps(data, indent=2) + ";\n", encoding="utf-8")
print(f"wrote {dst}")
PY
}

PY="${PY:-python3}"
if [[ -x /home/marley/anaconda2/envs/gaia-env/bin/python ]]; then
  PY=/home/marley/anaconda2/envs/gaia-env/bin/python
fi

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

run_cmd mkdir -p "$WEB_BASE" "$WEB_ROOT/tables" "$WEB_ROOT/stars"

# Static assets only; never delete existing tables/ or stars/.
run_cmd rsync -a \
  --exclude 'tables/' \
  --exclude 'stars/' \
  "$SRC/" "$WEB_ROOT/"

gen_sample_tags_js

if [[ -f "$SRC/tables/sample_tags.json" ]]; then
  run_cmd cp "$SRC/tables/sample_tags.json" "$WEB_ROOT/tables/"
fi

if [[ ! -f "$WEB_ROOT/tables/data.csv" ]]; then
  cat <<EOF
[WARN] $WEB_ROOT/tables/data.csv is missing.
       Run: bash scripts/bootstrap_website_tables.sh
       (copies tables from the legacy site, or set LEGACY_WEB_ROOT).
EOF
fi

cat <<EOF

Installed static site to: $WEB_ROOT
Open (after Apache maps this path): .../darkhunter/rv/index.html

Optional base-dir note (no index.html at $WEB_BASE):
  echo 'RV explorer: <a href="rv/">rv/</a>' > $WEB_BASE/README.html

EOF
