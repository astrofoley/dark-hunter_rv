#!/usr/bin/env bash
# Copy tables/*.csv from the legacy website into the new site tree.
#
# Usage:
#   bash scripts/bootstrap_website_tables.sh
#   LEGACY_WEB_ROOT=/var/www/html/ktaggart/rv_website_v1 bash scripts/bootstrap_website_tables.sh

set -euo pipefail

WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
LEGACY_WEB_ROOT="${LEGACY_WEB_ROOT:-/var/www/html/ktaggart/rv_website_v1}"
TABLES_DIR="$WEB_ROOT/tables"
LEGACY_TABLES="$LEGACY_WEB_ROOT/tables"

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

run_cmd mkdir -p "$TABLES_DIR"

if [[ ! -d "$LEGACY_TABLES" ]]; then
  echo "[ERROR] legacy tables dir not found: $LEGACY_TABLES" >&2
  echo "Set LEGACY_WEB_ROOT to the old rv_website_v1 directory." >&2
  exit 2
fi

echo "copying $LEGACY_TABLES -> $TABLES_DIR"
run_cmd rsync -a "$LEGACY_TABLES/" "$TABLES_DIR/"

if [[ ! -f "$TABLES_DIR/data.csv" ]]; then
  echo "[ERROR] data.csv still missing after bootstrap" >&2
  exit 2
fi

echo "OK: $(wc -l < "$TABLES_DIR/data.csv") lines in data.csv"

# Add new mass columns to header if absent (batch table updater expects them).
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
if [[ -x /home/marley/anaconda2/envs/gaia-env/bin/python ]]; then
  PY=/home/marley/anaconda2/envs/gaia-env/bin/python
fi

export DATA_CSV="$TABLES_DIR/data.csv"
run_cmd "$PY" - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["DATA_CSV"])
rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
if not rows:
    raise SystemExit("empty data.csv")
hdr = rows[0]
for col in (
    "M2sin i (Msun)",
    "(M2sin i)/(sin i) (Msun)",
    "INCLINATION (deg)",
    "G (mag)",
    "GAIA DATA",
    "DAYS SINCE LAST APF",
    "NEXT RV EVENT (DATE)",
):
    if col not in hdr:
        hdr.append(col)
        for r in rows[1:]:
            while len(r) < len(hdr):
                r.append("")
        print(f"added column: {col}")
with path.open("w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows(rows)
print("data.csv columns:", len(hdr))
PY
