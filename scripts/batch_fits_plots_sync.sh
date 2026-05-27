#!/usr/bin/env bash
# Website integration batch:
# 1) Keplerian fits
# 2) Regenerate APF plots
# 3) Stage files into website contract:
#    stars/Gaia_DR3_<id>/Gaia/{<id>_summary.txt,Plots/*,RV_Fit/*}
# 4) Update tables/data.csv M2 column from fit JSON
# 5) Produce QC reports
# 6) rsync staged content to web root
#
# Usage:
#   bash scripts/batch_fits_plots_sync.sh
#   DRY_RUN=1 bash scripts/batch_fits_plots_sync.sh
#   STAR_ID=1702370142434513152 bash scripts/batch_fits_plots_sync.sh   # canary

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
WEB_ROOT="${WEB_ROOT:-/data2/gaia_stars/dark-hunter_rv-kirsty}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
LOG="${LOG:-$REPO/batch_fits_plots_sync.log}"
MIN_POINTS="${MIN_POINTS:-7}"
FIT_FORCE="${FIT_FORCE:-1}"
QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAR_ID="${STAR_ID:-}" # optional canary mode

WEBSITE_STARS_DIR="$WEB_ROOT/stars"
WEBSITE_TABLES_DIR="$WEB_ROOT/tables"
DATA_CSV="$WEBSITE_TABLES_DIR/data.csv"
QC_DIR="$REPORTS_DIR/qc"
MISSING_ASSETS_CSV="$QC_DIR/missing_assets.csv"
STAGED_SUMMARY="$QC_DIR/staging_summary.json"

RSYNC_OPTS=(-rlptD --delete --omit-dir-times)

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

require_path() {
  local p="$1"
  local t="$2"
  if [[ "$t" == "dir" && ! -d "$p" ]]; then
    echo "[ERROR] missing directory: $p" >&2
    exit 2
  fi
  if [[ "$t" == "file" && ! -f "$p" ]]; then
    echo "[ERROR] missing file: $p" >&2
    exit 2
  fi
}

cd "$REPO"
export PYTHONPATH="$REPO"
export DARKHUNTER_OUTPUT_DIR="$OUT"

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) batch start (pid $$) ==="
echo "repo=$REPO out=$OUT web_root=$WEB_ROOT reports=$REPORTS_DIR dry_run=$DRY_RUN star_id=${STAR_ID:-ALL}"

require_path "$OUT" dir
require_path "$SPEC_ROOT" dir
require_path "$WEB_ROOT" dir
require_path "$WEBSITE_TABLES_DIR" dir
require_path "$DATA_CSV" file

run_cmd mkdir -p "$REPORTS_DIR" "$QC_DIR"
if [[ "$DRY_RUN" != "1" ]]; then
  echo "gaia_id,reason,path" > "$MISSING_ASSETS_CSV"
fi

if [[ -n "$STAR_ID" ]]; then
  SUMMARY_GLOB="Gaia_DR3_${STAR_ID}_summary.txt"
else
  SUMMARY_GLOB="Gaia_DR3_*_summary.txt"
fi

mapfile -t SUMMARY_FILES < <(find "$OUT" -maxdepth 1 -type f -name "$SUMMARY_GLOB" | sort)
if [[ "${#SUMMARY_FILES[@]}" -eq 0 ]]; then
  echo "[ERROR] No summary files matched: $OUT/$SUMMARY_GLOB" >&2
  exit 2
fi

echo "=== Epoch counts (PIPELINE RESULTS rows per summary) ==="
n_ge_min=0
for summ in "${SUMMARY_FILES[@]}"; do
  n=$(awk '/^\[PIPELINE RESULTS\]/{s=1;next} s && $0 !~ /^#/ && NF>=5 {c++} END{print c+0}' "$summ")
  if [[ "$n" -ge "$MIN_POINTS" ]]; then
    n_ge_min=$((n_ge_min + 1))
  fi
done
echo "summaries=${#SUMMARY_FILES[@]} with_epochs>=${MIN_POINTS}: $n_ge_min"

echo "=== Keplerian fits ==="
fit_args=(
  fit_apf_rv_keplerian.py
  --all
  --output-dir "$OUT"
  --use-gaia-nss
  --min-points "$MIN_POINTS"
  --reports-dir "$REPORTS_DIR"
)
if [[ "$FIT_FORCE" == "1" ]]; then
  fit_args+=(--force)
fi
if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
  fit_args+=(--query-gaia-online)
fi
if [[ -n "$STAR_ID" ]]; then
  fit_args=(fit_apf_rv_keplerian.py --summary "$OUT/Gaia_DR3_${STAR_ID}_summary.txt" --output-dir "$OUT" --use-gaia-nss --min-points "$MIN_POINTS" --reports-dir "$REPORTS_DIR")
  if [[ "$FIT_FORCE" == "1" ]]; then
    fit_args+=(--force)
  fi
  if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
    fit_args+=(--query-gaia-online)
  fi
fi
run_cmd "$PY" "${fit_args[@]}"

echo "=== Pipeline plots (per-star output subdir) ==="
n_specs=0
for summ in "${SUMMARY_FILES[@]}"; do
  gid=$(basename "$summ" _summary.txt | sed 's/^Gaia_DR3_//')
  plot_dir="$OUT/Gaia_DR3_${gid}"
  run_cmd mkdir -p "$plot_dir"
  export DARKHUNTER_PLOT_DIR="$plot_dir"
  while IFS= read -r bn; do
    [[ -n "$bn" ]] || continue
    spec=$(find "$SPEC_ROOT" -type f -name "$bn" -print -quit 2>/dev/null || true)
    if [[ -z "$spec" ]]; then
      echo "[WARN] spectrum not found: $bn (Gaia_DR3_${gid})"
      if [[ "$DRY_RUN" != "1" ]]; then
        echo "${gid},missing_spectrum,${bn}" >> "$MISSING_ASSETS_CSV"
      fi
      continue
    fi
    n_specs=$((n_specs + 1))
    run_cmd "$PY" -m darkhunter_rv.pipeline "$spec" \
      --instrument APF \
      --update \
      --plots \
      --plots-focus \
      --plots-only
  done < <(
    awk '/^\[PIPELINE RESULTS\]/{show=1; next}
         show && $0 !~ /^#/ && NF >= 5 { print $1 }' "$summ"
  )
done
echo "plot_pass: summaries=${#SUMMARY_FILES[@]} pipeline_invocations=$n_specs"

echo "=== Stage website files into stars/Gaia_DR3_<id>/Gaia/... ==="
staged_stars=0
staged_fit_png=0
staged_fit_json=0
staged_plots=0
for summ in "${SUMMARY_FILES[@]}"; do
  gid=$(basename "$summ" _summary.txt | sed 's/^Gaia_DR3_//')
  star_root="$WEBSITE_STARS_DIR/Gaia_DR3_${gid}/Gaia"
  star_plots="$star_root/Plots"
  star_fit="$star_root/RV_Fit"
  run_cmd mkdir -p "$star_plots" "$star_fit"

  run_cmd cp "$summ" "$star_root/${gid}_summary.txt"
  staged_stars=$((staged_stars + 1))

  fit_png="$REPORTS_DIR/${gid}_keplerian_fit.png"
  fit_json="$REPORTS_DIR/${gid}_keplerian_fit.json"
  if [[ -f "$fit_png" ]]; then
    run_cmd cp "$fit_png" "$star_fit/${gid}_keplerian_fit.png"
    staged_fit_png=$((staged_fit_png + 1))
  elif [[ "$DRY_RUN" != "1" ]]; then
    echo "${gid},missing_fit_png,${fit_png}" >> "$MISSING_ASSETS_CSV"
  fi
  if [[ -f "$fit_json" ]]; then
    run_cmd cp "$fit_json" "$star_fit/${gid}_keplerian_fit.json"
    staged_fit_json=$((staged_fit_json + 1))
  elif [[ "$DRY_RUN" != "1" ]]; then
    echo "${gid},missing_fit_json,${fit_json}" >> "$MISSING_ASSETS_CSV"
  fi

  src_plot_dir="$OUT/Gaia_DR3_${gid}"
  if [[ -d "$src_plot_dir" ]]; then
    mapfile -t pngs < <(find "$src_plot_dir" -maxdepth 1 -type f -name '*.png' | sort)
    if [[ "${#pngs[@]}" -gt 0 ]]; then
      for p in "${pngs[@]}"; do
        run_cmd cp "$p" "$star_plots/$(basename "$p")"
        staged_plots=$((staged_plots + 1))
      done
    elif [[ "$DRY_RUN" != "1" ]]; then
      echo "${gid},missing_plot_pngs,${src_plot_dir}" >> "$MISSING_ASSETS_CSV"
    fi
  elif [[ "$DRY_RUN" != "1" ]]; then
    echo "${gid},missing_plot_dir,${src_plot_dir}" >> "$MISSING_ASSETS_CSV"
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  export REPORTS_DIR DATA_CSV MISSING_ASSETS_CSV STAGED_SUMMARY
  "$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORTS_DIR"])
data_csv = Path(os.environ["DATA_CSV"])
missing_csv = Path(os.environ["MISSING_ASSETS_CSV"])
summary_json = Path(os.environ["STAGED_SUMMARY"])

reports = {}
for p in sorted(report_dir.glob("*_keplerian_fit.json")):
    sid = p.stem.replace("_keplerian_fit", "")
    try:
        reports[sid] = json.loads(p.read_text())
    except Exception:
        continue

rows = []
with data_csv.open(newline="", encoding="utf-8") as fh:
    reader = csv.reader(fh)
    rows = list(reader)

if not rows:
    raise SystemExit("tables/data.csv is empty")

hdr = rows[0]
try:
    gaia_i = hdr.index("GAIA NAME")
    m2_i = hdr.index("M2 (Msun)")
except ValueError as exc:
    raise SystemExit(f"Required data.csv column missing: {exc}")

# Add requested derived-mass columns when missing.
col_m2sini = "M2sin i (Msun)"
col_m2over = "(M2sin i)/(sin i) (Msun)"
if col_m2sini not in hdr:
    hdr.append(col_m2sini)
if col_m2over not in hdr:
    hdr.append(col_m2over)
for r in rows[1:]:
    while len(r) < len(hdr):
        r.append("")
m2sini_i = hdr.index(col_m2sini)
m2over_i = hdr.index(col_m2over)

updated = 0
for r in rows[1:]:
    if not r:
        continue
    gaia = (r[gaia_i] if gaia_i < len(r) else "").strip()
    sid = ""
    import re
    m = re.search(r"(\d{8,})", gaia)
    if m:
        sid = m.group(1)
    if not sid or sid not in reports:
        continue
    rep = reports[sid]
    m2i = rep.get("m2_given_inclination_msun")
    m2s = rep.get("m2sini_msun")
    m2 = m2i if isinstance(m2i, (int, float)) else m2s
    if isinstance(m2, (int, float)):
        while len(r) <= m2_i:
            r.append("")
        r[m2_i] = f"{float(m2):.5f}"
        updated += 1
    if isinstance(m2s, (int, float)):
        r[m2sini_i] = f"{float(m2s):.5f}"
    if isinstance(m2i, (int, float)):
        r[m2over_i] = f"{float(m2i):.5f}"

with data_csv.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerows(rows)

missing = 0
if missing_csv.exists():
    with missing_csv.open(encoding="utf-8") as fh:
        missing = max(0, sum(1 for _ in fh) - 1)

summary = {
    "reports_found": len(reports),
    "table_rows_updated_m2": updated,
    "missing_assets_records": missing,
}
summary_json.write_text(json.dumps(summary, indent=2))
print(f"table_update: m2_rows_updated={updated}, reports_found={len(reports)}, missing_assets={missing}")
PY
fi

echo "=== rsync to web tree ==="
run_cmd rsync "${RSYNC_OPTS[@]}" "$OUT/" "$WEB_ROOT/output/"
run_cmd rsync "${RSYNC_OPTS[@]}" "$REPORTS_DIR/" "$WEB_ROOT/rv_fit_reports/"

echo "staged: stars=$staged_stars fit_png=$staged_fit_png fit_json=$staged_fit_json plots=$staged_plots"
echo "qc_reports: $MISSING_ASSETS_CSV $STAGED_SUMMARY"
echo "=== $(date -Is) batch done ==="
