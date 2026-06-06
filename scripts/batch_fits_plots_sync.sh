#!/usr/bin/env bash
# Website integration batch:
# 1) Keplerian fits
# 2) Generate APF RV plots from summaries (no pipeline rerun)
# 3) Stage files into website contract:
#    stars/Gaia_DR3_<id>/Gaia/{<id>_summary.txt,Plots/*,RV_Fit/*}
# 4) Update tables/data.csv M2 / M2sin i / M2 at i from fit JSON
# 5) Produce QC reports
# 6) rsync staged content to web root
#
# Usage:
#   bash scripts/batch_fits_plots_sync.sh
#   WEB_ROOT=/var/www/html/darkhunter/rv RUN_FITS=0 MIN_POINTS=5 bash scripts/batch_fits_plots_sync.sh
#   DRY_RUN=1 bash scripts/batch_fits_plots_sync.sh
#   STAR_ID=1702370142434513152 bash scripts/batch_fits_plots_sync.sh   # canary
#
# One-time site setup (static HTML/JS/CSS): bash scripts/setup_website.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib/website_plot_sync.sh
source "$SCRIPT_DIR/lib/website_plot_sync.sh"

REPO="${REPO:-/data2/darkhunter/dark-hunter_rv}"
OUT="${OUT:-$REPO/output}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
REPORTS_DIR="${REPORTS_DIR:-$REPO/rv_fit_reports}"
LOG="${LOG:-$REPO/batch_fits_plots_sync.log}"
MIN_POINTS="${MIN_POINTS:-7}"
RUN_FITS="${RUN_FITS:-1}"
RUN_RV_PLOTS="${RUN_RV_PLOTS:-1}"
RUN_HBETA_PLOTS="${RUN_HBETA_PLOTS:-1}"
FIT_FORCE="${FIT_FORCE:-1}"
QUERY_GAIA_ONLINE="${QUERY_GAIA_ONLINE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STAR_ID="${STAR_ID:-}" # optional canary mode
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"

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

echo "=== APF observability windows cache ==="
obs_args=(
  scripts/build_apf_observability_cache.py
  --data-csv "$DATA_CSV"
  --output-dir "$OUT"
  --cache "$REPORTS_DIR/observability_windows_cache.json"
)
if [[ -n "$STAR_ID" ]]; then
  obs_args+=(--gaia-id "$STAR_ID")
fi
run_cmd "$PY" "${obs_args[@]}"

if [[ "$RUN_FITS" == "1" ]]; then
  echo "=== Keplerian fits ==="
  fit_args=(
    fit_apf_rv_keplerian.py
    --all
    --output-dir "$OUT"
    --use-gaia-nss
    --min-points "$MIN_POINTS"
    --reports-dir "$REPORTS_DIR"
    --data-csv "$DATA_CSV"
  )
  if [[ "$FIT_FORCE" == "1" ]]; then
    fit_args+=(--force)
  fi
  if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
    fit_args+=(--query-gaia-online)
  fi
  if [[ -n "$STAR_ID" ]]; then
    fit_args=(fit_apf_rv_keplerian.py --summary "$OUT/Gaia_DR3_${STAR_ID}_summary.txt" --output-dir "$OUT" --use-gaia-nss --min-points "$MIN_POINTS" --reports-dir "$REPORTS_DIR" --data-csv "$DATA_CSV")
    if [[ "$FIT_FORCE" == "1" ]]; then
      fit_args+=(--force)
    fi
    if [[ "$QUERY_GAIA_ONLINE" == "1" ]]; then
      fit_args+=(--query-gaia-online)
    fi
  fi
  run_cmd "$PY" "${fit_args[@]}"
else
  echo "=== Keplerian fits (skipped: RUN_FITS=0) ==="
fi

if [[ "$RUN_RV_PLOTS" == "1" && "$RUN_FITS" != "1" ]]; then
  echo "=== Summary-based RV data plots (no fits) ==="
  plot_args=(
    scripts/plot_rv_from_summaries.py
    --summary-dir "$OUT"
    --plots-root "$OUT"
  )
  if [[ -n "$STAR_ID" ]]; then
    plot_args+=(--star-id "$STAR_ID")
  fi
  run_cmd "$PY" "${plot_args[@]}"
fi

if [[ "$RUN_HBETA_PLOTS" == "1" ]]; then
  echo "=== Hβ epoch stack plots for website ==="
  hbeta_args=(
    scripts/build_hbeta_website_plots.py
    --summary-dir "$OUT"
    --plots-root "$OUT"
    --spec-root "${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
  )
  if [[ -n "$STAR_ID" ]]; then
    hbeta_args+=(--star-id "$STAR_ID")
  fi
  run_cmd "$PY" "${hbeta_args[@]}"
fi

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
    n_plots=$(website_stage_gaia_plots "$gid" "$src_plot_dir" "$star_plots")
    staged_plots=$((staged_plots + n_plots))
    if [[ "$n_plots" -eq 0 && "$DRY_RUN" != "1" ]]; then
      echo "${gid},missing_plot_pngs,${src_plot_dir}" >> "$MISSING_ASSETS_CSV"
    fi
  elif [[ "$DRY_RUN" != "1" ]]; then
    echo "${gid},missing_plot_dir,${src_plot_dir}" >> "$MISSING_ASSETS_CSV"
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  export REPORTS_DIR DATA_CSV MISSING_ASSETS_CSV STAGED_SUMMARY OUT
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

from scripts.update_website_table_columns import load_gaia_nss_cache, update_table_columns

report_dir = Path(os.environ["REPORTS_DIR"])
out_dir = Path(os.environ.get("OUT", report_dir.parent / "output"))
data_csv = Path(os.environ["DATA_CSV"])
missing_csv = Path(os.environ["MISSING_ASSETS_CSV"])
summary_json = Path(os.environ["STAGED_SUMMARY"])

gaia_cache = load_gaia_nss_cache(report_dir)
stats = update_table_columns(
    data_csv,
    out_dir=out_dir,
    reports_dir=report_dir,
    gaia_cache=gaia_cache,
)

missing = 0
if missing_csv.exists():
    with missing_csv.open(encoding="utf-8") as fh:
        missing = max(0, sum(1 for _ in fh) - 1)

summary = {
    "reports_found": stats["reports_loaded"],
    "table_rows_updated_m2": stats["m2_filled"],
    "inclination_filled": stats["inclination_filled"],
    "m2_at_i_filled": stats["m2_at_i_filled"],
    "missing_assets_records": missing,
}
summary_json.write_text(json.dumps(summary, indent=2))
print(
    f"table_update: m2_rows={stats['m2_filled']}, incl={stats['inclination_filled']}, "
    f"m2_at_i={stats['m2_at_i_filled']}, reports={stats['reports_loaded']}, missing_assets={missing}"
)
PY
fi

echo "=== rsync to web tree ==="
run_cmd rsync "${RSYNC_OPTS[@]}" "$OUT/" "$WEB_ROOT/output/"
run_cmd rsync "${RSYNC_OPTS[@]}" "$REPORTS_DIR/" "$WEB_ROOT/rv_fit_reports/"

echo "staged: stars=$staged_stars fit_png=$staged_fit_png fit_json=$staged_fit_json plots=$staged_plots"
echo "qc_reports: $MISSING_ASSETS_CSV $STAGED_SUMMARY"
echo "=== $(date -Is) batch done ==="
