# Stage one star into WEB_ROOT and refresh its data.csv row.
# Requires: REPO, WEB_ROOT, OUT, REPORTS_DIR, PY, WEBSITE_STARS_DIR, DATA_CSV, run_cmd (optional DRY_RUN).

# shellcheck source=scripts/lib/website_plot_sync.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/website_plot_sync.sh"

website_sync_one_star() {
  local gid="${1:?gaia id required}"
  local summ="$OUT/Gaia_DR3_${gid}_summary.txt"
  local star_root="$WEBSITE_STARS_DIR/Gaia_DR3_${gid}/Gaia"
  local star_plots="$star_root/Plots"
  local star_fit="$star_root/RV_Fit"

  if [[ ! -f "$summ" ]]; then
    echo "[WARN] no summary for Gaia_DR3_${gid}; skip website sync" >&2
    return 1
  fi

  run_cmd mkdir -p "$star_plots" "$star_fit"
  run_cmd cp "$summ" "$star_root/${gid}_summary.txt"

  local fit_png="$REPORTS_DIR/${gid}_keplerian_fit.png"
  local fit_json="$REPORTS_DIR/${gid}_keplerian_fit.json"
  if [[ -f "$fit_png" ]]; then
    run_cmd cp "$fit_png" "$star_fit/${gid}_keplerian_fit.png"
  elif [[ "${DRY_RUN:-0}" != "1" && -n "${MISSING_ASSETS_CSV:-}" ]]; then
    echo "${gid},missing_fit_png,${fit_png}" >> "$MISSING_ASSETS_CSV"
  fi
  if [[ -f "$fit_json" ]]; then
    run_cmd cp "$fit_json" "$star_fit/${gid}_keplerian_fit.json"
  elif [[ "${DRY_RUN:-0}" != "1" && -n "${MISSING_ASSETS_CSV:-}" ]]; then
    echo "${gid},missing_fit_json,${fit_json}" >> "$MISSING_ASSETS_CSV"
  fi

  local src_plot_dir="$OUT/Gaia_DR3_${gid}"
  if [[ -d "$src_plot_dir" ]]; then
    website_stage_gaia_plots "$gid" "$src_plot_dir" "$star_plots"
  elif [[ "${DRY_RUN:-0}" != "1" && -n "${MISSING_ASSETS_CSV:-}" ]]; then
    echo "${gid},missing_plot_dir,${src_plot_dir}" >> "$MISSING_ASSETS_CSV"
  fi

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    local csv_lock="${DATA_CSV}.lock"
    if command -v flock >/dev/null 2>&1; then
      run_cmd flock -w 600 "$csv_lock" "$PY" scripts/update_website_table_columns.py \
        --data-csv "$DATA_CSV" \
        --output-dir "$OUT" \
        --reports-dir "$REPORTS_DIR" \
        --gaia-id "$gid"
    else
      run_cmd "$PY" scripts/update_website_table_columns.py \
        --data-csv "$DATA_CSV" \
        --output-dir "$OUT" \
        --reports-dir "$REPORTS_DIR" \
        --gaia-id "$gid"
    fi
    if [[ -d "$WEBSITE_STARS_DIR/Gaia_DR3_${gid}" ]]; then
      run_cmd rsync -rlptD --omit-dir-times \
        "$WEBSITE_STARS_DIR/Gaia_DR3_${gid}/" \
        "$WEB_ROOT/stars/Gaia_DR3_${gid}/"
    fi
    if [[ -f "$REPORTS_DIR/${gid}_keplerian_fit.json" ]]; then
      run_cmd mkdir -p "$WEB_ROOT/rv_fit_reports"
      run_cmd cp "$REPORTS_DIR/${gid}_keplerian_fit.json" "$WEB_ROOT/rv_fit_reports/"
      [[ -f "$REPORTS_DIR/${gid}_keplerian_fit.png" ]] && \
        run_cmd cp "$REPORTS_DIR/${gid}_keplerian_fit.png" "$WEB_ROOT/rv_fit_reports/"
    fi
  fi
  return 0
}
