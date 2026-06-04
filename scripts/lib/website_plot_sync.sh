# Copy only website-contract Gaia plot PNGs; remove legacy pipeline Hβ diagnostics.
# Requires: run_cmd (optional DRY_RUN).

website_stage_gaia_plots() {
  local gid="${1:?gaia id}"
  local src_dir="${2:?source plot dir}"
  local dest_dir="${3:?dest Plots dir}"
  local n=0

  run_cmd mkdir -p "$dest_dir"
  local name
  for name in \
    "Gaia_DR3_${gid}_28_hbeta.png" \
    "Gaia_DR3_${gid}_rv_plot.png" \
    "Gaia_DR3_${gid}_keplerian_residuals.png"
  do
    if [[ -f "$src_dir/$name" ]]; then
      run_cmd cp "$src_dir/$name" "$dest_dir/$name"
      n=$((n + 1))
    fi
  done
  website_prune_legacy_gaia_plots "$dest_dir"
  echo "$n"
}

website_prune_legacy_gaia_plots() {
  local plot_dir="${1:?Plots directory}"
  [[ -d "$plot_dir" ]] || return 0
  local f
  while IFS= read -r -d '' f; do
    run_cmd rm -f "$f"
  done < <(
    find "$plot_dir" -maxdepth 1 -type f \( \
      -name '*_h_beta_rv.png' -o \
      -name '*_h_beta_order*.png' -o \
      -name '*_h_beta_three*.png' \
    \) -print0 2>/dev/null
  )
}
