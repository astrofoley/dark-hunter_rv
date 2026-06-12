# List unique Gaia DR3 source ids under a reduction tree (sorted, any depth).

discover_gaia_star_ids() {
  local root="${1:?SPEC_ROOT required}"
  local -a ids=()
  local d base

  if [[ ! -d "$root" ]]; then
    return 0
  fi

  while IFS= read -r d; do
    base=$(basename "$d")
    ids+=("${base#Gaia_DR3_}")
  done < <(find "$root" -type d -name 'Gaia_DR3_*' 2>/dev/null | sort)

  while IFS= read -r base; do
    base=$(basename "$base")
    if [[ "$base" =~ ^Gaia_DR3_([0-9]+)_epoch_[0-9]+\.txt$ ]]; then
      ids+=("${BASH_REMATCH[1]}")
    fi
  done < <(find "$root" -type f -name 'Gaia_DR3_*_epoch_*.txt' ! -name '*_order_*' 2>/dev/null | sort -u)

  if [[ "${#ids[@]}" -eq 0 ]]; then
    return 0
  fi
  printf '%s\n' "${ids[@]}" | awk '!seen[$0]++'
}
