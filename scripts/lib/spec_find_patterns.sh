# Shared find(1) for APF spectrum files under a reduction tree.
# Gaia_DR3_*_epoch_<N>.txt are full-epoch reduced spectra; exclude *_order_* extracts
# (e.g. epoch_1_order_28.txt is Hβ-only). *_ap1.* / *.fits are APF products.

find_apf_spectra_print0() {
  local root="${1:?root directory required}"
  find "$root" -type f \( \
    \( -name 'Gaia_DR3_*_epoch_*.txt' ! -name '*_order_*' \) -o \
    -name '*_ap1.flm' -o \
    -name '*_ap1.txt' -o \
    -name '*.fits' \
  \) -print0 2>/dev/null
}
