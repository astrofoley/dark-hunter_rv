"""Paths and constants; override with env vars for portability."""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent

MASK_DIRECTORY = Path(os.environ.get("DARKHUNTER_MASK_DIR", REPO_ROOT / "stellar_masks"))
OUTPUT_DIR = Path(os.environ.get("DARKHUNTER_OUTPUT_DIR", REPO_ROOT / "output"))
PLOT_DIR = Path(os.environ.get("DARKHUNTER_PLOT_DIR", REPO_ROOT / "plots"))

# Production mask-CCF chunk layout (chunk-campaign winner: equal 8-way pixel splits per order).
# Override with DARKHUNTER_CHUNK_LAYOUT; CLI --chunk-layout takes precedence over env/default.
_chunk_layout_env = os.environ.get("DARKHUNTER_CHUNK_LAYOUT")
if _chunk_layout_env:
    DEFAULT_CHUNK_LAYOUT: Path | None = Path(_chunk_layout_env)
else:
    _prod_layout = REPO_ROOT / "calibration" / "chunk_layouts" / "subchunks_8.yaml"
    DEFAULT_CHUNK_LAYOUT = _prod_layout if _prod_layout.is_file() else None

# Per-chunk debias table for APF (bias_dv, bias_err_stat, bias_rms_stat → b0, b1, b2).
BIAS_STATISTICS_FILE = Path(
    os.environ.get("DARKHUNTER_BIAS_FILE", REPO_ROOT / "bias_statistics.txt")
)

# HiRes grids: directory containing WAVE_PHOENIX-ACES-AGSS-COND-2011.fits (or WAVE_PHOENIX*.fits) and
# PHOENIX-ACES-AGSS-COND-2011/ (metallicity subfolders with lte*.fits flux files).
# Override with DARKHUNTER_PHOENIX_DIR. If unset, use ~/phoenix/HiResFITS when that tree exists
# (e.g. /Users/rfoley/phoenix/HiResFITS), else REPO_ROOT / "phoenix_models".
def _auto_hires_phoenix_dir() -> Path | None:
    p = Path.home() / "phoenix" / "HiResFITS"
    if not p.is_dir():
        return None
    wave = p / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    if not wave.is_file() and not list(p.glob("WAVE_PHOENIX*.fits")):
        return None
    if not (p / "PHOENIX-ACES-AGSS-COND-2011").is_dir():
        return None
    return p


_phoenix_env = os.environ.get("DARKHUNTER_PHOENIX_DIR")
if _phoenix_env:
    PHOENIX_BASE_DIR = Path(_phoenix_env)
else:
    _auto = _auto_hires_phoenix_dir()
    PHOENIX_BASE_DIR = _auto if _auto is not None else REPO_ROOT / "phoenix_models"

C_KMS = 299_792.458
# Used for CLI default and PHOENIX bank when Gaia ``teff_gspphot`` is missing; diagnostics CSV uses NaN
# for Teff in that case (see ``pipeline._attach_diagnostics_teff``) so plots are not stacked at 5800 K.
DEFAULT_TEFF = 5800.0
HOT_STAR_TEFF_THRESHOLD = 6500.0

# ``estimate_broadening`` returns ``(None, {"vsini_proxy_rejected_kms": x})`` when the inferred width
# exceeds VSINI_PROXY_MAX_KMS; the pipeline then uses VSINI_PROXY_REJECTED_GRID_KMS for the PHOENIX bank.
# Non-finite or missing estimates use VSINI_PROXY_NONFINITE_GRID_KMS (not the narrow FALLBACK).
VSINI_PROXY_MIN_KMS = 0.5
VSINI_PROXY_MAX_KMS = 200.0
# Used when ``estimate_broadening`` is non-finite (quick-normalize vs template mismatch, etc.).
VSINI_PROXY_FALLBACK_KMS = 10.0
# When the raw broadening estimate exceeds VSINI_PROXY_MAX_KMS we do not clamp to max (that would
# still build a huge rotational grid). Use a **moderately wide** proxy so fast rotators stay in play.
VSINI_PROXY_REJECTED_GRID_KMS = 75.0
# Non-finite broadening: wider than FALLBACK so unknown-width stars are not forced to a 10 km/s bank.
VSINI_PROXY_NONFINITE_GRID_KMS = 45.0
# Upper end of the linspace used in ``templates._broadening_velocity_grid(..., wide=True)``.
VSINI_WIDE_GRID_CAP_KMS = 160.0

# Spline continuum: exclude pixels near Balmer/strong lines from spline anchors (Å). Cool stars use a
# moderate width so line cores/wings do not pull the continuum; hot stars use a wider exclusion.
COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH = 55.0
HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH = 78.0

# Spline continuum: rolling upper-envelope estimate (percentile_filter) so noisy orders track the
# continuum **top** instead of a mix of lines+noise. Second pass smooths the envelope slightly.
# Floor ties final spline to the envelope so fits cannot run through the spectrum mid-level.
CONTINUUM_ENVELOPE_PERCENTILE = 88
CONTINUUM_ENVELOPE_PERCENTILE_REFINE = 93
CONTINUUM_ENVELOPE_MIN_WINDOW = 7
CONTINUUM_ENVELOPE_MAX_WINDOW = 65
CONTINUUM_ENVELOPE_FLOOR_FRAC = 0.72
# Noisy echelle orders: MAD/median of flux (pre-envelope) above this triggers gentler envelope tuning.
CONTINUUM_NOISY_MAD_TO_MEDIAN = 0.28
CONTINUUM_NOISY_WINDOW_EXTRA = 18
CONTINUUM_NOISY_PERCENTILE_ADD = 2
CONTINUUM_NOISY_FLOOR_FRAC = 0.80

# validation reports: default max exposure-level σ (km/s) for method comparison scatter plots.
COMPARISON_REPORT_MAX_RV_ERR_KMS = 2.5
# Fixed equal-width Teff bins for ``high_err_fraction_vs_teff`` (not quantiles of the sample).
COMPARISON_REPORT_TEFF_BIN_LO_K = 3500.0
COMPARISON_REPORT_TEFF_BIN_HI_K = 9000.0

# Method applicability regions for residual-vs-Teff plots (``rv_method_diagnostics_report``).
# Stellar mask: Teff < MASK_COOL_K OR (Teff < MASK_WARM_K AND log10(median mask CCF peak S/N) > threshold).
METHOD_REGION_MASK_COOL_TEFF_K = 5200.0
METHOD_REGION_MASK_WARM_TEFF_K = 7000.0
METHOD_REGION_LOG10_SNR_MIN = 0.65
# Strong lines (Hβ): Teff > STRONG_LINES_MIN_TEFF_K AND log10(S/N) > same threshold.
METHOD_REGION_STRONG_LINES_MIN_TEFF_K = 5500.0

# Adopted exposure RV cascade (mask → template → strong): prefer first method that is
# region-applicable, valid, and has σ ≤ this (km/s). If none meet σ, use first applicable valid
# method in order (report its σ, possibly large). See ``method_evaluation.recommend_adopted_rv``.
ADOPTED_CASCADE_MAX_SIGMA_KMS = float(
    os.environ.get(
        "DARKHUNTER_ADOPTED_MAX_SIGMA_KMS", str(COMPARISON_REPORT_MAX_RV_ERR_KMS)
    )
)

# Optional global method RV offsets file (mask = truth); template/strong shifted before adoption.
# Override with env DARKHUNTER_METHOD_OFFSETS_FILE. If unset, default path is only used when the file exists.
_METHOD_OFFSETS_ENV = os.environ.get("DARKHUNTER_METHOD_OFFSETS_FILE", "").strip()
METHOD_OFFSETS_FILE: Path | None
if _METHOD_OFFSETS_ENV:
    METHOD_OFFSETS_FILE = Path(_METHOD_OFFSETS_ENV)
else:
    _mo_default = REPO_ROOT / "method_rv_offsets.txt"
    METHOD_OFFSETS_FILE = _mo_default if _mo_default.is_file() else None

# Per-chunk template FFT: reject |RV| above cap before stacking (reduces spurious correlation peaks).
# Hot-star path keeps a wide cap; cool stars use a tighter default (binaries with |γ|≳100 km/s may need review).
TEMPLATE_FFT_MAX_ABS_RV_KMS_HOT = 400.0
TEMPLATE_FFT_MAX_ABS_RV_KMS_COOL = 100.0

# Cool stars only: when mask CCF passes chunk QC, restrict FFT lag search to mask RV ± this width
# (km/s), in the same frame as FFT raw output (add per-chunk bary offset b0 to diagnostics rv_m).
# Prevents spurious ±1000 km/s correlation peaks from beating the correct template+velocity.
TEMPLATE_FFT_MASK_SEED_HALF_WIDTH_KMS = 175.0

# Template FFT: optional coarse pass in (Teff, log g, [M/H]) — one vsini trial per atmosphere — then
# full vsini grid only for the top few atmospheres by correlation peak. Skipped automatically when it
# would not reduce FFT work vs scanning the whole bank.
FFT_COARSE_TOP_K = 8
FFT_TWO_PHASE_MIN_TEMPLATES = 14

# Cool-star template FFT: how the winning RV is chosen after the (possibly two-phase) bank is fixed.
# ``per_template_max`` picks the template whose *own* CCF has the highest peak (one outlier template
# can dominate). ``aggregate_median`` takes, at each lag, the median correlation across templates and
# peaks that curve — robust to a single spurious high-correlation template/velocity (mask-independent).
FFT_TEMPLATE_PEAK_PICK_COOL = "aggregate_median"
FFT_TEMPLATE_PEAK_PICK_HOT = "per_template_max"

# Exposure-level PHOENIX key vote runs before per-chunk mask CCF. Chooses the key with **lowest mean
# RMS** absorption residual on the FFT log grid; per chunk/template the RV is the CCF argmax in a
# window around FFT_EXPOSURE_VOTE_RV_SEED_KMS. Binaries with large |γ| may need a wider half-width or
# ``--no-fixed-exposure-template``.
FFT_EXPOSURE_VOTE_RV_SEED_KMS = 0.0
FFT_EXPOSURE_VOTE_RV_HALF_WIDTH_KMS_COOL = 280.0
FFT_EXPOSURE_VOTE_RV_HALF_WIDTH_KMS_HOT = 500.0

# Exposure-level template FFT stack: ignore failed chunks (qc_pass); need at least this many good chunks.
MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK = 3

# Exposure-level mask CCF stack (diagnostics + cool-star primary): need at least this many QC-good chunks.
# Below this, return NaN from inverse-variance aggregation (same idea as template_fft).
MIN_MASK_CCF_CHUNKS_FOR_STACK = 5

# Mask CCF Gaussian: reject fits narrower than this (noise spikes); widen lag if peak sits near grid edge.
MASK_CCF_DEFAULT_MAX_LAG = 420
MASK_CCF_MIN_GAUSS_SIGMA_KMS = 2.5
MASK_CCF_EDGE_REFIT_FRACTION = 0.08
MASK_CCF_EDGE_REFIT_LAG_MULT = 1.75
MASK_CCF_MAX_LAG_CAP = 700

# Per-chunk primary stack (used for plots / legacy ``--no-run-all-methods``): cool stars use mask CCF
# chunks; hot stars (Teff > HOT_STAR_TEFF_THRESHOLD) use template FFT chunks. With **default**
# multi-method diagnostics, exposure-level **adopted** RV follows ``method_evaluation.recommend_adopted_rv``
# (mask → template → strong_lines cascade with METHOD_REGION_* and ADOPTED_CASCADE_MAX_SIGMA_KMS).
# Legacy ``--no-run-all-methods``: adopted RV uses the hot/cool chunk stack + strong_lines centroid on hot
# stars; optional ``--hb-rv-fallback`` uses Hβ bundle ``rv_best_kms`` when the stack is unusable (and after
# cascade NaN when multi-method is on).
# Template FFT: by default a two-phase search in (Teff, log g, [M/H]) when the bank is large
# (``FFT_COARSE_TOP_K``, ``FFT_TWO_PHASE_MIN_TEMPLATES``); disable with ``--no-fft-two-phase``.
RV_METHOD_SELECTION_NOTES = (
    "Default: multi-method diagnostics (mask_ccf, template_fft, strong_lines Voigt+Lorentz Hβ). "
    "Adopted exposure RV: cascade mask → template → strong using METHOD_REGION_* and "
    "ADOPTED_CASCADE_MAX_SIGMA_KMS; optional method_rv_offsets.txt shifts template/strong vs mask truth. "
    "Legacy --no-run-all-methods: cool → mask chunk stack; hot → template chunks then strong_lines centroid; "
    "--hb-rv-fallback for Hβ bundle best RV when needed. "
    "Spline continuum: upper-envelope rolling percentile anchors + LSQ spline (not absorption_mask mid-level); "
    "excludes pixels near strong lines (cool/hot SPLINE_EXCLUDE_NEAR_LINES_WIDTH) and ISM Na D anchors. "
    "Cool template FFT chunks use TEMPLATE_FFT_MAX_ABS_RV_KMS_COOL (alias control); hot uses _HOT. "
    "Cool + mask QC pass: FFT lag search is limited to mask RV ± TEMPLATE_FFT_MASK_SEED_HALF_WIDTH_KMS. "
    "Template stack applies MAD outlier rejection when N chunks is large enough. "
    "Large PHOENIX banks: coarse atmosphere pass then full vsini on top-K; --no-fft-two-phase for full scan. "
    "Cool template FFT RV: median correlation across templates at each lag (FFT_TEMPLATE_PEAK_PICK_COOL) "
    "unless overridden; hot path uses per-template max. "
    "Exposure PHOENIX key vote: atmosphere choice uses narrowest v sin i per (Teff, log g, [M/H]) "
    "(templates.narrow_fft_subbank), then minimum mean RMS over the full vsini grid for that triple; "
    "per chunk RV = CCF peak in FFT_EXPOSURE_VOTE_RV_SEED_KMS ± half-width (before mask CCF). "
    "Default: one PHOENIX (Teff, log g, [M/H], vsini) key per exposure for all template_fft chunks "
    "(--no-fixed-exposure-template restores per-chunk template choice). "
    "vsini_proxy: above-max broadening → VSINI_PROXY_REJECTED_GRID_KMS; missing/invalid → "
    "VSINI_PROXY_NONFINITE_GRID_KMS; wide bank vsini linspace capped by VSINI_WIDE_GRID_CAP_KMS. "
    "Spline continuum: noisy orders (high MAD/median) use wider envelope window, higher percentiles, "
    "higher floor tie (CONTINUUM_NOISY_*). "
    "Mask CCF peak fit: offset + Gaussian (no linear velocity term). "
    "Telluric + Na D bands excluded from mask CCF sums, template FFT line flux, spline anchors (Na D), "
    "and telluric_fraction QC."
)
