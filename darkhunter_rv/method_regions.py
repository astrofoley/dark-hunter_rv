"""Method applicability regions (Teff, log10 median mask CCF peak S/N).

Used by validation reports, method-offset calibration, and adopted-RV cascade.
Future: a cheaper S/N proxy may replace dependence on mask CCF statistics for warm stars
(see ``docs/operations.md``).
"""
from __future__ import annotations

import numpy as np

from . import config


def region_mask_applicable(teff: np.ndarray, log10_snr: np.ndarray) -> np.ndarray:
    """Stellar mask applicable: Teff < cool OR (Teff < warm AND log10 S/N > min)."""
    t = np.asarray(teff, float)
    s = np.asarray(log10_snr, float)
    t_ok = np.isfinite(t)
    s_ok = np.isfinite(s)
    tc = float(config.METHOD_REGION_MASK_COOL_TEFF_K)
    tw = float(config.METHOD_REGION_MASK_WARM_TEFF_K)
    smin = float(config.METHOD_REGION_LOG10_SNR_MIN)
    return t_ok & ((t < tc) | ((t < tw) & s_ok & (s > smin)))


def region_template_applicable(teff: np.ndarray, log10_snr: np.ndarray) -> np.ndarray:
    """Template FFT applicable where Teff is finite (same convention as validation plots)."""
    return np.isfinite(np.asarray(teff, float))


def region_strong_lines_applicable(teff: np.ndarray, log10_snr: np.ndarray) -> np.ndarray:
    """Strong lines (Hβ) region: Teff and log10 S/N above configured minima."""
    t = np.asarray(teff, float)
    s = np.asarray(log10_snr, float)
    tmin = float(config.METHOD_REGION_STRONG_LINES_MIN_TEFF_K)
    smin = float(config.METHOD_REGION_LOG10_SNR_MIN)
    return np.isfinite(t) & np.isfinite(s) & (t > tmin) & (s > smin)
