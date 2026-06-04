"""Shared RV epoch validity rules."""

from __future__ import annotations

import numpy as np

# Legacy sentinels and other non-physical RVs (e.g. -9999 km/s).
RV_ABS_MAX_KMS = 5000.0
# Reject MJD=0 placeholders and pre-Gaia epoch dates.
MJD_MIN_VALID = 40000.0


def mjd_is_valid(mjd: float) -> bool:
    if not np.isfinite(mjd):
        return False
    return float(mjd) >= MJD_MIN_VALID


def rv_value_is_valid(rv: float) -> bool:
    if not np.isfinite(rv):
        return False
    if abs(float(rv)) >= RV_ABS_MAX_KMS:
        return False
    return True


def rv_epoch_is_valid(mjd: float, rv: float) -> bool:
    return mjd_is_valid(mjd) and rv_value_is_valid(rv)
