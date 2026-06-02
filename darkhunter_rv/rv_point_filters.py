"""Shared RV epoch validity rules."""

from __future__ import annotations

import numpy as np

# Legacy sentinels and other non-physical RVs (e.g. -9999 km/s).
RV_ABS_MAX_KMS = 5000.0


def rv_value_is_valid(rv: float) -> bool:
    if not np.isfinite(rv):
        return False
    if abs(float(rv)) >= RV_ABS_MAX_KMS:
        return False
    return True
