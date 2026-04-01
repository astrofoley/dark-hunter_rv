"""Spectroscopic binary SB1 circular-orbit RV fit (single dataset)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


@dataclass
class BinaryFitResult:
    gamma: float
    k: float
    period: float
    t0: float
    rv_pred: np.ndarray
    success: bool
    message: str


def keplerian_circular(t: np.ndarray, gamma: float, k: float, t0: float, p: float) -> np.ndarray:
    """RV for circular orbit: gamma + K sin(2π(t-t0)/P). times in days, RV km/s."""
    ph = 2.0 * np.pi * (t - t0) / p
    return gamma + k * np.sin(ph)


def fit_circular_binary(
    t_days: np.ndarray,
    rv_kms: np.ndarray,
    err_kms: np.ndarray,
    period_days: float,
    p0: Optional[tuple[float, float, float]] = None,
) -> BinaryFitResult:
    """Fit (gamma, K, t0) holding period fixed. Requires finite errors."""
    t_days = np.asarray(t_days, float)
    rv_kms = np.asarray(rv_kms, float)
    err_kms = np.asarray(err_kms, float)
    mask = np.isfinite(t_days) & np.isfinite(rv_kms) & (err_kms > 0) & np.isfinite(err_kms)
    t_days, rv_kms, err_kms = t_days[mask], rv_kms[mask], err_kms[mask]
    if len(t_days) < 4:
        return BinaryFitResult(0, 0, period_days, 0, rv_kms, False, "too few points")

    if p0 is None:
        gamma0 = float(np.median(rv_kms))
        k0 = float((np.percentile(rv_kms, 90) - np.percentile(rv_kms, 10)) / 2.0)
        k0 = max(abs(k0), 1.0)
        t0_0 = float(t_days[0])
        p0 = (gamma0, k0, t0_0)

    def resid(x):
        g, k, t0 = x
        m = keplerian_circular(t_days, g, k, t0, period_days)
        return (m - rv_kms) / err_kms

    res = least_squares(resid, p0, bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]))
    g, k, t0 = res.x
    pred = keplerian_circular(t_days, g, k, t0, period_days)
    ok = res.success
    logger.info("binary fit gamma=%.3f K=%.3f t0=%.4f P=%.4f ok=%s", g, k, t0, period_days, ok)
    return BinaryFitResult(g, k, period_days, t0, pred, bool(ok), res.message)
