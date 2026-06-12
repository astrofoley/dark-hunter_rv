"""Tests for mask-CCF RV estimators and S/N router."""
from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv import config
from darkhunter_rv.ccf_rv_estimators import (
    EstimatorConfig,
    estimate_ccf_rv,
    prepare_ccf_fit_slice,
    select_ccf_estimator,
)
from darkhunter_rv.rv_core import cross_correlate_stellar_mask


def _synthetic_ccf_slice(
    true_mu: float = 12.0,
    *,
    skew_kms: float = 0.0,
    noise: float = 0.0,
) -> tuple:
    vel = np.linspace(-80.0, 80.0, 321)
    y = 2.0 + 100.0 * np.exp(-0.5 * ((vel - true_mu) / 5.5) ** 2)
    if abs(skew_kms) > 0:
        # Red wing shoulder pulls symmetric Gaussian centroid away from true_mu.
        y += 0.55 * 100.0 * np.exp(-0.5 * ((vel - true_mu - skew_kms) / 3.0) ** 2)
    if noise > 0:
        rng = np.random.default_rng(0)
        y += rng.normal(0, noise, size=len(y))
    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    c_med = float(np.median(y))
    c_mad = float(np.median(np.abs(y - c_med))) + 1e-12
    peak_snr = float((peak_val - c_med) / (1.4826 * c_mad))
    sl = prepare_ccf_fit_slice(
        vel,
        y,
        peak_idx=peak_idx,
        peak_val=peak_val,
        peak_snr=peak_snr,
        fit_width=50,
        ccf_neg_spike_sigma=6.0,
    )
    assert sl is not None
    return sl, true_mu


def test_symmetric_estimators_near_true_shift():
    sl, true_mu = _synthetic_ccf_slice(true_mu=-18.5, skew_kms=0.0)
    cfg = EstimatorConfig()
    for est in ("gauss_offset", "parabolic_ls", "smooth_peak", "grid"):
        res = estimate_ccf_rv(est, sl, cfg=cfg)
        assert np.isfinite(res.rv_kms)
        assert abs(res.rv_kms - true_mu) < 0.5


def test_skewed_ccf_bi_gauss_beats_gaussian():
    sl, true_mu = _synthetic_ccf_slice(true_mu=10.0, skew_kms=14.0)
    cfg = EstimatorConfig(high_asymmetry_bi_gauss_threshold=0.1)
    gauss = estimate_ccf_rv("gauss_offset", sl, cfg=cfg)
    bi = estimate_ccf_rv("bi_gauss", sl, cfg=cfg)
    gauss_err = abs(gauss.rv_kms - true_mu)
    bi_err = abs(bi.rv_kms - true_mu)
    assert bi_err < gauss_err - 0.3


def test_select_ccf_estimator_low_snr():
    assert select_ccf_estimator(3.0, 0.1) == "parabolic_3pt"


def test_select_ccf_estimator_high_asymmetry():
    assert select_ccf_estimator(8.0, 0.35) == "bi_gauss"


def test_select_ccf_estimator_default_gaussian():
    assert select_ccf_estimator(7.0, 0.05) == "gauss_offset"


def test_rv_core_backward_compat_symmetric_line():
    wave = np.linspace(5000, 5100, 2048)
    rest = 5050.0
    true_rv = 12.3
    shifted = rest * (1 + true_rv / config.C_KMS)
    flux = 1.0 - 0.5 * np.exp(-0.5 * ((wave - shifted) / 0.3) ** 2)
    obs = 1.0 - flux
    mask_wave = np.linspace(5020.0, 5080.0, 40)
    mask_strength = np.ones_like(mask_wave)

    rv_gauss, _, _, _, _, _, _ = cross_correlate_stellar_mask(
        wave, obs, mask_wave, mask_strength, max_lag=200, min_peak_snr=1.5
    )
    assert np.isfinite(rv_gauss)
    assert abs(rv_gauss - true_rv) < 50
