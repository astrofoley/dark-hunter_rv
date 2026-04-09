"""Spline continuum should track an upper envelope, not a noisy mid-level."""

import numpy as np
import pytest

from darkhunter_rv.continuum import fit_continuum


def _gaussian(x: np.ndarray, center: float, depth: float, sigma: float) -> np.ndarray:
    return 1.0 - depth * np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.mark.parametrize("exclude_near_lines_width", [None, 40.0])
def test_noisy_order_median_norm_below_unity(exclude_near_lines_width):
    rng = np.random.default_rng(42)
    n = 512
    w = np.linspace(5200.0, 5300.0, n)
    # Slow continuum tilt + several absorption features + heavy noise
    cont = 1.05 + 0.03 * (w - w[0]) / (w[-1] - w[0])
    line = _gaussian(w, 5225.0, 0.35, 1.2) * _gaussian(w, 5268.0, 0.25, 2.5)
    flux = cont * line + rng.normal(0.0, 0.09, n)
    eflux = np.full(n, 0.09)

    _, nf, _ = fit_continuum(
        w,
        flux,
        eflux,
        continuum_mode="spline",
        exclude_near_lines_width=exclude_near_lines_width,
    )

    assert np.isfinite(nf).all()
    assert np.median(nf) < 0.99
    assert np.percentile(nf, 92) > 0.92


def test_broad_shallow_lines_stay_below_envelope_continuum():
    """Very broad features (like smeared lines): continuum stays near local tops, not line cores."""
    rng = np.random.default_rng(7)
    n = 600
    w = np.linspace(4400.0, 4500.0, n)
    cont = np.ones(n)
    # Broad trough ~40 Å FWHM-ish
    sigma = 12.0
    line = 1.0 - 0.22 * np.exp(-0.5 * ((w - 4450.0) / sigma) ** 2)
    flux = cont * line + rng.normal(0.0, 0.05, n)
    eflux = np.full(n, 0.05)

    w2, nf, _ = fit_continuum(w, flux, eflux, continuum_mode="spline", exclude_near_lines_width=None)

    mid = (w2 > 4435.0) & (w2 < 4465.0)
    assert np.median(nf[mid]) < 1.0
    # Wing pixels should sit closer to normalized unity than the line center
    wing = (w2 > 4405.0) & (w2 < 4425.0)
    assert np.median(nf[wing]) > np.percentile(nf[mid], 25)
