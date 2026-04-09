import numpy as np

from darkhunter_rv import config
from darkhunter_rv.rv_core import (
    cross_correlate_stellar_mask,
    fit_balmer_line_all_methods,
    mask_line_flux_in_excluded_wavelengths,
)


def test_ccf_peak_near_injected_shift():
    # synthetic: Gaussian absorption line
    wave = np.linspace(5000, 5100, 2048)
    rest = 5050.0
    true_rv = 12.3  # km/s
    shifted = rest * (1 + true_rv / config.C_KMS)

    flux = 1.0 - 0.5 * np.exp(-0.5 * ((wave - shifted) / 0.3) ** 2)
    obs = 1.0 - flux

    mask_wave = np.linspace(5020.0, 5080.0, 40)
    mask_strength = np.ones_like(mask_wave)

    rv, rv_err, vel, ccf, peak, gauss_p, _snr = cross_correlate_stellar_mask(
        wave, obs, mask_wave, mask_strength, max_lag=200, min_peak_snr=1.5
    )

    assert np.isfinite(rv)
    assert abs(rv - true_rv) < 50
    assert gauss_p is None or len(gauss_p) in (3, 4, 5)
    if gauss_p is not None and len(gauss_p) == 4:
        c0, amp, mu, sig = gauss_p
        assert np.isfinite(c0) and np.isfinite(amp) and np.isfinite(mu) and np.isfinite(sig)
        assert abs(mu - rv) < 1e-6
    if gauss_p is not None and len(gauss_p) == 5:
        c0, c1, amp, mu, sig = gauss_p
        assert np.isfinite(c0) and np.isfinite(c1) and np.isfinite(amp)
        assert abs(mu - rv) < 1e-6


def test_ccf_offset_gaussian_model_recovers_mu():
    """Same 4-parameter CCF model as mask fitting: constant + Gaussian peak."""
    from scipy.optimize import curve_fit

    v = np.linspace(-120.0, 120.0, 241)
    true_mu = -22.4
    y = 2.5 + 95.0 * np.exp(-0.5 * ((v - true_mu) / 6.8) ** 2)

    def model(x, c0p, amp, mu, sig):
        return c0p + amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    span = float(np.ptp(y))
    v_lo, v_hi = float(v[0]), float(v[-1])
    c0_med = float(np.median(y))
    amp0 = float(max(np.max(y) - c0_med, span * 0.02, 1e-9))
    sig0 = 8.0
    p0 = [c0_med, amp0, true_mu + 1.0, sig0]
    bounds = (
        [float(np.min(y)) - span, 1e-12, v_lo, 0.5],
        [float(np.max(y)) + span, 10.0 * span + abs(c0_med), v_hi, 400.0],
    )
    popt, _ = curve_fit(model, v, y, p0=p0, bounds=bounds, maxfev=8000)
    mu_fit = float(popt[2])
    assert abs(mu_fit - true_mu) < 0.15


def test_mask_line_flux_zeros_na_d_region():
    w = np.linspace(5885.0, 5905.0, 80)
    line = np.ones_like(w) * 0.15
    out = mask_line_flux_in_excluded_wavelengths(w, line)
    in_d = (w >= 5888.5) & (w <= 5898.0)
    assert np.all(out[in_d] == 0.0)
    assert np.any(out[~in_d] > 0.0)


def test_fit_balmer_sloped_continuum_full_order():
    rest = 4861.3
    w = np.linspace(4850.0, 4875.0, 900)
    true_rv = 22.0
    w_line = rest * (1.0 + true_rv / config.C_KMS)
    continuum = 0.92 + 12.0 * (w - w.mean()) / (np.ptp(w) + 1e-9) * 0.004
    line = 0.38 * np.exp(-0.5 * ((w - w_line) / 0.22) ** 2)
    f = continuum - line
    out = fit_balmer_line_all_methods(w, f, rest, "Hb", broad_lines=True)
    assert out is not None
    assert abs(float(out["rv_gauss_kms"]) - true_rv) < 18.0
    assert np.isfinite(out["err_gauss_kms"])
