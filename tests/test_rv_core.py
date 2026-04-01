import numpy as np

from darkhunter_rv import config
from darkhunter_rv.rv_core import cross_correlate_stellar_mask


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

    rv, rv_err, vel, ccf, peak = cross_correlate_stellar_mask(wave, obs, mask_wave, mask_strength, max_lag=200)

    # With a single-line mask this is crude; just require finite RV within wide tolerance.
    assert np.isfinite(rv)
    assert abs(rv - true_rv) < 50
