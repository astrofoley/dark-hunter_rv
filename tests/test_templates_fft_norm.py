"""PHOENIX template normalization for FFT (hot-star broad Balmer wings)."""

import numpy as np

from darkhunter_rv import config
from darkhunter_rv.templates import _norm_template_flux_fft


def test_hot_template_norm_keeps_broad_line_contrast():
    rng = np.random.default_rng(42)
    # Span must include wavelengths outside Hβ ±175 Å so continuum interpolation has anchors.
    wave = np.linspace(4780.0, 5120.0, 4000)
    w0 = 4861.3
    cont = 1.0 + 1.2e-4 * (wave - wave.mean())
    line = 0.5 * np.exp(-0.5 * ((wave - w0) / 3.1) ** 2)
    f = np.clip(cont - line + rng.normal(0, 0.006, wave.shape), 0.08, None)

    out = _norm_template_flux_fft(wave, f, float(config.HOT_STAR_TEFF_THRESHOLD + 800))
    assert np.isfinite(out).all()
    core = (wave >= 4845.0) & (wave <= 4885.0)
    assert np.nanstd(out[core]) > 0.035
