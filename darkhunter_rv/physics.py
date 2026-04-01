# physics.py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import config


def vac_to_air(wave):
    """Vacuum to air (standard approximation); wave in Angstrom."""
    s2 = (1e4 / wave) ** 2
    n = 1.0 + 0.0000834254 + 0.02406147 / (130.0 - s2) + 0.00015998 / (38.9 - s2)
    return wave / n


def broaden_spectrum(wave, flux, vbroad_kms):
    """Rotational broadening via Gaussian in log-wavelength."""
    if vbroad_kms is None or vbroad_kms <= 0:
        return flux.copy()

    logw = np.log(wave)
    dlogw = np.median(np.diff(logw))
    if dlogw == 0:
        return flux

    sigma_pix = vbroad_kms / (config.C_KMS * dlogw)
    padded = np.pad(flux, (50, 50), mode="edge")
    broadened = gaussian_filter1d(padded, sigma_pix)
    return broadened[50:-50]
