# continuum.py
import numpy as np
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import maximum_filter, percentile_filter, uniform_filter1d

STRONG_LINES = [6562.8, 4861.3, 4340.5, 4101.7, 3970.1, 3889.0]


def outlier_mask(wavelength, flux, hi_sigma=5, lo_sigma=20, max_iter=3):
    flux = np.array(flux, dtype=float)
    mask = np.ones_like(flux, dtype=bool)
    smooth = medfilt(flux, kernel_size=5)
    sg = savgol_filter(flux, window_length=7, polyorder=2, mode="interp")
    residuals = flux - sg
    mad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

    for _ in range(max_iter):
        hi = residuals > hi_sigma * mad
        lo = residuals < -lo_sigma * mad
        mask[hi | lo] = False
    return mask


def absorption_mask(wavelength, flux, mask=None, window=100, drop=0.5, sn=10, fast=False):
    flux = np.array(flux, dtype=float)
    med = np.nanmedian(flux if np.nanmedian(flux) > 0 else 1.0)
    norm = flux / med

    if fast:
        smooth = uniform_filter1d(norm, size=window)
        is_abs = norm < (smooth * drop)
        combined_mask = mask & (~is_abs) if mask is not None else ~is_abs
        return combined_mask

    smooth = uniform_filter1d(norm, size=9)
    rolling_min = percentile_filter(norm, 10, size=window)
    resid = np.abs(norm - smooth)
    sn_est = smooth / np.maximum(resid, 1e-6)

    is_abs = (norm < rolling_min * drop) & (sn_est > sn)
    if mask is None:
        return ~is_abs
    return mask & (~is_abs)


def _fit_continuum_spline(wavelength, flux, eflux, num_knots=5, order=3):
    wavelength = np.array(wavelength, float)
    flux = np.array(flux, float)

    if np.sum(eflux) != len(eflux):
        mask_cr = outlier_mask(wavelength, flux)
        mask_abs = absorption_mask(wavelength, flux, mask_cr)
    else:
        mask_cr = np.ones_like(flux, bool)
        mask_abs = absorption_mask(wavelength, flux, mask_cr, window=500, drop=1.0, fast=True)

    good = mask_cr & mask_abs
    x_val, y_val = wavelength[good], flux[good]

    if len(x_val) < order + 2:
        if len(x_val) > 1:
            p = np.poly1d(np.polyfit(x_val, y_val, 1))
            continuum = p(wavelength)
        else:
            continuum = np.full_like(wavelength, np.median(flux))
    else:
        knots = np.linspace(x_val[0], x_val[-1], num_knots)[1:-1]
        try:
            spl = LSQUnivariateSpline(x_val, y_val, t=knots, k=order)
            continuum = spl(wavelength)
        except Exception:
            try:
                spl = UnivariateSpline(x_val, y_val, k=min(order, 3), s=0)
                continuum = spl(wavelength)
            except Exception:
                p = np.poly1d(np.polyfit(x_val, y_val, min(3, len(x_val) - 1)))
                continuum = p(wavelength)

    continuum = savgol_filter(continuum, 11, 2, mode="interp")
    continuum[continuum <= 0] = np.nanmedian(continuum[continuum > 0])

    norm_flux = flux / continuum
    norm_eflux = np.array(eflux, float) / continuum
    final_mask = continuum > 0.1 * np.nanmedian(continuum)
    return wavelength[final_mask], norm_flux[final_mask], norm_eflux[final_mask]


def _fit_continuum_blaze(wavelength, flux, eflux, poly_order=4):
    """Polynomial through upper envelope — proxy for echelle blaze + slow continuum."""
    wavelength = np.array(wavelength, float)
    flux = np.array(flux, float)
    eflux = np.array(eflux, float)
    mask_cr = outlier_mask(wavelength, flux)
    w, f, e = wavelength[mask_cr], flux[mask_cr], eflux[mask_cr]
    if len(w) < poly_order + 3:
        return _fit_continuum_spline(wavelength, flux, eflux)
    win = min(31, max(5, len(f) // 10 | 1))
    env = percentile_filter(f, 90, size=win)
    try:
        coef = np.polyfit(w, env, poly_order)
        continuum = np.polyval(coef, wavelength)
    except Exception:
        return _fit_continuum_spline(wavelength, flux, eflux)
    continuum = np.maximum(continuum, np.nanmedian(continuum) * 0.05)
    norm_flux = flux / continuum
    norm_eflux = eflux / continuum
    final_mask = continuum > 0.1 * np.nanmedian(continuum)
    return wavelength[final_mask], norm_flux[final_mask], norm_eflux[final_mask]


def fit_continuum(wavelength, flux, eflux, num_knots=5, order=3, continuum_mode="spline"):
    """Normalize for line work: continuum_mode is 'spline' or 'blaze'."""
    if continuum_mode == "blaze":
        return _fit_continuum_blaze(wavelength, flux, eflux, poly_order=min(6, order + 2))
    return _fit_continuum_spline(wavelength, flux, eflux, num_knots=num_knots, order=order)


def quick_normalize(wave, flux):
    k = max(11, int(len(flux) * 0.02)) | 1
    cont = medfilt(flux, k)
    cont[cont <= 0] = 1.0
    return wave, flux / cont


def renormalize_local(wave, flux, cont_global, wl_min=None, wl_max=None, poly_order=2):
    if wl_min is None:
        wl_min = wave.min()
    if wl_max is None:
        wl_max = wave.max()
    sel = (wave >= wl_min) & (wave <= wl_max)
    w_seg = wave[sel]
    f_seg = flux[sel] / cont_global[sel]
    env = percentile_filter(f_seg, 95, size=31)
    keep = f_seg > (env * 0.9)
    if np.sum(keep) > poly_order + 2:
        p = np.polyfit(w_seg[keep], f_seg[keep], min(poly_order, 3))
        local_cont = np.polyval(p, w_seg)
    else:
        local_cont = env
    return w_seg, f_seg / local_cont, local_cont


def compute_template_global_continuum(
    wave, flux, mask_lines=STRONG_LINES, mask_width=20.0, percentile=95, window_frac=0.02
):
    n = len(wave)
    win = max(51, int(window_frac * n)) | 1
    mask = np.ones(n, bool)
    for c in mask_lines:
        mask &= ~((wave >= c - mask_width) & (wave <= c + mask_width))
    flux_filled = flux.copy()
    if not np.all(mask):
        flux_filled[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], flux[mask])
    env = percentile_filter(flux_filled, percentile, size=win)
    env = maximum_filter(env, size=win)
    sel = flux_filled >= 0.98 * env
    xs, ys = wave[sel], env[sel]
    if len(xs) > 10:
        if len(xs) > 500:
            idx = np.linspace(0, len(xs) - 1, 500, dtype=int)
            xs, ys = xs[idx], ys[idx]
        spl = UnivariateSpline(xs, ys, s=0, k=3)
        return spl(wave)
    return env
