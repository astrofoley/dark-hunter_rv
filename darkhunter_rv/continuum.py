# continuum.py
import numpy as np
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import maximum_filter, percentile_filter, uniform_filter1d

from . import config

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


def _pixels_far_from_strong_lines(wavelength: np.ndarray, half_width_angstrom: float) -> np.ndarray:
    """True where pixel is more than ``half_width_angstrom`` from any entry in STRONG_LINES."""
    w = np.asarray(wavelength, float)
    ok = np.ones(w.shape[0], dtype=bool)
    hw = float(half_width_angstrom)
    for c in STRONG_LINES:
        ok &= ~((w >= c - hw) & (w <= c + hw))
    return ok


def _pixels_far_from_ism_rv_bands(wavelength: np.ndarray) -> np.ndarray:
    """Drop Na D (etc.) from spline anchors so ISM-dominated cores do not tilt the continuum."""
    from . import qc

    w = np.asarray(wavelength, float)
    ok = np.ones(w.shape[0], dtype=bool)
    for lo, hi in qc.ISM_RV_EXCLUDE_BANDS:
        ok &= ~((w >= float(lo)) & (w <= float(hi)))
    return ok


def _savgol_safe(y: np.ndarray, window_length: int, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky–Golay if ``window_length`` is valid for ``len(y)``."""
    n = int(len(y))
    wl = int(window_length) | 1
    if wl < 5 or n < wl:
        return y
    wl = min(wl, n if n % 2 else n - 1)
    if wl < 5:
        return y
    return savgol_filter(y, wl, polyorder, mode="interp")


def _fit_continuum_spline(
    wavelength,
    flux,
    eflux,
    num_knots=5,
    order=3,
    exclude_near_lines_width: float | None = None,
):
    """
    Continuum via **upper envelope**: rolling high-percentile of flux defines anchor heights; LSQ spline
    follows those anchors on pixels outside CR/ISM/strong-line masks. Final continuum is floored to a
    fraction of the envelope so it cannot sit through the middle of noisy, line-filled orders.

    Broad-lined stars: ``exclude_near_lines_width`` removes Balmer/He cores from anchors; inter-line
    regions still supply high-percentile (continuum-top) samples.
    """
    wavelength = np.array(wavelength, float)
    flux = np.array(flux, float)
    n_pix = len(flux)

    if np.sum(eflux) != len(eflux):
        mask_cr = outlier_mask(wavelength, flux)
    else:
        mask_cr = np.ones_like(flux, bool)

    good = mask_cr & _pixels_far_from_ism_rv_bands(wavelength)
    if exclude_near_lines_width is not None and float(exclude_near_lines_width) > 0:
        span = float(np.ptp(wavelength))
        hw = float(exclude_near_lines_width)
        # Cap exclusion width vs order length so a chunk sitting on top of Hβ is not 100% masked.
        hw_eff = min(hw, max(14.0, 0.34 * span))
        good_narrow = good & _pixels_far_from_strong_lines(wavelength, hw_eff)
        if int(np.sum(good_narrow)) >= order + 4:
            good = good_narrow
        else:
            n = len(wavelength)
            nedge = max(8, int(0.13 * n))
            edge_ok = np.zeros(n, dtype=bool)
            edge_ok[:nedge] = True
            edge_ok[-nedge:] = True
            good_edge = good & edge_ok
            if int(np.sum(good_edge)) >= order + 4:
                good = good_edge

    fill = float(np.median(flux[mask_cr])) if np.any(mask_cr) else float(np.median(flux))
    f_fill = np.where(mask_cr, flux, fill)

    m0 = float(np.nanmedian(f_fill))
    mad_fl = float(np.nanmedian(np.abs(f_fill - m0)) * 1.4826) + 1e-12
    scale = float(max(abs(m0), float(np.nanpercentile(np.abs(f_fill), 90)), 1e-12))
    noise_ratio = mad_fl / scale
    noisy = noise_ratio > float(getattr(config, "CONTINUUM_NOISY_MAD_TO_MEDIAN", 0.28))

    wmax = int(config.CONTINUUM_ENVELOPE_MAX_WINDOW)
    if noisy:
        wmax = wmax + int(getattr(config, "CONTINUUM_NOISY_WINDOW_EXTRA", 18))
    win = max(
        int(config.CONTINUUM_ENVELOPE_MIN_WINDOW),
        min(wmax, max(5, (n_pix // 9) | 1)),
    )
    p1 = int(np.clip(round(float(config.CONTINUUM_ENVELOPE_PERCENTILE)), 60, 99))
    p2 = int(np.clip(round(float(config.CONTINUUM_ENVELOPE_PERCENTILE_REFINE)), 70, 99))
    if noisy:
        da = int(getattr(config, "CONTINUUM_NOISY_PERCENTILE_ADD", 2))
        p1 = int(np.clip(p1 + da, 60, 99))
        p2 = int(np.clip(p2 + da, 70, 99))
    env = percentile_filter(f_fill, p1, size=win)
    win2 = max(5, (win // 4) | 1)
    env = percentile_filter(env, p2, size=min(win2, win))
    env = _savgol_safe(env, min(max(5, (win // 2) | 1), 31))

    x_val = wavelength[good]
    y_val = env[good]

    nk = int(num_knots)
    if exclude_near_lines_width is not None and float(exclude_near_lines_width) > 0:
        nk = max(3, nk - 1)

    if len(x_val) < order + 2:
        if len(x_val) > 1:
            p = np.poly1d(np.polyfit(x_val, y_val, 1))
            continuum = p(wavelength)
        else:
            continuum = np.full_like(wavelength, fill)
    else:
        knots = np.linspace(x_val[0], x_val[-1], max(nk, 3))[1:-1]
        if len(knots) < 1:
            knots = np.linspace(x_val[0], x_val[-1], 4)[1:-1]
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

    floor_frac = float(np.clip(float(config.CONTINUUM_ENVELOPE_FLOOR_FRAC), 0.5, 0.99))
    if noisy:
        floor_frac = max(
            floor_frac,
            float(np.clip(float(getattr(config, "CONTINUUM_NOISY_FLOOR_FRAC", 0.80)), 0.5, 0.99)),
        )
    continuum = np.maximum(continuum, env * floor_frac)

    sg_post = 15 if exclude_near_lines_width is not None else 11
    continuum = _savgol_safe(continuum, sg_post)
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


def fit_continuum(
    wavelength,
    flux,
    eflux,
    num_knots=5,
    order=3,
    continuum_mode="spline",
    exclude_near_lines_width: float | None = None,
):
    """
    Normalize for line work: continuum_mode is 'spline' or 'blaze'.

    ``exclude_near_lines_width`` (Å): when set (e.g. 78 for very hot stars), pixels within this
    distance of strong Balmer/He lines are dropped from spline anchor points so broad wings do not
    pull the continuum; default None uses all inter-line pixels (subject to CR/ISM masks).
    """
    if continuum_mode == "blaze":
        return _fit_continuum_blaze(wavelength, flux, eflux, poly_order=min(6, order + 2))
    return _fit_continuum_spline(
        wavelength,
        flux,
        eflux,
        num_knots=num_knots,
        order=order,
        exclude_near_lines_width=exclude_near_lines_width,
    )


def despike_normalized_spectrum(wavelength, norm_flux, norm_eflux=None, hi_sigma=4.0, lo_sigma=12.0):
    """
    Interpolate cosmic-ray spikes in continuum-normalized flux (used before CCF).
    Uses the same local outlier logic as continuum fitting, slightly tighter than defaults.
    """
    wavelength = np.asarray(wavelength, float)
    norm_flux = np.asarray(norm_flux, float)
    mask = outlier_mask(wavelength, norm_flux, hi_sigma=hi_sigma, lo_sigma=lo_sigma, max_iter=3)
    if np.all(mask):
        return wavelength, norm_flux, norm_eflux
    bad = ~mask
    good = mask
    if np.sum(good) < 5:
        return wavelength, norm_flux, norm_eflux
    nf = norm_flux.copy()
    nf[bad] = np.interp(
        wavelength[bad],
        wavelength[good],
        norm_flux[good],
        left=float(np.median(norm_flux[good])),
        right=float(np.median(norm_flux[good])),
    )
    ne = norm_eflux
    if norm_eflux is not None and len(norm_eflux) == len(nf):
        ne = np.asarray(norm_eflux, float).copy()
        ne[bad] = float(np.median(ne[good]))
    return wavelength, nf, ne


def despike_normalized_spikes_only(
    wavelength,
    norm_flux,
    norm_eflux=None,
    hi_sigma=6.5,
    sg_window=7,
    max_norm_flux=2.75,
):
    """
    Remove cosmic-ray-like **positive** spikes in continuum-normalized flux only.
    Deep lines (negative residuals vs a smooth local trend) are never masked, unlike symmetric outlier rejection.

    Also flags pixels with norm_flux > max_norm_flux (narrow CRs that inflate MAD and evade sigma cuts).

    For stronger cleaning without harming lines, optionally chain a mild symmetric despike with large lo_sigma
    on the output of this function (see despike_normalized_pre_ccf).
    """
    wavelength = np.asarray(wavelength, float)
    norm_flux = np.asarray(norm_flux, float)
    wl = max(7, int(sg_window) | 1)
    smooth = savgol_filter(norm_flux, window_length=wl, polyorder=2, mode="interp")
    res = norm_flux - smooth
    mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
    bad = res > hi_sigma * mad
    if max_norm_flux is not None and np.isfinite(max_norm_flux):
        bad = bad | (norm_flux > float(max_norm_flux))
    if not np.any(bad):
        return wavelength, norm_flux, norm_eflux
    good = ~bad
    if np.sum(good) < 5:
        return wavelength, norm_flux, norm_eflux
    nf = norm_flux.copy()
    nf[bad] = np.interp(
        wavelength[bad],
        wavelength[good],
        norm_flux[good],
        left=float(np.median(norm_flux[good])),
        right=float(np.median(norm_flux[good])),
    )
    ne = norm_eflux
    if norm_eflux is not None and len(norm_eflux) == len(nf):
        ne = np.asarray(norm_eflux, float).copy()
        ne[bad] = float(np.median(ne[good]))
    return wavelength, nf, ne


def despike_normalized_pre_ccf(wavelength, norm_flux, norm_eflux=None):
    """
    CR cleaning before mask CCF: positive spike removal + hard flux cap, second MAD-only pass on
    interpolated spectrum, then mild symmetric pass with very loose lo_sigma so absorption lines
    are not clipped.
    """
    w, f, e = despike_normalized_spikes_only(
        wavelength, norm_flux, norm_eflux, hi_sigma=5.5, max_norm_flux=2.0
    )
    w, f, e = despike_normalized_spikes_only(w, f, e, hi_sigma=6.0, max_norm_flux=None)
    return despike_normalized_spectrum(w, f, e, hi_sigma=4.5, lo_sigma=48.0)


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
