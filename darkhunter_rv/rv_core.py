# rv_core.py
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.special import voigt_profile
from . import config, qc, templates as tpl_bank

# Mask-CCF quality: contrast of peak vs lag-to-lag noise; Gaussian center must stay near grid argmax.
_DEFAULT_MIN_CCF_PEAK_SNR = 3.2
_DEFAULT_MAX_GAUSS_OFFSET_FROM_GRID_KMS = 35.0


def cross_correlate_stellar_mask(
    obs_wave,
    obs_flux,
    mask_wave,
    mask_strength,
    max_lag=None,
    fit_width=50,
    min_peak_snr=_DEFAULT_MIN_CCF_PEAK_SNR,
    max_gauss_offset_kms=_DEFAULT_MAX_GAUSS_OFFSET_FROM_GRID_KMS,
    ccf_neg_spike_sigma=6.0,
    *,
    _refit_depth: int = 0,
):
    """
    Cross-correlate line-only flux with a mask. Excludes pixels in strong telluric bands from the sum.
    Gaussian fit uses a **constant offset** plus a positive Gaussian: ``c0 + amp*exp(...)`` (no linear
    term in velocity). When the lag grid is wide enough, ``c0`` bounds and initial guess are anchored
    to the median CCF on lags far from the grid argmax so broad peaks do not trade baseline against
    width inside the local fit slice. Deep negative CCF points (e.g. from unmasked CRs) are excluded
    from the fit via ``ccf_neg_spike_sigma``.

    On success, ``gauss_popt`` is ``(c0, amp, mu, sig)`` in the velocity frame of ``vel_shifts``.

    Rejects Gaussian widths below ``config.MASK_CCF_MIN_GAUSS_SIGMA_KMS`` (spike fits). If the fitted
    center lies within ``MASK_CCF_EDGE_REFIT_FRACTION`` of the velocity window edge, retries once with
    a wider lag grid (``MASK_CCF_EDGE_REFIT_LAG_MULT``).
    """
    if max_lag is None:
        max_lag = int(config.MASK_CCF_DEFAULT_MAX_LAG)
    max_lag = int(max_lag)
    obs_wave = np.asarray(obs_wave, float)
    obs_flux = np.asarray(obs_flux, float)
    ok_px = np.ones(len(obs_wave), dtype=bool)
    for lo, hi in qc.rv_contamination_bands():
        ok_px &= ~((obs_wave >= lo) & (obs_wave <= hi))
    obs_w = obs_flux * ok_px.astype(float)

    log_obs = np.log10(obs_wave)
    log_mask = np.log10(mask_wave)

    median_diff = np.median(np.diff(log_obs)) / 5.0
    log_shifts = np.arange(-max_lag, max_lag + 1) * median_diff
    ccf = np.zeros_like(log_shifts)

    vel_shifts = config.C_KMS * (10**log_shifts - 1)

    for i, shift in enumerate(log_shifts):
        shifted_mask = log_mask + shift
        idx = np.searchsorted(log_obs, shifted_mask)
        valid = (idx > 0) & (idx < len(log_obs))
        if np.sum(valid) < 5:
            continue
        ccf[i] = np.sum(obs_w[idx[valid]] * mask_strength[valid])

    if np.max(ccf) <= 0:
        return np.nan, np.nan, vel_shifts, ccf, 0.0, None, np.nan

    peak_idx = int(np.argmax(ccf))
    peak_val = float(ccf[peak_idx])
    v_grid = float(vel_shifts[peak_idx])
    v_lo = float(np.min(vel_shifts))
    v_hi = float(np.max(vel_shifts))

    c_med = float(np.median(ccf))
    c_mad = float(np.median(np.abs(ccf - c_med))) + 1e-12
    peak_snr = float((peak_val - c_med) / (1.4826 * c_mad))
    if peak_snr < min_peak_snr:
        return np.nan, np.nan, vel_shifts, ccf, peak_val, None, peak_snr

    sl = slice(max(0, peak_idx - fit_width), min(len(ccf), peak_idx + fit_width + 1))
    x_fit, y_fit = vel_shifts[sl], ccf[sl]

    if len(x_fit) < 5:
        return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr

    y = np.asarray(y_fit, float)
    x_arr = np.asarray(x_fit, float)
    med_y = float(np.median(y))
    mad_y = float(np.median(np.abs(y - med_y))) + 1e-12
    scale_y = 1.4826 * mad_y
    neg_floor = med_y - float(ccf_neg_spike_sigma) * scale_y
    use = y >= neg_floor
    if int(np.sum(use)) < 5:
        use = np.ones_like(y, dtype=bool)
    x_m = x_arr[use]
    y_m = y[use]

    span = float(np.ptp(y_m))
    span = max(span, 1e-9)
    c0_med = float(np.median(y_m))
    amp0 = float(max(np.max(y_m) - c0_med, span * 0.02, 1e-9))
    sig0 = float(max(2.0, min(80.0, 0.5 * (v_hi - v_lo) * fit_width / max(len(ccf), 1))))

    rough = c0_med + amp0 * np.exp(-0.5 * ((x_arr - v_grid) / max(sig0, 2.0)) ** 2)
    use = use & (y >= (rough - 4.0 * scale_y))
    if int(np.sum(use)) < 5:
        use = np.ones_like(y, dtype=bool)
    x_m = x_arr[use]
    y_m = y[use]
    span = float(np.ptp(y_m))
    span = max(span, 1e-9)
    sig_est = float(max(2.0, min(80.0, 0.5 * (v_hi - v_lo) * fit_width / max(len(ccf), 1))))
    c0_0 = float(np.median(y_m))
    amp0 = float(max(float(np.max(y_m)) - c0_0, span * 0.02, 1e-9))

    def ccf_offset_gauss(x, c0p, amp, mu, sig):
        return c0p + amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    c_lo = float(np.min(y_m) - 0.75 * span)
    c_hi = float(np.max(y_m) + 0.25 * span)
    # When the CCF peak is much wider than the fit slice, every point in the slice can sit on the
    # curved flank of the Gaussian; c0 + Gauss is then degenerate and the optimizer often picks a
    # baseline far from the true wings. Anchor c0 using medians on lags far from the grid argmax.
    span_v = float(v_hi - v_lo)
    if span_v > 25.0:
        wing_sep = max(28.0, min(0.11 * span_v, 140.0))
        wing_mask = np.abs(vel_shifts - v_grid) >= wing_sep
        n_wing = int(np.sum(wing_mask))
        if n_wing >= 6:
            y_w = np.asarray(ccf[wing_mask], float)
            c0_wing = float(np.median(y_w))
            mad_w = float(np.median(np.abs(y_w - c0_wing)) * 1.4826) + 1e-12
            ptp_w = float(np.ptp(y_w))
            band = max(10.0 * mad_w, 0.025 * abs(peak_val - c0_wing), 0.05 * max(ptp_w, mad_w))
            band = max(band, 1e-9)
            c_lo_w, c_hi_w = c0_wing - band, c0_wing + band
            c_lo_n = max(c_lo, c_lo_w)
            c_hi_n = min(c_hi, c_hi_w)
            if c_lo_n < c_hi_n - 1e-6:
                c_lo, c_hi = c_lo_n, c_hi_n
                c0_0 = float(np.clip(c0_wing, c_lo + 1e-9, c_hi - 1e-9))
                amp0 = float(max(float(np.max(y_m)) - c0_0, span * 0.02, 1e-9))
            else:
                c_lo = float(min(c_lo_w, float(np.min(y_m)) - 0.5 * span))
                c_hi = float(max(c_hi_w, float(np.max(y_m)) + 0.1 * span))
                c0_0 = float(np.clip(c0_wing, c_lo + 1e-9, c_hi - 1e-9))
                amp0 = float(max(float(np.max(y_m)) - c0_0, span * 0.02, 1e-9))
    sig_max = min(400.0, max(10.0, v_hi - v_lo))
    amp_hi = 10.0 * span + abs(c0_0)
    bounds = (
        [c_lo, 1e-12, v_lo, 0.5],
        [c_hi, amp_hi, v_hi, sig_max],
    )
    p0 = [c0_0, amp0, v_grid, sig_est]
    gauss_popt = None
    rv = v_grid
    rv_err = np.nan

    def _fit_soft_l1() -> tuple[float, float, float, float] | None:
        lb = np.array(bounds[0], dtype=float)
        ub = np.array(bounds[1], dtype=float)
        p0a = np.array(p0, dtype=float)
        p0a = np.clip(p0a, lb + 1e-12, ub - 1e-12)
        f_scale = max(span * 0.12, 1e-9)

        def resid(p):
            c0p, amp, mu, sig = p
            return (c0p + amp * np.exp(-0.5 * ((x_m - mu) / sig) ** 2)) - y_m

        try:
            sol = least_squares(
                resid,
                p0a,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=f_scale,
                max_nfev=8000,
            )
            if not np.all(np.isfinite(sol.x)):
                return None
            c0_f, a_f, mu_f, sig_f = (
                float(sol.x[0]),
                float(sol.x[1]),
                float(sol.x[2]),
                float(sol.x[3]),
            )
            return (c0_f, a_f, mu_f, sig_f)
        except Exception:
            return None

    try:
        popt, pcov = curve_fit(ccf_offset_gauss, x_m, y_m, p0=p0, bounds=bounds, maxfev=8000)
        c_fit, amp_fit, mu_fit, sig_fit = (
            float(popt[0]),
            float(popt[1]),
            float(popt[2]),
            float(popt[3]),
        )
        if amp_fit < 1e-11:
            sl = _fit_soft_l1()
            if sl is None:
                return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
            c_fit, amp_fit, mu_fit, sig_fit = sl
            pcov = None
        if amp_fit < 1e-11:
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        if abs(mu_fit - v_grid) > max_gauss_offset_kms:
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        if mu_fit < v_lo - 1e-3 or mu_fit > v_hi + 1e-3:
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        if float(sig_fit) < float(config.MASK_CCF_MIN_GAUSS_SIGMA_KMS):
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        rv = mu_fit
        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
            rv_err = float(perr[2]) if np.isfinite(perr[2]) else np.nan
        else:
            rv_err = np.nan
        gauss_popt = (c_fit, amp_fit, mu_fit, sig_fit)
    except Exception:
        sl = _fit_soft_l1()
        if sl is None:
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        c_fit, amp_fit, mu_fit, sig_fit = sl
        if amp_fit < 1e-11 or abs(mu_fit - v_grid) > max_gauss_offset_kms:
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        if mu_fit < v_lo - 1e-3 or mu_fit > v_hi + 1e-3:
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        if float(sig_fit) < float(config.MASK_CCF_MIN_GAUSS_SIGMA_KMS):
            return v_grid, np.nan, vel_shifts, ccf, peak_val, None, peak_snr
        rv = mu_fit
        rv_err = np.nan
        gauss_popt = (c_fit, amp_fit, mu_fit, sig_fit)

    edge_tol = float(config.MASK_CCF_EDGE_REFIT_FRACTION) * span_v
    if (
        _refit_depth < 1
        and span_v > 30.0
        and edge_tol > 1.0
        and (float(mu_fit) - v_lo < edge_tol or v_hi - float(mu_fit) < edge_tol)
    ):
        wider = min(
            max(int(max_lag * float(config.MASK_CCF_EDGE_REFIT_LAG_MULT)), max_lag + 50),
            int(config.MASK_CCF_MAX_LAG_CAP),
        )
        if wider > max_lag:
            return cross_correlate_stellar_mask(
                obs_wave,
                obs_flux,
                mask_wave,
                mask_strength,
                max_lag=wider,
                fit_width=fit_width,
                min_peak_snr=min_peak_snr,
                max_gauss_offset_kms=max_gauss_offset_kms,
                ccf_neg_spike_sigma=ccf_neg_spike_sigma,
                _refit_depth=_refit_depth + 1,
            )

    return rv, rv_err, vel_shifts, ccf, peak_val, gauss_popt, peak_snr


def mask_line_flux_in_excluded_wavelengths(obs_wave: np.ndarray, obs_line_flux: np.ndarray) -> np.ndarray:
    """
    Zero line-only flux (continuum-normalized absorption) in telluric + ISM bands so mask CCF sums and
    template FFT omit those pixels without changing the stored continuum normalization elsewhere.
    """
    m = qc.wavelength_band_mask(obs_wave, qc.rv_contamination_bands())
    out = np.asarray(obs_line_flux, float).copy()
    out[m] = 0.0
    return out


def _fft_absorption_affine_align(y_obs: np.ndarray, y_tpl: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Fit y_obs ≈ a * y_tpl + b on finite pixels; return aligned template copy and (a, b).
    """
    y_obs = np.asarray(y_obs, float)
    y_tpl = np.asarray(y_tpl, float)
    m = np.isfinite(y_obs) & np.isfinite(y_tpl)
    n = int(np.sum(m))
    if n < 12:
        out = np.where(np.isfinite(y_tpl), y_tpl, 0.0)
        return out, 1.0, 0.0
    X = np.column_stack([y_tpl[m], np.ones(n)])
    ab, _, _, _ = np.linalg.lstsq(X, y_obs[m], rcond=None)
    a, b = float(ab[0]), float(ab[1])
    aligned = a * y_tpl + b
    return aligned, a, b


def _fft_correlation_peak_for_template(
    obs_resamp: np.ndarray,
    window: np.ndarray,
    fft_obs: np.ndarray,
    mask_vel: np.ndarray,
    vel_win: np.ndarray,
    tpl_grid_wave: np.ndarray,
    t_wave: np.ndarray,
    t_flux: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    if t_wave[-1] < tpl_grid_wave[0] or t_wave[0] > tpl_grid_wave[-1]:
        return None
    t_resamp = np.interp(tpl_grid_wave, t_wave, t_flux, left=np.nan, right=np.nan)
    t_resamp = 1.0 - t_resamp
    valid = np.isfinite(t_resamp) & np.isfinite(obs_resamp)
    if int(np.sum(valid)) < 12:
        return None
    t_aligned, _a, _b = _fft_absorption_affine_align(obs_resamp, t_resamp)
    med_t = float(np.nanmedian(t_aligned[valid]))
    t_use = np.where(np.isfinite(t_aligned), t_aligned, med_t)
    tpl_z = (t_use - float(np.mean(t_use))) / (float(np.std(t_use)) + 1e-9)
    fft_tpl = np.fft.fft(tpl_z * window)
    ccf = np.fft.ifft(fft_obs * np.conj(fft_tpl)).real
    ccf = np.fft.fftshift(ccf)
    ccf_w = np.asarray(ccf[mask_vel], float)
    peak = float(np.max(ccf_w))
    idx = int(np.argmax(ccf_w))
    best_rv = float(vel_win[idx])
    return peak, best_rv, ccf_w


def _fft_velocity_window(
    obs_wave: np.ndarray,
    obs_flux: np.ndarray,
    *,
    rv_seed_kms: float | None = None,
    rv_search_half_width_kms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the log-λ FFT grid, z-scored observation FFT, and velocity mask for template correlation.

    Returns obs_resamp, window, fft_obs, vel_axis, mask_vel, vel_win, tpl_grid_wave.
    """
    loglam_obs = np.log10(obs_wave)
    npts = max(2 ** int(np.ceil(np.log2(len(obs_wave)))), 512)
    log_grid = np.linspace(loglam_obs.min(), loglam_obs.max(), npts)

    obs_resamp = np.interp(log_grid, loglam_obs, obs_flux)
    window = np.hanning(len(obs_resamp))
    obs_z = (obs_resamp - float(np.mean(obs_resamp))) / (float(np.std(obs_resamp)) + 1e-9)
    fft_obs = np.fft.fft(obs_z * window)

    delta_lnlam = (log_grid[1] - log_grid[0]) * np.log(10.0)
    dv_pix = config.C_KMS * delta_lnlam
    vel_axis = (np.arange(npts) - npts // 2) * dv_pix
    mask_vel = (vel_axis >= -1000) & (vel_axis <= 1000)
    if (
        rv_seed_kms is not None
        and np.isfinite(float(rv_seed_kms))
        and rv_search_half_width_kms is not None
        and np.isfinite(float(rv_search_half_width_kms))
        and float(rv_search_half_width_kms) > 5.0
    ):
        lo = float(rv_seed_kms) - float(rv_search_half_width_kms)
        hi = float(rv_seed_kms) + float(rv_search_half_width_kms)
        narrow = (vel_axis >= lo) & (vel_axis <= hi) & mask_vel
        if int(np.sum(narrow)) >= 16:
            mask_vel = narrow
    vel_win = np.asarray(vel_axis[mask_vel], float)
    tpl_grid_wave = 10**log_grid
    return obs_resamp, window, fft_obs, vel_axis, mask_vel, vel_win, tpl_grid_wave


def _collect_fft_ccf_rows(
    obs_resamp: np.ndarray,
    window: np.ndarray,
    fft_obs: np.ndarray,
    mask_vel: np.ndarray,
    vel_win: np.ndarray,
    tpl_grid_wave: np.ndarray,
    bank: dict,
) -> tuple[list[object], np.ndarray]:
    """Return (template_keys, ccf_matrix) with shape (n_templates, n_vel)."""
    keys: list[object] = []
    rows: list[np.ndarray] = []
    for key, (t_wave, t_flux) in bank.items():
        r = _fft_correlation_peak_for_template(
            obs_resamp,
            window,
            fft_obs,
            mask_vel,
            vel_win,
            tpl_grid_wave,
            np.asarray(t_wave, float),
            np.asarray(t_flux, float),
        )
        if r is None:
            continue
        _pk, _rv, ccf_w = r
        keys.append(key)
        rows.append(np.asarray(ccf_w, float))
    if not rows:
        return keys, np.zeros((0, len(vel_win)))
    return keys, np.stack(rows, axis=0)


def _estimate_rv_fft_best_in_bank(
    obs_resamp: np.ndarray,
    window: np.ndarray,
    fft_obs: np.ndarray,
    mask_vel: np.ndarray,
    vel_win: np.ndarray,
    tpl_grid_wave: np.ndarray,
    bank: dict,
    *,
    peak_pick: str = "per_template_max",
) -> tuple[float, object | None, np.ndarray | None]:
    if peak_pick == "aggregate_median":
        keys, mat = _collect_fft_ccf_rows(
            obs_resamp, window, fft_obs, mask_vel, vel_win, tpl_grid_wave, bank
        )
        if mat.size == 0 or mat.shape[0] == 0:
            return float("nan"), None, None
        agg = np.nanmedian(mat, axis=0)
        j = int(np.nanargmax(agg))
        best_rv = float(vel_win[j])
        col = mat[:, j]
        i_best = int(np.nanargmax(col))
        best_tpl_key = keys[i_best]
        best_ccf = mat[i_best].copy()
        return best_rv, best_tpl_key, best_ccf

    best_peak, best_rv, best_tpl_key = -np.inf, 0.0, None
    best_ccf = None
    for key, (t_wave, t_flux) in bank.items():
        r = _fft_correlation_peak_for_template(
            obs_resamp,
            window,
            fft_obs,
            mask_vel,
            vel_win,
            tpl_grid_wave,
            np.asarray(t_wave, float),
            np.asarray(t_flux, float),
        )
        if r is None:
            continue
        peak, rv, ccf_w = r
        if peak > best_peak:
            best_peak = peak
            best_rv = rv
            best_tpl_key = key
            best_ccf = ccf_w.copy()
    return best_rv, best_tpl_key, best_ccf


def template_fft_ccf_stack(
    obs_wave,
    obs_flux,
    bank: dict,
    *,
    rv_seed_kms: float | None = None,
    rv_search_half_width_kms: float | None = None,
) -> tuple[np.ndarray, list[object], np.ndarray]:
    """
    Cross-correlation curves for every template in ``bank`` on a common velocity grid.

    ``ccf_stack[i, j]`` is template ``keys[i]`` at ``vel_win[j]``. Used for failure-mode diagnostics.
    """
    ow = np.asarray(obs_wave, float)
    of = np.asarray(obs_flux, float)
    obs_resamp, window, fft_obs, _vel_axis, mask_vel, vel_win, tpl_grid_wave = _fft_velocity_window(
        ow,
        of,
        rv_seed_kms=rv_seed_kms,
        rv_search_half_width_kms=rv_search_half_width_kms,
    )
    keys, mat = _collect_fft_ccf_rows(
        obs_resamp, window, fft_obs, mask_vel, vel_win, tpl_grid_wave, bank
    )
    return vel_win, keys, mat


def estimate_rv_fft_with_ccf(
    obs_wave,
    obs_flux,
    templates,
    vsini_proxy,
    *,
    fft_two_phase: bool = True,
    fft_coarse_top_k: int | None = None,
    rv_seed_kms: float | None = None,
    rv_search_half_width_kms: float | None = None,
    fft_peak_pick: str = "per_template_max",
):
    """
    FFT template correlation; returns best RV, winning template key, and that template's CCF
    (velocity km/s, correlation) for diagnostics.

    Default lag search is ±1000 km/s. When ``rv_seed_kms`` and ``rv_search_half_width_kms`` are
    finite, the search is restricted to that interval so the global maximum is taken only among
    physically plausible Doppler shifts (e.g. mask CCF seed on cool stars).

    On the FFT log grid, each template absorption vector is affinely matched to the observation
    (scale + offset), then both are z-scored so correlation compares shape at matched amplitude.

    When ``fft_two_phase`` is True and the bank is large enough, a **coarse** pass runs on one
    broadening trial per (Teff, log g, [M/H]); the top few atmospheres by peak correlation are
    expanded to the full vsini set for a second pass. This is skipped when it would not reduce work.

    ``fft_peak_pick``: ``per_template_max`` picks the template with the highest single-template CCF
    peak; ``aggregate_median`` peaks the median CCF across templates at each lag (robust to outliers).
    """
    top_k = int(fft_coarse_top_k if fft_coarse_top_k is not None else config.FFT_COARSE_TOP_K)
    top_k = max(1, top_k)

    ow = np.asarray(obs_wave, float)
    of = np.asarray(obs_flux, float)
    obs_resamp, window, fft_obs, _vel_axis, mask_vel, vel_win, tpl_grid_wave = _fft_velocity_window(
        ow,
        of,
        rv_seed_kms=rv_seed_kms,
        rv_search_half_width_kms=rv_search_half_width_kms,
    )

    bank: dict = templates
    n_tot = len(templates)
    coarse = tpl_bank.coarse_fft_subbank(templates, vsini_proxy)
    n_coarse = len(coarse)
    max_vb = tpl_bank.max_vsini_variants_per_atmosphere(templates)
    est_two_phase_cost = n_coarse + top_k * max_vb

    use_two_phase = (
        fft_two_phase
        and n_tot >= int(config.FFT_TWO_PHASE_MIN_TEMPLATES)
        and n_coarse > top_k
        and est_two_phase_cost < n_tot
        and max_vb > 1
    )

    if use_two_phase:
        scored: list[tuple[float, object]] = []
        for key, (t_wave, t_flux) in coarse.items():
            r = _fft_correlation_peak_for_template(
                obs_resamp,
                window,
                fft_obs,
                mask_vel,
                vel_win,
                tpl_grid_wave,
                np.asarray(t_wave, float),
                np.asarray(t_flux, float),
            )
            if r is None:
                continue
            scored.append((r[0], key))
        scored.sort(key=lambda x: -x[0])
        picked: set[tuple[float, float, float]] = set()
        for _pk, key in scored:
            st = tpl_bank.template_key_stellar_tuple(key)
            if st is None or st in picked:
                continue
            picked.add(st)
            if len(picked) >= top_k:
                break
        refined = tpl_bank.refined_fft_subbank(templates, picked)
        if len(refined) >= 2:
            bank = refined

    best_rv, best_tpl_key, best_ccf = _estimate_rv_fft_best_in_bank(
        obs_resamp,
        window,
        fft_obs,
        mask_vel,
        vel_win,
        tpl_grid_wave,
        bank,
        peak_pick=fft_peak_pick,
    )

    if best_tpl_key is None or best_ccf is None:
        return float("nan"), None, None, None
    return best_rv, best_tpl_key, vel_win, best_ccf


def estimate_rv_fft_vectorized(obs_wave, obs_flux, templates, vsini_proxy, plot=False):
    rv, key, _, _ = estimate_rv_fft_with_ccf(obs_wave, obs_flux, templates, vsini_proxy)
    return rv, key


def fit_fft_ccf_models(
    vel: np.ndarray,
    ccf: np.ndarray,
    *,
    core_half_width_kms: float = 72.0,
    wide_half_width_kms: float = 220.0,
) -> dict:
    """
    Fit the FFT correlation peak: single Gaussian on a restricted core window, then optional
    double Gaussian on a wider window if that reduces reduced chi-squared clearly (broad/shoulder CCF).
    """
    v = np.asarray(vel, float)
    y = np.asarray(ccf, float)
    n = len(v)
    if n < 12:
        return {"ok": False}

    i0 = int(np.argmax(y))
    v_peak = float(v[i0])

    def ccf_one(vx, c, a, mu, sig):
        return c + np.abs(a) * np.exp(-0.5 * ((vx - mu) / (sig + 1e-6)) ** 2)

    def ccf_two(vx, c, a1, m1, s1, a2, m2, s2):
        return c + np.abs(a1) * np.exp(-0.5 * ((vx - m1) / (s1 + 1e-6)) ** 2) + np.abs(a2) * np.exp(
            -0.5 * ((vx - m2) / (s2 + 1e-6)) ** 2
        )

    def mad_sigma(resid: np.ndarray) -> float:
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-12
        return max(1.4826 * mad, 1e-9)

    m_core = np.abs(v - v_peak) <= float(core_half_width_kms)
    vc, yc = v[m_core], y[m_core]
    if len(vc) < 8:
        return {"ok": False, "peak_vel_grid": v_peak}

    edge = np.concatenate([yc[: min(4, len(yc) // 3)], yc[-min(4, len(yc) // 3) :]])
    c0 = float(np.median(edge)) if len(edge) else float(np.median(yc))
    a0 = max(float(np.max(yc) - c0), 1e-9)
    sig0 = max(float(core_half_width_kms) * 0.22, 5.0)
    p1 = [c0, a0, v_peak, sig0]
    lo_v, hi_v = v_peak - core_half_width_kms, v_peak + core_half_width_kms
    b1 = ([np.min(yc) - abs(a0), 1e-12, lo_v, 2.0], [np.max(yc) + abs(a0), abs(a0) * 50, hi_v, core_half_width_kms])
    p1[0] = float(np.clip(p1[0], b1[0][0] + 1e-6, b1[1][0] - 1e-6))
    p1[1] = float(np.clip(p1[1], b1[0][1] + 1e-9, b1[1][1] - 1e-9))
    p1[2] = float(np.clip(p1[2], b1[0][2] + 1e-3, b1[1][2] - 1e-3))
    p1[3] = float(np.clip(p1[3], b1[0][3] + 1e-3, b1[1][3] - 1e-3))

    single_core = None
    y1c = None
    try:
        popt, pcov = curve_fit(ccf_one, vc, yc, p0=p1, bounds=b1, maxfev=10000)
        pred = ccf_one(vc, *popt)
        sig = mad_sigma(yc - pred)
        chi1 = float(np.sum(((yc - pred) / sig) ** 2) / max(1, len(yc) - 4))
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(4, np.nan)
        single_core = {
            "popt": tuple(float(x) for x in popt),
            "mu_kms": float(popt[2]),
            "mu_err_kms": float(perr[2]) if np.isfinite(perr[2]) else float("nan"),
            "chi2_red": chi1,
        }
        vf = np.linspace(float(vc.min()), float(vc.max()), max(200, len(vc) * 4))
        y1c = ccf_one(vf, *popt)
    except Exception:
        vf = np.linspace(float(vc.min()), float(vc.max()), max(200, len(vc) * 4))
        y1c = None

    m_wide = np.abs(v - v_peak) <= float(wide_half_width_kms)
    vw, yw = v[m_wide], y[m_wide]
    use_double = False
    double_wide = None
    y2w = None
    vfw = np.linspace(float(vw.min()), float(vw.max()), max(250, len(vw) * 4))

    if len(vw) >= 14 and single_core is not None:
        c_f, a_f, mu_f, s_f = single_core["popt"]
        try:
            p0w = [c_f, a_f * 0.75, mu_f, max(s_f, 4.0), a_f * 0.22, mu_f + 42.0, max(s_f * 1.3, 8.0)]
            lo_w = float(v_peak - wide_half_width_kms)
            hi_w = float(v_peak + wide_half_width_kms)
            bw = (
                [np.min(yw) - abs(a_f), 1e-12, lo_w, 2.0, 1e-12, lo_w, 2.0],
                [np.max(yw) + abs(a_f) * 2, abs(a_f) * 80, hi_w, wide_half_width_kms * 0.55, abs(a_f) * 80, hi_w, wide_half_width_kms * 0.55],
            )
            p2, pc2 = curve_fit(ccf_two, vw, yw, p0=p0w, bounds=bw, maxfev=15000)
            pred2 = ccf_two(vw, *p2)
            pred1w = ccf_one(vw, c_f, a_f, mu_f, s_f)
            noise_w = mad_sigma(yw)
            chi2 = float(np.sum(((yw - pred2) / noise_w) ** 2) / max(1, len(yw) - 7))
            chi1w = float(np.sum(((yw - pred1w) / noise_w) ** 2) / max(1, len(yw) - 4))
            if chi2 < 0.72 * chi1w and chi2 < chi1w - 2.5:
                use_double = True
                pr2 = np.sqrt(np.diag(pc2)) if pc2 is not None else np.full(7, np.nan)
                double_wide = {
                    "popt": tuple(float(x) for x in p2),
                    "mu1_kms": float(p2[2]),
                    "mu1_err_kms": float(pr2[2]) if np.isfinite(pr2[2]) else float("nan"),
                    "mu2_kms": float(p2[5]),
                    "mu2_err_kms": float(pr2[5]) if np.isfinite(pr2[5]) else float("nan"),
                    "chi2_red": chi2,
                }
                y2w = ccf_two(vfw, *p2)
        except Exception:
            pass

    single_wide = None
    y1w = None
    try:
        if len(vw) < 8:
            raise ValueError("wide window too small")
        edge_w = np.concatenate([yw[: min(5, len(yw) // 3)], yw[-min(5, len(yw) // 3) :]])
        c0w = float(np.median(edge_w)) if len(edge_w) else float(np.median(yw))
        a0w = max(float(np.max(yw) - c0w), 1e-9)
        p1w = [c0w, a0w, v_peak, max(float(wide_half_width_kms) * 0.18, 8.0)]
        lo_vw, hi_vw = v_peak - wide_half_width_kms, v_peak + wide_half_width_kms
        b1w = ([np.min(yw) - abs(a0w), 1e-12, lo_vw, 3.0], [np.max(yw) + abs(a0w), abs(a0w) * 60, hi_vw, wide_half_width_kms * 0.5])
        p1w[0] = float(np.clip(p1w[0], b1w[0][0] + 1e-6, b1w[1][0] - 1e-6))
        p1w[1] = float(np.clip(p1w[1], b1w[0][1] + 1e-9, b1w[1][1] - 1e-9))
        p1w[2] = float(np.clip(p1w[2], b1w[0][2] + 1e-3, b1w[1][2] - 1e-3))
        p1w[3] = float(np.clip(p1w[3], b1w[0][3] + 1e-3, b1w[1][3] - 1e-3))
        pw, pcw = curve_fit(ccf_one, vw, yw, p0=p1w, bounds=b1w, maxfev=12000)
        predw = ccf_one(vw, *pw)
        sigw1 = mad_sigma(yw - predw)
        chi_w = float(np.sum(((yw - predw) / sigw1) ** 2) / max(1, len(yw) - 4))
        prw = np.sqrt(np.diag(pcw)) if pcw is not None else np.full(4, np.nan)
        single_wide = {
            "popt": tuple(float(x) for x in pw),
            "mu_kms": float(pw[2]),
            "mu_err_kms": float(prw[2]) if np.isfinite(prw[2]) else float("nan"),
            "chi2_red": chi_w,
        }
        y1w = ccf_one(vfw, *pw)
    except Exception:
        vfw = np.linspace(float(vw.min()), float(vw.max()), max(250, len(vw) * 4))
        y1w = None

    return {
        "ok": True,
        "peak_vel_grid": v_peak,
        "vel_full": v,
        "ccf_full": y,
        "vel_core": vc,
        "ccf_core": yc,
        "vel_wide": vw,
        "ccf_wide": yw,
        "v_fine_core": vf,
        "ccf_fit_core": y1c,
        "v_fine_wide": vfw,
        "ccf_fit_single_wide": y1w,
        "ccf_fit_double_wide": y2w,
        "single_core": single_core,
        "single_wide": single_wide,
        "double_wide": double_wide,
        "use_double": use_double,
    }


def degrade_template_flux_lsf(wavelength: np.ndarray, flux: np.ndarray, resolving_power: float) -> np.ndarray:
    """
    Apply a simple Gaussian LSF in wavelength space (σ from FWHM = λ/R, FWHM = 2.355σ).
    Operates on the sampled ``wavelength`` grid (e.g. observed pixel vector).
    """
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float).copy()
    m = np.isfinite(f) & np.isfinite(w)
    if int(np.sum(m)) < 3 or not np.isfinite(resolving_power) or resolving_power <= 1.0:
        return f
    dw = float(np.nanmedian(np.diff(w[m])))
    if not np.isfinite(dw) or dw <= 0:
        return f
    wm = float(np.nanmedian(w[m]))
    sigma_lam = wm / float(resolving_power) / 2.355
    sigma_pix = max(float(sigma_lam / dw), 0.25)
    fill = float(np.nanmedian(f[m]))
    f_work = np.where(m, f, fill)
    sm = gaussian_filter1d(f_work, sigma=sigma_pix, mode="nearest")
    out = sm.copy()
    out[~m] = f[~m]
    return out


def rms_absorption_residual_fft_grid(
    obs_wave: np.ndarray,
    obs_absorption_flux: np.ndarray,
    tpl_wave: np.ndarray,
    tpl_flux_norm: np.ndarray,
    rv_kms: float,
    resolving_power: float,
) -> float:
    """
    RMS of ``obs_absorption - (a·template + b)`` on the same log-λ grid used for template FFT, after
    Doppler shift, LSF degradation, and the affine line-strength match used in correlation.

    **Lower is better** shape agreement (e.g. discriminates rotational broadening). Used for
    exposure-level PHOENIX key selection.
    """
    obs_wave = np.asarray(obs_wave, float)
    obs_line = np.asarray(obs_absorption_flux, float)
    tpl_wave = np.asarray(tpl_wave, float)
    tpl_flux_norm = np.asarray(tpl_flux_norm, float)

    beta = 1.0 + float(rv_kms) / config.C_KMS
    w_map = tpl_wave * beta
    order = np.argsort(w_map)
    w_s, tf_s = w_map[order], tpl_flux_norm[order]
    tpl_on_obs = np.interp(obs_wave, w_s, tf_s, left=np.nan, right=np.nan)
    tpl_sm = degrade_template_flux_lsf(obs_wave, tpl_on_obs, resolving_power)
    tpl_line_raw = 1.0 - tpl_sm

    loglam_obs = np.log10(obs_wave)
    npts = max(2 ** int(np.ceil(np.log2(len(obs_wave)))), 512)
    log_grid = np.linspace(loglam_obs.min(), loglam_obs.max(), npts)
    obs_r = np.interp(log_grid, loglam_obs, obs_line)
    tpl_r = np.interp(log_grid, loglam_obs, tpl_line_raw)
    valid = np.isfinite(obs_r) & np.isfinite(tpl_r)
    if int(np.sum(valid)) < 12:
        return float("nan")
    tpl_r_al, _, _ = _fft_absorption_affine_align(obs_r, tpl_r)
    resid = obs_r - tpl_r_al
    return float(np.sqrt(np.mean((resid[valid]) ** 2)))


def build_fft_match_plot_series(
    obs_wave: np.ndarray,
    obs_absorption_flux: np.ndarray,
    tpl_wave: np.ndarray,
    tpl_flux_norm: np.ndarray,
    rv_kms: float,
    resolving_power: float,
) -> dict:
    """
    Match ``estimate_rv_fft_with_ccf``: Doppler-shift and LSF-degrade the template, map both
    spectra to the FFT log grid, **affine-match** template absorption to the observation, then
    **z-score** both (same as the live FFT correlation).
    """
    obs_wave = np.asarray(obs_wave, float)
    obs_absorption_flux = np.asarray(obs_absorption_flux, float)
    tpl_wave = np.asarray(tpl_wave, float)
    tpl_flux_norm = np.asarray(tpl_flux_norm, float)

    beta = 1.0 + float(rv_kms) / config.C_KMS
    w_map = tpl_wave * beta
    order = np.argsort(w_map)
    w_s, tf_s = w_map[order], tpl_flux_norm[order]
    tpl_on_obs = np.interp(obs_wave, w_s, tf_s, left=np.nan, right=np.nan)
    tpl_sm = degrade_template_flux_lsf(obs_wave, tpl_on_obs, resolving_power)

    obs_line = obs_absorption_flux
    tpl_line_raw = 1.0 - tpl_sm

    loglam_obs = np.log10(obs_wave)
    npts = max(2 ** int(np.ceil(np.log2(len(obs_wave)))), 512)
    log_grid = np.linspace(loglam_obs.min(), loglam_obs.max(), npts)
    tpl_grid_wave = 10.0**log_grid

    obs_r = np.interp(log_grid, loglam_obs, obs_line)
    tpl_r = np.interp(log_grid, loglam_obs, tpl_line_raw)
    valid = np.isfinite(obs_r) & np.isfinite(tpl_r)
    tpl_r_al, _a_ff, _b_ff = _fft_absorption_affine_align(obs_r, tpl_r)
    med_t = float(np.nanmedian(tpl_r_al[valid])) if int(np.sum(valid)) >= 8 else 0.0
    t_use = np.where(np.isfinite(tpl_r_al), tpl_r_al, med_t)
    obs_z = (obs_r - float(np.mean(obs_r))) / (float(np.std(obs_r)) + 1e-9)
    tpl_z = (t_use - float(np.mean(t_use))) / (float(np.std(t_use)) + 1e-9)

    tpl_on_obs_display = np.interp(obs_wave, tpl_grid_wave, t_use)

    window = np.hanning(len(obs_z))
    return {
        "wavelength_obs": obs_wave,
        "obs_absorption": obs_line,
        "tpl_absorption_on_obs": tpl_on_obs_display,
        "wavelength_fft_grid": tpl_grid_wave,
        "obs_zscore": obs_z,
        "tpl_mean_centered": tpl_z,
        "hanning_window": window,
        "obs_fft_input": obs_z * window,
        "tpl_fft_input": tpl_z * window,
    }


def estimate_broadening(obs_wave, obs_flux, tpl_wave, tpl_flux):
    log_min = max(np.log10(obs_wave.min()), np.log10(tpl_wave.min()))
    log_max = min(np.log10(obs_wave.max()), np.log10(tpl_wave.max()))
    if log_max <= log_min:
        return None, {}

    log_grid = np.linspace(log_min, log_max, 2048)
    obs_r = np.interp(log_grid, np.log10(obs_wave), obs_flux)
    tpl_r = np.interp(log_grid, np.log10(tpl_wave), tpl_flux)
    obs_r -= np.mean(obs_r)
    tpl_r -= np.mean(tpl_r)
    so = float(np.std(obs_r)) + 1e-12
    st = float(np.std(tpl_r)) + 1e-12
    obs_r /= so
    tpl_r /= st

    ccf_dt = np.fft.fftshift(np.fft.ifft(np.fft.fft(obs_r)*np.conj(np.fft.fft(tpl_r))).real)
    ccf_tt = np.fft.fftshift(np.fft.ifft(np.fft.fft(tpl_r)*np.conj(np.fft.fft(tpl_r))).real)

    def get_width(y):
        x = np.arange(len(y))
        mu = np.average(x, weights=np.abs(y))
        return np.sqrt(np.average((x-mu)**2, weights=np.abs(y)))

    sig_dt = get_width(ccf_dt[2048//2-50:2048//2+50])
    sig_tt = get_width(ccf_tt[2048//2-50:2048//2+50])

    if sig_dt > sig_tt:
        pix_broad = np.sqrt(max(sig_dt**2 - sig_tt**2, 0.0))
        dv = config.C_KMS * (log_grid[1] - log_grid[0]) * np.log(10)
        vb_kms = float(pix_broad * dv * 2.355)
        vmax = float(getattr(config, "VSINI_PROXY_MAX_KMS", 200.0))
        if not np.isfinite(vb_kms) or vb_kms < 0.0:
            return None, {}
        if vb_kms > vmax:
            # Pipeline uses config.VSINI_PROXY_REJECTED_GRID_KMS so the PHOENIX bank is not stuck at 10 km/s.
            return None, {"vsini_proxy_rejected_kms": float(vb_kms)}
        return vb_kms, {}
    return 10.0, {}


_STRONG_LINE_MAX_ABS_RV_KMS = 100.0
_STRONG_LINE_MAX_ABS_RV_KMS_BROAD = 150.0
_STRONG_LINE_MAX_REDUCED_CHI2 = 12.0
_STRONG_LINE_MAX_REDUCED_CHI2_BROAD = 45.0
_STRONG_LINE_MIN_DEPTH = 0.05


def measure_strong_line_centroids(wave, flux, broad_lines: bool = False):
    """
    Gaussian fits to Balmer segments in continuum-normalized flux.

    **Not** used as an exposure-level pipeline method: diagnostics ``strong_lines`` uses
    :func:`measure_h_beta_rv` (Voigt+Lorentz Hβ centroid). Kept for experiments or legacy scripts.
    Use broad_lines=True for hot/fast-rotating stars (wider windows, looser chi2/RV caps).
    """
    lines = {"Ha": 6562.8, "Hb": 4861.3, "Hg": 4340.5, "Hd": 4101.7}
    candidates: list[tuple[str, float, float, float]] = []

    win = 52.0 if broad_lines else 22.0
    max_rv = _STRONG_LINE_MAX_ABS_RV_KMS_BROAD if broad_lines else _STRONG_LINE_MAX_ABS_RV_KMS
    max_chi2 = _STRONG_LINE_MAX_REDUCED_CHI2_BROAD if broad_lines else _STRONG_LINE_MAX_REDUCED_CHI2
    sig_hi = 24.0 if broad_lines else 8.0
    maxfev = 6000 if broad_lines else 2500

    for name, rest in lines.items():
        mask = (wave > rest - win) & (wave < rest + win)
        w, f = wave[mask], flux[mask]
        if len(w) < 12:
            continue
        try:
            def model(x, a, mu, s, c):
                return c - a * np.exp(-0.5 * ((x - mu) / s) ** 2)

            fmin = float(np.min(f))
            mu0 = float(w[np.argmin(f)])
            span = float(np.max(w) - np.min(w))
            s0 = float(np.clip(0.08 * span, 1.2, sig_hi * 0.85))
            p0 = [max(0.15, 1.0 - fmin), mu0, s0, float(np.median(f))]
            lo_a, hi_a = 0.02, 2.8
            lo_mu, hi_mu = rest - win * 0.95, rest + win * 0.95
            lo_s, hi_s = 0.55, sig_hi
            lo_c, hi_c = 0.15, 1.45
            bounds = ([lo_a, lo_mu, lo_s, lo_c], [hi_a, hi_mu, hi_s, hi_c])
            p0 = [
                float(np.clip(p0[0], lo_a + 1e-3, hi_a - 1e-3)),
                float(np.clip(p0[1], lo_mu + 1e-3, hi_mu - 1e-3)),
                float(np.clip(p0[2], lo_s + 1e-3, hi_s - 1e-3)),
                float(np.clip(p0[3], lo_c + 1e-3, hi_c - 1e-3)),
            ]
            popt, _pcov = curve_fit(model, w, f, p0=p0, bounds=bounds, maxfev=maxfev)
            pred = model(w, *popt)
            res = f - pred
            mad = float(np.median(np.abs(res - np.median(res))))
            sigma = 1.4826 * mad if mad > 1e-9 else float(np.std(res) + 1e-9)
            chi2 = float(np.sum((res / sigma) ** 2))
            dof = max(1, len(w) - 4)
            chi2_red = chi2 / dof

            rv = float(config.C_KMS * (popt[1] - rest) / rest)
            err = float(config.C_KMS * np.sqrt(np.diag(_pcov))[1] / rest)
            if popt[0] <= _STRONG_LINE_MIN_DEPTH:
                continue
            if abs(rv) > max_rv:
                continue
            if not np.isfinite(err) or err <= 0:
                continue
            if chi2_red > max_chi2:
                continue
            candidates.append((name, rv, err, chi2_red))
        except Exception:
            continue

    if not candidates:
        return np.nan, np.nan

    rvs = np.array([c[1] for c in candidates], float)
    errs = np.array([c[2] for c in candidates], float)
    chi2rs = np.array([c[3] for c in candidates], float)
    keep = list(range(len(rvs)))

    if len(keep) >= 3:
        med = float(np.median(rvs[keep]))
        mad = float(np.median(np.abs(rvs[keep] - med)))
        scale = 1.4826 * mad if mad > 1e-6 else float(np.std(rvs[keep]))
        if scale > 1e-6:
            worst_j, worst_z = None, 0.0
            for j in keep:
                z = abs(rvs[j] - med) / max(scale, errs[j], 0.5)
                if z > worst_z:
                    worst_z, worst_j = z, j
            if worst_j is not None and worst_z > 3.0:
                keep.remove(worst_j)
    elif len(keep) == 2:
        i0, i1 = keep[0], keep[1]
        diff = abs(rvs[i0] - rvs[i1])
        sigma_comb = float(np.sqrt(errs[i0] ** 2 + errs[i1] ** 2 + 1e-18))
        if diff > 3.0 * sigma_comb:
            drop = i0 if chi2rs[i0] > chi2rs[i1] else i1
            keep.remove(drop)

    if not keep:
        return np.nan, np.nan

    rvs_k = rvs[keep]
    errs_k = errs[keep]
    weights = 1.0 / (errs_k**2 + 1e-9)
    mean_rv = float(np.average(rvs_k, weights=weights))
    mean_err = float(np.sqrt(1.0 / np.sum(weights)))
    return mean_rv, mean_err


# Strong-line diagnostics: per-line continuum flattening, then analytic fits on |v|≤cap (see below);
# reported uncertainties are nulled when |RV| exceeds the trust band.
_STRONG_RV_TRUST_HALF_WIDTH_KMS = 150.0
_STRONG_ZOOM_HALF_WIDTH_KMS = 100.0
_STRONG_LOCAL_CONT_EXCLUDE_KMS_NARROW = 280.0
_STRONG_LOCAL_CONT_EXCLUDE_KMS_BROAD = 420.0
# Match Hβ Voigt window: all strong-line profile fits use this |v| cap (not full echelle order).
_STRONG_FIT_VELOCITY_CAP_KMS_BROAD = 700.0
_STRONG_FIT_VELOCITY_CAP_KMS_NARROW = 320.0


def _local_continuum_divide_for_line(
    wave: np.ndarray,
    flux: np.ndarray,
    rest: float,
    exclude_half_width_kms: float,
) -> np.ndarray:
    """
    Divide flux by a linear continuum estimated from pixels with |v| >= exclude band, so broad
    Balmer wings and order-wide blaze residuals do not force line fits into a flat absorption model.
    """
    w = np.asarray(wave, float)
    f = np.asarray(flux, float)
    v = config.C_KMS * (w / rest - 1.0)
    ex = float(exclude_half_width_kms)
    m = (np.abs(v) >= ex) & np.isfinite(f) & np.isfinite(w)
    if int(np.sum(m)) < 14:
        med = float(np.nanmedian(f[np.isfinite(f)]))
        return f / max(abs(med), 1e-12)
    coef = np.polyfit(w[m], f[m], 1)
    cont = np.polyval(coef, w)
    cont = np.maximum(cont, float(np.nanpercentile(cont, 6)) * 0.06 + 1e-12)
    return f / cont


def _trust_rv_err(rv: float, err: float, half_width_kms: float) -> float:
    if not np.isfinite(rv) or not np.isfinite(err):
        return err
    if abs(float(rv)) > float(half_width_kms):
        return float("nan")
    return err


HB_REST_A = 4861.3


def h_beta_joint_line_model(
    wavelength: np.ndarray,
    params: np.ndarray | Sequence[float],
    *,
    rest: float | None = None,
) -> np.ndarray:
    """
    Evaluate the joint Hβ line model used by :func:`measure_h_beta_rv` (single fit, shared center).

    ``params`` must be length 8: ``amp_v, amp_l, center_angstrom, sigma, gamma_voigt, gamma_lorentz,
    c0, c1`` with flux ``c0 + c1 * (λ - λ_rest)`` minus a normalized Voigt depth and a Lorentzian wing
    term, both centered at ``center_angstrom``.
    """
    rest = float(HB_REST_A if rest is None else rest)
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 8:
        raise ValueError("h_beta_joint_line_model requires exactly 8 parameters")
    amp_v, amp_l, center, sig, gam_v, gam_l, c0, c1 = p
    x = np.asarray(wavelength, dtype=float)
    z = x - center
    sigp = abs(sig) + 1e-4
    gvp = abs(gam_v) + 1e-4
    vp = voigt_profile(z, sigp, gvp)
    vp0 = voigt_profile(0.0, sigp, gvp) + 1e-99
    vterm = abs(amp_v) * (vp / vp0)
    glp = abs(gam_l) + 1e-7
    lterm = abs(amp_l) * (glp**2) / (z**2 + glp**2)
    return c0 + c1 * (x - rest) - vterm - lterm


def measure_h_beta_rv(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    *,
    broad_lines: bool = False,
    tpl_wave: np.ndarray | None = None,
    tpl_flux_norm: np.ndarray | None = None,
    resolving_power: float | None = None,
) -> dict | None:
    """
    Third RV method (diagnostics ``strong_lines``): Hβ-only for now; extend to more rest wavelengths later.

    Local linear continuum from far wings, smoothed core minimum, capped-|v| **joint**
    Voigt+Lorentz model (single shared line center; both profiles subtracted from the same linear
    continuum — pressure/wing Lorentzian plus thermally dominated Voigt core), and optional template
    cross-correlation in the same wavelength slice.
    """
    w_full = np.asarray(wave, float)
    f_full = np.asarray(flux_norm, float)
    wmn, wmx = float(np.nanmin(w_full)), float(np.nanmax(w_full))
    if not (wmn <= HB_REST_A <= wmx):
        return None

    half_ang = 135.0 if broad_lines else 72.0
    mwin = (w_full >= HB_REST_A - half_ang) & (w_full <= HB_REST_A + half_ang) & np.isfinite(f_full)
    if int(np.sum(mwin)) < 22:
        return None
    w0 = w_full[mwin]
    f0 = f_full[mwin]
    o_s = np.argsort(w0)
    w0, f0 = w0[o_s], f0[o_s]

    ex_kms = 780.0 if broad_lines else 400.0
    f_flat = _local_continuum_divide_for_line(w0, f0, HB_REST_A, ex_kms)
    v0 = config.C_KMS * (w0 / HB_REST_A - 1.0)

    trust_core = 185.0 if broad_lines else 125.0
    sig_sm = max(3.5, float(len(f_flat)) * (0.045 if broad_lines else 0.032))
    fs = gaussian_filter1d(f_flat.astype(float), sigma=float(sig_sm), mode="nearest")
    m_tr = np.abs(v0) <= trust_core
    rv_sm, err_sm = float("nan"), float("nan")
    if np.any(m_tr):
        idx = np.where(m_tr)[0]
        j = int(idx[int(np.argmin(fs[m_tr]))])
        rv_sm = float(v0[j])
        vs = np.sort(v0[m_tr])
        dv = float(np.median(np.diff(vs))) if len(vs) > 2 else float("nan")
        err_sm = abs(dv) if np.isfinite(dv) else float("nan")
        err_sm = _trust_rv_err(rv_sm, err_sm, _STRONG_RV_TRUST_HALF_WIDTH_KMS)

    v_cap = _STRONG_FIT_VELOCITY_CAP_KMS_BROAD if broad_lines else _STRONG_FIT_VELOCITY_CAP_KMS_NARROW
    m_fit = (np.abs(v0) <= v_cap) & np.isfinite(f_flat)
    rv_v, err_v, f_joint = float("nan"), float("nan"), None
    wf = None
    popt_arr: np.ndarray | None = None
    w_f, ff = w0[m_fit], f_flat[m_fit]
    if len(w_f) >= 18:
        w_lo, w_hi = float(np.min(w_f)), float(np.max(w_f))
        if w_hi > w_lo + 1e-3:
            lo_mu, hi_mu = w_lo + 1e-4, w_hi - 1e-4
            max_sig_kms = 480.0 if broad_lines else 140.0
            sig_hi = max(HB_REST_A * max_sig_kms / config.C_KMS, 0.16)
            sig_lo = max(HB_REST_A * 10.0 / config.C_KMS, 0.065)
            span = max(w_hi - w_lo, 1e-6)
            c1max = 16.0 / span
            mh = np.abs(config.C_KMS * (w_f / HB_REST_A - 1.0)) <= (620.0 if broad_lines else 280.0)
            if np.any(mh):
                k0 = int(np.argmin(ff[mh]))
                mu0 = float(w_f[mh][k0])
                med_c = float(np.median(ff[mh]))
                fmin = float(ff[mh][k0])
            else:
                k0 = int(np.argmin(ff))
                mu0 = float(w_f[k0])
                med_c = float(np.median(ff))
                fmin = float(ff[k0])
            s0 = float(np.clip(0.1 * span, sig_lo, sig_hi * 0.9))
            p_lo_c, p_hi_c = np.nanpercentile(ff, [5, 95])

            def voigt_lin(x, amp, center, sig, gam, c0, c1):
                z = x - center
                sigp = abs(sig) + 1e-4
                gamp = abs(gam) + 1e-4
                vp = voigt_profile(z, sigp, gamp)
                vp0 = voigt_profile(0.0, sigp, gamp) + 1e-99
                return c0 + c1 * (x - HB_REST_A) - abs(amp) * vp / vp0

            def voigt_plus_lorentz_lin(x, amp_v, amp_l, center, sig, gam_v, gam_l, c0, c1):
                return h_beta_joint_line_model(
                    x, (amp_v, amp_l, center, sig, gam_v, gam_l, c0, c1), rest=HB_REST_A
                )

            p0v = [
                max(0.1, min(2.4, med_c - fmin)),
                mu0,
                s0,
                max(s0 * 0.45, sig_lo * 0.5),
                med_c,
                0.0,
            ]
            bounds_v = (
                [0.03, lo_mu, sig_lo, sig_lo * 0.35, max(0.04, p_lo_c - 0.5), -c1max],
                [2.8, hi_mu, sig_hi, sig_hi, min(2.85, p_hi_c + 0.5), c1max],
            )
            p0v = [
                float(np.clip(p0v[i], bounds_v[0][i] + 1e-4, bounds_v[1][i] - 1e-4)) for i in range(6)
            ]

            gam_l_hi = max(
                sig_hi * 2.6,
                HB_REST_A * ((650.0 if broad_lines else 220.0) / config.C_KMS),
            )
            try:
                pv_w, _cw = curve_fit(voigt_lin, w_f, ff, p0=p0v, bounds=bounds_v, maxfev=16000)
                adepth = float(max(pv_w[0], 0.08))
                p0_vl = [
                    max(0.04, adepth * 0.72),
                    max(0.0, adepth * 0.28),
                    float(pv_w[1]),
                    float(pv_w[2]),
                    float(pv_w[3]),
                    float(np.clip(max(float(pv_w[2]) * 1.8, sig_lo * 1.2), sig_lo * 0.5, gam_l_hi)),
                    float(pv_w[4]),
                    float(pv_w[5]),
                ]
            except Exception:
                depth = max(0.12, min(2.2, med_c - fmin))
                p0_vl = [
                    depth * 0.7,
                    depth * 0.3,
                    mu0,
                    s0,
                    max(s0 * 0.45, sig_lo * 0.5),
                    max(s0 * 1.5, sig_lo * 1.2),
                    med_c,
                    0.0,
                ]

            bounds_vl = (
                [0.03, 0.0, lo_mu, sig_lo, sig_lo * 0.25, sig_lo * 0.4, max(0.04, p_lo_c - 0.5), -c1max],
                [2.8, 2.8, hi_mu, sig_hi, sig_hi, gam_l_hi, min(2.85, p_hi_c + 0.5), c1max],
            )
            p0_vl = [
                float(np.clip(p0_vl[i], bounds_vl[0][i] + 1e-4, bounds_vl[1][i] - 1e-4))
                for i in range(8)
            ]

            try:
                popt, pcov = curve_fit(
                    voigt_plus_lorentz_lin,
                    w_f,
                    ff,
                    p0=p0_vl,
                    bounds=bounds_vl,
                    maxfev=32000,
                )
                rv_v = float(config.C_KMS * (popt[2] - HB_REST_A) / HB_REST_A)
                err_v = (
                    float(config.C_KMS * np.sqrt(np.diag(pcov))[2] / HB_REST_A)
                    if pcov is not None
                    else float("nan")
                )
                err_v = _trust_rv_err(rv_v, err_v, _STRONG_RV_TRUST_HALF_WIDTH_KMS)
                wf = np.linspace(w_lo, w_hi, max(300, len(w_f) * 5))
                popt_arr = np.asarray(popt, dtype=float).reshape(8)
                f_joint = h_beta_joint_line_model(wf, popt_arr, rest=HB_REST_A)
            except Exception:
                wf = None
                f_joint = None
                popt_arr = None
        else:
            wf = None
            popt_arr = None

    rv_ccf, cc_peak = float("nan"), None
    if (
        tpl_wave is not None
        and tpl_flux_norm is not None
        and len(tpl_wave) > 30
        and len(tpl_flux_norm) == len(tpl_wave)
    ):
        tw = np.asarray(tpl_wave, float)
        tf = np.asarray(tpl_flux_norm, float)
        ok = np.isfinite(tw) & np.isfinite(tf)
        tw, tf = tw[ok], tf[ok]
        if len(tw) > 40:
            order = np.argsort(tw)
            tw, tf = tw[order], tf[order]
            obs_abs = 1.0 - np.clip(f_flat, 0.02, 2.5)
            rv_grid = np.arange(-255.0, 256.0, 2.8 if broad_lines else 2.2)
            R = float(resolving_power) if resolving_power is not None and resolving_power > 1.0 else None
            best_c = -1e99
            for rv in rv_grid:
                beta = 1.0 + float(rv) / config.C_KMS
                w_map = tw * beta
                if w_map[-1] < w0[0] or w_map[0] > w0[-1]:
                    continue
                ti = np.interp(w0, w_map, tf, left=np.nan, right=np.nan)
                if R is not None:
                    ti = degrade_template_flux_lsf(w0, ti, R)
                m = np.isfinite(ti)
                if int(np.sum(m)) < 14:
                    continue
                t_abs = 1.0 - np.clip(ti, 0.02, 2.5)
                oa = obs_abs[m]
                ta = t_abs[m]
                if np.nanstd(oa) < 1e-9 or np.nanstd(ta) < 1e-9:
                    continue
                c = float(np.corrcoef(oa, ta)[0, 1])
                if np.isfinite(c) and c > best_c:
                    best_c = c
                    rv_ccf = float(rv)
            if best_c > -1e90:
                cc_peak = float(best_c)

    pool: list[float] = []
    if np.isfinite(rv_sm):
        pool.append(rv_sm)
    if np.isfinite(rv_v):
        pool.append(rv_v)
    method_used = "median(smooth,voigt+lorentz)"
    rv_best = float("nan")
    err_best = float("nan")
    if cc_peak is not None and cc_peak > 0.32 and np.isfinite(rv_ccf):
        rv_best = rv_ccf
        err_best = max(4.0, 90.0 / max(cc_peak, 0.35))
        method_used = "template_ccf"
    elif pool:
        rv_best = float(np.median(pool))
        spread = float(np.std(pool)) if len(pool) > 1 else float(err_sm if np.isfinite(err_sm) else err_v)
        err_best = float(
            np.hypot(
                spread if np.isfinite(spread) else 0.0,
                float(np.nanmedian([x for x in (err_sm, err_v) if np.isfinite(x)]) or 2.0),
            )
        )
        if not np.isfinite(err_best) or err_best <= 0:
            err_best = 5.0

    return {
        "rest_a": HB_REST_A,
        "v_kms_plot": v0,
        "flux_plot": f_flat,
        "wavelength_plot": w0,
        "rv_best_kms": rv_best,
        "err_best_kms": err_best,
        "rv_smoothed_min_kms": rv_sm,
        "err_smoothed_kms": err_sm,
        "rv_voigt_kms": rv_v,
        "err_voigt_kms": err_v,
        "rv_template_ccf_kms": rv_ccf,
        "template_ccf_peak": cc_peak,
        "method_used": method_used,
        "voigt_wave_fine": np.asarray(wf, dtype=float) if wf is not None else np.array([], dtype=float),
        "voigt_model_fine": f_joint,
        "hb_joint_fit_params": None if popt_arr is None else np.asarray(popt_arr, dtype=float).tolist(),
        "rv_lorentz_kms": float("nan"),
        "err_lorentz_kms": float("nan"),
        "lorentz_model_fine": None,
    }


def fit_balmer_line_all_methods(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    rest: float,
    line_name: str,
    broad_lines: bool = False,
) -> dict | None:
    """
    Per-line linear continuum division (excluding pixels near the line), then Gaussian / Lorentzian /
    Voigt / core Gaussian / smoothed-minimum estimators on pixels with
    ``|v| ≤`` :data:`_STRONG_FIT_VELOCITY_CAP_KMS_BROAD` or
    :data:`_STRONG_FIT_VELOCITY_CAP_KMS_NARROW` (same window as Hβ Voigt), each profile with
    ``c0 + c1 (λ - λ_rest)`` except the smoothed minimum. Trusted formal errors use
    |v| ≤ :data:`_STRONG_RV_TRUST_HALF_WIDTH_KMS`.
    """
    w = np.asarray(wave, float)
    f_in = np.asarray(flux_norm, float)
    ex_kms = _STRONG_LOCAL_CONT_EXCLUDE_KMS_BROAD if broad_lines else _STRONG_LOCAL_CONT_EXCLUDE_KMS_NARROW
    f = _local_continuum_divide_for_line(w, f_in, rest, ex_kms)

    v_all = config.C_KMS * (w / rest - 1.0)
    m_fit = np.isfinite(f) & np.isfinite(w)
    v_cap_kms = _STRONG_FIT_VELOCITY_CAP_KMS_BROAD if broad_lines else _STRONG_FIT_VELOCITY_CAP_KMS_NARROW
    m_cap = m_fit & (np.abs(v_all) <= v_cap_kms)
    if int(np.sum(m_cap)) < 18:
        return None

    w_fit, f_fit, v_fit = w[m_cap], f[m_cap], v_all[m_cap]

    o_sort = np.argsort(w_fit)
    v_plot_full = v_fit[o_sort]
    f_plot_full = f_fit[o_sort]

    w_lo, w_hi = float(np.min(w_fit)), float(np.max(w_fit))
    n_pix = len(f_fit)
    n_fine = max(400, n_pix * 6)
    w_fine_full = np.linspace(w_lo, w_hi, n_fine)
    v_fine_full = config.C_KMS * (w_fine_full / rest - 1.0)
    w_ref = float(rest)

    def gauss_lin(x, a, mu, s, c0, c1):
        return c0 + c1 * (x - w_ref) - a * np.exp(-0.5 * ((x - mu) / s) ** 2)

    def lorentz_lin(x, a, mu, gam, c0, c1):
        g = abs(gam) + 1e-7
        return c0 + c1 * (x - w_ref) - a * (g**2) / ((x - mu) ** 2 + g**2)

    def voigt_lin(x, amp, center, sig, gam, c0, c1):
        z = x - center
        sigp = abs(sig) + 1e-4
        gamp = abs(gam) + 1e-4
        vp = voigt_profile(z, sigp, gamp)
        vp0 = voigt_profile(0.0, sigp, gamp) + 1e-99
        return c0 + c1 * (x - w_ref) - abs(amp) * vp / vp0

    lo_mu, hi_mu = w_lo + 1e-4, w_hi - 1e-4
    if hi_mu <= lo_mu:
        return None

    max_sig_kms = 520.0 if broad_lines else 150.0
    sig_hi = max(rest * max_sig_kms / config.C_KMS, 0.18)
    sig_lo = max(rest * 10.0 / config.C_KMS, 0.07)
    gam_hi = max(sig_hi * 1.5, rest * (max_sig_kms * 1.1) / config.C_KMS)

    hint_w = 620.0 if broad_lines else 280.0
    mh = np.abs(v_fit) <= hint_w
    if np.any(mh):
        k = int(np.argmin(f_fit[mh]))
        mu0 = float(w_fit[mh][k])
        fmin = float(f_fit[mh][k])
        med_c = float(np.median(f_fit[mh]))
    else:
        k = int(np.argmin(f_fit))
        mu0 = float(w_fit[k])
        med_c = float(np.median(f_fit))

    span = max(w_hi - w_lo, 1e-6)
    c1max = 18.0 / span
    s0 = float(np.clip(0.09 * span, sig_lo, sig_hi * 0.92))
    p_lo_c, p_hi_c = np.nanpercentile(f_fit, [4, 96])
    p0g = [max(0.12, min(2.5, med_c - fmin)), mu0, s0, float(np.median(f_fit)), 0.0]
    bounds_g = (
        [0.03, lo_mu, sig_lo, max(0.05, p_lo_c - 0.55), -c1max],
        [2.8, hi_mu, sig_hi, min(2.85, p_hi_c + 0.55), c1max],
    )
    p0g = [
        float(np.clip(p0g[0], bounds_g[0][0] + 1e-4, bounds_g[1][0] - 1e-4)),
        float(np.clip(p0g[1], bounds_g[0][1] + 1e-4, bounds_g[1][1] - 1e-4)),
        float(np.clip(p0g[2], bounds_g[0][2] + 1e-4, bounds_g[1][2] - 1e-4)),
        float(np.clip(p0g[3], bounds_g[0][3] + 1e-4, bounds_g[1][3] - 1e-4)),
        float(np.clip(p0g[4], bounds_g[0][4] + 1e-5, bounds_g[1][4] - 1e-5)),
    ]
    try:
        pg, cg = curve_fit(gauss_lin, w_fit, f_fit, p0=p0g, bounds=bounds_g, maxfev=16000)
        rv_g = float(config.C_KMS * (pg[1] - rest) / rest)
        eg = float(config.C_KMS * np.sqrt(np.diag(cg))[1] / rest) if cg is not None else float("nan")
    except Exception:
        return None

    eg = _trust_rv_err(rv_g, eg, _STRONG_RV_TRUST_HALF_WIDTH_KMS)
    fg_full = gauss_lin(w_fine_full, *pg)

    p0l = [
        max(0.08, pg[0]),
        float(pg[1]),
        max(abs(float(pg[2])), sig_lo),
        float(pg[3]),
        float(pg[4]),
    ]
    bounds_l = (
        [0.03, lo_mu, sig_lo, bounds_g[0][3], -c1max],
        [2.8, hi_mu, gam_hi, bounds_g[1][3], c1max],
    )
    p0l = [
        float(np.clip(p0l[0], bounds_l[0][0] + 1e-4, bounds_l[1][0] - 1e-4)),
        float(np.clip(p0l[1], bounds_l[0][1] + 1e-4, bounds_l[1][1] - 1e-4)),
        float(np.clip(p0l[2], bounds_l[0][2] + 1e-4, bounds_l[1][2] - 1e-4)),
        float(np.clip(p0l[3], bounds_l[0][3] + 1e-4, bounds_l[1][3] - 1e-4)),
        float(np.clip(p0l[4], bounds_l[0][4] + 1e-5, bounds_l[1][4] - 1e-5)),
    ]
    rv_l, el = float("nan"), float("nan")
    fl_full = None
    try:
        pl, cl = curve_fit(lorentz_lin, w_fit, f_fit, p0=p0l, bounds=bounds_l, maxfev=20000)
        fl_full = lorentz_lin(w_fine_full, *pl)
        rv_l = float(config.C_KMS * (pl[1] - rest) / rest)
        el = float(config.C_KMS * np.sqrt(np.diag(cl))[1] / rest) if cl is not None else float("nan")
    except Exception:
        pass
    el = _trust_rv_err(rv_l, el, _STRONG_RV_TRUST_HALF_WIDTH_KMS)

    amp0 = float(max(pg[0], 0.08))
    p0v = [
        amp0,
        float(pg[1]),
        abs(float(pg[2])),
        max(abs(float(pg[2])) * 0.5, sig_lo),
        float(pg[3]),
        float(pg[4]),
    ]
    bounds_v = (
        [0.03, lo_mu, sig_lo, sig_lo * 0.5, bounds_g[0][3], -c1max],
        [2.8, hi_mu, sig_hi, sig_hi, bounds_g[1][3], c1max],
    )
    p0v = [
        float(np.clip(p0v[0], bounds_v[0][0] + 1e-4, bounds_v[1][0] - 1e-4)),
        float(np.clip(p0v[1], bounds_v[0][1] + 1e-4, bounds_v[1][1] - 1e-4)),
        float(np.clip(p0v[2], bounds_v[0][2] + 1e-4, bounds_v[1][2] - 1e-4)),
        float(np.clip(p0v[3], bounds_v[0][3] + 1e-4, bounds_v[1][3] - 1e-4)),
        float(np.clip(p0v[4], bounds_v[0][4] + 1e-4, bounds_v[1][4] - 1e-4)),
        float(np.clip(p0v[5], bounds_v[0][5] + 1e-5, bounds_v[1][5] - 1e-5)),
    ]
    rv_v, ev, fv_full = float("nan"), float("nan"), None
    try:
        pv, cv = curve_fit(voigt_lin, w_fit, f_fit, p0=p0v, bounds=bounds_v, maxfev=20000)
        fv_full = voigt_lin(w_fine_full, *pv)
        rv_v = float(config.C_KMS * (pv[1] - rest) / rest)
        ev = float(config.C_KMS * np.sqrt(np.diag(cv))[1] / rest) if cv is not None else float("nan")
    except Exception:
        pass
    ev = _trust_rv_err(rv_v, ev, _STRONG_RV_TRUST_HALF_WIDTH_KMS)

    core_kms = 52.0 if broad_lines else 16.0
    mc = np.abs(v_fit) <= core_kms
    wc, fc = w_fit[mc], f_fit[mc]
    rv_c, ec, f_core_full = float("nan"), float("nan"), None
    dl_c = rest * core_kms / config.C_KMS * 1.08
    lo_c, hi_c = max(w_lo, rest - dl_c), min(w_hi, rest + dl_c)
    sig_core_hi = min(sig_hi, 9.5 if broad_lines else 5.8)
    span_c = max(float(np.ptp(wc)), 1e-6) if len(wc) >= 2 else span
    c1max_c = 12.0 / span_c
    bounds_c_lo = [0.03, lo_c, sig_lo, max(0.05, p_lo_c - 0.45), -c1max_c]
    bounds_c_hi = [2.8, hi_c, sig_core_hi, min(2.85, p_hi_c + 0.45), c1max_c]
    if len(wc) >= 8 and lo_c < hi_c:
        mu_c = float(wc[np.argmin(fc)])
        p0c = [
            max(0.08, float(np.median(fc)) - float(np.min(fc))),
            mu_c,
            max(sig_lo, 0.06 * span_c),
            float(np.median(fc)),
            0.0,
        ]
        p0c = [
            float(np.clip(p0c[0], bounds_c_lo[0] + 1e-4, bounds_c_hi[0] - 1e-4)),
            float(np.clip(p0c[1], bounds_c_lo[1] + 1e-4, bounds_c_hi[1] - 1e-4)),
            float(np.clip(p0c[2], bounds_c_lo[2] + 1e-4, bounds_c_hi[2] - 1e-4)),
            float(np.clip(p0c[3], bounds_c_lo[3] + 1e-4, bounds_c_hi[3] - 1e-4)),
            float(np.clip(p0c[4], bounds_c_lo[4] + 1e-5, bounds_c_hi[4] - 1e-5)),
        ]
        try:
            pc, cc = curve_fit(gauss_lin, wc, fc, p0=p0c, bounds=(bounds_c_lo, bounds_c_hi), maxfev=12000)
            f_core_full = gauss_lin(w_fine_full, *pc)
            rv_c = float(config.C_KMS * (pc[1] - rest) / rest)
            ec = float(config.C_KMS * np.sqrt(np.diag(cc))[1] / rest) if cc is not None else float("nan")
        except Exception:
            pass
    ec = _trust_rv_err(rv_c, ec, _STRONG_RV_TRUST_HALF_WIDTH_KMS)

    sig_smooth_fit = max(3.0, float(n_pix) * (0.055 if broad_lines else 0.04))
    o_w = np.argsort(w_fit)
    w_s = w_fit[o_w]
    f_s = f_fit[o_w]
    v_s = v_fit[o_w]
    fs_cap = gaussian_filter1d(f_s.astype(float), sigma=float(sig_smooth_fit), mode="nearest")
    m_trust = np.abs(v_s) <= _STRONG_RV_TRUST_HALF_WIDTH_KMS
    if np.any(m_trust):
        idx = np.where(m_trust)[0]
        j = int(idx[int(np.argmin(fs_cap[m_trust]))])
        rv_sm = float(v_s[j])
    else:
        j = int(np.argmin(fs_cap))
        rv_sm = float(v_s[j])
    vs_tr = v_s[m_trust] if np.any(m_trust) else v_s
    dv_med = float(np.median(np.diff(np.sort(vs_tr)))) if len(vs_tr) > 2 else float("nan")
    err_sm = abs(dv_med) if np.isfinite(dv_med) else float("nan")
    err_sm = _trust_rv_err(rv_sm, err_sm, _STRONG_RV_TRUST_HALF_WIDTH_KMS)
    f_smooth_heavy_on_fine = np.interp(w_fine_full, w_s, fs_cap)

    v_trust = v_fit[np.abs(v_fit) <= _STRONG_RV_TRUST_HALF_WIDTH_KMS]
    f_trust = f_fit[np.abs(v_fit) <= _STRONG_RV_TRUST_HALF_WIDTH_KMS]

    return {
        "name": line_name,
        "rest": rest,
        "v_kms": v_trust,
        "flux": f_trust,
        "v_kms_plot_full": v_plot_full,
        "flux_plot_full": f_plot_full,
        "v_fine_full": v_fine_full,
        "flux_gauss_full": fg_full,
        "flux_lorentz_full": fl_full,
        "flux_voigt_full": fv_full,
        "flux_core_gauss_full": f_core_full,
        "flux_smooth_full": f_smooth_heavy_on_fine,
        "zoom_half_width_kms": float(_STRONG_ZOOM_HALF_WIDTH_KMS),
        "fit_half_width_kms": float(_STRONG_RV_TRUST_HALF_WIDTH_KMS),
        "fit_velocity_cap_kms": float(v_cap_kms),
        "rv_trust_half_width_kms": float(_STRONG_RV_TRUST_HALF_WIDTH_KMS),
        "rv_gauss_kms": rv_g,
        "err_gauss_kms": eg,
        "rv_lorentz_kms": rv_l,
        "err_lorentz_kms": el,
        "rv_voigt_kms": rv_v,
        "err_voigt_kms": ev,
        "rv_core_gauss_kms": rv_c,
        "err_core_gauss_kms": ec,
        "rv_smooth_min_kms": rv_sm,
        "err_smooth_min_kms": err_sm,
    }


def fit_balmer_line_gaussian_voigt(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    rest: float,
    line_name: str,
    broad_lines: bool = False,
) -> dict | None:
    """Backward-compatible name for :func:`fit_balmer_line_all_methods`."""
    return fit_balmer_line_all_methods(wave, flux_norm, rest, line_name, broad_lines=broad_lines)
