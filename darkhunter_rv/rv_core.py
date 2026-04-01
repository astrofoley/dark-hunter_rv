# rv_core.py
import numpy as np
from scipy.optimize import curve_fit
from . import config

def cross_correlate_stellar_mask(obs_wave, obs_flux, mask_wave, mask_strength, max_lag=350, fit_width=50):
    log_obs = np.log10(obs_wave)
    log_mask = np.log10(mask_wave)
    
    median_diff = np.median(np.diff(log_obs)) / 5.0
    log_shifts = np.arange(-max_lag, max_lag + 1) * median_diff
    ccf = np.zeros_like(log_shifts)

    vel_shifts = config.C_KMS * (10**log_shifts - 1)
    
    # Summation
    for i, shift in enumerate(log_shifts):
        shifted_mask = log_mask + shift
        idx = np.searchsorted(log_obs, shifted_mask)
        valid = (idx > 0) & (idx < len(log_obs))
        if np.sum(valid) < 5: continue
        ccf[i] = np.sum(obs_flux[idx[valid]] * mask_strength[valid])
        
    if np.max(ccf) <= 0:
        return np.nan, np.nan, vel_shifts, ccf, 0.0
    
    peak_idx = np.argmax(ccf)
    peak_val = ccf[peak_idx] # Metric for Tournament
    
    sl = slice(max(0, peak_idx - fit_width), min(len(ccf), peak_idx + fit_width + 1))
    x_fit, y_fit = vel_shifts[sl], ccf[sl]
    
    if len(x_fit) < 4: return vel_shifts[peak_idx], np.nan, vel_shifts, ccf, peak_val

    try:
        def gauss(x, a, mu, sig): return a * np.exp(-0.5*((x-mu)/sig)**2)
        p0 = [np.max(y_fit), vel_shifts[peak_idx], 10.0]
        popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=p0, maxfev=2000)
        
        rv = popt[1]
        perr = np.sqrt(np.diag(pcov))
        # Always report covariance-based uncertainty when fit converges;
        # downstream code can decide whether to accept based on diagnostics.
        rv_err = float(perr[1]) if np.isfinite(perr[1]) else np.nan
    except Exception:
        rv = vel_shifts[peak_idx]
        rv_err = np.nan
        
    return rv, rv_err, vel_shifts, ccf, peak_val

def estimate_rv_fft_vectorized(obs_wave, obs_flux, templates, vsini_proxy, plot=False):
    loglam_obs = np.log10(obs_wave)
    npts = max(2 ** int(np.ceil(np.log2(len(obs_wave)))), 512)
    log_grid = np.linspace(loglam_obs.min(), loglam_obs.max(), npts)
    
    obs_resamp = np.interp(log_grid, loglam_obs, obs_flux)
    obs_resamp = (obs_resamp - np.nanmean(obs_resamp)) / (np.nanstd(obs_resamp) + 1e-9)
    window = np.hanning(len(obs_resamp))
    fft_obs = np.fft.fft(obs_resamp * window)
    
    delta_lnlam = (log_grid[1] - log_grid[0]) * np.log(10.0)
    dv_pix = config.C_KMS * delta_lnlam
    vel_axis = (np.arange(npts) - npts//2) * dv_pix
    mask_vel = (vel_axis >= -1000) & (vel_axis <= 1000)
    
    best_peak, best_rv, best_tpl_key = -np.inf, 0.0, None
    tpl_grid_wave = 10**log_grid

    for key, (t_wave, t_flux) in templates.items():
        if t_wave[-1] < tpl_grid_wave[0] or t_wave[0] > tpl_grid_wave[-1]: continue
        t_resamp = np.interp(tpl_grid_wave, t_wave, t_flux, left=np.nan, right=np.nan)
        t_resamp = 1.0 - t_resamp # Abs Positive
        valid = np.isfinite(t_resamp)
        if np.sum(valid) < 10: continue
        t_resamp[~valid] = 0.0
        t_resamp -= np.nanmean(t_resamp)
        fft_tpl = np.fft.fft(t_resamp * window)
        
        ccf = np.fft.ifft(fft_obs * np.conj(fft_tpl)).real
        ccf = np.fft.fftshift(ccf)
        ccf_win = ccf[mask_vel]
        peak = np.max(ccf_win)
        
        if peak > best_peak:
            best_peak = peak
            idx = np.argmax(ccf_win)
            best_rv = vel_axis[mask_vel][idx]
            best_tpl_key = key
            
    return best_rv, best_tpl_key

def estimate_broadening(obs_wave, obs_flux, tpl_wave, tpl_flux):
    log_min = max(np.log10(obs_wave.min()), np.log10(tpl_wave.min()))
    log_max = min(np.log10(obs_wave.max()), np.log10(tpl_wave.max()))
    if log_max <= log_min: return None, {}
    
    log_grid = np.linspace(log_min, log_max, 2048)
    obs_r = np.interp(log_grid, np.log10(obs_wave), obs_flux)
    tpl_r = np.interp(log_grid, np.log10(tpl_wave), tpl_flux)
    obs_r -= np.mean(obs_r); tpl_r -= np.mean(tpl_r)
    
    ccf_dt = np.fft.fftshift(np.fft.ifft(np.fft.fft(obs_r)*np.conj(np.fft.fft(tpl_r))).real)
    ccf_tt = np.fft.fftshift(np.fft.ifft(np.fft.fft(tpl_r)*np.conj(np.fft.fft(tpl_r))).real)
    
    def get_width(y):
        x = np.arange(len(y))
        mu = np.average(x, weights=np.abs(y))
        return np.sqrt(np.average((x-mu)**2, weights=np.abs(y)))
        
    sig_dt = get_width(ccf_dt[2048//2-50:2048//2+50])
    sig_tt = get_width(ccf_tt[2048//2-50:2048//2+50])
    
    if sig_dt > sig_tt:
        pix_broad = np.sqrt(sig_dt**2 - sig_tt**2)
        dv = config.C_KMS * (log_grid[1]-log_grid[0]) * np.log(10)
        return pix_broad * dv * 2.355, {}
    return 10.0, {}

def measure_strong_line_centroids(wave, flux):
    lines = {"Ha": 6562.8, "Hb": 4861.3, "Hg": 4340.5, "Hd": 4101.7}
    rvs, errs = [], []
    
    for name, rest in lines.items():
        mask = (wave > rest - 20) & (wave < rest + 20)
        w, f = wave[mask], flux[mask]
        if len(w) < 10: continue
        try:
            # Inverted Gauss
            p0 = [1.0-np.min(f), w[np.argmin(f)], 2.0, 1.0]
            def model(x, a, mu, s, c): return c - a*np.exp(-0.5*((x-mu)/s)**2)
            popt, pcov = curve_fit(model, w, f, p0=p0, maxfev=2000)
            
            rv = config.C_KMS * (popt[1] - rest) / rest
            err = config.C_KMS * np.sqrt(np.diag(pcov))[1] / rest
            if popt[0] > 0.05: # Depth check
                rvs.append(rv); errs.append(err)
        except: continue
        
    if not rvs: return np.nan, np.nan
    weights = 1.0 / (np.array(errs)**2 + 1e-9)
    mean_rv = np.average(rvs, weights=weights)
    mean_err = np.sqrt(1.0/np.sum(weights))
    return mean_rv, mean_err
