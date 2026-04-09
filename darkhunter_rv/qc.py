"""Quality-control metrics and filtering for order/chunk RVs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# Broad optical telluric bands (A/B + strong H2O bands) in Angstrom
TELLURIC_BANDS = [
    (6270.0, 6330.0),
    (6860.0, 6970.0),
    (7160.0, 7340.0),
    (7580.0, 7710.0),
    (8100.0, 8400.0),
]

# Interstellar / circumstellar features that can bias RV if treated as photospheric (air Å).
ISM_RV_EXCLUDE_BANDS = [
    (5888.5, 5898.0),  # Na D (often ISM-dominated)
]


def rv_contamination_bands() -> list[Tuple[float, float]]:
    """Merged telluric + ISM regions excluded from mask CCF sums, template FFT, and contamination QC."""
    return list(TELLURIC_BANDS) + list(ISM_RV_EXCLUDE_BANDS)


def wavelength_band_mask(wave: np.ndarray, bands: list[Tuple[float, float]]) -> np.ndarray:
    w = np.asarray(wave, float)
    m = np.zeros(w.shape[0], dtype=bool)
    for lo, hi in bands:
        m |= (w >= float(lo)) & (w <= float(hi))
    return m

DEFAULT_QC = {
    "global": {
        "max_chunk_err_kms": 50.0,
        "min_mask_line_count": 8,
        "max_telluric_fraction": 0.25,
        "max_ccf_asymmetry": 0.35,
        "min_ccf_peak": 0.02,
        "min_ccf_peak_snr": 2.75,
        # Template FFT: reject chunk if RSS of Gaussian+baseline is not well below RSS of a flat mean.
        "max_fft_ccf_flat_rss_ratio": 0.88,
    },
    "instruments": {
        "APF": {
            "max_chunk_err_kms": 25.0,
            "min_mask_line_count": 8,
            "max_telluric_fraction": 0.22,
            "max_ccf_asymmetry": 0.42,
            "min_ccf_peak": 0.03,
            "min_ccf_peak_snr": 2.75,
            "max_fft_ccf_flat_rss_ratio": 0.88,
        }
    },
}


def ensure_qc_config(path: Path) -> None:
    path = Path(path)
    if path.exists() or yaml is None:
        return
    path.write_text(yaml.safe_dump(DEFAULT_QC, sort_keys=False))


def load_qc_config(path: Path, instrument: str) -> Dict:
    cfg = DEFAULT_QC.copy()
    p = Path(path)
    if p.exists() and yaml is not None:
        loaded = yaml.safe_load(p.read_text()) or {}
        if "global" in loaded:
            cfg["global"] = {**cfg["global"], **loaded.get("global", {})}
        if "instruments" in loaded:
            merged_inst = dict(cfg.get("instruments", {}))
            for k, v in loaded.get("instruments", {}).items():
                base = merged_inst.get(k, {})
                merged_inst[k] = {**base, **(v or {})}
            cfg["instruments"] = merged_inst
    inst = cfg.get("instruments", {}).get(instrument, {})
    return {**cfg.get("global", {}), **inst}


def telluric_fraction(wave: np.ndarray) -> float:
    """Fraction of pixels in telluric + ISM RV-exclusion bands (chunk contamination metric)."""
    if len(wave) == 0:
        return 1.0
    m = wavelength_band_mask(wave, rv_contamination_bands())
    return float(np.mean(m))


def mask_line_count_in_chunk(wave: np.ndarray, mask_wave: np.ndarray | None) -> int:
    if mask_wave is None or len(wave) == 0:
        return 0
    lo, hi = np.nanmin(wave), np.nanmax(wave)
    return int(np.sum((mask_wave >= lo) & (mask_wave <= hi)))


def ccf_shape_metrics(vel: np.ndarray | None, ccf: np.ndarray | None) -> Tuple[float, float]:
    """Return (width_kms, asymmetry). Uses median baseline and positive excess only so narrow
    negative CCF artifacts (e.g. CRs) do not dominate the normalization."""
    if vel is None or ccf is None or len(vel) < 5:
        return np.nan, np.nan
    y_raw = np.array(ccf, float)
    x = np.array(vel, float)
    baseline = float(np.nanmedian(y_raw))
    y = y_raw - baseline
    y = np.maximum(y, 0.0)
    if not np.isfinite(y).all() or np.nanmax(y) <= 0:
        return np.nan, np.nan
    y = y / (np.nanmax(y) + 1e-12)
    peak = int(np.nanargmax(y))
    half = 0.5

    left = np.where(y[:peak] <= half)[0]
    right = np.where(y[peak:] <= half)[0]
    if len(left) == 0 or len(right) == 0:
        width = np.nan
    else:
        i1 = left[-1]
        i2 = peak + right[0]
        width = float(abs(x[i2] - x[i1]))

    # asymmetry proxy: compare integrated area left/right of peak
    yl = y[:peak]
    yr = y[peak + 1 :]
    if len(yl) < 3 or len(yr) < 3:
        asym = np.nan
    else:
        al = float(np.trapz(yl, x[:peak]))
        ar = float(np.trapz(yr, x[peak + 1 :]))
        asym = float(abs(al - ar) / (abs(al) + abs(ar) + 1e-12))
    return width, asym


def _fft_ccf_gaussian_bump(x: np.ndarray, c: float, a: float, mu: float, sig: float) -> np.ndarray:
    sigp = abs(float(sig)) + 1e-6
    return c + a * np.exp(-0.5 * ((x - float(mu)) / sigp) ** 2)


def fft_ccf_passes_vs_flat(
    vel: np.ndarray | None,
    ccf: np.ndarray | None,
    *,
    max_rss_ratio: float = 0.88,
    min_points: int = 24,
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Aggressive template-FFT chunk gate: compare residual sum of squares (RSS) for a constant mean
    versus a 4-parameter Gaussian bump + offset on the CCF. If the bump does not reduce RSS by at
    least ``(1 - max_rss_ratio)``, treat the peak as noise-dominated and reject the chunk.
    """
    metrics: Dict[str, float] = {
        "fft_ccf_rss_flat": float("nan"),
        "fft_ccf_rss_gauss": float("nan"),
        "fft_ccf_rss_ratio": float("nan"),
    }
    if vel is None or ccf is None:
        return False, "no_ccf", metrics
    x = np.asarray(vel, dtype=float).ravel()
    y = np.asarray(ccf, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) != len(y) or len(x) < min_points:
        return False, "too_few_ccf_points", metrics

    o = np.argsort(x)
    x, y = x[o], y[o]
    n = len(y)
    mean_y = float(np.mean(y))
    rss0 = float(np.sum((y - mean_y) ** 2))
    metrics["fft_ccf_rss_flat"] = rss0
    if rss0 <= 0 or not np.isfinite(rss0):
        return False, "bad_rss_flat", metrics

    i_peak = int(np.argmax(y))
    mu0 = float(x[i_peak])
    c0 = float(np.median(y))
    a0 = float(max(y[i_peak] - c0, (np.max(y) - np.min(y)) * 0.05 + 1e-9))
    span = float(np.ptp(x)) + 1e-9
    sig0 = max(span * 0.06, float(np.median(np.diff(x))) * 3.0)
    p0 = [c0, a0, mu0, sig0]
    lo_x, hi_x = float(x[0]), float(x[-1])
    bounds = (
        [np.min(y) - abs(a0), -abs(a0) * 25.0, lo_x - 0.25 * span, span * 1e-4],
        [np.max(y) + abs(a0), abs(a0) * 25.0, hi_x + 0.25 * span, span * 0.55],
    )
    try:
        popt, _pcov = curve_fit(
            _fft_ccf_gaussian_bump,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        y_m = _fft_ccf_gaussian_bump(x, *popt)
        rss1 = float(np.sum((y - y_m) ** 2))
    except Exception:
        return False, "ccf_gauss_fit_failed", metrics

    metrics["fft_ccf_rss_gauss"] = rss1
    ratio = rss1 / rss0 if rss0 > 0 else float("nan")
    metrics["fft_ccf_rss_ratio"] = ratio
    if not np.isfinite(ratio) or ratio >= max_rss_ratio:
        return False, f"ccf_flat_like(rss_ratio={ratio:.3f})", metrics
    return True, "ok", metrics


def evaluate_chunk_qc(metrics: Dict, thresholds: Dict) -> Tuple[bool, str]:
    reasons = []
    if np.isfinite(metrics.get("rv_err_kms", np.nan)) and metrics["rv_err_kms"] > thresholds["max_chunk_err_kms"]:
        reasons.append("high_err")
    if metrics.get("mask_line_count", 0) < thresholds["min_mask_line_count"]:
        reasons.append("few_mask_lines")
    if metrics.get("telluric_fraction", 0.0) > thresholds["max_telluric_fraction"]:
        reasons.append("telluric_heavy")
    if np.isfinite(metrics.get("ccf_asymmetry", np.nan)) and metrics["ccf_asymmetry"] > thresholds["max_ccf_asymmetry"]:
        reasons.append("ccf_asymmetric")
    if np.isfinite(metrics.get("ccf_peak", np.nan)) and metrics["ccf_peak"] < thresholds["min_ccf_peak"]:
        reasons.append("weak_ccf")
    snr = metrics.get("ccf_peak_snr", np.nan)
    thr_snr = thresholds.get("min_ccf_peak_snr", np.nan)
    if np.isfinite(snr) and np.isfinite(thr_snr) and snr < thr_snr:
        reasons.append("weak_ccf_snr")
    ok = len(reasons) == 0
    return ok, ";".join(reasons) if reasons else "ok"
