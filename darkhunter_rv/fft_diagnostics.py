"""
Offline diagnostics for template-FFT failure modes (alias peaks, coarse subbank, shallow envelope).

Intended for dissecting discrepancies vs stellar-mask RV without changing the production pipeline
call graph; ``summarize_fft_chunk_failure_modes`` is safe to call from notebooks or
``validation/diagnose_template_fft_star.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from . import rv_core


def _envelope_peak_stats(vel: np.ndarray, y: np.ndarray, *, prominence_frac: float = 0.08) -> dict[str, Any]:
    """Local maxima on a 1D correlation curve (max or median across templates)."""
    v = np.asarray(vel, float)
    y = np.asarray(y, float)
    if len(y) < 16:
        return {"n_peaks": 0, "peak_velocities": [], "peak_heights": [], "second_to_first_ratio": float("nan")}
    y = np.nan_to_num(y, nan=-1e9)
    prom = max(float(prominence_frac) * (float(np.max(y)) - float(np.median(y)) + 1e-12), 1e-9)
    peaks, props = find_peaks(y, prominence=prom)
    if peaks.size == 0:
        j = int(np.argmax(y))
        return {
            "n_peaks": 1,
            "peak_velocities": [float(v[j])],
            "peak_heights": [float(y[j])],
            "second_to_first_ratio": float("nan"),
        }
    order = np.argsort(y[peaks])[::-1]
    pk = peaks[order]
    heights = y[pk].astype(float)
    vels = v[pk].astype(float)
    ratio = float(heights[1] / heights[0]) if heights.size > 1 else float("nan")
    return {
        "n_peaks": int(peaks.size),
        "peak_velocities": [float(x) for x in vels[:5]],
        "peak_heights": [float(x) for x in heights[:5]],
        "second_to_first_ratio": ratio,
    }


def summarize_fft_chunk_failure_modes(
    obs_wave: np.ndarray,
    obs_flux: np.ndarray,
    bank: dict,
    vsini_proxy: float,
    *,
    rv_truth_kms: float | None = None,
    fft_coarse_top_k: int | None = None,
) -> dict[str, Any]:
    """
    Compare estimators and envelope structure for one continuum-normalized chunk.

    Parameters
    ----------
    rv_truth_kms
        Reference velocity (e.g. mask CCF RV in the **same frame** as ``estimate_rv_fft_with_ccf``
        output: barycentric corrections must match how the pipeline calls the estimator).

    Returns
    -------
    dict
        Scalar diagnostics plus ``failure_mode_hints``: strings naming likely issues.
    """
    out: dict[str, Any] = {}
    hints: list[str] = []

    def _est(**kw):
        return rv_core.estimate_rv_fft_with_ccf(
            obs_wave,
            obs_flux,
            bank,
            vsini_proxy,
            fft_coarse_top_k=fft_coarse_top_k,
            **kw,
        )

    r_tp_max, _, _, _ = _est(fft_two_phase=True, fft_peak_pick="per_template_max")
    r_tp_med, _, _, _ = _est(fft_two_phase=True, fft_peak_pick="aggregate_median")
    r_full_max, _, _, _ = _est(fft_two_phase=False, fft_peak_pick="per_template_max")
    r_full_med, _, _, _ = _est(fft_two_phase=False, fft_peak_pick="aggregate_median")

    out["rv_two_phase_per_template_max_kms"] = float(r_tp_max) if np.isfinite(r_tp_max) else float("nan")
    out["rv_two_phase_aggregate_median_kms"] = float(r_tp_med) if np.isfinite(r_tp_med) else float("nan")
    out["rv_full_bank_per_template_max_kms"] = float(r_full_max) if np.isfinite(r_full_max) else float("nan")
    out["rv_full_bank_aggregate_median_kms"] = float(r_full_med) if np.isfinite(r_full_med) else float("nan")

    if np.isfinite(r_tp_max) and np.isfinite(r_full_max) and abs(r_tp_max - r_full_max) > 25.0:
        hints.append("two_phase_subbank: two-phase and full-bank per_template_max disagree (>25 km/s)")

    if np.isfinite(r_tp_max) and np.isfinite(r_tp_med) and abs(r_tp_max - r_tp_med) > 40.0:
        hints.append(
            "outlier_template_peak: per-template max disagrees strongly with median-across-templates peak"
        )

    vel_win, _keys, mat = rv_core.template_fft_ccf_stack(obs_wave, obs_flux, bank)
    if mat.size == 0:
        hints.append("empty_ccf_stack: no template produced a CCF on this chunk")
        out["failure_mode_hints"] = hints
        return out

    env_max = np.nanmax(mat, axis=0)
    env_med = np.nanmedian(mat, axis=0)
    st_max = _envelope_peak_stats(vel_win, env_max)
    st_med = _envelope_peak_stats(vel_win, env_med)
    out["envelope_max_ccf"] = st_max
    out["envelope_median_ccf"] = st_med

    if st_max["n_peaks"] >= 2 and st_max["second_to_first_ratio"] > 0.85:
        hints.append("multiple_similar_peaks: envelope max has competing peaks (alias risk)")

    med_peak_h = st_med["peak_heights"][0] if st_med["peak_heights"] else float("nan")
    med_floor = float(np.median(env_med))
    if np.isfinite(med_peak_h) and (med_peak_h - med_floor) < 0.02:
        hints.append("shallow_median_envelope: median CCF peak barely above floor")

    if rv_truth_kms is not None and np.isfinite(float(rv_truth_kms)):
        rt = float(rv_truth_kms)
        j = int(np.argmin(np.abs(vel_win - rt)))
        out["ccf_median_at_truth"] = float(env_med[j])
        out["ccf_max_template_at_truth"] = float(env_max[j])
        j_mx = int(np.argmax(env_max))
        out["delta_v_truth_to_env_max_kms"] = float(vel_win[j_mx] - rt)
        if np.isfinite(r_tp_max) and abs(r_tp_max - rt) > 50 and out["ccf_median_at_truth"] > med_floor + 0.03:
            hints.append(
                "truth_on_median_ridge: large per_template_max error but median CCF elevated near truth "
                "(median aggregation or better template subset may fix)"
            )

    out["failure_mode_hints"] = hints
    return out
