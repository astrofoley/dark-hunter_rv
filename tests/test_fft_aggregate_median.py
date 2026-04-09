"""Median-across-templates FFT peak picking (mask-independent alias control)."""

import numpy as np

from darkhunter_rv import rv_core


def test_median_aggregation_picks_majority_peak():
    """
    Mirrors ``_estimate_rv_fft_best_in_bank(..., peak_pick='aggregate_median')``: one template's CCF
    has a much higher spurious peak, but the median-across-templates curve still peaks with the majority.
    """
    vel = np.linspace(-400.0, 400.0, 801)
    mat = np.zeros((10, len(vel)))
    for i in range(9):
        mat[i] = np.exp(-0.5 * ((vel - 18.0) / 30.0) ** 2)
    mat[9] = 1.5 * np.exp(-0.5 * ((vel - 260.0) / 25.0) ** 2)
    agg = np.nanmedian(mat, axis=0)
    rv_med = float(vel[int(np.nanargmax(agg))])
    assert abs(rv_med - 18.0) < 12.0
    worst_i = int(np.argmax(np.max(mat, axis=1)))
    assert worst_i == 9
    rv_outlier = float(vel[int(np.argmax(mat[9]))])
    assert abs(rv_outlier - 260.0) < 15.0


def test_aggregate_median_full_fft_matches_per_template_for_single_template_bank():
    wave = np.linspace(6000.0, 6100.0, 512)
    rng = np.random.default_rng(42)
    obs = 1.0 - 0.08 * np.exp(-0.5 * ((wave - 6050.0) / 1.35) ** 2) + rng.normal(0, 0.002, wave.shape)
    tw = wave.copy()
    tpl = 1.0 - 0.09 * np.exp(-0.5 * ((wave - 6050.0) / 1.3) ** 2)
    bank = {(5800.0, 4.5, 0.0, 10.0): (tw, tpl)}
    r_max, _, _, _ = rv_core.estimate_rv_fft_with_ccf(
        wave, obs, bank, 10.0, fft_two_phase=False, fft_peak_pick="per_template_max"
    )
    r_med, _, _, _ = rv_core.estimate_rv_fft_with_ccf(
        wave, obs, bank, 10.0, fft_two_phase=False, fft_peak_pick="aggregate_median"
    )
    assert np.isfinite(r_max) and np.isfinite(r_med)
    assert abs(r_max - r_med) < 2.0


def test_template_fft_ccf_stack_shape_matches_bank():
    w = np.linspace(6000.0, 6100.0, 256)
    obs = 1.0 - 0.05 * np.exp(-0.5 * ((w - 6050.0) / 1.5) ** 2)
    tw = w.copy()
    tpl = 1.0 - 0.06 * np.exp(-0.5 * ((w - 6050.0) / 1.4) ** 2)
    bank = {(5800.0, 4.5, 0.0, 10.0): (tw, tpl), (6000.0, 4.5, 0.0, 10.0): (tw, tpl * 0.98)}
    vel, keys, mat = rv_core.template_fft_ccf_stack(w, obs, bank)
    assert mat.shape[0] == len(keys) == 2
    assert mat.shape[1] == len(vel)


def test_fft_diagnostics_hints_on_outlier_bank():
    from darkhunter_rv import fft_diagnostics

    rng = np.random.default_rng(1)
    wave = np.linspace(6000.0, 6100.0, 900)
    obs = 1.0 - 0.08 * np.exp(-0.5 * ((wave - 6050.0) / 1.35) ** 2) + rng.normal(0, 0.002, wave.shape)
    tpl_good = 1.0 - 0.09 * np.exp(-0.5 * ((wave - 6050.0) / 1.3) ** 2)
    tpl_out = 1.0 - 0.14 * np.exp(-0.5 * ((wave - 6038.0) / 1.1) ** 2)
    bank = {}
    for i in range(9):
        bank[(5000.0 + 50 * i, 4.5, 0.0, 10.0)] = (wave.copy(), tpl_good)
    bank[(6500.0, 4.5, 0.0, 10.0)] = (wave.copy(), tpl_out)

    d = fft_diagnostics.summarize_fft_chunk_failure_modes(wave, obs, bank, 10.0, rv_truth_kms=0.0)
    assert "failure_mode_hints" in d
    assert abs(d["rv_full_bank_aggregate_median_kms"]) < 40.0
