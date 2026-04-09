"""Hβ-only RV (windowed continuum + Voigt + optional template CCF)."""

from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv import config, plotting, rv_core


def test_hb_scale_template_to_local_cont_matches_wings():
    v = np.linspace(-400.0, 400.0, 81)
    f_obs = np.ones_like(v) * 1.05
    f_tpl = np.ones_like(v) * 0.35
    out = plotting._hb_scale_template_to_local_cont(v, f_obs, f_tpl)
    np.testing.assert_allclose(out, f_obs, rtol=0.05, atol=0.02)


def test_measure_h_beta_rv_synthetic_broad_line():
    rest = rv_core.HB_REST_A
    w = np.linspace(rest - 95.0, rest + 95.0, 500)
    true_rv = 18.0
    w_line = rest * (1.0 + true_rv / config.C_KMS)
    cont = 1.02 + 2e-4 * (w - w.mean())
    line = 0.42 * np.exp(-0.5 * ((w - w_line) / 2.1) ** 2)
    f = cont - line
    out = rv_core.measure_h_beta_rv(w, f, broad_lines=True)
    assert out is not None
    assert abs(float(out["rv_best_kms"]) - true_rv) < 22.0
    assert np.isfinite(out["rv_smoothed_min_kms"])
    assert np.isfinite(float(out["rv_voigt_kms"]))


def test_measure_h_beta_rv_outside_window_returns_none():
    w = np.linspace(4000.0, 4100.0, 80)
    f = np.ones_like(w)
    assert rv_core.measure_h_beta_rv(w, f, broad_lines=False) is None


def test_h_beta_joint_reconstruction_matches_stored_fine_model():
    """Plot/file model must match h_beta_joint_line_model(params) on voigt_wave_fine."""
    rest = rv_core.HB_REST_A
    w = np.linspace(rest - 95.0, rest + 95.0, 500)
    true_rv = 18.0
    w_line = rest * (1.0 + true_rv / config.C_KMS)
    cont = 1.02 + 2e-4 * (w - w.mean())
    line = 0.42 * np.exp(-0.5 * ((w - w_line) / 2.1) ** 2)
    f = cont - line
    out = rv_core.measure_h_beta_rv(w, f, broad_lines=True)
    assert out is not None
    p = out.get("hb_joint_fit_params")
    if p is None or len(p) != 8:
        pytest.skip("joint fit did not return 8 parameters")
    wf = np.asarray(out["voigt_wave_fine"], dtype=float)
    f_stored = np.asarray(out["voigt_model_fine"], dtype=float)
    assert wf.size > 5 and f_stored.shape == wf.shape
    f_calc = rv_core.h_beta_joint_line_model(wf, p, rest=rest)
    np.testing.assert_allclose(f_calc, f_stored, rtol=0.0, atol=1e-9)
    ctr = float(p[2])
    rv_from_ctr = float(config.C_KMS * (ctr - rest) / rest)
    assert abs(rv_from_ctr - float(out["rv_voigt_kms"])) < 1e-5


def test_plot_h_beta_rv_diagnostic_smoke(tmp_path: Path) -> None:
    """Mask/template/V+L overlays and black step data render without error."""
    rest = rv_core.HB_REST_A
    v = np.linspace(-400.0, 400.0, 120)
    w = rest * (1.0 + v / config.C_KMS)
    f = 1.0 - 0.28 * np.exp(-0.5 * (v / 55.0) ** 2)
    wf = np.linspace(float(w.min()), float(w.max()), 200)
    p8 = [0.25, 0.06, rest * (1.0 + 12.0 / config.C_KMS), 0.55, 0.35, 0.9, 1.0, 1e-5]
    hb = {
        "v_kms_plot": v,
        "flux_plot": f,
        "wavelength_plot": w,
        "rest_a": rest,
        "voigt_wave_fine": wf,
        "hb_joint_fit_params": p8,
        "rv_voigt_kms": 12.0,
        "err_voigt_kms": 4.0,
        "rv_best_kms": 10.0,
        "method_used": "test",
    }
    mw = np.array([rest - 1.2, rest + 1.5], dtype=float)
    ms = np.array([1.0, 1.0], dtype=float)
    out_png = tmp_path / "hb_diag.png"
    plotting.plot_h_beta_rv_diagnostic(
        "smoke",
        hb,
        out_png,
        order_num=9,
        rv_mask_kms=8.0,
        err_mask_kms=2.5,
        mask_wave=mw,
        mask_strength=ms,
        tpl_wave=w,
        tpl_flux_norm=f,
        rv_template_kms=11.0,
        err_template_kms=3.0,
        resolving_power=60_000.0,
    )
    assert out_png.is_file()
