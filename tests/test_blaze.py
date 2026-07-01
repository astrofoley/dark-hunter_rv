"""Parameterized echelle blaze model and Hβ-order fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from darkhunter_rv import blaze


def _synthetic_blaze_profile(
    center: float = 4900.0,
    width: float = 90.0,
    power: float = 2.0,
    amplitude: float = 1.4,
    n_pix: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.linspace(center - 120.0, center + 120.0, n_pix)
    f = blaze.eval_blaze_sinc2(w, center, width, power=power, amplitude=amplitude)
    return w, f


def test_eval_blaze_sinc2_peak_at_center():
    w, f = _synthetic_blaze_profile(center=4900.0)
    i = int(np.argmax(f))
    assert abs(w[i] - 4900.0) < 1.5
    assert f[i] == pytest.approx(1.4, rel=1e-3)


def test_blaze_line_mask_excludes_hbeta_core():
    w = np.linspace(4800.0, 4920.0, 300)
    mask = blaze.blaze_line_mask(w, half_width_angstrom=22.0)
    assert not bool(mask[np.argmin(np.abs(w - blaze.HB_REST_A))])


def test_order_covers_strong_line_and_clean_orders():
    assert blaze.order_covers_strong_line(4850.0, 4875.0)
    assert not blaze.order_covers_strong_line(5195.0, 5278.0)

    spec_data = {
        28: {"wavelength": np.linspace(4816.0, 4894.0, 120), "flux": np.ones(120)},
        35: {"wavelength": np.linspace(5195.0, 5278.0, 120), "flux": np.ones(120)},
    }
    clean = blaze.list_clean_orders(spec_data, bad_orders=[])
    assert [o for o, *_ in clean] == [35]
    picked = blaze.pick_clean_order_near_wavelength(spec_data, 5250.0, bad_orders=[])
    assert picked is not None and picked[0] == 35


def test_hbeta_absorption_depth_shallow_vs_deep():
    w = np.linspace(4790.0, 4935.0, 400)
    f_flat = np.ones_like(w) * 1.2 + 0.0001 * (w - w.mean())
    depth_flat = blaze.hbeta_absorption_depth_raw(w, f_flat)
    f_deep = f_flat.copy()
    core = np.abs(w - blaze.HB_REST_A) < 12.0
    f_deep[core] *= 0.55
    depth_deep = blaze.hbeta_absorption_depth_raw(w, f_deep)
    assert depth_flat < 0.05
    assert depth_deep > 0.25


def test_fit_order_blaze_recovers_shape():
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(42)
    for amp in (1.1, 1.25, 1.35, 1.2, 1.3):
        w, f = _synthetic_blaze_profile(center=4900.0, amplitude=amp)
        noise = 1.0 + 0.01 * rng.standard_normal(w.size)
        profiles.append((w, f * noise))

    model = blaze.fit_order_blaze_from_profiles(
        profiles,
        echelle_order=42,
        rest_lines=[],
    )
    assert model is not None
    assert model.echelle_order == 42
    assert model.center_angstrom == pytest.approx(4900.0, abs=3.0)
    assert model.width_angstrom == pytest.approx(90.0, rel=0.15)
    assert model.power == pytest.approx(2.0, abs=0.4)


def test_order_blaze_model_roundtrip(tmp_path: Path):
    w, f = _synthetic_blaze_profile()
    profiles = [(w, f), (w, f * 1.05), (w, f * 0.95)]
    model = blaze.fit_order_blaze_from_profiles(profiles, echelle_order=7, rest_lines=[])
    assert model is not None
    path = tmp_path / "blaze.json"
    model.save(path)
    loaded = blaze.OrderBlazeModel.load(path)
    assert loaded.center_angstrom == model.center_angstrom
    assert loaded.wavelength_min == model.wavelength_min
    data = json.loads(path.read_text())
    assert data["model"] == "sinc2"


def test_correct_flux_removes_blaze_envelope():
    w, f = _synthetic_blaze_profile(amplitude=2.0)
    profiles = [(w, f), (w, f * 1.02), (w, f * 0.98)]
    model = blaze.fit_order_blaze_from_profiles(profiles, echelle_order=1, rest_lines=[])
    assert model is not None
    fc = model.correct_flux(w, f)
    assert float(np.nanmedian(fc)) == pytest.approx(2.0, rel=0.08)
