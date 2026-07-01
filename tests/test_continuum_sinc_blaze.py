"""Sinc² blaze calibration applied before spline normalization."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv import blaze, continuum


def _mock_blaze_order(center: float = 5000.0, width: float = 80.0) -> blaze.OrderBlazeModel:
    w = np.linspace(center - 100, center + 100, 120)
    f = blaze.eval_blaze_sinc2(w, center, width, power=2.0, amplitude=2.0)
    profiles = [(w, f * a) for a in (1.0, 1.05, 0.98)]
    model = blaze.fit_order_blaze_from_profiles(profiles, echelle_order=12, rest_lines=[])
    assert model is not None
    return model


def test_sinc_blaze_only_flattens_median():
    model = _mock_blaze_order()
    center, width = model.center_angstrom, model.width_angstrom
    w = np.linspace(center - 90, center + 90, 150)
    raw = blaze.eval_blaze_sinc2(w, center, width, amplitude=3.2)
    nw, nf, _ = continuum.fit_continuum(
        w,
        raw,
        np.ones_like(raw),
        continuum_mode="sinc_blaze_only",
        blaze_model=model,
    )
    assert float(np.nanmedian(nf)) == pytest.approx(1.0, abs=0.05)


def test_sinc_blaze_applies_before_spline():
    model = _mock_blaze_order(center=5100.0)
    w = np.linspace(5010, 5190, 160)
    raw = blaze.eval_blaze_sinc2(w, model.center_angstrom, model.width_angstrom, amplitude=2.5)
    nw, nf, _ = continuum.fit_continuum(
        w,
        raw,
        np.ones_like(raw),
        continuum_mode="sinc_blaze",
        blaze_model=model,
    )
    assert np.nanmedian(nf) == pytest.approx(1.0, abs=0.15)
