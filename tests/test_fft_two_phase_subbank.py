"""PHOENIX bank sub-sampling for two-phase template FFT."""

import numpy as np
import pytest

from darkhunter_rv import config, rv_core
from darkhunter_rv.templates import (
    coarse_fft_subbank,
    max_vsini_variants_per_atmosphere,
    narrow_fft_subbank,
    refined_fft_subbank,
    template_key_stellar_tuple,
)


def test_template_key_stellar_tuple():
    assert template_key_stellar_tuple((5800.0, 4.5, 0.0, 10.0)) == (5800.0, 4.5, 0.0)
    assert template_key_stellar_tuple("bad") is None


def test_coarse_fft_subbank_picks_closest_vsini():
    w = np.linspace(5000, 5100, 50)
    f = np.ones(50)
    bank = {
        (5800.0, 4.5, 0.0, 5.0): (w, f),
        (5800.0, 4.5, 0.0, 20.0): (w, f * 2),
        (6000.0, 4.5, 0.0, 10.0): (w, f * 3),
    }
    c = coarse_fft_subbank(bank, 18.0)
    assert len(c) == 2
    assert (5800.0, 4.5, 0.0, 20.0) in c
    assert (6000.0, 4.5, 0.0, 10.0) in c


def test_narrow_fft_subbank_picks_minimum_vsini():
    w = np.linspace(5000, 5100, 50)
    f = np.ones(50)
    bank = {
        (5800.0, 4.5, 0.0, 5.0): (w, f),
        (5800.0, 4.5, 0.0, 200.0): (w, f * 2),
        (6000.0, 4.5, 0.0, 10.0): (w, f * 3),
    }
    n = narrow_fft_subbank(bank)
    assert len(n) == 2
    assert (5800.0, 4.5, 0.0, 5.0) in n
    assert (6000.0, 4.5, 0.0, 10.0) in n


def test_max_vsini_variants_per_atmosphere():
    w, f = np.linspace(5000, 5100, 10), np.ones(10)
    bank = {
        (5800.0, 4.5, 0.0, 5.0): (w, f),
        (5800.0, 4.5, 0.0, 10.0): (w, f),
        (5800.0, 4.5, 0.0, 15.0): (w, f),
        (6000.0, 4.5, 0.0, 10.0): (w, f),
    }
    assert max_vsini_variants_per_atmosphere(bank) == 3


def test_refined_fft_subbank():
    w, f = np.linspace(5000, 5100, 10), np.ones(10)
    bank = {
        (5800.0, 4.5, 0.0, 5.0): (w, f),
        (5800.0, 4.5, 0.0, 15.0): (w, f * 2),
        (6000.0, 4.5, 0.0, 10.0): (w, f),
    }
    r = refined_fft_subbank(bank, {(5800.0, 4.5, 0.0)})
    assert len(r) == 2


@pytest.mark.parametrize("two_phase", [True, False])
def test_estimate_rv_fft_two_phase_matches_single_for_small_bank(two_phase):
    """Small banks skip two-phase; toggling should not change outcome."""
    rng = np.random.default_rng(0)
    wave = np.linspace(6000.0, 6100.0, 1024)
    obs = 1.0 - 0.08 * np.exp(-0.5 * ((wave - 6050.0) / 1.2) ** 2) + rng.normal(0, 0.002, wave.shape)
    tw = wave.copy()
    tpl = 1.0 - 0.09 * np.exp(-0.5 * ((tw - 6050.0) / 1.1) ** 2)
    bank = {(5800.0, 4.5, 0.0, 10.0): (tw, tpl)}
    r1, k1, v1, c1 = rv_core.estimate_rv_fft_with_ccf(
        wave, obs, bank, 10.0, fft_two_phase=two_phase
    )
    r2, k2, v2, c2 = rv_core.estimate_rv_fft_with_ccf(
        wave, obs, bank, 10.0, fft_two_phase=not two_phase
    )
    assert k1 == k2
    assert np.isfinite(r1) and np.isfinite(r2)
    assert abs(r1 - r2) < 1.0


def test_two_phase_large_bank_does_not_exceed_single_scan_count():
    """Heuristic: two-phase estimated cost is below full bank size when enabled."""
    w = np.linspace(6000.0, 6100.0, 512)
    f = np.ones_like(w)
    bank = {}
    t_list = [5000 + 200 * i for i in range(20)]
    vb_list = [0.0, 10.0, 25.0]
    for t in t_list:
        for vb in vb_list:
            bank[(float(t), 4.5, 0.0, float(vb))] = (w.copy(), f * (1.0 + 0.001 * vb))
    coarse = coarse_fft_subbank(bank, 12.0)
    max_vb = max_vsini_variants_per_atmosphere(bank)
    top_k = config.FFT_COARSE_TOP_K
    n_tot = len(bank)
    n_coarse = len(coarse)
    est = n_coarse + top_k * max_vb
    assert max_vb == 3
    assert n_coarse == 20
    assert n_tot == 60
    assert n_coarse > top_k
    assert est < n_tot
