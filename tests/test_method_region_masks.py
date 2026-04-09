"""Applicability regions (package API; also used by validation reports)."""

import numpy as np

from darkhunter_rv.method_regions import (
    region_mask_applicable as _region_mask_applicable,
    region_strong_lines_applicable as _region_strong_lines_applicable,
    region_template_applicable as _region_template_applicable,
)


def test_mask_region_cool_star_no_snr_required():
    teff = np.array([5000.0])
    log10_snr = np.array([float("nan")])
    assert np.all(_region_mask_applicable(teff, log10_snr))


def test_mask_region_warm_requires_snr():
    teff = np.array([6000.0, 6000.0])
    s = np.array([0.5, 0.7])
    m = _region_mask_applicable(teff, s)
    assert not bool(m[0]) and bool(m[1])


def test_strong_lines_region():
    teff = np.array([5400.0, 5600.0])
    s = np.array([0.7, 0.7])
    m = _region_strong_lines_applicable(teff, s)
    assert not bool(m[0]) and bool(m[1])


def test_template_region_finite_teff():
    teff = np.array([3000.0, float("nan")])
    s = np.array([float("nan"), 1.0])
    m = _region_template_applicable(teff, s)
    assert bool(m[0]) and not bool(m[1])
