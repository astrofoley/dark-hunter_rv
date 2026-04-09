"""Tests for exposure-level method validity and adopted-RV cascade."""
from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv import method_evaluation as me


def _row(
    method: str,
    chunk_key: str,
    rv: float,
    err: float,
    qc: bool = True,
    ccf_peak_snr: float = 10.0,
) -> dict:
    return {
        "method": method,
        "chunk_key": chunk_key,
        "rv_kms": rv,
        "rv_err_kms": err,
        "qc_pass": qc,
        "ccf_peak_snr": ccf_peak_snr,
        "template_key": "t1",
    }


def test_mask_stack_requires_min_chunks_and_qc(monkeypatch):
    monkeypatch.setattr("darkhunter_rv.config.MIN_MASK_CCF_CHUNKS_FOR_STACK", 3)
    monkeypatch.setattr("darkhunter_rv.config.MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK", 2)
    # Only two mask chunks -> invalid mask stack
    rows = [
        _row("mask_ccf", "0_a", 1.0, 0.5),
        _row("mask_ccf", "0_b", 1.1, 0.5),
        _row("template_fft", "0_a", 1.0, 0.4),
        _row("template_fft", "0_b", 1.0, 0.4),
    ]
    fl = me.exposure_method_flags(rows)
    assert not fl["mask_valid"]
    assert fl["template_valid"]


def test_strong_lines_require_qc_on_all_row(monkeypatch):
    monkeypatch.setattr("darkhunter_rv.config.MIN_MASK_CCF_CHUNKS_FOR_STACK", 1)
    monkeypatch.setattr("darkhunter_rv.config.MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK", 1)
    rows = [
        _row("mask_ccf", "0_a", 0.0, 1.0),
        _row("template_fft", "0_a", 0.0, 1.0),
        {**_row("strong_lines", "all", 0.3, 0.15, qc=False)},
    ]
    fl = me.exposure_method_flags(rows)
    assert not fl["strong_lines_valid"]


def test_recommend_adopted_rv_cascade_prefers_mask_when_all_applicable():
    # Teff 6000, log10 SNR 0.7 → mask + template + strong regions all OK
    fl = {
        "mask_valid": True,
        "template_valid": True,
        "strong_lines_valid": True,
        "mask_rv_kms": 0.0,
        "mask_err_kms": 0.5,
        "template_rv_kms": 0.1,
        "template_err_kms": 0.2,
        "strong_lines_rv_kms": 0.0,
        "strong_lines_err_kms": 0.12,
        "median_mask_ccf_peak_snr": 10.0,
    }
    ad = me.recommend_adopted_rv(
        fl,
        teff=6000.0,
        log10_median_mask_ccf_peak_snr=0.7,
        max_sigma_kms=2.5,
    )
    assert ad["adopted_method"] == "mask_ccf"
    assert ad["adopted_err_kms"] == pytest.approx(0.5)


def test_recommend_adopted_rv_mask_high_sigma_falls_through_to_template():
    fl = {
        "mask_valid": True,
        "template_valid": True,
        "strong_lines_valid": True,
        "mask_rv_kms": 0.0,
        "mask_err_kms": 5.0,
        "template_rv_kms": 0.1,
        "template_err_kms": 0.2,
        "strong_lines_rv_kms": 0.0,
        "strong_lines_err_kms": 0.12,
        "median_mask_ccf_peak_snr": 10.0,
    }
    ad = me.recommend_adopted_rv(
        fl,
        teff=6000.0,
        log10_median_mask_ccf_peak_snr=0.7,
        max_sigma_kms=2.5,
    )
    assert ad["adopted_method"] == "template_fft"
    assert ad["adopted_err_kms"] == pytest.approx(0.2)


def test_recommend_adopted_rv_uses_loose_first_applicable_when_all_sigma_large():
    fl = {
        "mask_valid": True,
        "template_valid": True,
        "strong_lines_valid": True,
        "mask_rv_kms": 0.0,
        "mask_err_kms": 5.0,
        "template_rv_kms": 0.1,
        "template_err_kms": 4.0,
        "strong_lines_rv_kms": 0.0,
        "strong_lines_err_kms": 3.0,
        "median_mask_ccf_peak_snr": 10.0,
    }
    ad = me.recommend_adopted_rv(
        fl,
        teff=6000.0,
        log10_median_mask_ccf_peak_snr=0.7,
        max_sigma_kms=2.5,
    )
    assert ad["adopted_method"] == "mask_ccf"
    assert ad["adopted_err_kms"] == pytest.approx(5.0)


def test_flags_with_method_offsets_shifts_template_and_strong():
    fl = {
        "mask_valid": True,
        "template_valid": True,
        "strong_lines_valid": True,
        "mask_rv_kms": 0.0,
        "mask_err_kms": 0.5,
        "template_rv_kms": 1.0,
        "template_err_kms": 0.2,
        "strong_lines_rv_kms": 2.0,
        "strong_lines_err_kms": 0.3,
        "median_mask_ccf_peak_snr": 10.0,
    }
    row = {"offset_template_fft_kms": -0.5, "offset_strong_lines_kms": -1.0}
    out = me.flags_with_method_offsets(fl, row)
    assert out["template_rv_kms"] == pytest.approx(0.5)
    assert out["strong_lines_rv_kms"] == pytest.approx(1.0)


def test_median_mask_ccf_peak_snr():
    rows = [
        _row("mask_ccf", "0_a", 0.0, 1.0, ccf_peak_snr=100.0),
        _row("mask_ccf", "0_b", 0.0, 1.0, ccf_peak_snr=10.0),
        _row("template_fft", "0_a", 0.0, 1.0, ccf_peak_snr=999.0),
    ]
    assert me.median_mask_ccf_peak_snr(rows) == pytest.approx(55.0)
