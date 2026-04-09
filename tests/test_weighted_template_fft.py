"""Exposure-level template_fft aggregation: QC filtering and minimum chunk count."""

import numpy as np

from darkhunter_rv import config
from darkhunter_rv.pipeline import (
    _mask_ccf_stack_error_inflated,
    _weighted_method_rv_from_rows,
    _weighted_template_fft_for_order,
)


def _mask_row(chunk_key: str, rv: float, err: float, *, qc_pass: bool = True) -> dict:
    return {
        "chunk_key": chunk_key,
        "method": "mask_ccf",
        "rv_kms": rv,
        "rv_err_kms": err,
        "qc_pass": qc_pass,
        "qc_reason": "ok" if qc_pass else "test_fail",
    }


def _tpl_row(chunk_key: str, rv: float, err: float, *, qc_pass: bool = True) -> dict:
    return {
        "chunk_key": chunk_key,
        "method": "template_fft",
        "rv_kms": rv,
        "rv_err_kms": err,
        "qc_pass": qc_pass,
        "qc_reason": "ok" if qc_pass else "test_fail",
    }


def test_template_fft_weighted_mean_requires_min_good_chunks():
    rows = [_tpl_row("1", 10.0, 2.0), _tpl_row("2", 12.0, 2.0)]
    rv, er = _weighted_method_rv_from_rows(rows, "template_fft")
    assert not np.isfinite(rv) and not np.isfinite(er)

    need = int(config.MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK)
    rows_ok = [_tpl_row(str(i), 10.0 + i, 2.0) for i in range(need)]
    rv, er = _weighted_method_rv_from_rows(rows_ok, "template_fft")
    assert np.isfinite(rv) and np.isfinite(er) and er > 0


def test_template_fft_skips_qc_fail_rows():
    need = int(config.MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK)
    rows = [_tpl_row(str(i), 10.0, 2.0) for i in range(need)]
    rows.append(_tpl_row("bad", 999.0, 1.0, qc_pass=False))
    rv, er = _weighted_method_rv_from_rows(rows, "template_fft")
    assert np.isfinite(rv)
    assert abs(rv - 10.0) < 2.0


def test_mask_ccf_weighted_mean_requires_min_good_chunks():
    need = int(config.MIN_MASK_CCF_CHUNKS_FOR_STACK)
    rows_short = [_mask_row(str(i), 10.0, 0.5) for i in range(max(1, need - 1))]
    rv, er = _weighted_method_rv_from_rows(rows_short, "mask_ccf")
    assert not np.isfinite(rv) and not np.isfinite(er)

    rows_ok = [_mask_row(str(i), 10.0, 0.5) for i in range(need)]
    rv, er = _weighted_method_rv_from_rows(rows_ok, "mask_ccf")
    assert np.isfinite(rv) and np.isfinite(er) and er > 0


def test_mask_ccf_inflates_combined_err_when_chunks_disagree():
    """Formal Gaussian errors tiny but RVs spread: combined sigma must not be ~0."""
    need = int(config.MIN_MASK_CCF_CHUNKS_FOR_STACK)
    rvs = np.linspace(10.0, 18.0, need)
    rows = [_mask_row(str(i), float(rvs[i]), 0.02) for i in range(need)]
    rv, er = _weighted_method_rv_from_rows(rows, "mask_ccf")
    assert np.isfinite(rv) and np.isfinite(er)
    assert er > 0.4, f"expected inflation above formal IVAR error, got er={er}"


def test_mask_ccf_stack_inflation_helper():
    rv = np.array([10.0, 14.0, 12.0], float)
    er = np.array([0.01, 0.01, 0.01], float)
    w = 1.0 / (er**2)
    mu = float(np.sum(w * rv) / np.sum(w))
    formal = float(np.sqrt(1.0 / np.sum(1.0 / (er**2))))
    out = _mask_ccf_stack_error_inflated(rv, er, formal, mu_weighted=mu)
    assert out > formal * 5.0
    rms_term = float(np.sqrt(np.mean((rv - mu) ** 2)) / np.sqrt(len(rv)))
    assert out >= rms_term - 1e-9


def test_weighted_template_fft_for_order_ignores_min_chunk_rule():
    """Per-order stack for Hβ plot: two chunks on one order still yield a finite RV."""
    rows = [_tpl_row("28_0", 10.0, 2.0), _tpl_row("28_1", 14.0, 2.0), _tpl_row("29_0", 100.0, 2.0)]
    rv, er = _weighted_template_fft_for_order(rows, 28)
    assert np.isfinite(rv) and np.isfinite(er) and er > 0
    assert 10.0 < rv < 14.0
