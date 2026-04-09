import pytest
import numpy as np

from darkhunter_rv import qc


@pytest.mark.validation
def test_telluric_fraction_high_inside_band():
    w = np.linspace(7600, 7680, 100)
    f = qc.telluric_fraction(w)
    assert f > 0.8


@pytest.mark.validation
def test_mask_line_count():
    w = np.linspace(5000, 5100, 100)
    mw = np.array([4990, 5001, 5050, 5099, 5110])
    assert qc.mask_line_count_in_chunk(w, mw) == 3


@pytest.mark.validation
def test_qc_evaluate_flags():
    ok, reason = qc.evaluate_chunk_qc(
        {
            'rv_err_kms': 40,
            'mask_line_count': 2,
            'telluric_fraction': 0.4,
            'ccf_asymmetry': 0.5,
            'ccf_peak': 0.01,
        },
        {
            'max_chunk_err_kms': 25,
            'min_mask_line_count': 10,
            'max_telluric_fraction': 0.2,
            'max_ccf_asymmetry': 0.3,
            'min_ccf_peak': 0.03,
        },
    )
    assert not ok
    assert 'telluric_heavy' in reason
